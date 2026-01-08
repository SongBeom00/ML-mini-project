"""
간단한 에피소드 단위 좌표 예측 베이스라인 스크립트.
- train.csv: 에피소드별 이벤트 로그에서 마지막 이벤트 end_x/end_y를 타깃으로 사용
- test.csv: 각 game_episode의 이벤트 파일 경로를 읽어 동일한 피처화 후 예측
필요 라이브러리: pandas, numpy, scikit-learn
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def load_base(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(base_dir / "train.csv")
    test_index = pd.read_csv(base_dir / "test.csv")
    return train, test_index


def top_categories(train: pd.DataFrame, k: int = 15) -> Tuple[list[str], list[str]]:
    top_types = train["type_name"].value_counts().head(k).index.tolist()
    top_results = (
        train["result_name"].fillna("Unknown").value_counts().head(k).index.tolist()
    )
    return top_types, top_results


def _pivot_counts(df: pd.DataFrame, col: str, top: Iterable[str]) -> pd.DataFrame:
    pivot = (
        df.pivot_table(
            index="game_episode", columns=col, values="action_id", aggfunc="count", fill_value=0
        )
        .reindex(columns=list(top), fill_value=0)
        .add_prefix(f"{col}_cnt_")
    )
    return pivot


def build_features(df: pd.DataFrame, top_types: list[str], top_results: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["result_name_filled"] = df["result_name"].fillna("Unknown")
    df_sorted = df.sort_values(["game_episode", "period_id", "time_seconds"])

    # 기본 수치 집계
    agg = df.groupby("game_episode").agg(
        time_min=("time_seconds", "min"),
        time_max=("time_seconds", "max"),
        time_mean=("time_seconds", "mean"),
        time_std=("time_seconds", "std"),
        period_nuniq=("period_id", "nunique"),
        period_max=("period_id", "max"),
        start_x_mean=("start_x", "mean"),
        start_x_std=("start_x", "std"),
        start_y_mean=("start_y", "mean"),
        start_y_std=("start_y", "std"),
        end_x_mean=("end_x", "mean"),
        end_x_std=("end_x", "std"),
        end_y_mean=("end_y", "mean"),
        end_y_std=("end_y", "std"),
        episode_len=("episode_id", "max"),
        is_home_mean=("is_home", "mean"),
        team_nuniq=("team_id", "nunique"),
        player_nuniq=("player_id", "nunique"),
        n_events=("action_id", "size"),
    )

    type_counts = _pivot_counts(df, "type_name", top_types)
    result_counts = _pivot_counts(df, "result_name_filled", top_results)
    length = agg["n_events"].clip(lower=1)
    type_ratios = type_counts.div(length, axis=0).add_suffix("_ratio")
    result_ratios = result_counts.div(length, axis=0).add_suffix("_ratio")

    # 마지막 이벤트 좌표(타깃 라벨 추출 및 피처)
    last_events = (
        df_sorted.groupby("game_episode")
        .tail(1)
        .set_index("game_episode")[["end_x", "end_y", "start_x", "start_y"]]
        .rename(columns={"end_x": "last_end_x", "end_y": "last_end_y", "start_x": "last_start_x", "start_y": "last_start_y"})
    )

    features = pd.concat(
        [agg, type_counts, type_ratios, result_counts, result_ratios, last_events.drop(columns=["last_end_x", "last_end_y"])],
        axis=1,
    )
    return features, last_events[["last_end_x", "last_end_y"]]


def load_test_events(base_dir: Path, index_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in index_df.itertuples(index=False):
        path = Path(row.path.lstrip("./"))
        df = pd.read_csv(base_dir / path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def train_and_validate(X: pd.DataFrame, y: pd.DataFrame, random_state: int = 42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model_x = RandomForestRegressor(
        n_estimators=400, max_depth=None, random_state=random_state, n_jobs=-1
    )
    model_y = RandomForestRegressor(
        n_estimators=400, max_depth=None, random_state=random_state, n_jobs=-1
    )

    model_x.fit(X_train, y_train["last_end_x"])
    model_y.fit(X_train, y_train["last_end_y"])

    pred_x = model_x.predict(X_val)
    pred_y = model_y.predict(X_val)
    rmse_x = mean_squared_error(y_val["last_end_x"], pred_x) ** 0.5
    rmse_y = mean_squared_error(y_val["last_end_y"], pred_y) ** 0.5
    return model_x, model_y, (rmse_x, rmse_y)


def main(base_dir: Path):
    train_df, test_index = load_base(base_dir)
    top_types, top_results = top_categories(train_df)

    train_features, train_targets = build_features(train_df, top_types, top_results)
    # 검증
    model_x, model_y, (rmse_x, rmse_y) = train_and_validate(train_features, train_targets)
    print(f"Validation RMSE -> end_x: {rmse_x:.3f}, end_y: {rmse_y:.3f}")

    # 전체 학습
    model_x.fit(train_features, train_targets["last_end_x"])
    model_y.fit(train_features, train_targets["last_end_y"])

    # 테스트 피처 생성
    test_events = load_test_events(base_dir, test_index)
    test_features, _ = build_features(test_events, top_types, top_results)

    test_pred_x = model_x.predict(test_features)
    test_pred_y = model_y.predict(test_features)
    submission = pd.DataFrame(
        {
            "game_episode": test_features.index,
            "end_x": test_pred_x,
            "end_y": test_pred_y,
        }
    ).reset_index(drop=True)
    out_path = base_dir / "submission_baseline.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("./data/open_track1"),
        help="train.csv/test.csv가 있는 디렉터리 경로",
    )
    args = parser.parse_args()
    main(args.base_dir)
