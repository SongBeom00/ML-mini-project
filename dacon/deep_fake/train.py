import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 진행도 표시를 위해 추가 권장
from dacon.deep_fake.src.dataset import IntegratedDataset
from dacon.deep_fake.src.models import DeepfakeClassifier


def train():
    os.chdir('/Users/songbeom/PythonWorkSpace/ML_Project/dacon/deep_fake')
    print('현재 작업 디렉토리: ', os.getcwd())
    with open("./config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 2. 데이터 ID 기반 리스트 생성 (이미 잘 구현된 부분입니다!)
    real_path = os.path.join(cfg['data']['train_root'], 'real/original')
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {real_path}")

    video_ids = [f.split('.')[0] for f in os.listdir(real_path) if f.endswith('.mp4')]
    train_ids, val_ids = train_test_split(video_ids, test_size=0.2, random_state=42)

    def get_paths_labels(ids):
        paths, labels = [], []
        for vid in ids:
            # Real 추가
            paths.append(os.path.join(real_path, f"{vid}.mp4"))
            labels.append(0)
            # Fake(Deepfakes, Face2Face 등) 하위 폴더 탐색
            fake_base = os.path.join(cfg['data']['train_root'], 'fake')
            if os.path.exists(fake_base):
                for method in os.listdir(fake_base):
                    f_path = os.path.join(fake_base, method, f"{vid}.mp4")
                    if os.path.exists(f_path):
                        paths.append(f_path)
                        labels.append(1)
        return paths, labels

    t_paths, t_labels = get_paths_labels(train_ids)
    v_paths, v_labels = get_paths_labels(val_ids)

    print(f'학습 데이터 수: {len(t_paths)} | 검증 데이터 수: {len(v_paths)}')
    print(f'학습/검증 데이터 중복 확인: {len(set(t_paths).intersection(v_paths))}개')

    # 3. 데이터 로더 설정
    train_ds = IntegratedDataset(t_paths, t_labels, target_size=cfg['data']['img_size'])
    val_ds = IntegratedDataset(v_paths, v_labels, target_size=cfg['data']['img_size'])

    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False)

    # 4. 모델 및 도구 설정 (Mac MPS 및 CUDA 대응)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"사용 중인 장치: {device}")

    model = DeepfakeClassifier(cfg['model']['name']).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])
    # 학습률을 점진적으로 낮춰주는 스케줄러 추가
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    # 5. 학습 루프
    for epoch in range(cfg['train']['epochs']):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{cfg['train']['epochs']}]")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # 6. 검증 로직
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / len(val_ds)
        print(f"\n[Epoch {epoch + 1}] Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.4f}")

        # 스케줄러 단계 업데이트
        scheduler.step()

        # 7. Best Model 저장 (규칙 준수)
        if accuracy >= best_val_acc:
            best_val_acc = accuracy
            os.makedirs(os.path.dirname(cfg['model']['save_path']), exist_ok=True)
            torch.save(model.state_dict(), cfg['model']['save_path'])
            print(f"--> Best Model Saved with Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()