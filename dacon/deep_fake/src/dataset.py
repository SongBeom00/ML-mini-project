import cv2
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from dacon.deep_fake.src.preprocess import Preprocessor


class IntegratedDataset(Dataset):
    def __init__(self, file_paths, labels, target_size=(224, 224)):
        """
        통합 데이터셋 클래스
        file_paths: 이미지 파일 경로 리스트
        labels: 각 이미지에 대한 레이블 리스트
        transform: 이미지 전처리 함수
        """
        self.file_paths = file_paths
        self.labels = labels
        self.preprocessor = Preprocessor(target_size=target_size)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        ext = os.path.splitext(file_path)[-1].lower() # 파일 확장자 추출 및 소문자 변환

        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # 동영상은 랜덤하게 1프레임만 추출 (학습 효율 증대)
            cap = cv2.VideoCapture(file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(0, total_frames))
            ret, frame = cap.read()
            cap.release()
            if not ret: frame = np.zeros((224, 224, 3), dtype=np.uint8)

        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.jfif']:
            # 이미지 파일인 경우
            frame = cv2.imread(file_path)
            if frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)

        else:
            raise RuntimeError(f"지원하지 않는 파일 형식입니다: {file_path}")

        image_tensor = self.preprocessor(frame)
        return image_tensor, torch.tensor(label, dtype=torch.float32)