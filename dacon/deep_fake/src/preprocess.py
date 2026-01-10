import cv2
import torch
from torchvision import transforms
from PIL import Image

class Preprocessor:
    def __init__(self, target_size=(224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, frame):
        """
        OpenCV 프레임(BGR)을 입력받아 전치리된 텐서 변환
        """

        # BGR -> RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Transform 적용
        transform = self.transform(pil_img)
        return transform

def get_video_frames(video_path, sample_rate: int = 10):
    """
    비디어에서 일정 간격으로 프레임을 추출하는 유틸리티
    sample_rate: N 프레임당 1개씩 추출 (추론시간 조절)
    """

    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    if not cap.isOpened():
        raise ValueError(f"해당 비디오 파일을 열 수 없습니다: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_rate == 0:
            frames.append(frame)
        count += 1

    cap.release()
    return frames