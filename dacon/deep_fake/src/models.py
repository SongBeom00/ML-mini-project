import torch
import torch.nn as nn
from torchvision import models


class DeepfakeClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0'):
        super(DeepfakeClassifier, self).__init__()

        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

            num_features = self.backbone.classifier[1].in_features # EfficientNet-B0의 마지막 레이어 특징 수
            self.backbone.classifier[1] = nn.Sequential(
                nn.Linear(num_features, 1),
                nn.Sigmoid() # 이진 분류를 위한 시그모이드 활성화 함수 추가
            )

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    model = DeepfakeClassifier()
    print(model)