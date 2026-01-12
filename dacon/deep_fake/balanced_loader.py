import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models import resnet18
import torchvision.transforms as T


FAKE_TYPE_RATIOS = {
    "Deepfakes": 0.40,
    "Face2Face": 0.30,
    "FaceShifter": 0.30,
}


@dataclass(frozen=True)
class SampleItem:
    path: Path
    label: int
    fake_type: Optional[str]


def extract_frames(video_path: Path, max_frames: int, frame_stride: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    frames: List[Image.Image] = []
    frame_idx = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        frame_idx += 1
    cap.release()
    return frames


class VideoDataset(Dataset):
    def __init__(
        self,
        items: List[SampleItem],
        transform: Optional[T.Compose] = None,
        max_frames: int = 10,
        frame_stride: int = 10,
    ) -> None:
        self.items = items
        self.transform = transform
        self.max_frames = max_frames
        self.frame_stride = frame_stride

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        item = self.items[index]
        frames = extract_frames(item.path, self.max_frames, self.frame_stride)
        if not frames:
            frames = [Image.new("RGB", (224, 224), color=(0, 0, 0))]
        if self.transform is not None:
            frames = [self.transform(frame) for frame in frames]
        return torch.stack(frames), item.label


class RatioBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        items: List[SampleItem],
        batch_size: int,
        real_fraction: float,
        fake_type_ratios: Dict[str, float],
        seed: int,
    ) -> None:
        if not 0.0 <= real_fraction <= 1.0:
            raise ValueError("real_fraction must be between 0 and 1.")
        self.items = items
        self.batch_size = batch_size
        self.real_fraction = real_fraction
        self.fake_type_ratios = _normalize_ratios(fake_type_ratios)
        self.seed = seed
        self.num_batches = max(1, len(items) // batch_size)

        self.real_indices = [i for i, item in enumerate(items) if item.label == 0]
        self.fake_indices: Dict[str, List[int]] = {}
        for fake_type in self.fake_type_ratios:
            self.fake_indices[fake_type] = [
                i for i, item in enumerate(items) if item.fake_type == fake_type
            ]

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(self.seed)
        real_count = int(round(self.batch_size * self.real_fraction))
        fake_count = self.batch_size - real_count
        fake_counts = _allocate_fake_counts(fake_count, self.fake_type_ratios)

        for _ in range(self.num_batches):
            batch: List[int] = []
            if real_count > 0:
                batch.extend(_sample_indices(self.real_indices, real_count, rng))
            for fake_type, count in fake_counts.items():
                if count <= 0:
                    continue
                batch.extend(_sample_indices(self.fake_indices[fake_type], count, rng))
            rng.shuffle(batch)
            yield batch


class VideoClassifier(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        backbone = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = backbone
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, frames, channels, height, width = x.shape
        x = x.view(batch_size * frames, channels, height, width)
        feats = self.backbone(x)
        feats = feats.view(batch_size, frames, -1).mean(dim=1)
        return self.classifier(feats)


def _normalize_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("fake_type_ratios sum must be positive.")
    return {k: v / total for k, v in ratios.items()}


def _allocate_fake_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    if total <= 0:
        return {k: 0 for k in ratios}
    raw = {k: total * v for k, v in ratios.items()}
    counts = {k: int(raw[k]) for k in ratios}
    remaining = total - sum(counts.values())
    if remaining > 0:
        sorted_keys = sorted(raw.keys(), key=lambda k: raw[k] - counts[k], reverse=True)
        for key in sorted_keys[:remaining]:
            counts[key] += 1
    return counts


def _sample_indices(pool: List[int], count: int, rng: random.Random) -> List[int]:
    if not pool:
        return []
    if count <= len(pool):
        return rng.sample(pool, count)
    return rng.choices(pool, k=count)


def build_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collect_items(data_root: Path) -> List[SampleItem]:
    items: List[SampleItem] = []
    real_dir = data_root / "real" / "original"
    for path in sorted(real_dir.glob("*.mp4")):
        items.append(SampleItem(path=path, label=0, fake_type=None))

    fake_dir = data_root / "fake"
    for fake_type in FAKE_TYPE_RATIOS:
        type_dir = fake_dir / fake_type
        for path in sorted(type_dir.glob("*.mp4")):
            items.append(SampleItem(path=path, label=1, fake_type=fake_type))
    return items


def split_items(
    items: List[SampleItem],
    val_ratio: float,
    seed: int,
) -> Tuple[List[SampleItem], List[SampleItem]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[SampleItem]] = {
        "real": [],
        **{fake_type: [] for fake_type in FAKE_TYPE_RATIOS},
    }
    for item in items:
        key = "real" if item.label == 0 else item.fake_type
        grouped[key].append(item)

    train_items: List[SampleItem] = []
    val_items: List[SampleItem] = []
    for group_items in grouped.values():
        rng.shuffle(group_items)
        split_idx = int(round(len(group_items) * (1 - val_ratio)))
        train_items.extend(group_items[:split_idx])
        val_items.extend(group_items[split_idx:])
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    val_ratio: float,
    real_fraction: float,
    max_frames: int,
    frame_stride: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    items = collect_items(data_root)
    train_items, val_items = split_items(items, val_ratio, seed)

    transform = build_transforms()
    train_dataset = VideoDataset(
        train_items, transform=transform, max_frames=max_frames, frame_stride=frame_stride
    )
    val_dataset = VideoDataset(
        val_items, transform=transform, max_frames=max_frames, frame_stride=frame_stride
    )

    train_sampler = RatioBatchSampler(
        train_items,
        batch_size=batch_size,
        real_fraction=real_fraction,
        fake_type_ratios=FAKE_TYPE_RATIOS,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepfake balanced loader")
    parser.add_argument("--data-root", type=Path, default=Path("train_data"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--real-fraction", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-path", type=Path, default=Path("result/deep_fake_model.pt"))
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        logits = model(frames)
        loss = criterion(logits, labels)
        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main() -> None:
    args = parse_args()
    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        real_fraction=args.real_fraction,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = VideoClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch {epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"saved model to {args.save_path}")


if __name__ == "__main__":
    main()
