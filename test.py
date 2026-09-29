"""PyTorch 入门：使用 CNN 完成 MNIST 手写数字分类。

建议先运行快速模式：
    python test.py --quick

完整训练：
    python test.py --epochs 5

依赖安装（请根据你的 CUDA 环境选择合适的 PyTorch 版本）：
    pip install torch torchvision
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed: int = 42) -> None:
    """固定随机种子，使多次运行的结果尽量一致。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    """一个小型卷积神经网络。

    nn.Module 是所有 PyTorch 模型的基类：
    - __init__ 中定义“有哪些层”（带参数的层会被自动注册）。
    - forward 中定义“数据怎样流过这些层”。
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # 输入形状：[batch, 1, 28, 28]
            # Conv2d 参数：输入通道、输出通道、卷积核大小、padding。
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # 高和宽减半：28x28 -> 14x14
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # 14x14 -> 7x7
            nn.MaxPool2d(kernel_size=2),
        )
        #这是一个全连接层的分类器，将卷积特征映射到 10 个类别。   
        self.classifier = nn.Sequential(
            # 把 [batch, 32, 7, 7] 展平为 [batch, 32*7*7]。
            # 从第 1 维开始展平，保留第 0 维 batch。
            nn.Flatten(start_dim=1),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            # 训练时随机将部分神经元置零，缓解过拟合。
            nn.Dropout(p=0.3),
            # 10 个输出分别对应数字 0~9；输出是 logits，不需要手动 softmax。
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def make_dataloaders(
    data_dir: Path, batch_size: int, quick: bool
) -> tuple[DataLoader, DataLoader]:
    """下载数据，并用 DataLoader 按批次提供样本。"""
    transform = transforms.Compose(
        [
            # PIL 图片 -> Tensor，形状为 [C, H, W]，像素缩放到 [0, 1]。
            transforms.ToTensor(),
            # 使用 MNIST 的常用均值和标准差做标准化，有助于模型训练。
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    if quick:
        # 学习/调试时只取少量数据；这不是为了得到最佳准确率。
        train_dataset = Subset(train_dataset, range(2_000))
        test_dataset = Subset(test_dataset, range(1_000))

    # shuffle=True：每轮训练前打乱样本，避免模型记住数据顺序。
    # Windows 下 num_workers=0 最省心；后续可自行调大来加速数据读取。
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """完成一轮训练，返回平均损失和准确率。"""
    model.train()  # 开启训练模式：Dropout 生效。
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        # 将数据搬到与模型相同的设备（CPU 或 GPU）。
        images, labels = images.to(device), labels.to(device)

        # PyTorch 默认累积梯度，因此每个 batch 都要先清空旧梯度。
        optimizer.zero_grad()

        logits = model(images)          # 前向传播
        loss = loss_fn(logits, labels)  # 计算损失
        loss.backward()                 # 自动微分：计算每个参数的梯度
        optimizer.step()                # 根据梯度更新参数

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """在测试集评估；inference_mode 会关闭梯度以节省内存和计算量。"""
    model.eval()  # 评估模式：关闭 Dropout 等训练专用行为。
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 CNN 训练 MNIST 分类器")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--quick", action="store_true", help="只用少量数据快速试跑")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()

    # CUDA 是 NVIDIA GPU；没有可用 GPU 时自动回退到 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    train_loader, test_loader = make_dataloaders(
        Path("data"), args.batch_size, args.quick
    )
    model = SimpleCNN().to(device)
    print(model)

    # CrossEntropyLoss 内部已包含 LogSoftmax，因此模型末尾不要再加 Softmax。
    loss_fn = nn.CrossEntropyLoss()
    # optimizer 持有 model.parameters()，step() 时会更新这些可学习参数。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"训练 loss={train_loss:.4f}, acc={train_acc:.2%} | "
            f"测试 loss={test_loss:.4f}, acc={test_acc:.2%}"
        )

    # 推荐保存 state_dict（参数字典），而不是直接保存整个模型对象。
    save_path = Path("mnist_cnn_state_dict.pt")
    torch.save(model.state_dict(), save_path)
    print(f"模型参数已保存到：{save_path.resolve()}")

    # 加载方法：先创建相同结构的模型，再读取参数。
    loaded_model = SimpleCNN().to(device)
    loaded_model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    loaded_model.eval()

    # 展示一次真实推理：取一个 batch 中的第一张图片进行预测。
    images, labels = next(iter(test_loader))
    with torch.inference_mode():
        prediction = loaded_model(images[:1].to(device)).argmax(dim=1).item()
    print(f"推理示例：真实数字={labels[0].item()}，模型预测={prediction}")


if __name__ == "__main__":
    # Windows 使用 DataLoader 时，把入口放在此判断内是一个重要习惯。
    main()
