"""加载训练好的 MNIST CNN 参数并验证模型。

直接运行（验证整个 MNIST 测试集）：
    python validate_model.py

验证自己的手写数字图片：
    python validate_model.py --image my_digit.png

如果图片是白底黑字，增加 --invert：
    python validate_model.py --image my_digit.png --invert
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 网络结构必须与训练时完全一致。直接导入可以避免重复定义。
# test.py 中的训练代码位于 if __name__ == "__main__" 下，所以导入时不会重新训练。
from test import SimpleCNN


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def load_model(weights_path: Path, device: torch.device) -> SimpleCNN:
    """创建网络并加载 state_dict 参数。"""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"找不到参数文件：{weights_path}\n请先运行 python test.py 完成训练。"
        )

    model = SimpleCNN().to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # eval() 会关闭 Dropout。忘记调用它会导致同一图片的预测结果可能发生变化。
    model.eval()
    return model


@torch.inference_mode()
def validate_test_set(
    model: SimpleCNN, data_dir: Path, batch_size: int, device: torch.device
) -> None:
    """验证全部 10,000 张测试图片，并统计整体及各类别准确率。"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # confusion[真实类别, 预测类别] 表示对应的样本数量。
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    total_loss = 0.0
    sample_predictions: list[tuple[int, int, float]] = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)
        total_loss += loss_fn(logits, labels).item()

        # bincount 一次性统计一个 batch，避免逐个样本修改矩阵。
        indices = (labels * 10 + predictions).cpu()
        confusion += torch.bincount(indices, minlength=100).reshape(10, 10)

        # 保存前 10 个预测，便于直观看到“标签、预测、置信度”。
        remaining = 10 - len(sample_predictions)
        if remaining > 0:
            sample_predictions.extend(
                zip(
                    labels[:remaining].cpu().tolist(),
                    predictions[:remaining].cpu().tolist(),
                    confidences[:remaining].cpu().tolist(),
                )
            )

    correct = confusion.diag().sum().item()
    total = confusion.sum().item()
    print(f"\n测试集样本数：{total}")
    print(f"平均损失：{total_loss / total:.4f}")
    print(f"整体准确率：{correct / total:.2%} ({correct}/{total})")

    print("\n各数字准确率：")
    for digit in range(10):
        class_total = confusion[digit].sum().item()
        class_correct = confusion[digit, digit].item()
        print(
            f"  数字 {digit}: {class_correct / class_total:6.2%} "
            f"({class_correct}/{class_total})"
        )

    print("\n前 10 个样本的预测：")
    for index, (label, prediction, confidence) in enumerate(sample_predictions, start=1):
        mark = "√" if label == prediction else "×"
        print(
            f"  {index:02d}. 真实={label}，预测={prediction}，"
            f"置信度={confidence:.2%}  {mark}"
        )

    print("\n混淆矩阵（行=真实数字，列=预测数字）：")
    print(confusion.numpy())


@torch.inference_mode()
def predict_image(
    model: SimpleCNN, image_path: Path, invert: bool, device: torch.device
) -> None:
    """预测一张本地图片，并输出最可能的三个数字。"""
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片：{image_path}")

    steps: list[object] = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
    # MNIST 是黑底白字；普通纸张照片往往是白底黑字，需要反色。
    if invert:
        steps.append(transforms.Lambda(lambda tensor: 1.0 - tensor))
    steps.append(transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    transform = transforms.Compose(steps)

    image = Image.open(image_path)
    # unsqueeze(0) 添加 batch 维：[1, 28, 28] -> [1, 1, 28, 28]。
    image_tensor = transform(image).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(image_tensor), dim=1)[0]
    top_probabilities, top_classes = probabilities.topk(3)

    print(f"\n图片：{image_path.resolve()}")
    print("最可能的三个结果：")
    for digit, probability in zip(top_classes.tolist(), top_probabilities.tolist()):
        print(f"  数字 {digit}: {probability:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="加载并验证训练好的 MNIST CNN")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("mnist_cnn_state_dict.pt"),
        help="模型参数文件路径",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image", type=Path, help="可选：预测一张自己的数字图片")
    parser.add_argument(
        "--invert", action="store_true", help="将图片颜色反转，适合白底黑字图片"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print(f"加载参数：{args.weights.resolve()}")

    model = load_model(args.weights, device)
    validate_test_set(model, args.data_dir, args.batch_size, device)

    if args.image is not None:
        predict_image(model, args.image, args.invert, device)


if __name__ == "__main__":
    main()
