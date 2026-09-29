# MNIST 手写数字识别（PyTorch CNN）

用一个小型卷积神经网络完成 MNIST 手写数字分类，代码带详细的中文注释，适合作为 PyTorch 入门示例。

## 功能

- `test.py`：定义网络结构并完成训练，保存模型参数
- `validate_model.py`：加载参数，输出测试集准确率、各类别准确率、混淆矩阵；也可预测你自己手写的数字图片

## 环境要求

- Python 3.10+
- PyTorch 2.x（按你的 CUDA 环境选择版本）

```bash
pip install torch torchvision pillow
```

CPU 环境直接安装即可，脚本会自动检测 GPU：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA 版本按需调整
```

## 快速开始

```powershell
# 1. 快速试跑（只用少量数据，几秒到几十秒）
python test.py --quick

# 2. 完整训练（默认 3 轮）
python test.py --epochs 5

# 3. 验证模型（整个 10000 张测试集）
python validate_model.py
```

训练完成后会生成 `mnist_cnn_state_dict.pt`，**首次运行会自动下载 MNIST 数据集**到 `data/` 目录（约 11 MB）。

### 训练参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | 3 | 训练轮数 |
| `--batch-size` | 64 | 批大小 |
| `--learning-rate` | 1e-3 | 学习率（Adam） |
| `--quick` | 关闭 | 只取 2000 训练 / 1000 测试样本，用于调试 |

### 验证参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--weights` | `mnist_cnn_state_dict.pt` | 模型参数文件 |
| `--data-dir` | `data` | 数据集目录 |
| `--batch-size` | 256 | 批大小 |
| `--image` | 无 | 预测一张本地图片 |
| `--invert` | 关闭 | 图片颜色反转（适合白底黑字的照片） |

### 预测你手写的数字

```powershell
python validate_model.py --image my_digit.png --invert
```

输出最可能的三个数字及置信度。注意：普通纸张照片是白底黑字，需要加 `--invert`。

## 网络结构

```
输入 [batch, 1, 28, 28]
  ├─ Conv2d(1→16, 3x3, pad=1) → ReLU → MaxPool2d(2)    # 28×28 → 14×14
  ├─ Conv2d(16→32, 3x3, pad=1) → ReLU → MaxPool2d(2)   # 14×14 → 7×7
  ├─ Flatten                                          # 32×7×7 = 1568
  ├─ Linear(1568→128) → ReLU → Dropout(0.3)
  └─ Linear(128→10)                                   # 输出 logits
```

- 损失函数：`CrossEntropyLoss`（内部已含 Softmax，网络末端不再加激活）
- 优化器：`Adam`
- 评估：`model.eval()` + `torch.inference_mode()` 关闭 Dropout 与梯度

## 项目结构

```
CNN/
├── test.py                     # 网络定义 + 训练流程
├── validate_model.py           # 加载参数并验证
├── mnist_cnn_state_dict.pt     # 训练得到的参数（本地生成，未入库）
├── data/                       # MNIST 数据集（自动下载，未入库）
├── ajupytertest.ipynb          # 环境测试笔记
└── .gitignore
```

## 实现要点

- **固定随机种子**：`set_seed()` 让多次运行结果尽量一致
- **保存 `state_dict`** 而非整个模型对象，加载时先建同结构网络再 `load_state_dict`
- **`validate_model.py` 直接 `from test import SimpleCNN`**，保证网络结构与训练时完全一致，避免重复定义
- **混淆矩阵**用 `bincount` 批量统计，避免逐样本修改
- **Windows 下 `num_workers=0`**，并把入口放在 `if __name__ == "__main__":` 内

## License

本项目代码用于学习交流。
