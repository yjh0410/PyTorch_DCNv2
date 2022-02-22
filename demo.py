import torch
import torch.nn as nn
from DCN.dcnv2 import DeformableConv2d as DCN

model = nn.Sequential(
    DCN(3, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    DCN(32, 32, kernel_size=3, stride=1, padding=1),
    DCN(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    DCN(64, 64, kernel_size=3, stride=1, padding=1),
    DCN(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    DCN(128, 128, kernel_size=3, stride=1, padding=1),
    DCN(128, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2)
)

x = torch.randn(2, 3, 64, 64)
y = model(x)
print(y.size())