import torch
import torchinfo
from torch import nn
from torchinfo import summary

class LeNet(nn.Module):
    """支持不同棋盘尺寸的策略—价值网络，保留原来的类名。"""

    def __init__(self):
        super().__init__()
        # 保留每个落点的位置：不缩小棋盘，两个卷积都使用 padding=1。
        self.relu = nn.ReLU()
        self.c1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        # 每个格子产生一个分数，输出的格子数自动跟随输入尺寸。
        self.policy_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Flatten()
        )
        

        # 每个通道聚合为一个数，后面的全连接层只依赖通道数。
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """输入 (B, 2, H, W)，返回 logits (B, H*W)、价值 (B, 1)。

        logits 未做 softmax;选择动作前由调用方屏蔽非法位置。
        同一批次的棋盘尺寸需一致，不同批次可以使用不同尺寸。
        """
        x = self.relu(self.c1(x))
        x = self.relu(self.c2(x))
 
        policy_logits = self.policy_head(x)
        value = self.value_head(self.value_pool(x))
        return policy_logits, value


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet().to(device)
    summary(model,(1,2,15,15))