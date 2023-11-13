import gymnasium
import torch
from torch import nn
from torch.nn import functional as F

# env = gymnasium.make('ALE/Breakout-v5')

class DoubleDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DoubleDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

def create_model(env: gymnasium.Env):
    return DoubleDQN(env.observation_space.shape, env.action_space.n)