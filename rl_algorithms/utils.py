"""强化学习通用工具：经验回放缓冲区、网络基类等"""

import random
from collections import deque, namedtuple
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# 经验回放缓冲区（适用于 DQN / SAC / TD3）
# ──────────────────────────────────────────────

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """固定大小的经验回放缓冲区"""

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(np.array(actions), dtype=torch.float32),
            torch.as_tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.array(next_states), dtype=torch.float32),
            torch.as_tensor(np.array(dones), dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ──────────────────────────────────────────────
# 网络构建工具
# ──────────────────────────────────────────────

def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: nn.Module = nn.ReLU,
    output_activation: Optional[nn.Module] = None,
    init_weight: bool = True,
) -> nn.Sequential:
    """构建多层感知机"""
    layers = []
    dims = [input_dim] + hidden_dims
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-1], output_dim))
    if output_activation is not None:
        layers.append(output_activation())

    net = nn.Sequential(*layers)

    if init_weight:
        _init_weights(net)
    return net


def _init_weights(module: nn.Module, gain: float = np.sqrt(2)):
    """正交初始化"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ──────────────────────────────────────────────
# 训练辅助
# ──────────────────────────────────────────────

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Polyak 软更新目标网络"""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def hard_update(target: nn.Module, source: nn.Module):
    """硬拷贝：目标网络 ← 源网络"""
    target.load_state_dict(source.state_dict())


def explain_action(action: np.ndarray, env_type: str = "discrete") -> str:
    """对智能体决策进行简单解释"""
    if env_type == "discrete":
        mapping = {0: "向左", 1: "不动/无操作", 2: "向右"}
        return f"选择动作 {int(action)} ({mapping.get(int(action), '未知')})"
    else:
        return f"输出连续动作: {np.round(action, 4)}"


# ──────────────────────────────────────────────
# 环境包装 —— 统一 Gymnasium 接口
# ──────────────────────────────────────────────

def make_env(env_id: str, seed: Optional[int] = None):
    """创建 Gymnasium 环境并设置种子"""
    import gymnasium as gym

    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env
