"""DQN (Deep Q-Network) — 从零实现

算法要点：
  • 用深度神经网络近似 Q 函数 Q(s,a;θ)
  • 经验回放打破数据相关性
  • 目标网络稳定训练：每 C 步复制 Q → Q_target
  • ε-贪心探索策略
  • 支持 Double DQN（可选）
"""

from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import ReplayBuffer, build_mlp, hard_update, make_env


class QNetwork(nn.Module):
    """Q 网络：状态 → 各动作的 Q 值"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super().__init__()
        self.net = build_mlp(state_dim, hidden_dims, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQN:
    """Deep Q-Network 智能体

    参数
    ----------
    state_dim : int          状态维度
    action_dim : int         动作维度
    lr : float               学习率
    gamma : float            折扣因子
    epsilon_start : float    初始探索率
    epsilon_end : float      最终探索率
    epsilon_decay : float    探索率衰减步数
    buffer_size : int        回放缓冲区容量
    batch_size : int         批次大小
    target_update : int      目标网络更新间隔（步数）
    double_dqn : bool        是否使用 Double DQN
    hidden_dims : list       隐藏层神经元数
    device : str             运行设备
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10_000,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update: int = 100,
        double_dqn: bool = True,
        hidden_dims: list = [128, 128],
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.double_dqn = double_dqn
        self.device = torch.device(device)
        self._steps = 0

        # 在线网络 & 目标网络
        self.q = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        hard_update(self.q_target, self.q)

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """ε-贪心选择动作"""
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q(state_t)
        return q_values.argmax().item()

    def _decay_epsilon(self):
        """线性衰减 ε"""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> dict:
        """从经验回放中采样一个批次进行更新"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        actions = actions.long().squeeze(1)

        # 当前 Q 值
        q_values = self.q(states).gather(1, actions.unsqueeze(1))

        # 目标 Q 值
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: 用在线网络选动作，目标网络评估
                next_actions = self.q(next_states).argmax(dim=1, keepdim=True)
                next_q = self.q_target(next_states).gather(1, next_actions)
            else:
                next_q = self.q_target(next_states).max(dim=1, keepdim=True).values
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._steps += 1
        self._decay_epsilon()

        # 硬更新目标网络
        if self._steps % self.target_update == 0:
            hard_update(self.q_target, self.q)

        return {"loss": loss.item(), "epsilon": self.epsilon, "q_value": q_values.mean().item()}

    def save(self, path: str):
        torch.save(self.q.state_dict(), f"{path}_q.pt")
        torch.save(self.q_target.state_dict(), f"{path}_q_target.pt")

    def load(self, path: str):
        self.q.load_state_dict(torch.load(f"{path}_q.pt", map_location=self.device))
        self.q_target.load_state_dict(torch.load(f"{path}_q_target.pt", map_location=self.device))


# ──────────────────────────────────────────────
# 训练入口
# ──────────────────────────────────────────────

def train_dqn(
    env_id: str = "CartPole-v1",
    num_episodes: int = 500,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_decay: int = 10_000,
    batch_size: int = 64,
    target_update: int = 100,
    double_dqn: bool = True,
    hidden_dims: list = [128, 128],
    seed: Optional[int] = None,
    render: bool = False,
    device: str = "cpu",
) -> dict:
    """训练 DQN 并返回训练记录"""
    env = make_env(env_id, seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        target_update=target_update,
        double_dqn=double_dqn,
        hidden_dims=hidden_dims,
        device=device,
    )

    rewards_log = []
    loss_log = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.push(state, action, reward, next_state, done)
            info = agent.update()
            state = next_state
            ep_reward += reward
            if render:
                env.render()

        rewards_log.append(ep_reward)
        if info and "loss" in info:
            loss_log.append(info["loss"])

        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards_log[-50:]) if rewards_log else 0
            eps = agent.epsilon
            print(f"Ep {ep+1:4d} | Reward: {ep_reward:6.1f} | Avg50: {avg:6.2f} | ε: {eps:.3f}")

    env.close()
    return {"rewards": rewards_log, "losses": loss_log}


if __name__ == "__main__":
    train_dqn(env_id="CartPole-v1", num_episodes=400, double_dqn=True)
