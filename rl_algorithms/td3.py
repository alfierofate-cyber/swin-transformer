"""TD3 (Twin Delayed DDPG) — 从零实现

算法要点：
  • DDPG 的改进版本，解决过估计问题
  • 双重 Q 网络 (Clipped Double Q)
  • 延迟策略更新（每 2 步 Q 更新才更新 1 步 Actor）
  • 目标策略平滑正则化 (Target Policy Smoothing)
  • 经验回放 + 软更新
  • 适合连续控制任务
"""

from typing import Optional
import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import ReplayBuffer, make_env


def build_mlp_relu(dims: list, output_dim: int) -> nn.Sequential:
    """构建 ReLU MLP"""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """确定性策略网络 μ(s) → 连续动作"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [400, 300],
                 max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        self.net = build_mlp_relu([state_dim] + hidden_dims, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x)) * self.max_action


class Critic(nn.Module):
    """双重 Q 网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [400, 300]):
        super().__init__()
        # Q1: [s, a] → Q 值
        self.q1 = build_mlp_relu([state_dim + action_dim] + hidden_dims, 1)
        # Q2: [s, a] → Q 值
        self.q2 = build_mlp_relu([state_dim + action_dim] + hidden_dims, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class TD3:
    """Twin Delayed DDPG 智能体

    参数
    ----------
    state_dim : int
    action_dim : int
    max_action : float          动作最大值（绝对值）
    lr : float
    gamma : float
    tau : float                 软更新系数
    policy_noise : float        目标策略噪声标准差
    noise_clip : float          噪声裁剪范围
    policy_delay : int          策略更新延迟步数
    buffer_size : int
    batch_size : int
    hidden_dims : list
    device : str
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        hidden_dims: list = [400, 300],
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.device = torch.device(device)
        self._total_steps = 0

        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dims, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def get_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """选择动作，可选加探索噪声"""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t)
        action = action.cpu().numpy().flatten()
        if noise_scale > 0:
            action += np.random.normal(0, noise_scale * self.max_action, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)
        return action

    def push(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self._total_steps += 1

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        actions = actions.to(self.device)

        with torch.no_grad():
            # 目标策略平滑
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Clipped Double Q
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            td_target = rewards + self.gamma * q_target * (1 - dones)

        q1, q2 = self.critic(states, actions)
        q1_loss = nn.MSELoss()(q1, td_target)
        q2_loss = nn.MSELoss()(q2, td_target)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        info = {
            "critic_loss": critic_loss.item(),
            "q1": q1.mean().item(),
        }

        # 延迟策略更新
        if self._total_steps % self.policy_delay == 0:
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            for tp, sp in zip(self.actor_target.parameters(), self.actor.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
            for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

            info["actor_loss"] = actor_loss.item()

        return info

    def save(self, path: str):
        torch.save(self.actor.state_dict(), f"{path}_actor.pt")
        torch.save(self.critic.state_dict(), f"{path}_critic.pt")

    def load(self, path: str):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pt", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pt", map_location=self.device))


# ──────────────────────────────────────────────
# 训练入口
# ──────────────────────────────────────────────

def train_td3(
    env_id: str = "HalfCheetah-v4",
    num_steps: int = 1_000_000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    batch_size: int = 256,
    start_steps: int = 25_000,
    update_after: int = 1_000,
    update_every: int = 1,
    noise_scale: float = 0.1,
    hidden_dims: list = [400, 300],
    seed: Optional[int] = None,
    render: bool = False,
    device: str = "cpu",
) -> dict:
    """训练 TD3 并返回训练记录"""
    env = make_env(env_id, seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_action = float(env.action_space.high[0])

    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        hidden_dims=hidden_dims,
        device=device,
    )

    state, _ = env.reset()
    ep_reward = 0
    episode = 0
    total_steps = 0
    rewards_log = []

    while total_steps < num_steps:
        if total_steps < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state, noise_scale)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        total_steps += 1

        if render:
            env.render()

        if total_steps >= update_after:
            agent.update()

        if done:
            episode += 1
            rewards_log.append(ep_reward)
            if episode % 10 == 0:
                avg = np.mean(rewards_log[-10:]) if rewards_log else 0
                print(f"Ep {episode:4d} | Steps: {total_steps:7d} | Reward: {ep_reward:8.1f} | Avg10: {avg:8.2f}")
            state, _ = env.reset()
            ep_reward = 0

    env.close()
    return {"rewards": rewards_log}


if __name__ == "__main__":
    train_td3(env_id="Hopper-v4", num_steps=200_000, start_steps=5_000)
