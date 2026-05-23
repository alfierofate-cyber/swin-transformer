"""SAC (Soft Actor-Critic) — 从零实现

算法要点：
  • 最大熵强化学习框架
  • Actor: 重参数化技巧采样连续动作
  • Critic: 双 Q 网络 (Clipped Double Q)
  • 自动调节温度系数 α
  • 经验回放 + 软更新目标网络
  • 适合 MuJoCo / PyBullet 等连续控制任务
"""

from typing import Optional
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import ReplayBuffer, build_mlp, hard_update, soft_update, make_env


class SquashedGaussianActor(nn.Module):
    """高斯策略网络：输出均值 + 对数标准差，通过 tanh 压缩动作"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256],
                 log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = build_mlp(state_dim, hidden_dims, hidden_dims[-1])
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)

        # 初始化
        nn.init.xavier_uniform_(self.mean.weight, gain=0.01)
        nn.init.xavier_uniform_(self.log_std.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> tuple:
        features = self.net(x)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, x: torch.Tensor, deterministic: bool = False) -> tuple:
        """采样动作，返回 (动作, 对数概率, 原始均值)"""
        mean, log_std = self.forward(x)
        std = log_std.exp()

        if deterministic:
            return torch.tanh(mean), torch.zeros_like(mean), mean

        # 重参数化：μ + ε * σ
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        # 对数概率：修正 tanh 变换后的概率密度
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean


class TwinQNetwork(nn.Module):
    """双 Q 网络 (Clipped Double Q)"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        self.q1 = build_mlp(state_dim + action_dim, hidden_dims, 1)
        self.q2 = build_mlp(state_dim + action_dim, hidden_dims, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class ValueNetwork(nn.Module):
    """软状态价值网络 V(s)"""

    def __init__(self, state_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        self.net = build_mlp(state_dim, hidden_dims, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SAC:
    """Soft Actor-Critic 智能体

    参数
    ----------
    state_dim : int
    action_dim : int
    action_scale : float        动作缩放因子（实际范围）
    lr : float
    gamma : float
    tau : float                 软更新系数
    alpha : float               初始温度
    auto_alpha : bool           是否自动调节温度
    target_entropy : float      目标熵（自动调节用）
    buffer_size : int
    batch_size : int
    hidden_dims : list
    device : str
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        hidden_dims: list = [256, 256],
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Actor
        self.actor = SquashedGaussianActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Twin Q
        self.q = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q_target = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        hard_update(self.q_target, self.q)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=lr)

        # 温度 α
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.log_alpha = torch.tensor(math.log(alpha), dtype=torch.float32,
                                          device=self.device, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.target_entropy = target_entropy or -action_dim
        else:
            self.alpha = alpha

        self.replay_buffer = ReplayBuffer(buffer_size)

    @property
    def alpha(self):
        if self.auto_alpha:
            return self.log_alpha.exp().detach()
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.actor.sample(state_t, deterministic=deterministic)
        return action.cpu().numpy().flatten() * self.action_scale

    def push(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action / self.action_scale, reward, next_state, done)

    def update(self) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        actions = actions.to(self.device)

        # ── Q 更新 ──
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_target, q2_target = self.q_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            value_target = q_target - self.alpha * next_log_probs
            td_target = rewards + self.gamma * value_target * (1 - dones)

        q1, q2 = self.q(states, actions)
        q1_loss = nn.MSELoss()(q1, td_target)
        q2_loss = nn.MSELoss()(q2, td_target)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ── Actor 更新 ──
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new, q2_new = self.q(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新
        soft_update(self.q_target, self.q, self.tau)

        info = {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item() if hasattr(self.alpha, 'item') else self.alpha,
        }

        # ── 温度更新 ──
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            info["alpha_loss"] = alpha_loss.item()
            info["alpha"] = self.alpha.item()

        return info

    def save(self, path: str):
        torch.save(self.actor.state_dict(), f"{path}_actor.pt")
        torch.save(self.q.state_dict(), f"{path}_q.pt")
        torch.save(self.q_target.state_dict(), f"{path}_q_target.pt")

    def load(self, path: str):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pt", map_location=self.device))
        self.q.load_state_dict(torch.load(f"{path}_q.pt", map_location=self.device))
        self.q_target.load_state_dict(torch.load(f"{path}_q_target.pt", map_location=self.device))


# ──────────────────────────────────────────────
# 训练入口
# ──────────────────────────────────────────────

def train_sac(
    env_id: str = "HalfCheetah-v4",
    num_episodes: int = 200,
    num_steps: int = 1_000_000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    batch_size: int = 256,
    start_steps: int = 10_000,
    update_after: int = 1_000,
    update_every: int = 50,
    hidden_dims: list = [256, 256],
    seed: Optional[int] = None,
    render: bool = False,
    device: str = "cpu",
) -> dict:
    """训练 SAC 并返回训练记录"""
    env = make_env(env_id, seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    action_scale = float(env.action_space.high[0])

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        hidden_dims=hidden_dims,
        device=device,
    )

    rewards_log = []
    state, _ = env.reset()
    ep_reward = 0
    episode = 0
    total_steps = 0

    while total_steps < num_steps:
        if total_steps < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        total_steps += 1

        if render:
            env.render()

        if total_steps >= update_after and total_steps % update_every == 0:
            for _ in range(update_every):
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
    train_sac(env_id="Hopper-v4", num_steps=200_000, start_steps=5_000)
