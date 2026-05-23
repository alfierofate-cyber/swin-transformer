"""REINFORCE (Monte Carlo Policy Gradient) — 从零实现

算法要点：
  • 基于回合制的策略梯度方法
  • 用蒙特卡洛采样整个轨迹，计算累计折扣回报 G_t
  • 梯度上升优化策略网络 π_θ(a|s)
  • 加入基线（baseline）降低方差
"""

from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import build_mlp, make_env


class PolicyNet(nn.Module):
    """策略网络：状态 → 动作概率分布"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super().__init__()
        self.net = build_mlp(state_dim, hidden_dims, action_dim)
        # 对连续动作使用高斯策略时需要的对数标准差
        self.log_std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class REINFORCE:
    """REINFORCE with Baseline 智能体

    参数
    ----------
    state_dim : int          状态空间维度
    action_dim : int         动作空间维度（离散）
    lr : float               学习率
    gamma : float            折扣因子
    hidden_dims : list       隐藏层神经元数
    use_baseline : bool      是否使用状态值基线
    device : str             运行设备
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        hidden_dims: list = [128, 128],
        use_baseline: bool = True,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.device = torch.device(device)

        # 策略网络
        self.policy = PolicyNet(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 基线网络（可选）
        self.baseline = None
        if use_baseline:
            self.baseline = build_mlp(state_dim, hidden_dims, 1).to(self.device)
            self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=lr)

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """根据策略采样动作"""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.policy(state_t)
        if deterministic:
            return probs.argmax().item()
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def _compute_returns(self, rewards: list) -> torch.Tensor:
        """计算折扣回报 G_t"""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(self, trajectory: list) -> dict:
        """用完整轨迹更新策略（和基线）

        参数
        ----------
        trajectory : list of (state, action, reward)
        """
        states, actions, rewards = zip(*trajectory)

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.long, device=self.device)
        returns = self._compute_returns(rewards)

        # 优势：G_t - V(s_t)
        advantages = returns.clone()
        if self.use_baseline:
            values = self.baseline(states_t).squeeze(1)
            advantages = returns - values.detach()

        # 标准化优势（稳定训练）
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 策略梯度损失
        probs = self.policy(states_t)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_t)
        policy_loss = -(log_probs * advantages).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        info = {"policy_loss": policy_loss.item()}

        # 更新基线网络（MSE）
        if self.use_baseline:
            values = self.baseline(states_t).squeeze(1)
            baseline_loss = nn.MSELoss()(values, returns)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()
            info["baseline_loss"] = baseline_loss.item()

        return info

    def save(self, path: str):
        torch.save(self.policy.state_dict(), f"{path}_policy.pt")
        if self.baseline is not None:
            torch.save(self.baseline.state_dict(), f"{path}_baseline.pt")

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(f"{path}_policy.pt", map_location=self.device))
        if self.baseline is not None:
            self.baseline.load_state_dict(torch.load(f"{path}_baseline.pt", map_location=self.device))


# ──────────────────────────────────────────────
# 训练入口
# ──────────────────────────────────────────────

def train_reinforce(
    env_id: str = "CartPole-v1",
    num_episodes: int = 1000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    hidden_dims: list = [128, 128],
    use_baseline: bool = True,
    seed: Optional[int] = None,
    render: bool = False,
    device: str = "cpu",
) -> dict:
    """训练 REINFORCE 并返回训练记录"""
    env = make_env(env_id, seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCE(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        hidden_dims=hidden_dims,
        use_baseline=use_baseline,
        device=device,
    )

    rewards_log = []
    success_count = 0

    for ep in range(num_episodes):
        state, _ = env.reset()
        trajectory = []
        ep_reward = 0

        while True:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append((state, action, reward))
            state = next_state
            ep_reward += reward
            if render:
                env.render()
            if done:
                break

        # 更新智能体
        info = agent.update(trajectory)
        rewards_log.append(ep_reward)
        solved = ep_reward >= env.spec.reward_threshold if hasattr(env.spec, "reward_threshold") else False
        if solved:
            success_count += 1

        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards_log[-50:]) if rewards_log else 0
            print(f"Ep {ep+1:4d} | Reward: {ep_reward:6.1f} | Avg50: {avg:6.2f} | Loss: {info['policy_loss']:.4f}")

    env.close()
    return {"rewards": rewards_log, "success_rate": success_count / num_episodes}


if __name__ == "__main__":
    train_reinforce(env_id="CartPole-v1", num_episodes=500, render=False)
