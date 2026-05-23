"""A2C (Advantage Actor-Critic) — 从零实现

算法要点：
  • 同步版 Actor-Critic
  • Actor (策略网络) + Critic (价值网络) 共享或分离
  • N步优势估计 (N-step advantage)
  • 熵正则化鼓励探索
  • 并行环境加速采样（可选）
"""

from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import build_mlp, make_env


class ActorCriticNet(nn.Module):
    """共享特征提取的 Actor-Critic 网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super().__init__()
        encoder_dims = [state_dim] + hidden_dims
        layers = []
        for i in range(len(encoder_dims) - 1):
            layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*layers)

        # Actor 头
        self.actor = nn.Linear(hidden_dims[-1], action_dim)
        # Critic 头
        self.critic = nn.Linear(hidden_dims[-1], 1)

        # 初始化
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple:
        features = self.encoder(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class A2C:
    """A2C 智能体

    参数
    ----------
    state_dim : int
    action_dim : int
    lr : float
    gamma : float
    n_steps : int            N步优势估计长度
    entropy_coef : float     熵正则系数
    value_coef : float       价值损失系数
    max_grad_norm : float    梯度裁剪范数
    hidden_dims : list
    device : str
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 5,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_dims: list = [128, 128],
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.net = ActorCriticNet(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # 用于 N步 存储的滚动缓冲区
        self._rollout = []

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(state_t)

        if deterministic:
            return logits.argmax().item()

        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item()

    def store(self, state: np.ndarray, action: int, reward: float, done: bool):
        """存储一步经验到滚动缓冲区"""
        self._rollout.append((state, action, reward, done))

    def _process_rollout(self, final_state: np.ndarray) -> tuple:
        """处理 N步 经验，计算优势"""
        states, actions, rewards, dones = zip(*self._rollout)
        states_a = np.array(states)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)

        # 获取价值预测
        states_t = torch.as_tensor(states_a, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, values = self.net(states_t)
        values = values.squeeze(1)

        # 获取最后状态的 V(s_{t+n})，如果结束则为 0
        with torch.no_grad():
            final_t = torch.as_tensor(final_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, last_value = self.net(final_t)
        last_value = last_value.squeeze(1).item()

        # 计算 N步 回报
        returns = []
        G = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = returns_t - values

        self._rollout.clear()
        return states_t, actions_t, returns_t, advantages

    def update(self, final_state: np.ndarray) -> dict:
        """用 N步 累积经验更新网络"""
        if len(self._rollout) == 0:
            return {}

        states_t, actions_t, returns_t, advantages = self._process_rollout(final_state)

        # 前向传播
        logits, values = self.net(states_t)
        dist = torch.distributions.Categorical(logits=logits)

        # Actor 损失：优势 * -logπ
        log_probs = dist.log_prob(actions_t)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # 熵正则
        entropy = dist.entropy().mean()
        actor_loss -= self.entropy_coef * entropy

        # Critic 损失
        value_loss = self.value_coef * nn.MSELoss()(values.squeeze(1), returns_t.detach())

        total_loss = actor_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "advantages": advantages.mean().item(),
        }

    def save(self, path: str):
        torch.save(self.net.state_dict(), f"{path}_a2c.pt")

    def load(self, path: str):
        self.net.load_state_dict(torch.load(f"{path}_a2c.pt", map_location=self.device))


# ──────────────────────────────────────────────
# 训练入口
# ──────────────────────────────────────────────

def train_a2c(
    env_id: str = "CartPole-v1",
    num_episodes: int = 1000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    n_steps: int = 5,
    entropy_coef: float = 0.01,
    hidden_dims: list = [128, 128],
    seed: Optional[int] = None,
    render: bool = False,
    device: str = "cpu",
) -> dict:
    """训练 A2C 并返回训练记录"""
    env = make_env(env_id, seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2C(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        n_steps=n_steps,
        entropy_coef=entropy_coef,
        hidden_dims=hidden_dims,
        device=device,
    )

    rewards_log = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            # 收集 N步 经验
            for _ in range(agent.n_steps):
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.store(state, action, reward, done)
                state = next_state
                ep_reward += reward
                if render:
                    env.render()
                if done:
                    break

            # 更新
            info = agent.update(state if not done else np.zeros(state_dim))

        rewards_log.append(ep_reward)

        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards_log[-50:]) if rewards_log else 0
            adv = info.get("advantages", 0)
            ent = info.get("entropy", 0)
            print(f"Ep {ep+1:4d} | Reward: {ep_reward:6.1f} | Avg50: {avg:6.2f} | Adv: {adv:.3f} | Ent: {ent:.4f}")

    env.close()
    return {"rewards": rewards_log}


if __name__ == "__main__":
    train_a2c(env_id="CartPole-v1", num_episodes=500, n_steps=5)
