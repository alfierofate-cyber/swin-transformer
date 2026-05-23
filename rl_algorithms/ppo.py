"""PPO (Proximal Policy Optimization) — 从零实现

算法要点：
  • Clip-based 信赖域约束：L^CLIP(θ) = min(r_t(θ)Â_t, clip(r_t, 1-ε, 1+ε)Â_t)
  • 重要性采样比率 r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
  • GAE (Generalized Advantage Estimation) 计算优势
  • 多 epochs 小批量更新
  • 熵正则化
  • 适合离散和连续动作空间
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
        self.encoder = build_mlp(state_dim, hidden_dims, hidden_dims[-1], activation=nn.Tanh)

        self.actor = nn.Linear(hidden_dims[-1], action_dim)
        self.critic = nn.Linear(hidden_dims[-1], 1)

        # 特殊初始化（PPO 常用小初始值）
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple:
        features = torch.tanh(self.encoder(x))
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class RolloutBuffer:
    """PPO 滚动缓冲区：存储完整轨迹"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


class PPO:
    """PPO 智能体 (Clip-based)

    参数
    ----------
    state_dim : int
    action_dim : int
    lr : float
    gamma : float
    gae_lambda : float       GAE λ 参数
    clip_epsilon : float      PPO clip 范围
    value_coef : float        价值损失系数
    entropy_coef : float      熵正则系数
    max_grad_norm : float
    update_epochs : int      每次更新迭代次数
    batch_size : int
    hidden_dims : list
    device : str
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        hidden_dims: list = [128, 128],
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.net = ActorCriticNet(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.rollout = RolloutBuffer()

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> tuple:
        """返回 (动作, 对数概率, 价值)"""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(state_t)

        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax().item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device))
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store(self, state, action, reward, done, log_prob, value):
        """存储一步经验"""
        self.rollout.states.append(state)
        self.rollout.actions.append(action)
        self.rollout.rewards.append(reward)
        self.rollout.dones.append(done)
        self.rollout.log_probs.append(log_prob)
        self.rollout.values.append(value)

    def _compute_gae(self, last_value: float):
        """计算 GAE 优势"""
        rewards = np.array(self.rollout.rewards, dtype=np.float32)
        dones = np.array(self.rollout.dones, dtype=np.float32)
        values = np.array(self.rollout.values + [last_value], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def update(self, last_value: float) -> dict:
        """用完整轨迹更新策略"""
        advantages, returns = self._compute_gae(last_value)

        # 标准化优势
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8

        states_t = torch.as_tensor(np.array(self.rollout.states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.array(self.rollout.actions), dtype=torch.long, device=self.device)
        old_log_probs_t = torch.as_tensor(np.array(self.rollout.log_probs), dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor((advantages - adv_mean) / adv_std, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        dataset_size = len(self.rollout)
        indices = np.arange(dataset_size)
        policy_losses, value_losses, entropies = [], [], []

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                batch = indices[start : start + self.batch_size]

                logits, values = self.net(states_t[batch])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t[batch])
                entropy = dist.entropy().mean()

                # 重要性采样比率
                ratio = torch.exp(new_log_probs - old_log_probs_t[batch])

                # Clip 损失
                surr1 = ratio * advantages_t[batch]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t[batch]
                actor_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = self.value_coef * nn.MSELoss()(values.squeeze(1), returns_t[batch])

                total_loss = actor_loss + value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(actor_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        self.rollout.clear()

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
        }

    def save(self, path: str):
        torch.save(self.net.state_dict(), f"{path}_ppo.pt")

    def load(self, path: str):
        self.net.load_state_dict(torch.load(f"{path}_ppo.pt", map_location=self.device))


# ──────────────────────────────────────────────
# 训练入口
# ──────────────────────────────────────────────

def train_ppo(
    env_id: str = "CartPole-v1",
    num_episodes: int = 1000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    update_epochs: int = 10,
    batch_size: int = 64,
    hidden_dims: list = [128, 128],
    seed: Optional[int] = None,
    render: bool = False,
    device: str = "cpu",
) -> dict:
    """训练 PPO 并返回训练记录"""
    env = make_env(env_id, seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        update_epochs=update_epochs,
        batch_size=batch_size,
        hidden_dims=hidden_dims,
        device=device,
    )

    rewards_log = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, done, log_prob, value)
            state = next_state
            ep_reward += reward
            if render:
                env.render()

        # 获取最后状态的价值（用于 GAE 计算）
        _, _, last_value = agent.get_action(state, deterministic=True)

        # 更新
        info = agent.update(last_value)

        rewards_log.append(ep_reward)

        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards_log[-50:]) if rewards_log else 0
            print(f"Ep {ep+1:4d} | Reward: {ep_reward:6.1f} | Avg50: {avg:6.2f} | "
                  f"p_loss: {info['policy_loss']:.4f} | v_loss: {info['value_loss']:.4f}")

    env.close()
    return {"rewards": rewards_log}


if __name__ == "__main__":
    train_ppo(env_id="CartPole-v1", num_episodes=500, update_epochs=10)
