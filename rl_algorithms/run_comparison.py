"""多算法对比实验脚本

在同一个 Gymnasium 环境上运行多个 RL 算法，绘制 reward 曲线对比。
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np

# 确保包可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_algorithms.reinforce import train_reinforce
from rl_algorithms.dqn import train_dqn
from rl_algorithms.a2c import train_a2c
from rl_algorithms.ppo import train_ppo

ALGORITHMS = {
    "reinforce": train_reinforce,
    "dqn": train_dqn,
    "a2c": train_a2c,
    "ppo": train_ppo,
}


def moving_average(data: list, window: int = 20) -> np.ndarray:
    """滑动平均"""
    data = np.array(data)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def run_comparison(
    env_id: str = "CartPole-v1",
    algs: List[str] = None,
    num_episodes: int = 500,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, dict]:
    """运行多个算法并返回结果"""
    if algs is None:
        algs = list(ALGORITHMS.keys())

    results = {}
    for name in algs:
        if name not in ALGORITHMS:
            print(f"⚠ 未知算法: {name}，跳过")
            continue

        print(f"\n{'='*50}")
        print(f"  训练: {name.upper()} | 环境: {env_id} | 回合: {num_episodes}")
        print(f"{'='*50}")

        train_fn = ALGORITHMS[name]
        result = train_fn(
            env_id=env_id,
            num_episodes=num_episodes,
            seed=seed,
            device=device,
        )
        results[name] = result

    return results


def plot_results(results: Dict[str, dict], window: int = 20, save_path: str = None):
    """绘制对比曲线"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图。pip install matplotlib")
        return

    plt.figure(figsize=(12, 6))

    for name, data in results.items():
        rewards = data.get("rewards", [])
        if len(rewards) < window:
            plt.plot(rewards, label=name.upper(), alpha=0.6)
        else:
            smoothed = moving_average(rewards, window)
            plt.plot(smoothed, label=f"{name.upper()} (MA{window})", linewidth=2)
            plt.plot(rewards, alpha=0.15, color=plt.gca().lines[-1].get_color())

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Algorithm Comparison — {list(results.keys())[0] if results else ''}")
    plt.legend()
    plt.grid(alpha=0.3)

    # 标注环境最优分数阈值
    try:
        import gymnasium as gym
        env = gym.make(list(results.keys())[0])
        if hasattr(env.spec, "reward_threshold") and env.spec.reward_threshold is not None:
            plt.axhline(y=env.spec.reward_threshold, color="r", linestyle="--",
                        alpha=0.7, label=f"Solved ({env.spec.reward_threshold})")
            plt.legend()
        env.close()
    except Exception:
        pass

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()


def print_summary(results: Dict[str, dict]):
    """打印结果摘要"""
    print("\n" + "=" * 60)
    print("  训练结果摘要")
    print("=" * 60)

    if not results:
        print("  无结果")
        return

    for name, data in results.items():
        rewards = data.get("rewards", [])
        if not rewards:
            continue
        print(f"  {name.upper():>10s}: "
              f"Avg Reward = {np.mean(rewards[-100:]):7.2f} "
              f"(最后100回合) | "
              f"Max = {max(rewards):6.1f} | "
              f"Success Rate = {data.get('success_rate', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="多算法 RL 对比实验")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gymnasium 环境 ID")
    parser.add_argument("--algs", type=str, nargs="+",
                        default=["reinforce", "dqn", "a2c", "ppo"],
                        help="要运行的算法列表")
    parser.add_argument("--episodes", type=int, default=500,
                        help="每个算法的训练回合数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--device", type=str, default="cpu",
                        help="运行设备 (cpu/cuda)")
    parser.add_argument("--save", type=str, default=None,
                        help="图表保存路径")
    parser.add_argument("--no-plot", action="store_true",
                        help="禁止绘图")
    args = parser.parse_args()

    results = run_comparison(
        env_id=args.env,
        algs=args.algs,
        num_episodes=args.episodes,
        seed=args.seed,
        device=args.device,
    )

    print_summary(results)

    if not args.no_plot:
        plot_results(results, save_path=args.save or f"rl_comparison_{args.env}.png")


if __name__ == "__main__":
    main()
