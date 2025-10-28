"""
Train Shared Policy Agent with Sparse Reward Signal

One agent is trained on all 10 trucks using sparse rewards (+1/-1).
The sparse signal compares policy performance against greedy baseline.

Observation: priority, deadline_urgency, distance, truck_load (no zone info)
Action: scores for each package (agent ranks them)

Training loop:
- 100 days per episode
- 10 trucks per day
- 1,000 experiences per episode all update the same agent
- 2 models: PPO and SAC
- Sparse reward: +1 if policy beats baseline, -1 if worse, 0 if equal
"""

import os
import argparse
import numpy as np
from typing import Dict, Tuple
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

from constrained_problem import create_constrained_problem
from per_truck_delivery_env import PerTruckDeliveryEnv
from delivery_penalties import DeliveryPenalties


class RewardCallback(BaseCallback):
    """Callback for tracking rewards during training."""

    def __init__(self, name: str = "SparsePolicy", verbose: int = 0):
        super().__init__(verbose)
        self.name = name
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """Called at each step."""
        return True


class SparsePolicyTrainer:
    """Train shared policy agent with sparse rewards."""

    def __init__(self, num_days: int = 100, seed: int = 42):
        """
        Initialize trainer.

        Args:
            num_days: Days per training episode
            seed: Random seed for reproducibility
        """
        self.num_days = num_days
        self.seed = seed

        # Create problem once (shared between algorithms)
        self.problem = create_constrained_problem(
            num_days=num_days,
            num_trucks=10,
            truck_capacity=2,
            packages_per_day=30,
            num_zones=10,
            seed=seed
        )

        self.penalties = DeliveryPenalties()

    def _create_environment(self):
        """Create per-truck environment with sparse rewards."""
        env = PerTruckDeliveryEnv(
            problem=self.problem,
            penalties=self.penalties,
            use_sparse_rewards=True  # Enable sparse reward mode
        )
        return env

    def run_episode(self, agent, env, deterministic: bool = False) -> Tuple[Dict, list]:
        """
        Run one episode with given agent.

        Args:
            agent: PPO or SAC agent
            env: Environment (manages trucks internally, uses sparse rewards)
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (metrics, sparse_rewards)
        """
        obs, _ = env.reset()

        sparse_rewards = []
        total_sparse_signal = 0.0  # For tracking +1/-1 signals
        day_rewards = []

        # Run full episode (env manages which truck acts each step)
        done = False
        while not done:
            # Get action for current truck from agent
            action, _ = agent.predict(obs, deterministic=deterministic)

            # Execute step (env returns sparse reward signal)
            obs, reward, done, truncated, info = env.step(action)

            day_rewards.append(reward)
            total_sparse_signal += reward

            # Check if day is complete
            if info.get('day_complete', False):
                daily_reward = sum(day_rewards)
                sparse_rewards.append(daily_reward)
                day_rewards = []

        # Calculate metrics
        # Count positive, negative, and zero signals
        num_positive = sum(1 for r in sparse_rewards if r > 0)
        num_negative = sum(1 for r in sparse_rewards if r < 0)
        num_zero = sum(1 for r in sparse_rewards if r == 0)

        metrics = {
            'num_days': self.num_days,
            'total_sparse_signal': total_sparse_signal,
            'avg_sparse_signal': total_sparse_signal / self.num_days if self.num_days > 0 else 0.0,
            'days_beating_baseline': num_positive,
            'days_below_baseline': num_negative,
            'days_equal_baseline': num_zero,
            'baseline_beat_rate': (num_positive / self.num_days * 100 if self.num_days > 0 else 0.0),
        }

        return metrics, sparse_rewards

    def train_ppo(self, total_timesteps: int = 500000) -> None:
        """Train PPO agent with sparse rewards."""
        print("\n" + "=" * 100)
        print("TRAINING SHARED POLICY: PPO AGENT (SPARSE REWARDS)")
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  Algorithm: PPO (on-policy)")
        print(f"  Reward Signal: Sparse (+1/-1 vs greedy baseline)")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Days per episode: {self.num_days}")
        print(f"  Trucks per day: 10")
        print(f"  Decisions per episode: {self.num_days * 10}")
        print(f"  Observation: Zone-agnostic (priority, deadline, distance, load)")
        print(f"  Action: Continuous scores [0,1] for each package")
        print()

        env = self._create_environment()

        agent = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            clip_range=0.2,
            verbose=0
        )

        callback = RewardCallback(name="PPO-Sparse")

        print("Starting PPO training with sparse rewards...\n")

        # Train with checkpoints
        checkpoint_interval = 50000
        checkpoints = []

        for step in range(0, total_timesteps, checkpoint_interval):
            steps_remaining = min(checkpoint_interval, total_timesteps - step)

            print(f"Training: {step:6d} / {total_timesteps} timesteps...", end="", flush=True)
            agent.learn(total_timesteps=steps_remaining, callback=callback)
            print(" done")

            # Evaluate
            print(f"  Evaluating...", end="", flush=True)
            metrics, rewards = self.run_episode(agent, env, deterministic=True)
            print(f" Baseline beat rate: {metrics['baseline_beat_rate']:.1f}%, Avg signal: {metrics['avg_sparse_signal']:.2f}")

            # Save checkpoint
            os.makedirs("ppo_sparse_policy", exist_ok=True)
            checkpoint_path = f"ppo_sparse_policy/agent_{step:06d}"
            agent.save(checkpoint_path)
            checkpoints.append(checkpoint_path)

        print("\n" + "=" * 100)
        print("PPO SPARSE REWARD TRAINING COMPLETE")
        print("=" * 100)
        print(f"\nCheckpoints saved to: ppo_sparse_policy/")
        print(f"Final model: ppo_sparse_policy/agent_{total_timesteps:06d}")

        # Save final model
        agent.save("ppo_sparse_policy/agent_final")

    def train_sac(self, total_timesteps: int = 500000) -> None:
        """Train SAC agent with sparse rewards."""
        print("\n" + "=" * 100)
        print("TRAINING SHARED POLICY: SAC AGENT (SPARSE REWARDS)")
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  Algorithm: SAC (off-policy)")
        print(f"  Reward Signal: Sparse (+1/-1 vs greedy baseline)")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Days per episode: {self.num_days}")
        print(f"  Trucks per day: 10")
        print(f"  Decisions per episode: {self.num_days * 10}")
        print(f"  Observation: Zone-agnostic (priority, deadline, distance, load)")
        print(f"  Action: Continuous scores [0,1] for each package")
        print()

        env = self._create_environment()

        agent = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=1000000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=-1,
            ent_coef='auto',
            target_entropy='auto',
            verbose=0
        )

        callback = RewardCallback(name="SAC-Sparse")

        print("Starting SAC training with sparse rewards...\n")

        # Train with checkpoints
        checkpoint_interval = 50000
        checkpoints = []

        for step in range(0, total_timesteps, checkpoint_interval):
            steps_remaining = min(checkpoint_interval, total_timesteps - step)

            print(f"Training: {step:6d} / {total_timesteps} timesteps...", end="", flush=True)
            agent.learn(total_timesteps=steps_remaining, callback=callback)
            print(" done")

            # Evaluate
            print(f"  Evaluating...", end="", flush=True)
            metrics, rewards = self.run_episode(agent, env, deterministic=True)
            print(f" Baseline beat rate: {metrics['baseline_beat_rate']:.1f}%, Avg signal: {metrics['avg_sparse_signal']:.2f}")

            # Save checkpoint
            os.makedirs("sac_sparse_policy", exist_ok=True)
            checkpoint_path = f"sac_sparse_policy/agent_{step:06d}"
            agent.save(checkpoint_path)
            checkpoints.append(checkpoint_path)

        print("\n" + "=" * 100)
        print("SAC SPARSE REWARD TRAINING COMPLETE")
        print("=" * 100)
        print(f"\nCheckpoints saved to: sac_sparse_policy/")
        print(f"Final model: sac_sparse_policy/agent_{total_timesteps:06d}")

        # Save final model
        agent.save("sac_sparse_policy/agent_final")


def main():
    parser = argparse.ArgumentParser(
        description="Train shared policy agent with sparse reward signal"
    )
    parser.add_argument(
        "--ppo-only",
        action="store_true",
        help="Train only PPO agent"
    )
    parser.add_argument(
        "--sac-only",
        action="store_true",
        help="Train only SAC agent"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500000,
        help="Total training timesteps (default: 500000)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 100)
    print("SPARSE REWARD TRAINING: ONE AGENT FOR ALL TRUCKS")
    print("=" * 100)
    print("\nArchitecture:")
    print("  - Single agent learns zone-agnostic policy")
    print("  - Observation: priority, deadline_urgency, distance, truck_load")
    print("  - Action: scores for each available package (agent ranks them)")
    print("  - Training: 10 trucks * 100 days = 1,000 decisions per episode")
    print("  - Reward Signal: +1/-1 based on policy vs greedy baseline comparison")
    print("  - Result: One model used for all trucks")

    trainer = SparsePolicyTrainer(num_days=100, seed=42)

    if args.ppo_only:
        trainer.train_ppo(total_timesteps=args.timesteps)
    elif args.sac_only:
        trainer.train_sac(total_timesteps=args.timesteps)
    else:
        # Train both
        trainer.train_ppo(total_timesteps=args.timesteps)
        trainer.train_sac(total_timesteps=args.timesteps)

    print("\n" + "=" * 100)
    print("SPARSE REWARD TRAINING COMPLETE")
    print("=" * 100)
    print("\nModels saved to:")
    print("  PPO: ppo_sparse_policy/agent_final.zip")
    print("  SAC: sac_sparse_policy/agent_final.zip")
    print("\nUse 'python test_sparse_policy_benchmark.py' to evaluate")


if __name__ == "__main__":
    main()
