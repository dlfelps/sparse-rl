"""
Train Shared Policy Agent for Per-Truck Delivery Optimization

One agent is trained on all 10 trucks. The agent learns zone-agnostic policy:
"Which 2 packages should this truck deliver today?"

Observation: priority, deadline_urgency, distance, truck_load (no zone info)
Action: scores for each package (agent ranks them)

Training loop:
- 100 days per episode
- 10 trucks per day
- 1,000 experiences per episode all update the same agent
- 2 models: PPO and SAC
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

    def __init__(self, name: str = "SharedPolicy", verbose: int = 0):
        super().__init__(verbose)
        self.name = name
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """Called at each step."""
        return True


class SharedPolicyTrainer:
    """Train shared policy agent for all trucks."""

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
        """Create per-truck environment."""
        env = PerTruckDeliveryEnv(
            problem=self.problem,
            penalties=self.penalties
        )
        return env

    def run_episode(self, agent, env, deterministic: bool = False) -> Tuple[Dict, list]:
        """
        Run one episode with given agent.

        Args:
            agent: PPO or SAC agent
            env: Environment (manages trucks internally)
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (metrics, daily_rewards)
        """
        obs, _ = env.reset()

        daily_rewards = []
        total_delivered = 0
        total_missed = 0
        day_rewards = []

        # Run full episode (env manages which truck acts each step)
        done = False
        while not done:
            # Get action for current truck from agent
            action, _ = agent.predict(obs, deterministic=deterministic)

            # Execute step (env advances to next truck)
            obs, reward, done, truncated, info = env.step(action)

            day_rewards.append(reward)
            total_delivered += info['delivered']
            total_missed += info['missed']

            # Check if day is complete
            if info.get('day_complete', False):
                daily_reward = sum(day_rewards)
                daily_rewards.append(daily_reward)
                day_rewards = []

        # Calculate metrics
        on_time_rate = (
            total_delivered / (total_delivered + total_missed) * 100
            if (total_delivered + total_missed) > 0 else 0.0
        )

        metrics = {
            'num_days': self.num_days,
            'total_delivered': total_delivered,
            'total_missed': total_missed,
            'on_time_delivery_rate': on_time_rate,
            'total_reward': sum(daily_rewards),
            'avg_daily_reward': sum(daily_rewards) / self.num_days if self.num_days > 0 else 0.0,
        }

        return metrics, daily_rewards

    def train_ppo(self, total_timesteps: int = 500000) -> None:
        """Train PPO agent."""
        print("\n" + "=" * 100)
        print("TRAINING SHARED POLICY: PPO AGENT")
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  Algorithm: PPO (on-policy)")
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

        callback = RewardCallback(name="PPO")

        print("Starting PPO training...\n")

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
            print(f" Reward: {metrics['avg_daily_reward']:.1f}, On-time: {metrics['on_time_delivery_rate']:.1f}%")

            # Save checkpoint
            os.makedirs("ppo_shared_policy", exist_ok=True)
            checkpoint_path = f"ppo_shared_policy/agent_{step:06d}"
            agent.save(checkpoint_path)
            checkpoints.append(checkpoint_path)

        print("\n" + "=" * 100)
        print("PPO TRAINING COMPLETE")
        print("=" * 100)
        print(f"\nCheckpoints saved to: ppo_shared_policy/")
        print(f"Final model: ppo_shared_policy/agent_{total_timesteps:06d}")

        # Save final model
        agent.save("ppo_shared_policy/agent_final")

    def train_sac(self, total_timesteps: int = 500000) -> None:
        """Train SAC agent."""
        print("\n" + "=" * 100)
        print("TRAINING SHARED POLICY: SAC AGENT")
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  Algorithm: SAC (off-policy)")
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

        callback = RewardCallback(name="SAC")

        print("Starting SAC training...\n")

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
            print(f" Reward: {metrics['avg_daily_reward']:.1f}, On-time: {metrics['on_time_delivery_rate']:.1f}%")

            # Save checkpoint
            os.makedirs("sac_shared_policy", exist_ok=True)
            checkpoint_path = f"sac_shared_policy/agent_{step:06d}"
            agent.save(checkpoint_path)
            checkpoints.append(checkpoint_path)

        print("\n" + "=" * 100)
        print("SAC TRAINING COMPLETE")
        print("=" * 100)
        print(f"\nCheckpoints saved to: sac_shared_policy/")
        print(f"Final model: sac_shared_policy/agent_{total_timesteps:06d}")

        # Save final model
        agent.save("sac_shared_policy/agent_final")


def main():
    parser = argparse.ArgumentParser(
        description="Train shared policy agent for per-truck delivery optimization"
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
    print("SHARED POLICY TRAINING: ONE AGENT FOR ALL TRUCKS")
    print("=" * 100)
    print("\nArchitecture:")
    print("  - Single agent learns zone-agnostic policy")
    print("  - Observation: priority, deadline_urgency, distance, truck_load")
    print("  - Action: scores for each available package (agent ranks them)")
    print("  - Training: 10 trucks * 100 days = 1,000 decisions per episode")
    print("  - Result: One model used for all trucks")

    trainer = SharedPolicyTrainer(num_days=100, seed=42)

    if args.ppo_only:
        trainer.train_ppo(total_timesteps=args.timesteps)
    elif args.sac_only:
        trainer.train_sac(total_timesteps=args.timesteps)
    else:
        # Train both
        trainer.train_ppo(total_timesteps=args.timesteps)
        trainer.train_sac(total_timesteps=args.timesteps)

    print("\n" + "=" * 100)
    print("TRAINING COMPLETE")
    print("=" * 100)
    print("\nModels saved to:")
    print("  PPO: ppo_shared_policy/agent_final.zip")
    print("  SAC: sac_shared_policy/agent_final.zip")
    print("\nUse 'python test_shared_policy_benchmark.py' to evaluate")


if __name__ == "__main__":
    main()
