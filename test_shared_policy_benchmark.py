"""
Benchmark Shared Policy Agents vs Greedy Baseline

Tests:
- Greedy baseline (deliver max from each truck's zone)
- Shared policy PPO
- Shared policy SAC

Compares on-time delivery, reward, and efficiency.
Generates plots for visualization.
"""

import os
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC

from constrained_problem import create_constrained_problem
from per_truck_delivery_env import PerTruckDeliveryEnv
from delivery_penalties import DeliveryPenalties
from evaluation_metrics import EvaluationMetrics


class SharedPolicyBenchmark:
    """Benchmark shared policy agents."""

    def __init__(self, num_problems: int = 5, num_days: int = 100, seed: int = 42):
        """
        Initialize benchmark.

        Args:
            num_problems: Number of test problems
            num_days: Days per problem
            seed: Random seed
        """
        self.num_problems = num_problems
        self.num_days = num_days
        self.seed = seed
        self.penalties = DeliveryPenalties()

    def run_greedy_baseline(self, problem: Dict) -> Tuple[Dict, List]:
        """
        Run greedy baseline: each truck delivers all available packages (up to 2).

        Args:
            problem: Problem instance

        Returns:
            Tuple of (metrics, daily_rewards)
        """
        env = PerTruckDeliveryEnv(problem=problem, penalties=self.penalties)
        obs, _ = env.reset()

        metrics = EvaluationMetrics()
        daily_rewards = []
        total_delivered = 0
        total_missed = 0

        for day in range(self.num_days):
            day_rewards = []

            for truck_id in range(10):
                # Greedy: deliver ALL available packages (max 2)
                available = env._get_available_packages(truck_id)
                packages_to_deliver = available[:2]  # Take up to 2

                if packages_to_deliver:
                    env.trucks[truck_id].current_load += len(packages_to_deliver)

                    # Calculate reward
                    delivered_ids = {p.id for p in packages_to_deliver}
                    env.pending_packages[truck_id] = [
                        p for p in env.pending_packages[truck_id]
                        if p.id not in delivered_ids
                    ]

                    all_truck_packages = env.pending_packages[truck_id] + [
                        p for p in env.all_packages_today
                        if env._get_truck_zone(truck_id) == env._get_package_zone(p)
                    ]

                    missed = [
                        p for p in all_truck_packages
                        if p.deadline == env.current_day and p.id not in delivered_ids
                    ]

                    distance = sum(
                        2 * env._get_distance(p.destination)
                        for p in packages_to_deliver
                    )

                    reward_dict = self.penalties.calculate_daily_reward(
                        packages_to_deliver,
                        missed,
                        distance
                    )

                    day_rewards.append(reward_dict['total_reward'])
                    total_delivered += len(packages_to_deliver)
                    total_missed += len(missed)

                    # Track metrics
                    class StubPackage:
                        def __init__(self, priority=5):
                            self.priority = priority

                    stub_delivered = [StubPackage(priority=5) for _ in packages_to_deliver]
                    stub_missed = [StubPackage(priority=5) for _ in missed]

                    metrics.track_day(
                        day=day,
                        delivered_packages=stub_delivered,
                        missed_packages=stub_missed,
                        deferred_packages=[],
                        trucks=env.trucks,
                        total_distance=distance,
                        daily_reward=reward_dict['total_reward']
                    )

            # End of day
            daily_reward = sum(day_rewards) if day_rewards else 0.0
            daily_rewards.append(daily_reward)

            if not env.is_episode_done():
                env.advance_day()

        return metrics.get_episode_summary(), daily_rewards

    def run_shared_policy_agent(self, agent, problem: Dict, agent_name: str = "Agent") -> Tuple[Dict, List]:
        """
        Run shared policy agent.

        Args:
            agent: PPO or SAC agent
            problem: Problem instance
            agent_name: Name for logging

        Returns:
            Tuple of (metrics, daily_rewards)
        """
        env = PerTruckDeliveryEnv(problem=problem, penalties=self.penalties)
        obs, _ = env.reset()

        metrics = EvaluationMetrics()
        daily_rewards = []
        total_delivered = 0
        total_missed = 0
        day_rewards = []
        current_day = 0

        # Run full episode (env manages which truck acts each step)
        done = False
        while not done:
            # Get action from shared policy
            action, _ = agent.predict(obs, deterministic=True)

            # Execute step (env advances to next truck)
            obs, reward, done, truncated, info = env.step(action)

            day_rewards.append(reward)
            total_delivered += info['delivered']
            total_missed += info['missed']

            # Check if day is complete
            if info.get('day_complete', False):
                # Track metrics for completed day
                class StubPackage:
                    def __init__(self, priority=5):
                        self.priority = priority

                # Create stub packages for metrics tracking
                stub_delivered = [StubPackage(priority=5) for _ in range(info.get('delivered', 0))]
                stub_missed = [StubPackage(priority=5) for _ in range(info.get('missed', 0))]

                # Sum up rewards for the day
                daily_reward = sum(day_rewards)
                daily_rewards.append(daily_reward)

                # Track day in metrics
                metrics.track_day(
                    day=current_day,
                    delivered_packages=stub_delivered,
                    missed_packages=stub_missed,
                    deferred_packages=[],
                    trucks=env.trucks,
                    total_distance=0,  # Not tracking distance in shared policy
                    daily_reward=daily_reward
                )

                day_rewards = []
                current_day += 1

        return metrics.get_episode_summary(), daily_rewards

    def run_all_benchmarks(self):
        """Run complete benchmark suite."""
        print("\n" + "=" * 100)
        print("SHARED POLICY BENCHMARK")
        print("=" * 100)

        results = {
            'greedy': {'metrics': [], 'rewards': []},
            'ppo': {'metrics': [], 'rewards': []},
            'sac': {'metrics': [], 'rewards': []},
        }

        # Load agents
        ppo_agent = None
        sac_agent = None

        if os.path.exists("ppo_shared_policy/agent_final.zip"):
            ppo_agent = PPO.load("ppo_shared_policy/agent_final")
            print("Loaded PPO agent")
        else:
            print("Warning: PPO agent not found at ppo_shared_policy/agent_final.zip")

        if os.path.exists("sac_shared_policy/agent_final.zip"):
            sac_agent = SAC.load("sac_shared_policy/agent_final")
            print("Loaded SAC agent")
        else:
            print("Warning: SAC agent not found at sac_shared_policy/agent_final.zip")

        # Run benchmarks
        for problem_idx in range(self.num_problems):
            print(f"\nProblem {problem_idx + 1}/{self.num_problems}")
            print("-" * 100)

            problem = create_constrained_problem(
                num_days=self.num_days,
                num_trucks=10,
                truck_capacity=2,
                packages_per_day=30,
                num_zones=10,
                seed=self.seed + problem_idx
            )

            # Greedy
            print("  Running Greedy (baseline)...", end="", flush=True)
            metrics, rewards = self.run_greedy_baseline(problem)
            results['greedy']['metrics'].append(metrics)
            results['greedy']['rewards'].append(rewards)
            print(f" On-time: {metrics['on_time_delivery_rate']:.1f}%")

            # PPO
            if ppo_agent:
                print("  Running Shared Policy PPO...", end="", flush=True)
                metrics, rewards = self.run_shared_policy_agent(ppo_agent, problem, "PPO")
                results['ppo']['metrics'].append(metrics)
                results['ppo']['rewards'].append(rewards)
                print(f" On-time: {metrics['on_time_delivery_rate']:.1f}%")
            else:
                print("  Shared Policy PPO... SKIPPED (not found)")

            # SAC
            if sac_agent:
                print("  Running Shared Policy SAC...", end="", flush=True)
                metrics, rewards = self.run_shared_policy_agent(sac_agent, problem, "SAC")
                results['sac']['metrics'].append(metrics)
                results['sac']['rewards'].append(rewards)
                print(f" On-time: {metrics['on_time_delivery_rate']:.1f}%")
            else:
                print("  Shared Policy SAC... SKIPPED (not found)")

        # Print summary
        self._print_summary(results)

        # Generate and save plots
        self._generate_plots(results)

    def _print_summary(self, results):
        """Print benchmark summary."""
        print("\n" + "=" * 100)
        print("BENCHMARK SUMMARY")
        print("=" * 100)

        for agent_name in ['greedy', 'ppo', 'sac']:
            if not results[agent_name]['metrics']:
                continue

            metrics_list = results[agent_name]['metrics']
            on_times = [m['on_time_delivery_rate'] for m in metrics_list]
            rewards = [sum(r) for r in results[agent_name]['rewards']]

            print(f"\n{agent_name.upper()}:")
            print(f"  On-time delivery:  {np.mean(on_times):.1f}% (±{np.std(on_times):.1f}%)")
            print(f"  Total reward:      {np.mean(rewards):.1f} (±{np.std(rewards):.1f})")
            print(f"  Avg daily reward:  {np.mean(rewards) / self.num_days:.1f}")

    def _generate_plots(self, results):
        """Generate and save benchmark plots."""
        # Create output directory
        os.makedirs('benchmark_results', exist_ok=True)

        print("\n" + "=" * 100)
        print("GENERATING PLOTS")
        print("=" * 100)

        # Plot 1: On-time delivery comparison
        self._plot_on_time_delivery(results)

        # Plot 2: Total reward comparison
        self._plot_total_reward(results)

        # Plot 3: Daily reward evolution
        self._plot_daily_reward_evolution(results)

        print("\nPlots saved to benchmark_results/")

    def _plot_on_time_delivery(self, results):
        """Plot on-time delivery rates comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        agent_names = []
        on_time_means = []
        on_time_stds = []

        for agent_name in ['greedy', 'ppo', 'sac']:
            if not results[agent_name]['metrics']:
                continue

            metrics_list = results[agent_name]['metrics']
            on_times = [m['on_time_delivery_rate'] for m in metrics_list]

            agent_names.append(agent_name.upper())
            on_time_means.append(np.mean(on_times))
            on_time_stds.append(np.std(on_times))

        x_pos = np.arange(len(agent_names))
        ax.bar(x_pos, on_time_means, yerr=on_time_stds, capsize=10, alpha=0.7, color=['blue', 'green', 'red'])
        ax.set_ylabel('On-Time Delivery Rate (%)', fontsize=12)
        ax.set_title('On-Time Delivery Rate Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agent_names)
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(on_time_means, on_time_stds)):
            ax.text(i, mean + std + 2, f'{mean:.1f}%', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('benchmark_results/01_on_time_delivery.png', dpi=300, bbox_inches='tight')
        print("  Saved: 01_on_time_delivery.png")
        plt.close()

    def _plot_total_reward(self, results):
        """Plot total reward comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        agent_names = []
        reward_means = []
        reward_stds = []

        for agent_name in ['greedy', 'ppo', 'sac']:
            if not results[agent_name]['metrics']:
                continue

            rewards = [sum(r) for r in results[agent_name]['rewards']]

            agent_names.append(agent_name.upper())
            reward_means.append(np.mean(rewards))
            reward_stds.append(np.std(rewards))

        x_pos = np.arange(len(agent_names))
        ax.bar(x_pos, reward_means, yerr=reward_stds, capsize=10, alpha=0.7, color=['blue', 'green', 'red'])
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('Total Episode Reward Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agent_names)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(reward_means, reward_stds)):
            ax.text(i, mean + std + 50, f'{mean:.0f}', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('benchmark_results/02_total_reward.png', dpi=300, bbox_inches='tight')
        print("  Saved: 02_total_reward.png")
        plt.close()

    def _plot_daily_reward_evolution(self, results):
        """Plot daily reward evolution over episodes."""
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = {'greedy': 'blue', 'ppo': 'green', 'sac': 'red'}

        for agent_name in ['greedy', 'ppo', 'sac']:
            if not results[agent_name]['rewards']:
                continue

            # Average daily rewards across problems
            all_daily_rewards = results[agent_name]['rewards']
            if all_daily_rewards:
                # Get average reward for each day
                avg_daily = np.mean(all_daily_rewards, axis=0)
                ax.plot(range(len(avg_daily)), avg_daily, marker='o', label=agent_name.upper(),
                       color=colors[agent_name], linewidth=2, markersize=4, alpha=0.7)

        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Average Daily Reward', fontsize=12)
        ax.set_title('Daily Reward Evolution During Episode', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('benchmark_results/03_daily_reward_evolution.png', dpi=300, bbox_inches='tight')
        print("  Saved: 03_daily_reward_evolution.png")
        plt.close()


def main():
    benchmark = SharedPolicyBenchmark(num_problems=5, num_days=100, seed=42)
    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()
