"""
Comprehensive Benchmark: All Agents vs Greedy Baseline

Compares 5 agents:
1. Greedy Baseline (reference)
2. Dense Reward PPO (trained on -100 to +100 scaled rewards)
3. Dense Reward SAC (trained on -100 to +100 scaled rewards)
4. Sparse Reward PPO (trained on +1/-1 signals)
5. Sparse Reward SAC (trained on +1/-1 signals)

All agents evaluated using RELATIVE VALUE metric:
- For each scenario: value_metric = agent_reward - baseline_reward
- Positive value = agent beats baseline
- Negative value = agent underperforms baseline

This allows fair comparison between agents trained with different reward signals.
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


class ComprehensiveBenchmark:
    """Comprehensive benchmark comparing all agents with relative value metrics."""

    def __init__(self, num_problems: int = 5, num_days: int = 100, seed: int = 42):
        """
        Initialize benchmark.

        Args:
            num_problems: Number of test problem instances
            num_days: Days per problem
            seed: Random seed
        """
        self.num_problems = num_problems
        self.num_days = num_days
        self.seed = seed
        self.penalties = DeliveryPenalties()

        # Store baseline and agent rewards for comparison
        self.baseline_rewards = []
        self.agent_rewards = {}

    def run_greedy_baseline(self, problem: Dict, problem_idx: int) -> Tuple[Dict, List]:
        """
        Run greedy baseline: deliver first 2 available packages per truck.

        Args:
            problem: Problem instance
            problem_idx: Index of problem (for caching)

        Returns:
            Tuple of (metrics, daily_rewards)
        """
        # Use dense reward environment (sparse=False)
        env = PerTruckDeliveryEnv(problem=problem, penalties=self.penalties, use_sparse_rewards=False)
        obs, _ = env.reset()

        metrics = EvaluationMetrics()
        daily_rewards = []
        day_reward_list = []

        for day in range(self.num_days):
            for truck_id in range(10):
                available = env._get_available_packages(truck_id)
                packages_to_deliver = available[:2]

                if packages_to_deliver:
                    env.trucks[truck_id].current_load += len(packages_to_deliver)

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
                        packages_to_deliver, missed, distance)

                    day_reward_list.append(reward_dict['total_reward'])

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
            if day_reward_list:
                daily_rewards.append(sum(day_reward_list))
                day_reward_list = []

            if day < self.num_days - 1:
                env.advance_day()

        # Cache baseline rewards for later comparison
        self.baseline_rewards.append(daily_rewards)

        return metrics.get_episode_summary(), daily_rewards

    def run_dense_policy_agent(self, agent, problem: Dict, agent_name: str = "Agent") -> Tuple[Dict, List]:
        """
        Run dense reward policy agent (PPO or SAC trained on priority-weighted rewards).

        Args:
            agent: PPO or SAC agent
            problem: Problem instance
            agent_name: Name for logging

        Returns:
            Tuple of (metrics, daily_rewards)
        """
        # Use dense reward environment (sparse=False)
        env = PerTruckDeliveryEnv(problem=problem, penalties=self.penalties, use_sparse_rewards=False)
        obs, _ = env.reset()

        metrics = EvaluationMetrics()
        daily_rewards = []
        day_reward_list = []

        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            # Dense reward from environment
            day_reward_list.append(reward)

            if info.get('day_complete', False):
                if day_reward_list:
                    daily_rewards.append(sum(day_reward_list))
                    day_reward_list = []

                # Track metrics with stubs
                class StubPackage:
                    def __init__(self, priority=5):
                        self.priority = priority

                stub_delivered = [StubPackage(priority=5) for _ in range(info.get('delivered', 0))]
                stub_missed = [StubPackage(priority=5) for _ in range(info.get('missed', 0))]

                metrics.track_day(
                    day=len(daily_rewards) - 1,
                    delivered_packages=stub_delivered,
                    missed_packages=stub_missed,
                    deferred_packages=[],
                    trucks=env.trucks,
                    total_distance=0,
                    daily_reward=sum(day_reward_list) if day_reward_list else 0
                )

        return metrics.get_episode_summary(), daily_rewards

    def run_sparse_policy_agent(self, agent, problem: Dict, problem_idx: int,
                                agent_name: str = "Agent") -> Tuple[Dict, List]:
        """
        Run sparse reward policy agent (PPO or SAC trained on +1/-1 signals).

        Key difference: Return full reward values (not sparse signals) for fair comparison.

        Args:
            agent: PPO or SAC agent
            problem: Problem instance
            problem_idx: Index of problem (for accessing baseline)
            agent_name: Name for logging

        Returns:
            Tuple of (metrics, relative_daily_rewards)
        """
        # Use sparse reward environment for training consistency,
        # but we'll calculate full rewards for evaluation
        env_sparse = PerTruckDeliveryEnv(problem=problem, penalties=self.penalties, use_sparse_rewards=True)
        env_dense = PerTruckDeliveryEnv(problem=problem, penalties=self.penalties, use_sparse_rewards=False)

        obs, _ = env_sparse.reset()
        obs_dense, _ = env_dense.reset()

        metrics = EvaluationMetrics()
        daily_rewards = []
        daily_relative_rewards = []  # Value relative to baseline
        day_reward_list = []
        day_reward_dense_list = []

        done = False
        day_idx = 0

        while not done:
            # Get action from sparse-trained agent
            action, _ = agent.predict(obs, deterministic=True)

            # Execute on both environments
            obs, sparse_reward, done, truncated, info = env_sparse.step(action)
            obs_dense, dense_reward, _, _, info_dense = env_dense.step(action)

            # Collect full dense rewards (not sparse signals)
            day_reward_dense_list.append(dense_reward)

            if info.get('day_complete', False):
                if day_reward_dense_list:
                    daily_reward = sum(day_reward_dense_list)
                    daily_rewards.append(daily_reward)

                    # Calculate relative value vs baseline
                    baseline_reward = self.baseline_rewards[problem_idx][day_idx]
                    relative_value = daily_reward - baseline_reward
                    daily_relative_rewards.append(relative_value)

                    day_reward_dense_list = []
                    day_idx += 1

                # Track metrics with stubs
                class StubPackage:
                    def __init__(self, priority=5):
                        self.priority = priority

                stub_delivered = [StubPackage(priority=5) for _ in range(info.get('delivered', 0))]
                stub_missed = [StubPackage(priority=5) for _ in range(info.get('missed', 0))]

                metrics.track_day(
                    day=day_idx - 1,
                    delivered_packages=stub_delivered,
                    missed_packages=stub_missed,
                    deferred_packages=[],
                    trucks=env_sparse.trucks,
                    total_distance=0,
                    daily_reward=sum(day_reward_dense_list) if day_reward_dense_list else 0
                )

        return metrics.get_episode_summary(), daily_relative_rewards

    def run_all_benchmarks(self):
        """Run complete benchmark suite comparing all 5 agents."""
        print("\n" + "=" * 120)
        print("COMPREHENSIVE BENCHMARK: ALL AGENTS WITH RELATIVE VALUE METRICS")
        print("=" * 120)

        results = {
            'baseline': {'metrics': [], 'rewards': [], 'relative_rewards': []},
            'dense_ppo': {'metrics': [], 'rewards': [], 'relative_rewards': []},
            'dense_sac': {'metrics': [], 'rewards': [], 'relative_rewards': []},
            'sparse_ppo': {'metrics': [], 'rewards': [], 'relative_rewards': []},
            'sparse_sac': {'metrics': [], 'rewards': [], 'relative_rewards': []},
        }

        # Load all trained agents
        agents = {
            'dense_ppo': None,
            'dense_sac': None,
            'sparse_ppo': None,
            'sparse_sac': None,
        }

        if os.path.exists("ppo_shared_policy/agent_final.zip"):
            agents['dense_ppo'] = PPO.load("ppo_shared_policy/agent_final")
            print("[OK] Loaded Dense Reward PPO agent")
        else:
            print("[SKIP] Dense Reward PPO not found")

        if os.path.exists("sac_shared_policy/agent_final.zip"):
            agents['dense_sac'] = SAC.load("sac_shared_policy/agent_final")
            print("[OK] Loaded Dense Reward SAC agent")
        else:
            print("[SKIP] Dense Reward SAC not found")

        if os.path.exists("ppo_sparse_policy/agent_final.zip"):
            agents['sparse_ppo'] = PPO.load("ppo_sparse_policy/agent_final")
            print("[OK] Loaded Sparse Reward PPO agent")
        else:
            print("[SKIP] Sparse Reward PPO not found")

        if os.path.exists("sac_sparse_policy/agent_final.zip"):
            agents['sparse_sac'] = SAC.load("sac_sparse_policy/agent_final")
            print("[OK] Loaded Sparse Reward SAC agent")
        else:
            print("[SKIP] Sparse Reward SAC not found")

        # Run benchmarks
        for problem_idx in range(self.num_problems):
            print(f"\n{'=' * 120}")
            print(f"Problem {problem_idx + 1}/{self.num_problems}")
            print('=' * 120)

            problem = create_constrained_problem(
                num_days=self.num_days,
                num_trucks=10,
                truck_capacity=2,
                packages_per_day=30,
                num_zones=10,
                seed=self.seed + problem_idx
            )

            # Greedy baseline
            print("  [BASELINE] Running greedy baseline...", end="", flush=True)
            metrics, rewards = self.run_greedy_baseline(problem, problem_idx)
            results['baseline']['metrics'].append(metrics)
            results['baseline']['rewards'].append(rewards)
            # Baseline relative reward is always 0 (it's the reference)
            results['baseline']['relative_rewards'].append([0.0] * len(rewards))
            print(f" [OK] Total reward: {sum(rewards):.0f}")

            # Dense Reward PPO
            if agents['dense_ppo']:
                print("  [DENSE PPO] Running dense reward PPO...", end="", flush=True)
                metrics, rewards = self.run_dense_policy_agent(agents['dense_ppo'], problem, "DensePPO")
                results['dense_ppo']['metrics'].append(metrics)
                results['dense_ppo']['rewards'].append(rewards)

                # Calculate relative rewards
                baseline_rewards = results['baseline']['rewards'][problem_idx]
                relative = [agent_r - base_r for agent_r, base_r in zip(rewards, baseline_rewards)]
                results['dense_ppo']['relative_rewards'].append(relative)
                avg_relative = np.mean(relative)
                print(f" [OK] Avg relative value: {avg_relative:+.1f}")

            # Dense Reward SAC
            if agents['dense_sac']:
                print("  [DENSE SAC] Running dense reward SAC...", end="", flush=True)
                metrics, rewards = self.run_dense_policy_agent(agents['dense_sac'], problem, "DenseSAC")
                results['dense_sac']['metrics'].append(metrics)
                results['dense_sac']['rewards'].append(rewards)

                # Calculate relative rewards
                baseline_rewards = results['baseline']['rewards'][problem_idx]
                relative = [agent_r - base_r for agent_r, base_r in zip(rewards, baseline_rewards)]
                results['dense_sac']['relative_rewards'].append(relative)
                avg_relative = np.mean(relative)
                print(f" [OK] Avg relative value: {avg_relative:+.1f}")

            # Sparse Reward PPO
            if agents['sparse_ppo']:
                print("  [SPARSE PPO] Running sparse reward PPO...", end="", flush=True)
                metrics, relative = self.run_sparse_policy_agent(agents['sparse_ppo'], problem,
                                                                 problem_idx, "SparsePPO")
                results['sparse_ppo']['metrics'].append(metrics)
                # For sparse, we already have relative rewards
                results['sparse_ppo']['relative_rewards'].append(relative)
                # Store absolute rewards by adding back baseline
                baseline_rewards = results['baseline']['rewards'][problem_idx]
                absolute = [rel + base for rel, base in zip(relative, baseline_rewards)]
                results['sparse_ppo']['rewards'].append(absolute)
                avg_relative = np.mean(relative)
                print(f" [OK] Avg relative value: {avg_relative:+.1f}")

            # Sparse Reward SAC
            if agents['sparse_sac']:
                print("  [SPARSE SAC] Running sparse reward SAC...", end="", flush=True)
                metrics, relative = self.run_sparse_policy_agent(agents['sparse_sac'], problem,
                                                                problem_idx, "SparseSAC")
                results['sparse_sac']['metrics'].append(metrics)
                # For sparse, we already have relative rewards
                results['sparse_sac']['relative_rewards'].append(relative)
                # Store absolute rewards by adding back baseline
                baseline_rewards = results['baseline']['rewards'][problem_idx]
                absolute = [rel + base for rel, base in zip(relative, baseline_rewards)]
                results['sparse_sac']['rewards'].append(absolute)
                avg_relative = np.mean(relative)
                print(f" [OK] Avg relative value: {avg_relative:+.1f}")

        # Print summary
        self._print_summary(results)

        # Generate and save plots
        self._generate_plots(results)

    def _print_summary(self, results):
        """Print comprehensive benchmark summary."""
        print("\n" + "=" * 120)
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print("=" * 120)

        agent_names = ['baseline', 'dense_ppo', 'dense_sac', 'sparse_ppo', 'sparse_sac']
        labels = ['GREEDY BASELINE', 'DENSE PPO', 'DENSE SAC', 'SPARSE PPO', 'SPARSE SAC']

        # Absolute rewards
        print("\nABSOLUTE REWARD METRICS:")
        print("-" * 120)
        for agent, label in zip(agent_names, labels):
            if not results[agent]['rewards']:
                continue

            total_rewards = [sum(r) for r in results[agent]['rewards']]
            print(f"\n{label}:")
            print(f"  Total episode reward:  {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
            print(f"  Avg daily reward:      {np.mean(total_rewards) / self.num_days:.1f}")

        # Relative value metrics
        print("\n\nRELATIVE VALUE METRICS (vs Greedy Baseline):")
        print("-" * 120)
        for agent, label in zip(agent_names[1:], labels[1:]):  # Skip baseline
            if not results[agent]['relative_rewards']:
                continue

            # Flatten all relative rewards across all days and problems
            all_relative = []
            for problem_relative in results[agent]['relative_rewards']:
                all_relative.extend(problem_relative)

            positive_days = sum(1 for r in all_relative if r > 0)
            negative_days = sum(1 for r in all_relative if r < 0)
            total_days = len(all_relative)

            print(f"\n{label}:")
            print(f"  Avg daily relative value: {np.mean(all_relative):+.1f}")
            print(f"  Std deviation:            {np.std(all_relative):.1f}")
            print(f"  Days beating baseline:    {positive_days}/{total_days} ({positive_days/total_days*100:.1f}%)")
            print(f"  Days below baseline:      {negative_days}/{total_days} ({negative_days/total_days*100:.1f}%)")
            print(f"  Max daily advantage:      {max(all_relative):+.1f}")
            print(f"  Max daily disadvantage:   {min(all_relative):+.1f}")

    def _generate_plots(self, results):
        """Generate and save comprehensive plots."""
        os.makedirs('comprehensive_results', exist_ok=True)

        print("\n" + "=" * 120)
        print("GENERATING COMPREHENSIVE PLOTS")
        print("=" * 120)

        # Plot 1: Absolute total rewards comparison
        self._plot_absolute_rewards(results)

        # Plot 2: Relative value comparison (advantage over baseline)
        self._plot_relative_values(results)

        # Plot 3: Win rate comparison (% days beating baseline)
        self._plot_win_rates(results)

        # Plot 4: Daily relative value evolution
        self._plot_relative_evolution(results)

        print("\nPlots saved to comprehensive_results/")

    def _plot_absolute_rewards(self, results):
        """Plot absolute total episode rewards."""
        fig, ax = plt.subplots(figsize=(12, 6))

        agent_names = ['baseline', 'dense_ppo', 'dense_sac', 'sparse_ppo', 'sparse_sac']
        labels = ['GREEDY\nBASELINE', 'DENSE\nPPO', 'DENSE\nSAC', 'SPARSE\nPPO', 'SPARSE\nSAC']
        colors = ['gray', 'green', 'darkgreen', 'orange', 'darkorange']

        means = []
        stds = []

        for agent in agent_names:
            if not results[agent]['rewards']:
                continue
            total_rewards = [sum(r) for r in results[agent]['rewards']]
            means.append(np.mean(total_rewards))
            stds.append(np.std(total_rewards))

        x_pos = np.arange(len(means))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.8, color=colors)

        ax.set_ylabel('Total Episode Reward', fontsize=12, fontweight='bold')
        ax.set_title('Total Episode Reward: All Agents', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 100, f'{mean:.0f}', ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('comprehensive_results/01_absolute_rewards.png', dpi=300, bbox_inches='tight')
        print("  Saved: 01_absolute_rewards.png")
        plt.close()

    def _plot_relative_values(self, results):
        """Plot average relative value (advantage) over baseline."""
        fig, ax = plt.subplots(figsize=(12, 6))

        agent_names = ['dense_ppo', 'dense_sac', 'sparse_ppo', 'sparse_sac']
        labels = ['DENSE\nPPO', 'DENSE\nSAC', 'SPARSE\nPPO', 'SPARSE\nSAC']
        colors = ['green', 'darkgreen', 'orange', 'darkorange']

        means = []
        stds = []

        for agent in agent_names:
            if not results[agent]['relative_rewards']:
                continue

            # Flatten all relative rewards
            all_relative = []
            for problem_relative in results[agent]['relative_rewards']:
                all_relative.extend(problem_relative)

            means.append(np.mean(all_relative))
            stds.append(np.std(all_relative))

        x_pos = np.arange(len(means))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.8, color=colors)

        # Add baseline reference line
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, label='Baseline (0)')

        ax.set_ylabel('Avg Daily Relative Value', fontsize=12, fontweight='bold')
        ax.set_title('Relative Value vs Greedy Baseline (Positive = Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=10)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            offset = std + 5 if mean >= 0 else -std - 15
            ax.text(i, mean + offset, f'{mean:+.1f}', ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('comprehensive_results/02_relative_values.png', dpi=300, bbox_inches='tight')
        print("  Saved: 02_relative_values.png")
        plt.close()

    def _plot_win_rates(self, results):
        """Plot percentage of days beating baseline."""
        fig, ax = plt.subplots(figsize=(12, 6))

        agent_names = ['dense_ppo', 'dense_sac', 'sparse_ppo', 'sparse_sac']
        labels = ['DENSE\nPPO', 'DENSE\nSAC', 'SPARSE\nPPO', 'SPARSE\nSAC']
        colors = ['green', 'darkgreen', 'orange', 'darkorange']

        win_rates = []

        for agent in agent_names:
            if not results[agent]['relative_rewards']:
                continue

            # Flatten all relative rewards
            all_relative = []
            for problem_relative in results[agent]['relative_rewards']:
                all_relative.extend(problem_relative)

            win_rate = sum(1 for r in all_relative if r > 0) / len(all_relative) * 100
            win_rates.append(win_rate)

        x_pos = np.arange(len(win_rates))
        bars = ax.bar(x_pos, win_rates, alpha=0.8, color=colors)

        # Add 50% reference line
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=2, label='50% (Fair)')

        ax.set_ylabel('% Days Beating Baseline', fontsize=12, fontweight='bold')
        ax.set_title('Win Rate vs Greedy Baseline', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=10)

        # Add value labels on bars
        for i, rate in enumerate(win_rates):
            ax.text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('comprehensive_results/03_win_rates.png', dpi=300, bbox_inches='tight')
        print("  Saved: 03_win_rates.png")
        plt.close()

    def _plot_relative_evolution(self, results):
        """Plot relative value evolution over 100-day episodes."""
        fig, ax = plt.subplots(figsize=(14, 7))

        agent_names = ['dense_ppo', 'dense_sac', 'sparse_ppo', 'sparse_sac']
        labels = ['Dense PPO', 'Dense SAC', 'Sparse PPO', 'Sparse SAC']
        colors = ['green', 'darkgreen', 'orange', 'darkorange']

        for agent, label, color in zip(agent_names, labels, colors):
            if not results[agent]['relative_rewards']:
                continue

            # Average relative values across all problems for each day
            avg_relative_by_day = np.mean(results[agent]['relative_rewards'], axis=0)

            # Plot with moving average for smoothness
            days = np.arange(1, len(avg_relative_by_day) + 1)
            ax.plot(days, avg_relative_by_day, marker='o', label=label, color=color,
                   linewidth=2.5, markersize=4, alpha=0.8)

        # Baseline reference line
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, label='Baseline')

        ax.set_xlabel('Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Relative Value', fontsize=12, fontweight='bold')
        ax.set_title('Relative Value Evolution During Episode', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('comprehensive_results/04_relative_evolution.png', dpi=300, bbox_inches='tight')
        print("  Saved: 04_relative_evolution.png")
        plt.close()


def main():
    benchmark = ComprehensiveBenchmark(num_problems=5, num_days=100, seed=42)
    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()
