"""
Benchmark Sparse Reward Agents vs Greedy Baseline

Tests:
- Greedy baseline (deliver max from each truck's zone)
- Sparse reward PPO (trained with +1/-1 signals)
- Sparse reward SAC (trained with +1/-1 signals)

Compares how often agents beat the baseline.
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
from sparse_penalty_system import SparsePenaltySystem


class SparseRewardBenchmark:
    """Benchmark sparse reward policy agents."""

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
        self.sparse_system = SparsePenaltySystem()

    def run_greedy_baseline(self, problem: Dict) -> Tuple[Dict, List]:
        """
        Run greedy baseline: each truck delivers first available 2 packages.

        Args:
            problem: Problem instance

        Returns:
            Tuple of (metrics, sparse_signals)
        """
        # Create environment WITHOUT sparse rewards for greedy baseline
        env = PerTruckDeliveryEnv(problem=problem, penalties=self.penalties, use_sparse_rewards=False)
        obs, _ = env.reset()

        metrics = EvaluationMetrics()
        sparse_signals = []
        day_sparse_signals = []

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

                    # For greedy baseline, sparse signal is always 0 (it is the baseline)
                    day_sparse_signals.append(0.0)

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
            if day_sparse_signals:
                sparse_signals.append(sum(day_sparse_signals))
                day_sparse_signals = []

            if day < self.num_days - 1:
                env.advance_day()

        return metrics.get_episode_summary(), sparse_signals

    def run_sparse_policy_agent(self, agent, problem: Dict, agent_name: str = "Agent") -> Tuple[Dict, List]:
        """
        Run sparse reward policy agent.

        Args:
            agent: PPO or SAC agent
            problem: Problem instance
            agent_name: Name for logging

        Returns:
            Tuple of (metrics, sparse_signals)
        """
        # Create environment WITH sparse rewards enabled
        env = PerTruckDeliveryEnv(problem=problem, penalties=self.penalties, use_sparse_rewards=True)
        obs, _ = env.reset()

        metrics = EvaluationMetrics()
        sparse_signals = []
        day_sparse_signals = []

        done = False
        day_idx = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            # Sparse reward from environment
            day_sparse_signals.append(reward)

            if info.get('day_complete', False):
                if day_sparse_signals:
                    sparse_signals.append(sum(day_sparse_signals))
                    day_sparse_signals = []

                # Track metrics with stubs
                class StubPackage:
                    def __init__(self, priority=5):
                        self.priority = priority

                stub_delivered = [StubPackage(priority=5) for _ in range(info.get('delivered', 0))]
                stub_missed = [StubPackage(priority=5) for _ in range(info.get('missed', 0))]

                metrics.track_day(
                    day=day_idx,
                    delivered_packages=stub_delivered,
                    missed_packages=stub_missed,
                    deferred_packages=[],
                    trucks=env.trucks,
                    total_distance=0,
                    daily_reward=info.get('reward', 0.0)
                )

                day_idx += 1

        return metrics.get_episode_summary(), sparse_signals

    def run_all_benchmarks(self):
        """Run complete benchmark suite."""
        print("\n" + "=" * 100)
        print("SPARSE REWARD POLICY BENCHMARK")
        print("=" * 100)

        results = {
            'greedy': {'metrics': [], 'signals': []},
            'ppo_sparse': {'metrics': [], 'signals': []},
            'sac_sparse': {'metrics': [], 'signals': []},
        }

        # Load agents
        ppo_agent = None
        sac_agent = None

        if os.path.exists("ppo_sparse_policy/agent_final.zip"):
            ppo_agent = PPO.load("ppo_sparse_policy/agent_final")
            print("Loaded PPO sparse reward agent")
        else:
            print("Warning: PPO sparse agent not found at ppo_sparse_policy/agent_final.zip")

        if os.path.exists("sac_sparse_policy/agent_final.zip"):
            sac_agent = SAC.load("sac_sparse_policy/agent_final")
            print("Loaded SAC sparse reward agent")
        else:
            print("Warning: SAC sparse agent not found at sac_sparse_policy/agent_final.zip")

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
            metrics, signals = self.run_greedy_baseline(problem)
            results['greedy']['metrics'].append(metrics)
            results['greedy']['signals'].append(signals)
            print(f" On-time: {metrics['on_time_delivery_rate']:.1f}%")

            # PPO Sparse
            if ppo_agent:
                print("  Running PPO (sparse rewards)...", end="", flush=True)
                metrics, signals = self.run_sparse_policy_agent(ppo_agent, problem, "PPO-Sparse")
                results['ppo_sparse']['metrics'].append(metrics)
                results['ppo_sparse']['signals'].append(signals)
                beat_rate = (sum(1 for s in signals if s > 0) / len(signals) * 100 if signals else 0)
                print(f" Beats baseline: {beat_rate:.1f}%")
            else:
                print("  PPO sparse rewards... SKIPPED (not found)")

            # SAC Sparse
            if sac_agent:
                print("  Running SAC (sparse rewards)...", end="", flush=True)
                metrics, signals = self.run_sparse_policy_agent(sac_agent, problem, "SAC-Sparse")
                results['sac_sparse']['metrics'].append(metrics)
                results['sac_sparse']['signals'].append(signals)
                beat_rate = (sum(1 for s in signals if s > 0) / len(signals) * 100 if signals else 0)
                print(f" Beats baseline: {beat_rate:.1f}%")
            else:
                print("  SAC sparse rewards... SKIPPED (not found)")

        # Print summary
        self._print_summary(results)

        # Generate and save plots
        self._generate_plots(results)

    def _print_summary(self, results):
        """Print benchmark summary."""
        print("\n" + "=" * 100)
        print("SPARSE REWARD BENCHMARK SUMMARY")
        print("=" * 100)

        for agent_name in ['greedy', 'ppo_sparse', 'sac_sparse']:
            if not results[agent_name]['metrics']:
                continue

            metrics_list = results[agent_name]['metrics']
            signals_list = results[agent_name]['signals']

            on_times = [m['on_time_delivery_rate'] for m in metrics_list]

            print(f"\n{agent_name.upper()}:")
            print(f"  On-time delivery:  {np.mean(on_times):.1f}% (±{np.std(on_times):.1f}%)")

            # Calculate beat rate if applicable
            if agent_name != 'greedy':
                beat_rates = []
                for signals in signals_list:
                    beat_rate = (sum(1 for s in signals if s > 0) / len(signals) * 100 if signals else 0)
                    beat_rates.append(beat_rate)
                print(f"  Baseline beat rate: {np.mean(beat_rates):.1f}% (±{np.std(beat_rates):.1f}%)")
                print(f"  Avg sparse signal:  {np.mean([np.mean(s) for s in signals_list]):.2f}")

    def _generate_plots(self, results):
        """Generate and save benchmark plots."""
        os.makedirs('sparse_benchmark_results', exist_ok=True)

        print("\n" + "=" * 100)
        print("GENERATING PLOTS")
        print("=" * 100)

        # Plot 1: Baseline beat rates
        self._plot_baseline_beat_rates(results)

        # Plot 2: On-time delivery comparison
        self._plot_on_time_delivery(results)

        # Plot 3: Sparse signal evolution
        self._plot_sparse_signal_evolution(results)

        print("\nPlots saved to sparse_benchmark_results/")

    def _plot_baseline_beat_rates(self, results):
        """Plot how often agents beat baseline."""
        fig, ax = plt.subplots(figsize=(10, 6))

        agent_names = []
        beat_rates = []
        beat_stds = []

        for agent_name in ['ppo_sparse', 'sac_sparse']:
            if not results[agent_name]['signals']:
                continue

            signals_list = results[agent_name]['signals']
            rates = []
            for signals in signals_list:
                beat_rate = (sum(1 for s in signals if s > 0) / len(signals) * 100 if signals else 0)
                rates.append(beat_rate)

            agent_names.append(agent_name.replace('_sparse', '').upper())
            beat_rates.append(np.mean(rates))
            beat_stds.append(np.std(rates))

        x_pos = np.arange(len(agent_names))
        ax.bar(x_pos, beat_rates, yerr=beat_stds, capsize=10, alpha=0.7, color=['green', 'red'])
        ax.set_ylabel('Days Beating Baseline (%)', fontsize=12)
        ax.set_title('Sparse Reward Agents vs Greedy Baseline', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agent_names)
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(beat_rates, beat_stds)):
            ax.text(i, mean + std + 2, f'{mean:.1f}%', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('sparse_benchmark_results/01_baseline_beat_rates.png', dpi=300, bbox_inches='tight')
        print("  Saved: 01_baseline_beat_rates.png")
        plt.close()

    def _plot_on_time_delivery(self, results):
        """Plot on-time delivery rates comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        agent_names = []
        on_time_means = []
        on_time_stds = []

        for agent_name in ['greedy', 'ppo_sparse', 'sac_sparse']:
            if not results[agent_name]['metrics']:
                continue

            metrics_list = results[agent_name]['metrics']
            on_times = [m['on_time_delivery_rate'] for m in metrics_list]

            agent_names.append(agent_name.replace('_sparse', '').upper())
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
        plt.savefig('sparse_benchmark_results/02_on_time_delivery.png', dpi=300, bbox_inches='tight')
        print("  Saved: 02_on_time_delivery.png")
        plt.close()

    def _plot_sparse_signal_evolution(self, results):
        """Plot sparse signal evolution over episodes."""
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = {'ppo_sparse': 'green', 'sac_sparse': 'red'}

        for agent_name in ['ppo_sparse', 'sac_sparse']:
            if not results[agent_name]['signals']:
                continue

            signals_list = results[agent_name]['signals']
            if signals_list:
                avg_signals = np.mean(signals_list, axis=0)
                ax.plot(range(len(avg_signals)), avg_signals, marker='o',
                       label=agent_name.replace('_sparse', '').upper(),
                       color=colors[agent_name], linewidth=2, markersize=4, alpha=0.7)

        ax.axhline(y=0, color='blue', linestyle='--', linewidth=2, label='GREEDY BASELINE', alpha=0.5)
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Average Sparse Signal', fontsize=12)
        ax.set_title('Sparse Signal Evolution During Episode', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('sparse_benchmark_results/03_sparse_signal_evolution.png', dpi=300, bbox_inches='tight')
        print("  Saved: 03_sparse_signal_evolution.png")
        plt.close()


def main():
    benchmark = SparseRewardBenchmark(num_problems=5, num_days=100, seed=42)
    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()
