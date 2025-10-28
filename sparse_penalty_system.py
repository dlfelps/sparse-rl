"""
Sparse Reward System for Delivery Optimization

Compares policy performance against greedy baseline.
Returns sparse signals: +1 if policy better, -1 if worse, 0 if equal.
"""

from typing import List, Dict
from data_models_v2 import Package
from delivery_penalties import DeliveryPenalties


class SparsePenaltySystem:
    """Calculate sparse rewards by comparing policy to baseline."""

    def __init__(self):
        """Initialize with dense reward calculator for comparison."""
        self.penalties = DeliveryPenalties()

    def calculate_sparse_reward(self,
                                policy_delivered: List[Package],
                                policy_missed: List[Package],
                                policy_distance: float,
                                greedy_delivered: List[Package],
                                greedy_missed: List[Package],
                                greedy_distance: float) -> float:
        """
        Compare policy performance against greedy baseline.

        Args:
            policy_delivered: Packages delivered by policy
            policy_missed: Packages missed by policy
            policy_distance: Distance traveled by policy
            greedy_delivered: Packages delivered by greedy baseline
            greedy_missed: Packages missed by greedy baseline
            greedy_distance: Distance traveled by greedy baseline

        Returns:
            +1.0 if policy outperforms baseline
            -1.0 if policy underperforms baseline
             0.0 if equal
        """
        # Calculate dense rewards for both
        policy_reward_dict = self.penalties.calculate_daily_reward(
            policy_delivered, policy_missed, policy_distance)
        policy_reward = policy_reward_dict['total_reward']

        greedy_reward_dict = self.penalties.calculate_daily_reward(
            greedy_delivered, greedy_missed, greedy_distance)
        greedy_reward = greedy_reward_dict['total_reward']

        # Compare and return sparse signal
        if policy_reward > greedy_reward:
            return 1.0
        elif policy_reward < greedy_reward:
            return -1.0
        else:
            return 0.0

    def get_baseline_reward_components(self,
                                       greedy_delivered: List[Package],
                                       greedy_missed: List[Package],
                                       greedy_distance: float) -> Dict[str, float]:
        """
        Get detailed reward breakdown for baseline for debugging.

        Args:
            greedy_delivered: Packages delivered by greedy
            greedy_missed: Packages missed by greedy
            greedy_distance: Distance traveled by greedy

        Returns:
            Dict with on_time_reward, missed_penalty, distance_cost, total_reward
        """
        return self.penalties.calculate_daily_reward(
            greedy_delivered, greedy_missed, greedy_distance)
