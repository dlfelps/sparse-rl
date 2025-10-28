"""
Penalty and Reward System for Constrained Delivery Problem

Reward components:
- on_time_delivery_reward: +100 per on-time delivery (priority-weighted 1-10)
- missed_deadline_penalty: -500 per expired package
- distance_cost: -0.1 per km
- Total: sum(on_time) - missed_penalties - distance_cost
"""

from typing import List, Dict, Tuple
from data_models_v2 import Package, Truck


class DeliveryPenalties:
    """Calculate penalties and rewards for constrained delivery problem."""

    def __init__(self,
                 on_time_base_reward: float = 100.0,
                 missed_deadline_penalty: float = -50.0,  # Reduced from -500 (only daily now)
                 distance_cost: float = -0.01,  # Reduced from -0.1 (minor penalty)
                 reference_distance: float = 1000.0):
        """
        Initialize penalty system.

        Args:
            on_time_base_reward: Base reward for on-time delivery (+100)
            missed_deadline_penalty: Penalty per missed deadline (-500)
            distance_cost: Cost per km (-0.1)
            reference_distance: Reference distance for normalization
        """
        self.on_time_base_reward = on_time_base_reward
        self.missed_deadline_penalty = missed_deadline_penalty
        self.distance_cost = distance_cost
        self.reference_distance = reference_distance

    def calculate_on_time_reward(self, delivered_packages: List[Package]) -> float:
        """
        Calculate reward for on-time deliveries.

        Reward is priority-weighted: package with priority 10 worth 2x more than priority 5.

        Args:
            delivered_packages: List of packages delivered on time

        Returns:
            Total on-time reward (sum of priority-weighted rewards)
        """
        if not delivered_packages:
            return 0.0

        # Priority-weighted reward
        # Priority 1-10 → reward multiplier 0.1-1.0
        total_reward = 0.0
        for pkg in delivered_packages:
            priority_weight = pkg.priority / 10.0  # 1-10 → 0.1-1.0
            total_reward += self.on_time_base_reward * priority_weight

        return total_reward

    def calculate_missed_deadline_penalty(self, missed_packages: List[Package]) -> float:
        """
        Calculate penalty for packages that missed deadline.

        Args:
            missed_packages: List of packages that expired

        Returns:
            Total penalty (negative value)
        """
        if not missed_packages:
            return 0.0

        # Fixed penalty per missed package (priority-weighted)
        total_penalty = 0.0
        for pkg in missed_packages:
            priority_weight = pkg.priority / 10.0
            total_penalty += self.missed_deadline_penalty * priority_weight

        return total_penalty

    def calculate_distance_cost(self, total_distance: float) -> float:
        """
        Calculate cost for distance traveled.

        Args:
            total_distance: Total distance in km

        Returns:
            Distance cost (negative value, -0.1 per km)
        """
        return self.distance_cost * total_distance

    def calculate_daily_reward(self,
                               delivered_packages: List[Package],
                               missed_packages: List[Package],
                               total_distance: float) -> Dict[str, float]:
        """
        Calculate total daily reward with component breakdown.

        Args:
            delivered_packages: Packages delivered on time
            missed_packages: Packages that missed deadline
            total_distance: Total distance traveled

        Returns:
            Dict with:
            - on_time_reward: Reward for deliveries
            - missed_penalty: Penalty for missed deadlines
            - distance_cost: Cost for distance
            - total_reward: Sum of all components
        """
        on_time_reward = self.calculate_on_time_reward(delivered_packages)
        missed_penalty = self.calculate_missed_deadline_penalty(missed_packages)
        distance_cost = self.calculate_distance_cost(total_distance)

        total_reward = on_time_reward + missed_penalty + distance_cost

        return {
            'on_time_reward': on_time_reward,
            'missed_penalty': missed_penalty,
            'distance_cost': distance_cost,
            'total_reward': total_reward,
            'num_delivered': len(delivered_packages),
            'num_missed': len(missed_packages),
            'total_distance': total_distance
        }

    def calculate_episode_metrics(self,
                                  episode_rewards: List[Dict]) -> Dict[str, float]:
        """
        Calculate aggregate metrics over an episode (multiple days).

        Args:
            episode_rewards: List of daily reward dicts from calculate_daily_reward()

        Returns:
            Dict with episode-level metrics
        """
        num_days = len(episode_rewards)
        total_delivered = sum(r['num_delivered'] for r in episode_rewards)
        total_missed = sum(r['num_missed'] for r in episode_rewards)
        total_distance = sum(r['total_distance'] for r in episode_rewards)
        total_reward = sum(r['total_reward'] for r in episode_rewards)

        on_time_rate = (total_delivered / (total_delivered + total_missed) * 100
                       if (total_delivered + total_missed) > 0 else 0.0)

        return {
            'num_days': num_days,
            'total_delivered': total_delivered,
            'total_missed': total_missed,
            'on_time_delivery_rate': on_time_rate,
            'missed_deadline_rate': 100.0 - on_time_rate,
            'total_distance': total_distance,
            'avg_distance_per_day': total_distance / num_days if num_days > 0 else 0.0,
            'total_reward': total_reward,
            'avg_daily_reward': total_reward / num_days if num_days > 0 else 0.0
        }


def demonstrate_penalty_system():
    """Demonstrate the penalty system with example scenarios."""

    penalties = DeliveryPenalties()

    print("\n" + "=" * 80)
    print("DELIVERY PENALTY SYSTEM DEMONSTRATION")
    print("=" * 80)

    # Example 1: Successful deliveries
    print("\nExample 1: On-time deliveries only")
    delivered = [
        Package(id=1, priority=10, destination=1, deadline=5, created_at=0),
        Package(id=2, priority=5, destination=2, deadline=5, created_at=0),
        Package(id=3, priority=8, destination=3, deadline=5, created_at=0),
    ]
    reward_dict = penalties.calculate_daily_reward(delivered, [], 150.0)
    print(f"  Delivered: {len(delivered)} packages")
    print(f"  On-time reward: +{reward_dict['on_time_reward']:.1f}")
    print(f"  Distance cost: {reward_dict['distance_cost']:.1f}")
    print(f"  Total reward: {reward_dict['total_reward']:.1f}")

    # Example 2: Some missed deadlines
    print("\nExample 2: Mixed - some delivered, some missed")
    delivered = [
        Package(id=1, priority=10, destination=1, deadline=5, created_at=0),
        Package(id=2, priority=8, destination=2, deadline=5, created_at=0),
    ]
    missed = [
        Package(id=3, priority=9, destination=3, deadline=2, created_at=0),
    ]
    reward_dict = penalties.calculate_daily_reward(delivered, missed, 200.0)
    print(f"  Delivered: {len(delivered)} packages")
    print(f"  Missed: {len(missed)} packages")
    print(f"  On-time reward: +{reward_dict['on_time_reward']:.1f}")
    print(f"  Missed penalty: {reward_dict['missed_penalty']:.1f}")
    print(f"  Distance cost: {reward_dict['distance_cost']:.1f}")
    print(f"  Total reward: {reward_dict['total_reward']:.1f}")

    # Example 3: Many missed deadlines
    print("\nExample 3: Heavy missed deadlines (bad scenario)")
    delivered = [
        Package(id=1, priority=8, destination=1, deadline=5, created_at=0),
    ]
    missed = [
        Package(id=2, priority=10, destination=2, deadline=2, created_at=0),
        Package(id=3, priority=9, destination=3, deadline=2, created_at=0),
        Package(id=4, priority=7, destination=4, deadline=2, created_at=0),
    ]
    reward_dict = penalties.calculate_daily_reward(delivered, missed, 250.0)
    print(f"  Delivered: {len(delivered)} packages")
    print(f"  Missed: {len(missed)} packages")
    print(f"  On-time reward: +{reward_dict['on_time_reward']:.1f}")
    print(f"  Missed penalty: {reward_dict['missed_penalty']:.1f}")
    print(f"  Distance cost: {reward_dict['distance_cost']:.1f}")
    print(f"  Total reward: {reward_dict['total_reward']:.1f}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("- High priority packages (10) worth 2x more than low priority (5)")
    print("- Missed deadlines have significant penalty (-500 base)")
    print("- Distance cost is relatively small (-0.1/km)")
    print("- Missing even 1 high-priority package (-450) > delivering 5 low-priority (+50)")
    print("- Total reward typically ranges from -2000 to +500 per day")


if __name__ == "__main__":
    demonstrate_penalty_system()
