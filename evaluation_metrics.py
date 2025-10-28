"""
Evaluation Metrics for Delivery Optimization

Tracks performance metrics across episodes:
- On-time delivery rate
- Missed deadlines
- Total reward
"""

from typing import List, Dict
from data_models_v2 import Package, Truck


class EvaluationMetrics:
    """Track and analyze performance metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.daily_metrics = []
        self.episode_packages_delivered = []
        self.episode_packages_missed = []

    def track_day(self,
                  day: int,
                  delivered_packages: List[Package],
                  missed_packages: List[Package],
                  deferred_packages: List[Package],
                  trucks: List[Truck],
                  total_distance: float,
                  daily_reward: float):
        """
        Track metrics for a single day.

        Args:
            day: Current day number
            delivered_packages: Packages delivered on time
            missed_packages: Packages that missed deadline
            deferred_packages: Packages deferred to next day
            trucks: All trucks (with current loads)
            total_distance: Total distance traveled
            daily_reward: Daily reward value
        """
        metrics = {
            'day': day,
            'num_delivered': len(delivered_packages),
            'num_missed': len(missed_packages),
            'num_deferred': len(deferred_packages),
            'total_packages_processed': len(delivered_packages) + len(missed_packages),
            'on_time_rate': (len(delivered_packages) / (len(delivered_packages) + len(missed_packages)) * 100
                            if (len(delivered_packages) + len(missed_packages)) > 0 else 0.0),
            'avg_priority_delivered': (sum(p.priority for p in delivered_packages) / len(delivered_packages)
                                      if delivered_packages else 0.0),
            'avg_priority_missed': (sum(p.priority for p in missed_packages) / len(missed_packages)
                                   if missed_packages else 0.0),
            'total_distance': total_distance,
            'daily_reward': daily_reward,
            'truck_utilizations': [truck.current_load / truck.capacity * 100 for truck in trucks],
            'avg_truck_utilization': sum(truck.current_load / truck.capacity * 100 for truck in trucks) / len(trucks)
                                    if trucks else 0.0,
        }

        self.daily_metrics.append(metrics)
        self.episode_packages_delivered.extend(delivered_packages)
        self.episode_packages_missed.extend(missed_packages)

    def get_episode_summary(self) -> Dict:
        """
        Get aggregate metrics for an episode (multiple days).

        Returns:
            Dict with episode-level statistics
        """
        if not self.daily_metrics:
            return {}

        num_days = len(self.daily_metrics)
        total_delivered = sum(m['num_delivered'] for m in self.daily_metrics)
        total_missed = sum(m['num_missed'] for m in self.daily_metrics)
        total_deferred = sum(m['num_deferred'] for m in self.daily_metrics)
        total_distance = sum(m['total_distance'] for m in self.daily_metrics)
        total_reward = sum(m['daily_reward'] for m in self.daily_metrics)

        on_time_rate = (total_delivered / (total_delivered + total_missed) * 100
                       if (total_delivered + total_missed) > 0 else 0.0)

        avg_priority_delivered = (sum(p.priority for p in self.episode_packages_delivered) / len(self.episode_packages_delivered)
                                 if self.episode_packages_delivered else 0.0)

        avg_priority_missed = (sum(p.priority for p in self.episode_packages_missed) / len(self.episode_packages_missed)
                              if self.episode_packages_missed else 0.0)

        return {
            'num_days': num_days,
            'total_delivered': total_delivered,
            'total_missed': total_missed,
            'total_deferred': total_deferred,
            'on_time_delivery_rate': on_time_rate,
            'missed_deadline_rate': 100.0 - on_time_rate,
            'avg_priority_delivered': avg_priority_delivered,
            'avg_priority_missed': avg_priority_missed,
            'total_distance': total_distance,
            'avg_distance_per_day': total_distance / num_days if num_days > 0 else 0.0,
            'total_reward': total_reward,
            'avg_daily_reward': total_reward / num_days if num_days > 0 else 0.0,
        }
