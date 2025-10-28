"""
Per-Truck Delivery Environment for Shared Policy Learning

Each truck independently decides which 2 packages to deliver from its available set.
A single agent is trained on all trucks, learning a zone-agnostic policy.

Key differences from centralized:
- Action: scores for each available package (agent ranks them)
- Observation: zone-agnostic (priority, deadline, distance, truck load)
- Training: 1 agent sees 10 trucks per day (1000 decisions per episode)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any
from constrained_problem import create_constrained_problem
from delivery_penalties import DeliveryPenalties
from data_models_v2 import Package, Truck
from sparse_penalty_system import SparsePenaltySystem


class PerTruckDeliveryEnv(gym.Env):
    """Per-truck environment where each truck decides which 2 packages to deliver."""

    metadata = {"render_modes": []}

    def __init__(self,
                 problem: Dict,
                 penalties: DeliveryPenalties,
                 max_packages_per_truck: int = 8,
                 use_sparse_rewards: bool = False):
        """
        Initialize per-truck environment.

        Args:
            problem: Problem setup dict from constrained_problem.py
            penalties: Penalty system for reward calculation
            max_packages_per_truck: Maximum packages per truck (for padding)
            use_sparse_rewards: If True, return +1/-1 sparse signals vs dense rewards
        """
        self.problem = problem
        self.penalties = penalties
        self.max_packages_per_truck = max_packages_per_truck
        self.use_sparse_rewards = use_sparse_rewards

        # Initialize sparse penalty system if needed
        if self.use_sparse_rewards:
            self.sparse_system = SparsePenaltySystem()

        self.distance_matrix = problem['distance_matrix']
        self.packages_by_day = problem['packages_by_day']
        self.truck_capacity = problem['truck_capacity']
        self.num_trucks = problem['num_trucks']
        self.num_zones = problem['num_zones']
        self.num_days = len(self.packages_by_day)

        # State tracking
        self.current_day = 0
        self.current_truck_idx = 0  # Which truck is acting this step
        self.pending_packages = {i: [] for i in range(self.num_trucks)}  # Per-truck backlog
        self.trucks = None
        self.all_packages_today = []

        # Zone to addresses mapping
        self.zone_addresses = {}
        for zone_id in range(self.num_zones):
            self.zone_addresses[zone_id] = list(range(zone_id * 4 + 1, zone_id * 4 + 5))

        # Observation space: zone-agnostic features for up to max_packages_per_truck
        # Per package: [priority, deadline_urgency, distance] = 3 features
        # Truck state: [current_load, capacity, day, total_pending] = 4 features
        state_size = (self.max_packages_per_truck * 3) + 4

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(state_size,),
            dtype=np.float32
        )

        # Action space: continuous scores for each package (agent ranks them)
        self.action_space = spaces.Box(
            low=0, high=1,
            shape=(self.max_packages_per_truck,),
            dtype=np.float32
        )

    def _get_package_zone(self, package: Package) -> int:
        """Get zone ID for a package destination."""
        for zone_id, addresses in self.zone_addresses.items():
            if package.destination in addresses:
                return zone_id
        return 0

    def _get_distance(self, location_id: int) -> float:
        """Get distance from warehouse (location 0) to a location."""
        return self.distance_matrix[0, location_id]

    def _get_truck_zone(self, truck_id: int) -> int:
        """Get zone ID for a truck."""
        return truck_id % self.num_zones

    def _get_available_packages(self, truck_id: int) -> List[Package]:
        """
        Get packages available for a specific truck.

        Includes:
        - Deferred packages from this truck
        - New arrivals in this truck's zone

        Returns:
            List of available packages (up to max_packages_per_truck)
        """
        truck_zone = self._get_truck_zone(truck_id)

        # Deferred from this truck
        available = self.pending_packages[truck_id].copy()

        # New arrivals in this truck's zone
        zone_packages = [
            p for p in self.all_packages_today
            if self._get_package_zone(p) == truck_zone
        ]

        available.extend(zone_packages)

        # Filter to packages that haven't expired yet
        available = [p for p in available if p.deadline > self.current_day]

        # Sort by ID for consistent (but NOT by priority) ordering
        # Agent will rank them via action scores, not by pre-sorted priority
        available = sorted(available, key=lambda p: p.id)

        return available[:self.max_packages_per_truck]

    def _encode_observation(self, truck_id: int) -> np.ndarray:
        """
        Encode observation for a specific truck (zone-agnostic).

        Features:
        - Package features (up to max): priority, deadline_urgency, distance
        - Truck state: current_load, capacity, day, total_pending

        Returns:
            Normalized feature vector [0, 1]
        """
        features = []

        available_packages = self._get_available_packages(truck_id)

        # Package features (zone-agnostic!)
        for idx in range(self.max_packages_per_truck):
            if idx < len(available_packages):
                pkg = available_packages[idx]

                # Normalize features
                norm_priority = pkg.priority / 10.0  # [0, 1]
                days_to_deadline = max(0, pkg.deadline - self.current_day)
                # Urgency increases as deadline approaches: 0 days left = 1.0, 3+ days = 0.0
                norm_urgency = max(0.0, 1.0 - (days_to_deadline / 3.0))  # [0, 1]
                distance = self._get_distance(pkg.destination)
                norm_distance = min(distance / 100.0, 1.0)  # [0, 1]

                features.extend([
                    norm_priority,
                    norm_urgency,
                    norm_distance
                ])
            else:
                # Padding for missing packages
                features.extend([0.0, 0.0, 0.0])

        # Truck state (zone-agnostic!)
        truck = self.trucks[truck_id]
        norm_load = truck.current_load / truck.capacity
        norm_capacity = 1.0  # Always 2 (constant)
        norm_day = self.current_day / self.num_days
        total_pending = sum(len(self.pending_packages[i]) for i in range(self.num_trucks))
        norm_pending = min(total_pending / 200.0, 1.0)

        features.extend([
            norm_load,
            norm_capacity,
            norm_day,
            norm_pending
        ])

        features = np.array(features, dtype=np.float32)
        return features[:self.observation_space.shape[0]].astype(np.float32)

    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        self.current_day = 0
        self.current_truck_idx = 0
        self.pending_packages = {i: [] for i in range(self.num_trucks)}
        self.all_packages_today = self.packages_by_day[0].copy()

        # Initialize trucks
        self.trucks = [
            Truck(id=i, capacity=self.truck_capacity)
            for i in range(self.num_trucks)
        ]

        # Get observation for first truck
        observation = self._encode_observation(truck_id=self.current_truck_idx)
        info = {}

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step (one truck acts).

        The environment manages which truck acts internally (rotating through all trucks per day).

        Args:
            action: Continuous scores [0, 1] for each available package
                   Agent learns to rank packages by score

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Current truck acting
        truck_id = self.current_truck_idx

        # Handle both continuous (array) and scalar inputs
        if isinstance(action, np.ndarray):
            action_scores = action
        else:
            action_scores = np.array([action])

        # Get available packages for this truck
        available_packages = self._get_available_packages(truck_id)

        if len(available_packages) == 0:
            # No packages available, deliver nothing

            # Advance to next truck
            self.current_truck_idx += 1

            # Check if day is complete
            terminated = False
            if self.current_truck_idx >= self.num_trucks:
                self.current_truck_idx = 0
                self.advance_day()
                if self.is_episode_done():
                    terminated = True

            next_truck_idx = self.current_truck_idx
            if next_truck_idx >= self.num_trucks:
                next_truck_idx = 0

            observation = self._encode_observation(truck_id=next_truck_idx)

            info = {
                'delivered': 0,
                'missed': 0,
                'reward': 0.0,
                'truck_id': truck_id,
                'day_complete': next_truck_idx == 0
            }

            return observation, 0.0, terminated, False, info

        # Score each available package, pick top 2
        num_available = len(available_packages)
        scores = action_scores[:num_available]

        # Get indices of top 2 packages
        if num_available >= 2:
            top_indices = np.argsort(scores)[-2:][::-1]  # Top 2, descending
            packages_to_deliver = [available_packages[i] for i in top_indices]
        elif num_available == 1:
            packages_to_deliver = [available_packages[0]]
        else:
            packages_to_deliver = []

        # Update truck load
        self.trucks[truck_id].current_load += len(packages_to_deliver)

        # Remove delivered packages from pending
        delivered_ids = {pkg.id for pkg in packages_to_deliver}
        self.pending_packages[truck_id] = [
            p for p in self.pending_packages[truck_id] if p.id not in delivered_ids
        ]

        # Calculate reward for this truck only
        delivered = packages_to_deliver

        # Count misses: packages that expire TODAY from this truck's inventory
        all_truck_packages = self.pending_packages[truck_id] + [
            p for p in self.all_packages_today
            if self._get_truck_zone(truck_id) == self._get_package_zone(p)
        ]

        missed = [
            p for p in all_truck_packages
            if p.deadline == self.current_day and p.id not in delivered_ids
        ]

        # Distance cost for delivered packages
        total_distance = sum(
            2 * self._get_distance(pkg.destination)
            for pkg in delivered
        )

        # Get reward for this truck
        reward_dict = self.penalties.calculate_daily_reward(
            delivered,
            missed,
            total_distance
        )

        truck_reward = reward_dict['total_reward']

        # Calculate greedy baseline for sparse reward comparison
        if self.use_sparse_rewards:
            # Greedy baseline: deliver first 2 packages (by ID order)
            greedy_to_deliver = available_packages[:min(2, len(available_packages))]
            greedy_distance = sum(
                2 * self._get_distance(pkg.destination)
                for pkg in greedy_to_deliver
            )

            # Count greedy misses
            greedy_delivered_ids = {pkg.id for pkg in greedy_to_deliver}
            greedy_missed = [
                p for p in all_truck_packages
                if p.deadline == self.current_day and p.id not in greedy_delivered_ids
            ]

            # Calculate sparse reward
            truck_reward = self.sparse_system.calculate_sparse_reward(
                delivered, missed, total_distance,
                greedy_to_deliver, greedy_missed, greedy_distance
            )

        # Advance to next truck
        self.current_truck_idx += 1

        # Check if day is complete (all trucks acted)
        terminated = False
        if self.current_truck_idx >= self.num_trucks:
            # End of day: advance to next day
            self.current_truck_idx = 0
            self.advance_day()

            # Check if episode is done
            if self.is_episode_done():
                terminated = True

        # Get next observation for next truck (or first truck of next day)
        next_truck_idx = self.current_truck_idx
        if next_truck_idx >= self.num_trucks:
            next_truck_idx = 0

        observation = self._encode_observation(truck_id=next_truck_idx)

        info = {
            'delivered': len(delivered),
            'missed': len(missed),
            'reward': truck_reward,
            'truck_id': truck_id,
            'day_complete': next_truck_idx == 0
        }

        return observation, truck_reward, terminated, False, info

    def advance_day(self) -> None:
        """
        Advance to next day (called after all trucks have acted).

        Updates:
        - Current day
        - Packages for new day
        - Truck loads reset
        """
        self.current_day += 1

        # Get next day's packages
        if self.current_day < self.num_days:
            self.all_packages_today = self.packages_by_day[self.current_day].copy()
            # Add new arrivals to each truck's zone
            for truck_id in range(self.num_trucks):
                truck_zone = self._get_truck_zone(truck_id)
                zone_packages = [
                    p for p in self.all_packages_today
                    if self._get_package_zone(p) == truck_zone
                ]
                self.pending_packages[truck_id].extend(zone_packages)
        else:
            self.all_packages_today = []

        # Reset truck loads for next day
        for truck in self.trucks:
            truck.current_load = 0

    def is_episode_done(self) -> bool:
        """Check if episode is complete."""
        return self.current_day >= self.num_days
