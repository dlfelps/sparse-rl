"""
Constrained Capacity Problem Setup for RL Training

Problem definition:
- 10 trucks, 10 zones, 40 addresses (4 per zone) + warehouse
- Each truck capacity: 2 packages
- Daily arrivals: 30 packages (33% shortage)
- Deadlines: 1-7 days per package (INCREASED from 1-3 for variance)
- Priorities: Bimodal 1-10 scale (40% low 1-3, 40% high 8-10, 20% medium 4-7)

VARIANCE INCREASED (v2):
- Deadline range expanded to create real urgency differentiation
- Priority distribution made bimodal (extremes) instead of uniform
- This makes "which 2 packages to deliver" a meaningful decision for RL

This creates a constraint where packages > capacity, forcing daily decisions
about which to deliver and which to defer, with deadline risks.
"""

import numpy as np
from typing import List, Dict, Tuple
from data_models_v2 import Package, Truck


def create_distance_matrix(num_locations: int, seed: int = 42) -> np.ndarray:
    """
    Create a symmetric distance matrix for 40 locations.

    Locations:
    - 0: warehouse
    - 1-40: delivery addresses (4 per zone, 10 zones)

    Args:
        num_locations: Total locations (41 = warehouse + 40 addresses)
        seed: Random seed for reproducibility

    Returns:
        41x41 symmetric distance matrix
    """
    np.random.seed(seed)

    # Generate random coordinates
    coords = np.random.uniform(0, 100, size=(num_locations, 2))
    coords[0] = [50, 50]  # Warehouse at center

    # Arrange addresses in zones (4 per zone, 10 zones)
    # Zone i: addresses 4*i+1 to 4*i+4
    for zone in range(10):
        zone_center_x = 20 + (zone % 5) * 20
        zone_center_y = 30 + (zone // 5) * 40
        for addr_in_zone in range(4):
            addr_idx = zone * 4 + addr_in_zone + 1
            coords[addr_idx] = [
                zone_center_x + np.random.uniform(-5, 5),
                zone_center_y + np.random.uniform(-5, 5)
            ]

    # Calculate distance matrix
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

    # Add slight noise for realism
    noise = np.random.uniform(0.95, 1.05, size=(num_locations, num_locations))
    distance_matrix = distance_matrix * noise

    # Symmetrize and zero diagonal
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix


def create_zones() -> Dict[int, List[int]]:
    """
    Create 10 zones, each with 4 addresses.

    Returns:
        Dict: {zone_id: [address_ids]}
        - Zone 0: addresses 1-4
        - Zone 1: addresses 5-8
        - ...
        - Zone 9: addresses 37-40
    """
    zones = {}
    for zone_id in range(10):
        zones[zone_id] = list(range(zone_id * 4 + 1, zone_id * 4 + 5))
    return zones


def create_trucks(num_trucks: int = 10, capacity_per_truck: int = 2) -> List[Truck]:
    """
    Create 10 trucks, one per zone.

    Args:
        num_trucks: Number of trucks (should be 10)
        capacity_per_truck: Capacity per truck (2 packages)

    Returns:
        List of Truck objects
    """
    trucks = []
    for truck_id in range(num_trucks):
        trucks.append(Truck(id=truck_id, capacity=capacity_per_truck))
    return trucks


def generate_packages_for_day(
    day: int,
    num_packages: int = 30,
    num_zones: int = 10,
    seed_offset: int = 0
) -> List[Package]:
    """
    Generate packages for a given day.

    VARIANCE INCREASED (from constrained_problem.py v2):
    - Deadline: 1-7 days (was 1-3) - creates real urgency differentiation
    - Priority: Bimodal distribution (40% low, 40% high, 20% medium)
      - Was uniform 1-10, now skewed to extremes
    - This makes "which 2 packages to deliver" a meaningful decision

    Args:
        day: Current day (0-indexed)
        num_packages: Number of packages to generate (30 per day)
        num_zones: Number of zones (10)
        seed_offset: Offset for random seed (for variety across days)

    Returns:
        List of Package objects
    """
    np.random.seed(42 + day + seed_offset)

    packages = []

    for pkg_idx in range(num_packages):
        pkg_id = day * 10000 + pkg_idx

        # Random zone assignment (uniform distribution)
        destination_zone = np.random.randint(0, num_zones)
        zone_addresses = list(range(destination_zone * 4 + 1, destination_zone * 4 + 5))
        destination = np.random.choice(zone_addresses)

        # Priority: Bimodal distribution (increased variance)
        # 40% low (1-3), 40% high (8-10), 20% medium (4-7)
        priority_rand = np.random.rand()
        if priority_rand < 0.4:
            priority = np.random.randint(1, 4)    # Low: 1-3
        elif priority_rand < 0.8:
            priority = np.random.randint(8, 11)   # High: 8-10
        else:
            priority = np.random.randint(4, 8)    # Medium: 4-7

        # Deadline: 1-7 days from now (increased from 1-3)
        # Creates real urgency: some expire today, others have 6 days
        deadline = day + np.random.randint(1, 8)

        packages.append(Package(
            id=pkg_id,
            priority=priority,
            destination=destination,
            deadline=deadline,
            created_at=day
        ))

    return packages


def create_constrained_problem(
    num_days: int = 100,
    num_trucks: int = 10,
    truck_capacity: int = 2,
    packages_per_day: int = 30,
    num_zones: int = 10,
    seed: int = 42
) -> Dict:
    """
    Create a complete constrained capacity problem setup.

    Configuration:
    - 10 trucks, 10 zones, 40 addresses + warehouse
    - Truck capacity: 2 packages each (20 total capacity)
    - Daily arrivals: 30 packages (33% shortage)
    - Deadlines: 1-3 days
    - Priorities: 1-10

    Args:
        num_days: Number of simulation days
        num_trucks: Number of trucks
        truck_capacity: Capacity per truck
        packages_per_day: Packages arriving daily
        num_zones: Number of zones
        seed: Random seed

    Returns:
        Dict with:
        - distance_matrix: 41x41 distance matrix
        - zones: zone definitions
        - trucks: initial truck list
        - packages_by_day: List[List[Package]] for each day
    """
    problem = {
        'distance_matrix': create_distance_matrix(num_zones * 4 + 1, seed=seed),
        'zones': create_zones(),
        'trucks': create_trucks(num_trucks, truck_capacity),
        'num_trucks': num_trucks,
        'num_zones': num_zones,
        'truck_capacity': truck_capacity,
        'packages_per_day': packages_per_day,
        'total_capacity': num_trucks * truck_capacity,
        'shortage_ratio': 1.0 - (num_trucks * truck_capacity / packages_per_day),
        'packages_by_day': [
            generate_packages_for_day(day, packages_per_day, num_zones, seed)
            for day in range(num_days)
        ]
    }

    return problem


def print_problem_summary(problem: Dict):
    """Print summary of the constrained problem."""
    print("\n" + "=" * 80)
    print("CONSTRAINED CAPACITY PROBLEM SETUP")
    print("=" * 80)
    print(f"\nTrucks: {problem['num_trucks']}")
    print(f"Zones: {problem['num_zones']} (one per truck)")
    print(f"Addresses: {problem['num_zones'] * 4} (4 per zone) + warehouse")
    print(f"Distance matrix: {problem['distance_matrix'].shape[0]}x{problem['distance_matrix'].shape[0]}")

    print(f"\nCapacity:")
    print(f"  Per truck: {problem['truck_capacity']} packages")
    print(f"  Total fleet: {problem['total_capacity']} packages")

    print(f"\nDaily arrivals:")
    print(f"  Packages: {problem['packages_per_day']} per day")
    print(f"  Shortage: {problem['shortage_ratio']*100:.1f}%")
    print(f"  Excess: {problem['packages_per_day'] - problem['total_capacity']} packages/day")

    print(f"\nPackage properties:")
    print(f"  Priority: 1-10 scale (10 = most urgent)")
    print(f"  Deadline: 1-3 days from arrival")
    print(f"  Zones: uniformly distributed across 10 zones")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage
    problem = create_constrained_problem()
    print_problem_summary(problem)

    print(f"\nPackages on day 0:")
    for pkg in problem['packages_by_day'][0][:5]:
        print(f"  ID: {pkg.id}, Priority: {pkg.priority}, Dest: {pkg.destination}, "
              f"Deadline: {pkg.deadline}, Created: {pkg.created_at}")
