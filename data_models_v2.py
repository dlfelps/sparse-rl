"""
Data Models V2 - Simplified without weight

Packages and trucks are now count-based:
- Package: no weight field (just id, priority, destination, deadline, created_at)
- Truck: capacity is number of packages (not weight units)
- Utilization: (packages_assigned / total_capacity) × 100%

This makes analysis clearer and removes the ambiguity of weight distributions.
"""

from dataclasses import dataclass
from typing import Set


@dataclass
class Package:
    """A delivery package with no weight constraint."""
    id: int              # Unique identifier
    priority: int        # 1-10 (10=urgent)
    destination: int     # Location ID
    deadline: int        # Day by which to deliver
    created_at: int      # Day package arrived

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Package):
            return self.id == other.id
        return False


@dataclass
class Truck:
    """A delivery truck with capacity measured in package count."""
    id: int              # Unique identifier
    capacity: int        # Number of packages it can carry
    current_load: int = 0        # Current number of packages assigned
    assigned_packages: Set[int] = None  # Package IDs assigned

    def __post_init__(self):
        if self.assigned_packages is None:
            self.assigned_packages = set()

    @property
    def available_capacity(self) -> int:
        """Remaining package slots."""
        return self.capacity - self.current_load

    def utilization_percent(self) -> float:
        """Current utilization as percentage."""
        return (self.current_load / self.capacity * 100) if self.capacity > 0 else 0.0


@dataclass
class State:
    """Simulation state snapshot."""
    distance_matrix: object  # N×N distance matrix
    trucks: list            # Available trucks
    current_timestep: int   # Current day
    packages_pending: list  # Undelivered packages
