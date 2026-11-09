"""Compatibility entry point for isolated rigid-object settling."""
from .isolated_object_stability import assess_settling, settling_summary, run, main

__all__ = ["assess_settling", "settling_summary", "run"]

if __name__ == "__main__":
    main()
