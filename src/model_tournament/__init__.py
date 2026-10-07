"""Read-only model tournament primitives.

The package operates only on local evaluation panels.  It does not connect to
ClickHouse, activate forecast runs, or publish forecasts.
"""

from src.model_tournament.runner import TournamentConfig, run_tournament

__all__ = ["TournamentConfig", "run_tournament"]
