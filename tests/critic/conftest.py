"""Shared fixtures for the critic acceptance tests (Round 1). Fault IDs refer to docs/CRITIC_LOG.md."""
import warnings

import pandas as pd
import pytest

from src.config import ROOT, load_assumptions
from src.model import risk as R
from tests.critic.simworld import fresh_lots

warnings.filterwarnings("ignore")
OUTCOMES = ["metaphylaxis_given", "brd_cases_45d", "morbidity_45d", "high_risk"]


def arrivals_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith("_")] + [c for c in OUTCOMES if c in df.columns])


@pytest.fixture(scope="session")
def cfg():
    return load_assumptions()


@pytest.fixture(scope="session")
def hist():
    return pd.read_csv(ROOT / "data" / "historical_cohorts.csv", parse_dates=["arrival_date"])


@pytest.fixture(scope="session")
def cur():
    return pd.read_csv(ROOT / "data" / "current_arrivals.csv")


@pytest.fixture(scope="session")
def model(hist):
    return R.fit(hist, n_boot=30, seed=0)


@pytest.fixture(scope="session")
def fresh_untreated(cfg):
    """1,000 fresh lots from the seed-42 world, nobody treated (so observed = untreated outcome)."""
    return fresh_lots(1000, seed=777, cfg=cfg, meta_probs=(0.0, 0.0))


@pytest.fixture(scope="session")
def fresh_assessed(model, fresh_untreated, cfg):
    r, D = R.assess(model, arrivals_only(fresh_untreated), cfg, n_draws=40)
    return r, D


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: critic round-2 tests that refit several models (about 1 minute each)")
