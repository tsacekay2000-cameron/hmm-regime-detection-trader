"""Monte Carlo simulation for strategy robustness and risk analysis.

A single backtest is one path out of many that the same edge could have
produced. This module generates many alternative return paths and summarizes
the resulting distribution of outcomes (final return, CAGR, max drawdown,
Sharpe, risk of ruin).

Path generators
---------------
- ``shuffle_paths``          reorder historical returns (same total, new drawdowns)
- ``bootstrap_paths``        i.i.d. resampling with replacement
- ``block_bootstrap_paths``  resample contiguous blocks (keeps volatility clustering)
- ``simulate_hmm``           sample synthetic regimes + returns from a Gaussian HMM
- ``simulate_hmm_strategy``  run a regime-based strategy on HMM-sampled paths,
                             detecting the regime with a causal forward filter

All generators return arrays shaped ``(n_sims, horizon)`` of simple per-period
returns, which ``summarize`` turns into a ``MonteCarloResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


# ---------------------------------------------------------------------------
# Path generators
# ---------------------------------------------------------------------------

def _as_returns(returns: ArrayLike) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns must be finite")
    if np.any(arr <= -1.0):
        raise ValueError("simple returns must be greater than -1")
    return arr


def shuffle_paths(
    returns: ArrayLike,
    n_sims: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Randomly permute the return sequence ``n_sims`` times.

    Every path has the same compounded total return as the original; only the
    ordering (and therefore the drawdown profile) changes.
    """
    arr = _as_returns(returns)
    rng = rng or np.random.default_rng()
    return rng.permuted(np.tile(arr, (n_sims, 1)), axis=1)


def bootstrap_paths(
    returns: ArrayLike,
    n_sims: int = 1000,
    horizon: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Resample returns i.i.d. with replacement.

    Assumes returns are independent, so it understates streaks in markets with
    volatility clustering. Prefer ``block_bootstrap_paths`` for daily data.
    """
    arr = _as_returns(returns)
    rng = rng or np.random.default_rng()
    horizon = horizon or arr.size
    return arr[rng.integers(0, arr.size, size=(n_sims, horizon))]


def block_bootstrap_paths(
    returns: ArrayLike,
    n_sims: int = 1000,
    horizon: Optional[int] = None,
    block_size: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Circular block bootstrap: stitch together random contiguous blocks.

    Preserves short-range dependence (volatility clustering, momentum) within
    each block of ``block_size`` periods.
    """
    arr = _as_returns(returns)
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    rng = rng or np.random.default_rng()
    horizon = horizon or arr.size
    n_blocks = -(-horizon // block_size)  # ceil division
    starts = rng.integers(0, arr.size, size=(n_sims, n_blocks, 1))
    idx = (starts + np.arange(block_size)) % arr.size  # wrap around the end
    return arr[idx.reshape(n_sims, -1)[:, :horizon]]


# ---------------------------------------------------------------------------
# HMM simulation
# ---------------------------------------------------------------------------

@dataclass
class GaussianHMMParams:
    """Parameters of a Gaussian HMM over per-period simple returns.

    Maps directly onto a fitted ``hmmlearn.hmm.GaussianHMM`` with 1-D
    observations: ``startprob_``, ``transmat_``, ``means_.ravel()`` and
    ``np.sqrt(covars_.ravel())``.
    """

    transmat: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    startprob: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.transmat = np.asarray(self.transmat, dtype=float)
        self.means = np.asarray(self.means, dtype=float).ravel()
        self.stds = np.asarray(self.stds, dtype=float).ravel()
        k = self.means.size
        if self.transmat.shape != (k, k):
            raise ValueError(f"transmat must be {k}x{k}")
        if not np.allclose(self.transmat.sum(axis=1), 1.0):
            raise ValueError("transmat rows must sum to 1")
        if self.stds.size != k or np.any(self.stds <= 0):
            raise ValueError("stds must be positive, one per state")
        if self.startprob is None:
            self.startprob = stationary_distribution(self.transmat)
        self.startprob = np.asarray(self.startprob, dtype=float).ravel()
        if self.startprob.size != k or not np.isclose(self.startprob.sum(), 1.0):
            raise ValueError("startprob must have one entry per state and sum to 1")

    @property
    def n_states(self) -> int:
        return self.means.size

    @classmethod
    def from_hmmlearn(cls, model) -> "GaussianHMMParams":
        """Build params from a fitted 1-D ``hmmlearn`` GaussianHMM."""
        covars = np.asarray(model.covars_, dtype=float).reshape(model.n_components, -1)
        return cls(
            transmat=model.transmat_,
            means=np.asarray(model.means_).ravel(),
            stds=np.sqrt(covars[:, 0]),
            startprob=model.startprob_,
        )


def stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Long-run fraction of time spent in each regime."""
    transmat = np.asarray(transmat, dtype=float)
    vals, vecs = np.linalg.eig(transmat.T)
    vec = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return vec / vec.sum()


def simulate_hmm(
    params: GaussianHMMParams,
    n_sims: int = 1000,
    horizon: int = 252,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample regime sequences and returns from a Gaussian HMM.

    Returns ``(returns, states)``, both shaped ``(n_sims, horizon)``. Returns are
    clipped just above -100% so compounding stays well defined.
    """
    rng = rng or np.random.default_rng()
    cum_trans = np.cumsum(params.transmat, axis=1)
    states = np.empty((n_sims, horizon), dtype=int)
    states[:, 0] = rng.choice(params.n_states, size=n_sims, p=params.startprob)
    u = rng.random((n_sims, horizon))
    for t in range(1, horizon):
        rows = cum_trans[states[:, t - 1]]
        states[:, t] = np.minimum((u[:, t, None] > rows).sum(axis=1), params.n_states - 1)
    noise = rng.standard_normal((n_sims, horizon))
    returns = params.means[states] + params.stds[states] * noise
    return np.maximum(returns, -0.999), states


def forward_filter(returns: np.ndarray, params: GaussianHMMParams) -> np.ndarray:
    """Causal regime probabilities P(state_t | returns_0..t), vectorized over paths.

    This is what a live trader can actually know: no look-ahead, unlike Viterbi
    or smoothed posteriors. Output shape is ``(n_sims, horizon, n_states)``.
    """
    returns = np.atleast_2d(returns)
    n_sims, horizon = returns.shape
    probs = np.empty((n_sims, horizon, params.n_states))
    prior = np.broadcast_to(params.startprob, (n_sims, params.n_states))
    for t in range(horizon):
        z = (returns[:, t, None] - params.means) / params.stds
        # log-likelihood, shifted per path for numerical stability
        loglik = -0.5 * z**2 - np.log(params.stds)
        lik = np.exp(loglik - loglik.max(axis=1, keepdims=True))
        post = prior * lik
        post /= post.sum(axis=1, keepdims=True)
        probs[:, t] = post
        prior = post @ params.transmat
    return probs


def simulate_hmm_strategy(
    params: GaussianHMMParams,
    regime_exposure: ArrayLike,
    n_sims: int = 1000,
    horizon: int = 252,
    cost_per_turnover: float = 0.0,
    use_true_states: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a regime-switching strategy on HMM-sampled market paths.

    The position for period ``t`` is set from information available at the end
    of ``t - 1`` (flat for the first period). By default the regime is estimated
    with ``forward_filter`` and exposure is the probability-weighted average of
    ``regime_exposure``, so detection lag and misclassification are included.
    ``use_true_states=True`` gives an optimistic upper bound (perfect detection).

    ``cost_per_turnover`` is charged per unit of absolute position change, e.g.
    ``0.0005`` for 5 bps.

    Returns ``(strategy_returns, market_returns)``.
    """
    exposure = np.asarray(regime_exposure, dtype=float).ravel()
    if exposure.size != params.n_states:
        raise ValueError("regime_exposure needs one value per state")
    market, states = simulate_hmm(params, n_sims, horizon, rng)
    if use_true_states:
        signal = exposure[states]
    else:
        signal = forward_filter(market, params) @ exposure
    positions = np.zeros_like(market)
    positions[:, 1:] = signal[:, :-1]
    turnover = np.abs(np.diff(positions, axis=1, prepend=0.0))
    strat = positions * market - cost_per_turnover * turnover
    return np.maximum(strat, -0.999), market


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def equity_curves(paths: np.ndarray, initial: float = 1.0) -> np.ndarray:
    """Compound per-period returns into equity curves (excluding the start value)."""
    return initial * np.cumprod(1.0 + np.atleast_2d(paths), axis=1)


def max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """Maximum peak-to-trough drawdown per path, as a positive fraction."""
    equity = equity_curves(paths)
    equity = np.concatenate([np.ones((equity.shape[0], 1)), equity], axis=1)
    peaks = np.maximum.accumulate(equity, axis=1)
    return (1.0 - equity / peaks).max(axis=1)


DEFAULT_PERCENTILES = (5, 25, 50, 75, 95)


@dataclass
class MonteCarloResult:
    """Distribution of outcomes across simulated paths."""

    final_returns: np.ndarray
    cagr: np.ndarray
    max_drawdowns: np.ndarray
    sharpe: np.ndarray
    prob_ruin: float
    prob_loss: float
    ruin_threshold: float
    percentiles: tuple[int, ...] = field(default=DEFAULT_PERCENTILES)

    @property
    def n_sims(self) -> int:
        return self.final_returns.size

    def table(self) -> dict[str, dict[int, float]]:
        """Percentiles of each metric, e.g. ``table()["max_drawdown"][95]``."""
        metrics = {
            "final_return": self.final_returns,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdowns,
            "sharpe": self.sharpe,
        }
        return {
            name: dict(zip(self.percentiles, np.percentile(vals, self.percentiles)))
            for name, vals in metrics.items()
        }

    def report(self) -> str:
        """Human-readable summary table."""
        header = f"{'metric':<14}" + "".join(f"{f'p{p}':>10}" for p in self.percentiles)
        lines = [f"Monte Carlo: {self.n_sims} simulations", header, "-" * len(header)]
        for name, row in self.table().items():
            fmt = "{:>10.2f}" if name == "sharpe" else "{:>10.1%}"
            lines.append(f"{name:<14}" + "".join(fmt.format(v) for v in row.values()))
        lines.append(f"P(loss)            {self.prob_loss:.1%}")
        lines.append(f"P(ruin: DD >= {self.ruin_threshold:.0%}) {self.prob_ruin:.1%}")
        return "\n".join(lines)


def summarize(
    paths: np.ndarray,
    periods_per_year: int = 252,
    ruin_threshold: float = 0.5,
    percentiles: Sequence[int] = DEFAULT_PERCENTILES,
) -> MonteCarloResult:
    """Compute outcome distributions for ``(n_sims, horizon)`` return paths.

    ``ruin_threshold`` is the drawdown (e.g. 0.5 = -50% from peak) at which you
    would stop trading; ``prob_ruin`` is the share of paths that ever reach it.
    """
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    horizon = paths.shape[1]
    final = np.prod(1.0 + paths, axis=1) - 1.0
    years = horizon / periods_per_year
    cagr = (1.0 + final) ** (1.0 / years) - 1.0
    std = paths.std(axis=1, ddof=1) if horizon > 1 else np.zeros(paths.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 0, paths.mean(axis=1) / std * np.sqrt(periods_per_year), 0.0)
    dd = max_drawdowns(paths)
    return MonteCarloResult(
        final_returns=final,
        cagr=cagr,
        max_drawdowns=dd,
        sharpe=sharpe,
        prob_ruin=float(np.mean(dd >= ruin_threshold)),
        prob_loss=float(np.mean(final < 0)),
        ruin_threshold=ruin_threshold,
        percentiles=tuple(percentiles),
    )


def run(
    returns: ArrayLike,
    method: str = "block_bootstrap",
    n_sims: int = 1000,
    horizon: Optional[int] = None,
    block_size: int = 20,
    periods_per_year: int = 252,
    ruin_threshold: float = 0.5,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """One-call Monte Carlo on a backtest's per-period strategy returns.

    ``method`` is one of ``"shuffle"``, ``"bootstrap"`` or ``"block_bootstrap"``.
    """
    rng = np.random.default_rng(seed)
    generators: dict[str, Callable[[], np.ndarray]] = {
        "shuffle": lambda: shuffle_paths(returns, n_sims, rng),
        "bootstrap": lambda: bootstrap_paths(returns, n_sims, horizon, rng),
        "block_bootstrap": lambda: block_bootstrap_paths(
            returns, n_sims, horizon, block_size, rng
        ),
    }
    if method not in generators:
        raise ValueError(f"method must be one of {sorted(generators)}")
    return summarize(generators[method](), periods_per_year, ruin_threshold)
