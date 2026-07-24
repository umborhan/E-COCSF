#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECLCS.py  --  Endogenous Closed-Loop Conformal Safety Filtering (E-COCSF)

CARLA 0.9.15 control revision:
  * conservative headroom capping precedes always-returning q=0/max-h
    restoration; verified caps are attributable and restoration remains invalid;
  * rate/jerk history is rebased after emergency CARLA guard overrides;
  * velocity-swept oriented-box guard for junction traffic and NPC cut-ins;
  * conflict-based junction release replaces radial proximity deadlock;
  * legal lane-change permission and marking checks before overtaking;
  * physical stop-waypoint red-light stopping without premature far-range crawl;
  * precision creep throttle isolation near stop lines and queued vehicles;
  * lane-edge heading correction no longer imposes a hidden speed cap;
  * continuity-bounded traffic-light route projection for Town10HD junctions;
  * explicit raw-front-actor validity prevents cross-street/beyond-line leakage;
  * red-light idle reward exemption applies only near the intended stop pose;
  * nearest-valid signal ranking + local affected-lane matching;
  * dropout hysteresis, continuous creep transition, and explicit red violations;
  * negative acceleration never maps to throttle.

Reference implementation aligned with the paper:
    "Conformal Safety Filtering under Enforcement Mismatch"

The method calibrates a control-barrier safety margin *inside* the control
loop, where the margin is endogenous (performative): it selects the projected
action, the action changes the next state, and the next state changes the
residual that the margin is calibrated against.

Runtime loop (Algorithm 1 in the paper):
    1. black-box policy proposes  u_t^pi = pi(x_t)
    2. form the executed margin    qtilde_t = Proj[q_t + zeta_t]   (bounded probe)
    3. project at qtilde_t; on strict failure estimate headroom and re-solve at
       a conservative capped margin before attempting invalid restoration
    4. execute u_t; observe x_{t+1}
    5. compute shortfall           R_{t+1} = [hhat(x_t,u_t) - h(x_{t+1})]_+
    6. update only after attributable enforcement, with conditional integration
       and back-calculation anti-windup on verified capped transitions
    7. audit past direct-enforcement records only; report operational support
       without treating the dependent-data diagnostic as a finite-sample certificate.

Paper-aligned components implemented here:
  * Executed-margin calibration (martingale-difference noise; Lemma 1).
  * Conservative headroom capping and enforcement-valid anti-windup.
  * Local one-sided dissipativity check via a kernel local-linear regression
    with a one-sided numerical-instability bound on the slope.
  * Gain-optimal schedule  eta* = 2 (Dbar^2 / mu)^(1/3)  with online estimates
    of mu (negative slope) and Dbar (root drift)  -- Corollary (cube-root law).
  * Failure-explicit metrics: validity, capping, restoration, operational audit
    support, supported displacement, and exceedance accounting.
  * Two simulation settings (paper rescoped to driving):
      - CARLA adapter interface (primary; high-fidelity, ground-truth h).
      - Analytic ACC companion with an affine CBF (controlled drift sweep for
        verifying the cube-root law, the U-shaped gain, and the probing floor).
  * Shared projection baselines selected through ECLCSConfig.method.

Quick checks:
    python ECLCS.py --scaling --out_dir ./eclcs_scaling
    python ECLCS.py --sweep --out_dir ./eclcs_sweep
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import sys

from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # torch is optional (only the MLP policy needs it)
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


ArrayLike = Union[np.ndarray, Sequence[float], float]


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def as_vector(value: Optional[ArrayLike], dim: int, default: float, name: str,
              allow_inf: bool = False) -> np.ndarray:
    if value is None:
        arr = np.full(dim, default, dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 1 and dim > 1:
            arr = np.full(dim, float(arr.item()), dtype=np.float64)
    if arr.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},), got {arr.shape}.")
    if not allow_inf and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values.")
    return arr.astype(np.float64, copy=False)


def clip_scalar(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


def weighted_norm(x: np.ndarray, weight: np.ndarray) -> float:
    return float(math.sqrt(float(np.sum(weight * x * x))))


# =============================================================================
# Configuration and records
# =============================================================================

@dataclass
class ECLCSConfig:
    """Configuration for E-COCSF. Defaults are for the 2-D driving command."""

    action_dim: int = 2
    dt: float = 0.05

    # --- Online conformal target & update ------------------------------------
    epsilon: float = 0.10                 # target exceedance level
    eta: float = 0.03                     # base adaptation gain
    q_init: float = 0.10
    q_max: float = 5.0
    ramp_tau: float = 0.05                # soft-exceedance ramp width tau

    # --- Method used in a fair shared-filter benchmark ----------------------
    # All methods below use the same barrier, projection, actuator limits, and
    # restoration.  Only the rule that selects/updates the robustness margin
    # changes.  ``naive_aci`` intentionally updates on contaminated/restored
    # transitions and is therefore the ablation of E-COCSF's validity gate.
    method: str = "ecocsf"               # ecocsf | naive_aci | fixed | uncertainty
    fixed_margin: float = 0.01
    uncertainty_quantile: float = 0.90
    uncertainty_scale: float = 1.0
    uncertainty_min_samples: int = 30

    # Executed-margin probing for local identifiability (zeta in the paper).
    zeta_max: float = 0.02
    probe_probability: float = 0.10

    # --- Discrete-time barrier projection ------------------------------------
    alpha: float = 0.35                   # allowed barrier decay (1-alpha)h + q
    projection_tol: float = 1e-5
    hhat_grad_eps: float = 1e-3
    barrier_linearization_eps: float = 1e-4
    projection_scp_iterations: int = 3
    fallback_mode: str = "previous"       # "previous" | "neutral"

    # --- Conservative headroom capping and feasibility restoration ----------
    # After strict projection fails, a bounded search returns an admissible
    # action and therefore a conservative lower estimate of the available
    # margin headroom.  The filter re-solves at a buffered capped margin.  A
    # verified capped projection is enforcement-valid, while q=0/max-h recovery
    # used after a failed capped re-solve remains invalid for calibration.
    feasibility_restoration: bool = True
    restoration_grid_points: int = 5
    restoration_h_tolerance: float = 1e-8
    headroom_buffer: float = 1e-6
    # Compatibility aliases used by the Safety-Gymnasium evaluator.  When
    # ``headroom_cap_delta`` is supplied it overrides ``headroom_buffer``.
    headroom_margin_cap: bool = True
    headroom_cap_delta: Optional[float] = None
    margin_comparison_tolerance: float = 1e-8
    anti_windup_gamma: float = 0.01
    # Fraction of positive innovation retained on a verified capped transition.
    # 0.0 reproduces strict conditional integration; 1.0 makes the recursion
    # respond to every enforcement-valid loss so pooled valid loss can track
    # epsilon.  Capped transitions remain attributable because q_enforced is
    # the re-solved and finally verified margin.
    capped_positive_integration: float = 1.0
    # A high-loss capped transition must not simultaneously pull q downward.
    # Back-calculation is therefore applied only when its observed loss is at
    # or below the target.  This avoids the sign-inconsistent update in which
    # loss > epsilon nevertheless produced q_{t+1} < q_t.
    freeze_backcalc_on_positive_cap: bool = True
    rebase_history_on_guard_override: bool = True

    # --- Action bounds / limits (length action_dim, or scalar broadcast) -----
    action_low: ArrayLike = (-0.60, -4.00)
    action_high: ArrayLike = (0.60, 2.50)
    rate_limit: ArrayLike = (0.12, 0.80)
    jerk_limit: ArrayLike = (0.08, 0.50)
    neutral_action: ArrayLike = (0.0, 0.0)
    action_weight: ArrayLike = (1.0, 0.20)

    # --- Prospective response-support audit ---------------------------------
    audit_window: int = 120
    audit_min_samples: int = 30
    audit_min_q_range: float = 1e-3       # need spread in executed margins
    audit_bandwidth: float = 0.05          # fixed kernel bandwidth b_aud
    audit_conf_z: float = 1.645           # instability multiplier c_aud
    audit_min_mu: float = 1e-3            # require slope diagnostic <= -mu_aud
    audit_crossing_slack: float = 0.02    # eps band for the local crossing
    audit_min_weight_mass: float = 5.0
    audit_min_abs_slope: float = 1e-6
    audit_ridge_lambda: float = 1e-6
    audit_intercept_ridge_scale: float = 1e-3
    certified_tube_delta: float = 0.10    # half-width delta of C_t(delta) around q_star

    # --- Gain-optimal schedule (Corollary; cube-root law) --------------------
    use_gain_schedule: bool = False
    L_g: float = 1.0                      # Lipschitz estimate of g_t (for stability cap)
    eta_min: float = 1e-3
    eta_max: float = 0.30
    drift_window: int = 60                # window to estimate Dbar from root drift

    # --- Logging -------------------------------------------------------------
    seed: int = 42
    save_jsonl: bool = True
    run_audit: bool = True               # disable for fast controlled scaling studies

    def validate(self) -> None:
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if not (0.0 < self.epsilon < 1.0):
            raise ValueError("epsilon must lie in (0, 1).")
        if self.eta <= 0:
            raise ValueError("eta must be positive.")
        if self.q_max <= 0:
            raise ValueError("q_max must be positive.")
        if not (0.0 <= float(self.q_init) <= float(self.q_max)):
            raise ValueError("q_init must lie in [0, q_max].")
        if self.ramp_tau <= 0:
            raise ValueError("ramp_tau must be positive.")
        if self.method not in {"ecocsf", "naive_aci", "fixed", "uncertainty"}:
            raise ValueError(
                "method must be one of: ecocsf, naive_aci, fixed, uncertainty."
            )
        if not (0.0 <= float(self.fixed_margin) <= float(self.q_max)):
            raise ValueError("fixed_margin must lie in [0, q_max].")
        if not (0.0 < float(self.uncertainty_quantile) < 1.0):
            raise ValueError("uncertainty_quantile must lie in (0, 1).")
        if float(self.uncertainty_scale) < 0.0:
            raise ValueError("uncertainty_scale must be non-negative.")
        if int(self.uncertainty_min_samples) < 1:
            raise ValueError("uncertainty_min_samples must be >= 1.")
        if self.zeta_max < 0:
            raise ValueError("zeta_max must be non-negative.")
        if not (0.0 <= self.probe_probability <= 1.0):
            raise ValueError("probe_probability must lie in [0, 1].")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must lie in (0, 1].")
        if self.fallback_mode not in {"previous", "neutral"}:
            raise ValueError("fallback_mode must be 'previous' or 'neutral'.")
        if int(self.projection_scp_iterations) < 1:
            raise ValueError("projection_scp_iterations must be >= 1.")
        if int(self.restoration_grid_points) < 2:
            raise ValueError("restoration_grid_points must be >= 2.")
        if float(self.restoration_h_tolerance) < 0.0:
            raise ValueError("restoration_h_tolerance must be non-negative.")
        self.headroom_margin_cap = bool(self.headroom_margin_cap)
        if self.headroom_cap_delta is not None:
            if float(self.headroom_cap_delta) <= 0.0:
                raise ValueError("headroom_cap_delta must be positive when provided.")
            self.headroom_buffer = float(self.headroom_cap_delta)
        if float(self.headroom_buffer) <= 0.0:
            raise ValueError("headroom_buffer must be positive.")
        # Keep both public names synchronized for logging and external callers.
        self.headroom_cap_delta = float(self.headroom_buffer)
        if float(self.margin_comparison_tolerance) < 0.0:
            raise ValueError("margin_comparison_tolerance must be non-negative.")
        if not (0.0 < float(self.anti_windup_gamma) < 1.0):
            raise ValueError("anti_windup_gamma must lie in (0, 1).")
        if not (0.0 <= float(self.capped_positive_integration) <= 1.0):
            raise ValueError(
                "capped_positive_integration must lie in [0, 1]."
            )
        self.freeze_backcalc_on_positive_cap = bool(
            self.freeze_backcalc_on_positive_cap
        )
        if self.certified_tube_delta <= 0:
            raise ValueError("certified_tube_delta must be positive.")
        if int(self.audit_window) < 3:
            raise ValueError("audit_window must be >= 3.")
        if not (3 <= int(self.audit_min_samples) <= int(self.audit_window)):
            raise ValueError("audit_min_samples must lie in [3, audit_window].")
        if float(self.audit_min_q_range) <= 0.0:
            raise ValueError("audit_min_q_range must be positive.")
        if float(self.audit_bandwidth) <= 0.0:
            raise ValueError("audit_bandwidth must be positive.")
        if float(self.audit_min_weight_mass) <= 0.0:
            raise ValueError("audit_min_weight_mass must be positive.")
        if float(self.audit_min_abs_slope) <= 0.0:
            raise ValueError("audit_min_abs_slope must be positive.")
        if float(self.audit_ridge_lambda) <= 0.0:
            raise ValueError("audit_ridge_lambda must be positive.")
        if not (0.0 < float(self.audit_intercept_ridge_scale) <= 1.0):
            raise ValueError("audit_intercept_ridge_scale must lie in (0, 1].")
        if int(self.drift_window) < 2:
            raise ValueError("drift_window must be >= 2.")
        if not (0.0 < float(self.eta_min) <= float(self.eta_max)):
            raise ValueError("eta_min and eta_max must satisfy 0 < eta_min <= eta_max.")
        if float(self.L_g) <= 0.0:
            raise ValueError("L_g must be positive.")


@dataclass
class MachineCard:
    """Physical command limits for the actuator interface (any action_dim)."""

    action_low: ArrayLike = (-0.60, -4.00)
    action_high: ArrayLike = (0.60, 2.50)
    rate_limit: ArrayLike = (0.12, 0.80)
    jerk_limit: ArrayLike = (0.08, 0.50)
    neutral_action: ArrayLike = (0.0, 0.0)
    action_names: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # infer dimension from action_low
        low = np.asarray(self.action_low, dtype=np.float64).reshape(-1)
        self.action_dim = int(low.size)

        def _match(val, default, name, allow_inf=False):
            a = np.asarray(val, dtype=np.float64).reshape(-1)
            if a.size != self.action_dim:
                a = np.full(self.action_dim, float(a.flat[0]) if a.size else default)
            return as_vector(a, self.action_dim, default, name, allow_inf=allow_inf)

        self.low = as_vector(self.action_low, self.action_dim, -1.0, "action_low")
        self.high = _match(self.action_high, 1.0, "action_high")
        self.rate = _match(self.rate_limit, np.inf, "rate_limit", allow_inf=True)
        self.jerk = _match(self.jerk_limit, np.inf, "jerk_limit", allow_inf=True)
        self.neutral = np.clip(_match(self.neutral_action, 0.0, "neutral_action"),
                               self.low, self.high)
        if np.any(self.high <= self.low):
            raise ValueError("Every action_high must exceed action_low.")

    def effective_bounds_with_status(
        self,
        u_prev: np.ndarray,
        u_prevprev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Return the exact box induced by absolute, rate, and jerk limits.

        The returned ``feasible`` flag is false when the true issued-command
        history makes the intersection empty. In that case callers must not
        collapse the box to an arbitrary action and call it constraint-feasible;
        the transition lies outside attributable enforcement semantics.
        """
        u_prev = np.asarray(u_prev, dtype=np.float64).reshape(self.action_dim)
        u_prevprev = np.asarray(u_prevprev, dtype=np.float64).reshape(self.action_dim)
        rate = np.where(np.isfinite(self.rate), self.rate, 1e9)
        jerk = np.where(np.isfinite(self.jerk), self.jerk, 1e9)

        low = np.maximum(self.low, u_prev - rate)
        high = np.minimum(self.high, u_prev + rate)
        jerk_center = 2.0 * u_prev - u_prevprev
        low = np.maximum(low, jerk_center - jerk)
        high = np.minimum(high, jerk_center + jerk)
        feasible = bool(np.all(low <= high))
        return low.astype(np.float64), high.astype(np.float64), feasible

    def effective_bounds(
        self,
        u_prev: np.ndarray,
        u_prevprev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Backward-compatible two-array bounds helper.

        Internal filter code uses :meth:`effective_bounds_with_status` and
        handles an empty intersection explicitly.
        """
        low, high, feasible = self.effective_bounds_with_status(
            u_prev, u_prevprev
        )
        if not feasible:
            fb = np.clip(
                np.asarray(u_prev, dtype=np.float64).reshape(self.action_dim),
                self.low,
                self.high,
            )
            conflict = low > high
            low = np.where(conflict, fb, low)
            high = np.where(conflict, fb, high)
        return low.astype(np.float64), high.astype(np.float64)


@dataclass
class FilterDecision:
    action: np.ndarray
    nominal_action: np.ndarray
    q: float
    q_tilde: float
    zeta: float
    h_current: float
    hhat_projected: float
    barrier_rhs: float
    feasible: bool
    certified_before_execution: bool
    intervention_norm: float
    projection_margin: float
    active_constraints: int
    mode: str
    restoration_used: bool = False
    restoration_kind: str = "none"
    restoration_slack: float = 0.0
    effective_q: float = 0.0
    strict_feasible: bool = False
    margin_capped: bool = False
    headroom_search_succeeded: bool = False
    headroom_estimate: float = float("nan")
    active_barrier_component: str = "unknown"
    active_barrier_value: float = float("nan")
    linear_feasible: bool = False
    exact_candidate_hhat: float = float("nan")
    barrier_context: Dict[str, float] = field(default_factory=dict)
    barrier_component_names: List[str] = field(default_factory=list)
    audit_context: Dict[str, Any] = field(default_factory=dict)
    h_current_components: List[float] = field(default_factory=list)
    hhat_projected_components: List[float] = field(default_factory=list)
    barrier_rhs_components: List[float] = field(default_factory=list)
    requested_barrier_rhs_components: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["action"] = np.asarray(self.action, dtype=float).tolist()
        out["nominal_action"] = np.asarray(self.nominal_action, dtype=float).tolist()
        return out


@dataclass
class TransitionUpdate:
    residual: float
    hard_exceedance: bool
    hard_exceedance_requested: bool
    soft_exceedance: float
    q_before: float
    q_tilde: float
    q_after: float
    eta_used: float
    h_next: float
    violation: bool
    audit_status: str
    certified: bool
    infeasible: bool
    crossing_test_failure: bool
    e_out: bool
    audit_slope: float
    audit_slope_ub: float
    mu_hat: float
    Dbar_hat: float
    q_star: float
    certified_tube_low: float
    certified_tube_high: float
    in_certified_tube: bool
    executed_hhat: float
    executed_feasible: bool
    history_rebased: bool = False
    calibration_margin: float = float("nan")
    q_enforced: float = float("nan")
    calibration_valid: bool = False
    direct_enforcement_valid: bool = False
    cap_attempted: bool = False
    margin_capped: bool = False
    anti_windup_backcalc: float = 0.0
    audit_supported: bool = False
    execution_mode: str = "unverified"
    audit_status_after: str = "uncertified"
    certified_after_update: bool = False
    component_semantics_consistent: bool = True
    component_count_before: int = 0
    component_count_after: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Barrier models
# =============================================================================

@dataclass
class AutonomousDrivingBarrierModel:
    """Multi-margin driving barrier (action = [steering, acceleration])."""

    dt: float = 0.05
    lane_half_width: float = 1.75
    heading_limit_rad: float = 0.60
    speed_limit_mps: float = 35.0
    min_speed_mps: float = 0.0
    min_headway_m: float = 2.0
    time_headway_s: float = 0.45
    min_obstacle_distance_m: float = 3.0
    max_abs_lateral_for_optional_margin: float = 20.0
    action_dim: int = 2

    def _pad(self, state: ArrayLike) -> np.ndarray:
        x = np.asarray(state, dtype=np.float64).reshape(-1)
        if x.size < 6:
            raise ValueError("Driving barrier expects >= 6 state entries.")
        if x.size < 8:
            x = np.pad(x, (0, 8 - x.size), constant_values=0.0)
            x[6] = self.lane_half_width - float(x[0])
            x[7] = self.lane_half_width + float(x[0])
        return x


    def predict_next_state(self, state: ArrayLike, action: ArrayLike) -> np.ndarray:
        """Predict one high-level step while preserving explicit road-edge margins.

        State entries x[6] and x[7] are the signed distances to the right and left
        legal corridor boundaries.  For ordinary lane keeping they equal
        ``lane_half_width-lat`` and ``lane_half_width+lat``.  During an approved
        overtake they may describe a wider two-lane legal corridor while x[0]
        remains route-relative, preserving observation semantics.
        """
        x = self._pad(state).copy()
        u = np.asarray(action, dtype=np.float64).reshape(2)
        steering, accel = float(u[0]), float(u[1])
        lat, heading = float(x[0]), float(x[1])
        speed = max(float(x[2]), 0.0)
        lead, rel = max(float(x[3]), 0.0), float(x[4])
        obs = max(float(x[5]), 0.0)

        old_right_margin = float(x[6])
        old_left_margin = float(x[7])
        if not (np.isfinite(old_right_margin) and np.isfinite(old_left_margin)):
            old_right_margin = float(self.lane_half_width - lat)
            old_left_margin = float(self.lane_half_width + lat)

        nxt_speed = float(np.clip(speed + accel * self.dt, self.min_speed_mps,
                                  1.5 * self.speed_limit_mps))
        nxt_head = heading + steering * self.dt
        nxt_lat = lat + speed * math.sin(heading) * self.dt + 0.5 * steering * self.dt
        nxt_rel = rel - accel * self.dt
        nxt_lead = max(0.0, lead + rel * self.dt)
        nxt_obs = max(0.0, obs - speed * self.dt)

        delta_lat = float(nxt_lat - lat)
        x[0], x[1], x[2], x[3], x[4], x[5] = (
            nxt_lat, nxt_head, nxt_speed, nxt_lead, nxt_rel, nxt_obs
        )
        x[6] = old_right_margin - delta_lat
        x[7] = old_left_margin + delta_lat
        return x


    def barrier_values(self, state: ArrayLike) -> np.ndarray:
        x = self._pad(state)
        lat, heading = float(x[0]), float(x[1])
        speed = max(float(x[2]), 0.0)
        lead = max(float(x[3]), 0.0)
        obs = max(float(x[5]), 0.0)

        speed_min_margin = (
            float(self.speed_limit_mps)
            if float(self.min_speed_mps) <= 0.0
            else float(speed - self.min_speed_mps)
        )

        right_margin = float(x[6])
        left_margin = float(x[7])
        if np.isfinite(right_margin) and np.isfinite(left_margin):
            lane_margin = min(right_margin, left_margin)
        else:
            lane_margin = self.lane_half_width - abs(lat)

        v = [
            lane_margin,
            self.heading_limit_rad - abs(heading),
            self.speed_limit_mps - speed,
            speed_min_margin,
            lead - (self.min_headway_m + self.time_headway_s * speed),
            obs - self.min_obstacle_distance_m,
        ]
        return np.asarray(v, dtype=np.float64)

    COMPONENT_NAMES = ("lane", "heading", "speed_max", "speed_min",
                       "headway", "obstacle")

    def h(self, state: ArrayLike) -> float:
        return float(np.min(self.barrier_values(state)))

    def h_hat(self, state: ArrayLike, action: ArrayLike) -> float:
        return self.h(self.predict_next_state(state, action))

    def shortfall(self, state, action, next_state) -> float:
        return float(max(0.0, self.h_hat(state, action) - self.h(next_state)))


@dataclass
class ACCBarrierModel:
    """
    Affine-CBF longitudinal adaptive-cruise-control barrier (action = [a_ego]).

    State convention: x = [gap, v_ego, v_lead].
    Time-headway barrier:  h(x) = gap - (d0 + T * v_ego)   (affine in x).
    The one-step predictor assumes the lead vehicle holds speed (a_lead = 0);
    the conformal residual then absorbs the unknown lead behaviour, which is the
    controlled drift source for the cube-root sweep.
    """

    dt: float = 0.05
    d0: float = 5.0          # standstill distance
    T: float = 1.2          # time headway
    v_max: float = 35.0
    action_dim: int = 1

    def predict_next_state(self, state: ArrayLike, action: ArrayLike) -> np.ndarray:
        x = np.asarray(state, dtype=np.float64).reshape(3).copy()
        a = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        gap, v_ego, v_lead = float(x[0]), float(x[1]), float(x[2])
        nxt_v_ego = float(np.clip(v_ego + a * self.dt, 0.0, self.v_max))
        nxt_v_lead = float(np.clip(v_lead, 0.0, self.v_max))  # predictor: a_lead=0
        nxt_gap = gap + (nxt_v_lead - nxt_v_ego) * self.dt
        return np.asarray([nxt_gap, nxt_v_ego, nxt_v_lead], dtype=np.float64)

    def h(self, state: ArrayLike) -> float:
        x = np.asarray(state, dtype=np.float64).reshape(3)
        return float(x[0] - (self.d0 + self.T * x[1]))

    def h_hat(self, state: ArrayLike, action: ArrayLike) -> float:
        return self.h(self.predict_next_state(state, action))

    def shortfall(self, state, action, next_state) -> float:
        return float(max(0.0, self.h_hat(state, action) - self.h(next_state)))


# =============================================================================
# Weighted half-space projection (the barrier QP for small action_dim)
# =============================================================================

def project_weighted_halfspaces(u_ref: np.ndarray, A: np.ndarray, b: np.ndarray,
                                weight_diag: np.ndarray, tol: float = 1e-6
                                ) -> Tuple[np.ndarray, bool, float, int]:
    """
    min_u 0.5 (u-u_ref)^T W (u-u_ref)  s.t.  A u >= b ,  via active-set enumeration.
    Exact for low-dimensional action spaces (1-D ACC, 2-D driving).
    """
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    w = np.asarray(weight_diag, dtype=np.float64).reshape(-1)
    if A.ndim != 2 or A.shape[1] != u_ref.size:
        raise ValueError("A must have shape (n_constraints, action_dim).")
    if A.shape[0] != b.size:
        raise ValueError("A and b constraint counts mismatch.")
    if np.any(w <= 0):
        raise ValueError("weight_diag must be positive.")

    if np.all(A @ u_ref - b >= -tol):
        return u_ref, True, float(np.min(A @ u_ref - b)), 0

    n, m = A.shape
    winv = np.diag(1.0 / w)
    best_u, best_obj, best_active = None, float("inf"), 0
    for r in range(1, min(m, n) + 1):
        for active in itertools.combinations(range(n), r):
            Ai, bi = A[list(active), :], b[list(active)]
            M = Ai @ winv @ Ai.T
            try:
                lam = np.linalg.pinv(M) @ (bi - Ai @ u_ref)
            except np.linalg.LinAlgError:
                continue
            cand = u_ref + winv @ Ai.T @ lam
            if np.all(A @ cand - b >= -tol):
                obj = 0.5 * float(np.sum(w * (cand - u_ref) ** 2))
                if obj < best_obj:
                    best_obj, best_u, best_active = obj, cand, r
    if best_u is None:
        return u_ref, False, float(np.min(A @ u_ref - b)), 0
    return best_u, True, float(np.min(A @ best_u - b)), best_active


# =============================================================================
# Closed-loop calibration audit (local-linear response diagnostic)
# =============================================================================

class ClosedLoopCalibrationAudit:
    """
    Online verification of the local self-correction (dissipativity) condition.

    A kernel-weighted local-linear regression of the soft exceedance on the
    directly enforced margin gives ghat(q0) and the local slope ghat'(q0).
    The slope-instability scale is an operational numerical diagnostic rather
    than an independent-sample standard error. Response support is reported
    only when the one-sided slope diagnostic is negative,
        ghat'(q0) + c_aud * s_hat(q0) <= -mu_aud ,
    and the fitted map locally crosses the target eps. The estimated root and
    its drift feed the gain-optimal schedule.
    """

    def __init__(self, cfg: ECLCSConfig):
        self.cfg = cfg
        self.records: Deque[Dict[str, float]] = deque(maxlen=int(cfg.audit_window))
        # Store (filter tick, estimated root) pairs.  Certified roots need not
        # occur on consecutive ticks, so drift must be normalized by elapsed
        # ticks rather than treating adjacent stored roots as adjacent samples.
        self.root_history: Deque[Tuple[int, float]] = deque(
            maxlen=int(cfg.drift_window)
        )
        self.last_root: Optional[float] = None
        self.step_count: int = 0

    def update(self, q_tilde: float, residual: float, soft_exceedance: float,
               infeasible: bool) -> Dict[str, Any]:
        self.step_count += 1
        # A restored or externally overridden action was not generated by the
        # requested executed margin.  Keep the clock moving, but never let that
        # sample contaminate the local closed-loop regression.
        if infeasible:
            return self.current_status(q0=float(q_tilde), infeasible=True)
        self.records.append({
            "q_tilde": float(q_tilde),
            "soft": float(soft_exceedance),
            "hard": float(residual > q_tilde),
        })
        status = self.current_status(q0=float(q_tilde), infeasible=False)
        q_star = float(status.get("q_star", float("nan")))
        if bool(status.get("certified", False)) and np.isfinite(q_star):
            self.root_history.append((int(self.step_count), q_star))
            self.last_root = q_star
            status = dict(status)
            status["Dbar_hat"] = self._Dbar()
        return status

    def _nan_status(self, status: str) -> Dict[str, Any]:
        return {"status": status, "certified": False, "supported": False,
                "crossing_test_failure": status == "crossing_test_failure",
                "slope": float("nan"), "slope_ub": float("nan"), "mu_hat": float("nan"),
                "Dbar_hat": self._Dbar(), "q_star": float("nan"),
                "q_range": 0.0, "weight_mass": 0.0, "n": len(self.records)}

    def _Dbar(self) -> float:
        if len(self.root_history) < 2:
            return float("nan")
        hist = np.asarray(self.root_history, dtype=np.float64)
        dt = np.diff(hist[:, 0])
        dq = np.diff(hist[:, 1])
        valid = np.isfinite(dt) & np.isfinite(dq) & (dt > 0.0)
        if not np.any(valid):
            return float("nan")
        velocity = dq[valid] / dt[valid]
        return float(math.sqrt(float(np.mean(velocity * velocity))))


    def current_status(self, q0: float, infeasible: bool = False) -> Dict[str, Any]:
        if infeasible:
            return self._nan_status("infeasible")
        if len(self.records) < int(self.cfg.audit_min_samples):
            return self._nan_status("insufficient_samples")

        qs = np.asarray([r["q_tilde"] for r in self.records], dtype=np.float64)
        ys = np.asarray([r["soft"] for r in self.records], dtype=np.float64)
        if not (np.all(np.isfinite(qs)) and np.all(np.isfinite(ys))):
            return self._nan_status("nonfinite_records")
        bw = float(self.cfg.audit_bandwidth)

        wts = np.exp(-0.5 * ((qs - float(q0)) / bw) ** 2)
        weight_mass = float(np.sum(wts))
        if weight_mass < float(self.cfg.audit_min_weight_mass):
            status = self._nan_status("insufficient_weight")
            status["weight_mass"] = weight_mass
            return status

        q_range = float(qs.max() - qs.min())
        if q_range < float(self.cfg.audit_min_q_range):
            status = self._nan_status("insufficient_excitation")
            status["q_range"] = q_range
            status["weight_mass"] = weight_mass
            return status
        X = np.stack([np.ones_like(qs), qs - float(q0)], axis=1)
        XtW = X.T * wts[None, :]
        XtWX = XtW @ X
        ridge = float(self.cfg.audit_ridge_lambda) * np.diag([
            float(self.cfg.audit_intercept_ridge_scale),
            1.0,
        ])
        try:
            gram_inv = np.linalg.inv(XtWX + ridge)
        except np.linalg.LinAlgError:
            return self._nan_status("singular_fit")
        beta = gram_inv @ (XtW @ ys)
        a_hat, b_hat = float(beta[0]), float(beta[1])

        resid = ys - X @ beta
        sigma2 = float(np.sum(wts * resid * resid) / max(1.0, weight_mass))
        slope_scale = float(math.sqrt(
            max(sigma2 * float(gram_inv[1, 1]), 0.0)
        ))
        slope_ub = float(
            b_hat + float(self.cfg.audit_conf_z) * slope_scale
        )

        eps = float(self.cfg.epsilon)
        negative = bool(slope_ub <= -float(self.cfg.audit_min_mu))
        y_lo = a_hat + b_hat * (float(qs.min()) - float(q0))
        y_hi = a_hat + b_hat * (float(qs.max()) - float(q0))
        crosses = bool(
            min(y_lo, y_hi) <= eps + float(self.cfg.audit_crossing_slack)
            and max(y_lo, y_hi) >= eps - float(self.cfg.audit_crossing_slack)
        )

        # Only an audit-supported one-sided negative slope supplies the empirical
        # dissipativity estimate used by the optional next-step gain schedule.
        mu_hat = float(max(-slope_ub, float(self.cfg.audit_min_mu))) if negative else float("nan")
        q_star = float("nan")
        root_in_range = False
        slope_identified = bool(
            abs(b_hat) >= float(self.cfg.audit_min_abs_slope)
        )
        if slope_identified:
            candidate_root = float(float(q0) + (eps - a_hat) / b_hat)
            root_in_range = bool(
                np.isfinite(candidate_root)
                and float(qs.min()) <= candidate_root <= float(qs.max())
                and 0.0 <= candidate_root <= float(self.cfg.q_max)
            )
            if root_in_range:
                q_star = candidate_root

        if not slope_identified:
            status, supported, cross_fail = "flat_response", False, False
        elif not negative:
            status, supported, cross_fail = "nonnegative_slope", False, False
        elif not crosses:
            status, supported, cross_fail = "crossing_test_failure", False, True
        elif not root_in_range:
            status, supported, cross_fail = "root_out_of_range", False, False
        else:
            status, supported, cross_fail = "supported", True, False

        return {
            "status": status,
            # ``certified`` is retained as a backward-compatible field name for
            # existing result readers.  It denotes operational audit support,
            # not a finite-sample statistical certificate.
            "certified": supported,
            "supported": supported,
            "crossing_test_failure": cross_fail,
            "slope": b_hat,
            "slope_ub": slope_ub,
            "mu_hat": mu_hat,
            "Dbar_hat": self._Dbar(),
            "q_star": q_star,
            "q_range": q_range,
            "weight_mass": weight_mass,
            "n": len(self.records),
        }


# =============================================================================
# E-COCSF runtime filter
# =============================================================================

class EndogenousClosedLoopConformalSafetyFilter:
    """Policy-agnostic E-COCSF filter. Works with any BarrierModel/action_dim."""

    def __init__(self, cfg: Optional[ECLCSConfig] = None,
                 machine: Optional[MachineCard] = None,
                 barrier: Optional[Any] = None):
        self.cfg = cfg if cfg is not None else ECLCSConfig()
        self.cfg.validate()
        set_seed(self.cfg.seed)

        self.machine = machine if machine is not None else MachineCard(
            action_low=self.cfg.action_low, action_high=self.cfg.action_high,
            rate_limit=self.cfg.rate_limit, jerk_limit=self.cfg.jerk_limit,
            neutral_action=self.cfg.neutral_action)
        self.barrier = barrier if barrier is not None else AutonomousDrivingBarrierModel(dt=self.cfg.dt)
        self.action_dim = self.machine.action_dim

        self.q = float(self.cfg.q_init)
        self.prev_action = self.machine.neutral.copy()
        self.prevprev_action = self.machine.neutral.copy()
        self.audit = ClosedLoopCalibrationAudit(self.cfg)
        self.residual_window: Deque[float] = deque(maxlen=int(self.cfg.audit_window))
        self.eta_current = float(self.cfg.eta)
        self.log: List[Dict[str, Any]] = []

    def _weight(self) -> np.ndarray:
        """Action-distortion weight matched to action_dim (broadcast if mismatched)."""
        w = np.asarray(self.cfg.action_weight, dtype=np.float64).reshape(-1)
        if w.size == self.action_dim:
            return w
        return np.full(self.action_dim, float(w[0]))

    # -- lifecycle ------------------------------------------------------------
    def reset(self, q: Optional[float] = None, clear_log: bool = False) -> None:
        """Reset episode-local filter state without losing aggregate metrics.

        Evaluation episodes are independent CARLA trials.  Carrying previous
        actuator history, residual windows, audit state, or an adapted q into a
        new episode can create artificial startup infeasibility and makes method
        comparisons depend on episode order.  By default the long-run ``log`` is
        preserved so ``metrics()`` can still aggregate all evaluation episodes.
        Pass ``clear_log=True`` only when a completely fresh run is intended.
        """
        self.q = float(self.cfg.q_init if q is None else np.clip(q, 0.0, self.cfg.q_max))
        self.prev_action = self.machine.neutral.copy()
        self.prevprev_action = self.machine.neutral.copy()
        self.audit = ClosedLoopCalibrationAudit(self.cfg)
        self.residual_window.clear()
        self.eta_current = float(self.cfg.eta)
        if bool(clear_log):
            self.log.clear()

    # -- pieces ---------------------------------------------------------------
    def soft_exceedance_loss(self, z: float) -> float:
        tau, z = float(self.cfg.ramp_tau), float(z)
        if z <= -tau:
            return 0.0
        if z < 0.0:
            return float(1.0 + z / tau)
        return 1.0

    def _sample_probe(self, deterministic: bool = False) -> float:
        """Sample the E-COCSF executed-margin identification probe."""
        if self.cfg.method != "ecocsf":
            return 0.0
        if deterministic or self.cfg.zeta_max <= 0 or self.cfg.probe_probability <= 0:
            return 0.0
        if random.random() > float(self.cfg.probe_probability):
            return 0.0
        return float(np.random.uniform(-self.cfg.zeta_max, self.cfg.zeta_max))

    def _grad_hhat(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        grad = np.zeros(self.action_dim, dtype=np.float64)
        eps = float(self.cfg.hhat_grad_eps)
        for j in range(self.action_dim):
            du = np.zeros(self.action_dim); du[j] = eps
            grad[j] = (self.barrier.h_hat(state, u + du) - self.barrier.h_hat(state, u - du)) / (2.0 * eps)
        return grad

    def _h_components(self, state: np.ndarray) -> np.ndarray:
        """Return all available barrier components, with scalar fallback.

        Component-wise enforcement must not depend on whether the barrier also
        implements ``h_hat_components``.  The previous implementation made that
        dependency and silently collapsed the built-in driving barrier to one
        scalar minimum, which invalidated component-wise verification.
        """
        fn = getattr(self.barrier, "h_components", None)
        if fn is None:
            fn = getattr(self.barrier, "barrier_values", None)
        if fn is None:
            values = np.asarray([self.barrier.h(state)], dtype=np.float64)
        else:
            values = np.asarray(fn(state), dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise RuntimeError("Barrier components must be non-empty and finite")
        return values

    def _hhat_components(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Return predicted components under the same component semantics.

        A barrier may expose either a direct ``h_hat_components`` method or a
        one-step predictor plus state-component evaluator.  Only when neither is
        available is the scalar ``h_hat`` interface used.
        """
        fn = getattr(self.barrier, "h_hat_components", None)
        if fn is not None:
            values = np.asarray(fn(state, action), dtype=np.float64).reshape(-1)
        else:
            predictor = getattr(self.barrier, "predict_next_state", None)
            component_fn = getattr(self.barrier, "h_components", None)
            if component_fn is None:
                component_fn = getattr(self.barrier, "barrier_values", None)
            if predictor is not None and component_fn is not None:
                predicted_state = predictor(state, action)
                values = np.asarray(component_fn(predicted_state), dtype=np.float64).reshape(-1)
            else:
                values = np.asarray([self.barrier.h_hat(state, action)], dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise RuntimeError("Predicted barrier components must be non-empty and finite")
        return values

    def _barrier_component_names(self, state: np.ndarray, count: int) -> Tuple[str, ...]:
        """Return stable diagnostic names for the active component ordering."""
        names_fn = getattr(self.barrier, "component_names", None)
        names: Tuple[str, ...] = tuple()
        if callable(names_fn):
            try:
                names = tuple(str(v) for v in names_fn(state))
            except Exception:
                names = tuple()
        if not names:
            try:
                names = tuple(str(v) for v in getattr(self.barrier, "COMPONENT_NAMES", ()))
            except Exception:
                names = tuple()
        if len(names) != int(count):
            names = tuple(f"component_{i}" for i in range(int(count)))
        return names

    def _grad_hhat_components(
        self, state: np.ndarray, action: np.ndarray, component_count: int
    ) -> np.ndarray:
        gradients = np.zeros((component_count, self.action_dim), dtype=np.float64)
        eps = float(self.cfg.hhat_grad_eps)
        for j in range(self.action_dim):
            du = np.zeros(self.action_dim, dtype=np.float64)
            du[j] = eps
            plus = self._hhat_components(state, action + du)
            minus = self._hhat_components(state, action - du)
            if plus.size != component_count or minus.size != component_count:
                raise RuntimeError("Barrier component count changed under perturbation")
            gradients[:, j] = (plus - minus) / (2.0 * eps)
        return gradients

    def _build_constraints(self, state, u_ref, q_tilde, low, high):
        h_components = self._h_components(state)
        rhs_components = (
            (1.0 - float(self.cfg.alpha)) * h_components + float(q_tilde)
        )
        hhat_components = self._hhat_components(state, u_ref)
        if hhat_components.size != h_components.size:
            raise RuntimeError(
                "Current and predicted barrier component counts do not match"
            )
        gradients = self._grad_hhat_components(
            state, u_ref, int(h_components.size)
        )

        A_rows, b_rows = [], []
        for value, rhs, gradient in zip(
            hhat_components, rhs_components, gradients
        ):
            if np.linalg.norm(gradient) > float(self.cfg.barrier_linearization_eps):
                A_rows.append(gradient.copy())
                b_rows.append(float(rhs - value + np.dot(gradient, u_ref)))
            elif value < rhs:
                # A contradictory zero-gradient row makes the linear problem
                # explicitly infeasible instead of silently dropping safety.
                A_rows.append(np.zeros(self.action_dim))
                b_rows.append(1.0)

        eye = np.eye(self.action_dim)
        for j in range(self.action_dim):
            A_rows.append(eye[j]); b_rows.append(float(low[j]))
            A_rows.append(-eye[j]); b_rows.append(float(-high[j]))
        return (
            np.vstack(A_rows),
            np.asarray(b_rows),
            h_components,
            rhs_components,
            gradients,
        )

    def _active_barrier_component(self, state: np.ndarray) -> Tuple[str, float]:
        """Return the currently smallest barrier component for diagnostics/recovery."""
        try:
            values = np.asarray(self.barrier.barrier_values(state), dtype=np.float64).reshape(-1)
            if values.size == 0 or not np.any(np.isfinite(values)):
                return "unknown", float("nan")
            safe_values = np.where(np.isfinite(values), values, np.inf)
            idx = int(np.argmin(safe_values))
            names_fn = getattr(self.barrier, "component_names", None)
            names = (
                tuple(names_fn(state)) if names_fn is not None
                else tuple(getattr(self.barrier, "COMPONENT_NAMES", ()))
            )
            name = str(names[idx]) if idx < len(names) else f"component_{idx}"
            return name, float(values[idx])
        except Exception:
            return "unknown", float("nan")

    def _project_for_margin(self, x: np.ndarray, u_ref: np.ndarray,
                            q_margin: float, low: np.ndarray,
                            high: np.ndarray) -> Dict[str, Any]:
        """Sequentially linearize all barrier components, then verify exactly."""
        candidate = np.clip(u_ref.copy(), low, high)
        linear_feasible = False
        margin = float("-inf")
        active = 0
        h_cur_components = self._h_components(x)
        rhs_components = (
            (1.0 - float(self.cfg.alpha)) * h_cur_components + float(q_margin)
        )
        gradients = np.zeros(
            (h_cur_components.size, self.action_dim), dtype=np.float64
        )
        for _ in range(int(self.cfg.projection_scp_iterations)):
            A, b, h_cur_components, rhs_components, gradients = (
                self._build_constraints(
                    x, candidate, float(q_margin), low, high
                )
            )
            candidate_next, linear_feasible, margin, active = (
                project_weighted_halfspaces(
                    u_ref, A, b, self._weight(),
                    tol=float(self.cfg.projection_tol),
                )
            )
            candidate_next = np.clip(
                np.asarray(candidate_next, dtype=np.float64).reshape(self.action_dim),
                low,
                high,
            )
            change = float(np.linalg.norm(candidate_next - candidate))
            candidate = candidate_next
            if not bool(linear_feasible) or change <= float(self.cfg.projection_tol):
                break

        hhat_components = np.full(
            h_cur_components.shape, float("-inf"), dtype=np.float64
        )
        exact = False
        if bool(linear_feasible):
            try:
                hhat_components = self._hhat_components(x, candidate)
                exact = bool(
                    hhat_components.size == rhs_components.size
                    and np.all(np.isfinite(hhat_components))
                    and np.all(
                        hhat_components + float(self.cfg.projection_tol)
                        >= rhs_components
                    )
                )
            except Exception:
                exact = False
        hhat = float(np.min(hhat_components))
        h_cur = float(np.min(h_cur_components))
        rhs = float(np.min(rhs_components))
        return {
            "action": candidate,
            "linear_feasible": bool(linear_feasible),
            "exact_feasible": bool(exact),
            "hhat": float(hhat),
            "h_current": float(h_cur),
            "rhs": float(rhs),
            "margin": float(margin),
            "active_constraints": int(active),
            "grad": np.asarray(gradients, dtype=np.float64),
            "h_current_components": h_cur_components,
            "rhs_components": rhs_components,
            "hhat_components": hhat_components,
        }

    def _max_h_restoration_action(self, x: np.ndarray, u_ref: np.ndarray,
                                  low: np.ndarray, high: np.ndarray,
                                  seeds: Sequence[np.ndarray],
                                  active_component: str,
                                  active_value: float) -> Tuple[np.ndarray, float]:
        """Bounded deterministic recovery when even the q=0 projection fails.

        The search is deliberately small because the driving interface is 2-D.
        It first maximizes the exact one-step barrier, then minimizes forward
        acceleration near headway/obstacle boundaries, and finally minimizes
        distortion from the nominal action.  This makes progress on lane/heading
        recovery without accelerating into an action-insensitive close obstacle.
        """
        candidates: List[np.ndarray] = []

        def add(value: ArrayLike) -> None:
            try:
                u = np.clip(
                    np.asarray(value, dtype=np.float64).reshape(self.action_dim),
                    low, high,
                )
                if np.all(np.isfinite(u)):
                    candidates.append(u)
            except Exception:
                pass

        for seed in seeds:
            add(seed)
        add(u_ref)
        add(self.prev_action)
        add(self.machine.neutral)
        add(0.5 * (low + high))

        points = max(2, int(self.cfg.restoration_grid_points))
        axes = [np.linspace(float(low[j]), float(high[j]), points)
                for j in range(self.action_dim)]
        if points ** self.action_dim <= 625:
            for values in itertools.product(*axes):
                add(values)
        else:
            # Avoid exponential growth for non-driving action spaces.
            for j, axis in enumerate(axes):
                for value in axis:
                    u = np.clip(u_ref.copy(), low, high)
                    u[j] = float(value)
                    add(u)

        best_u = np.clip(u_ref, low, high)
        best_h = float("-inf")
        best_risk = float("inf")
        best_dist = float("inf")
        weight = self._weight()
        close_longitudinal = bool(
            active_component in {"headway", "obstacle"}
            and np.isfinite(active_value)
            and active_value < 0.50
            and self.action_dim >= 2
        )
        h_tol = float(self.cfg.restoration_h_tolerance)

        # Deduplicate without changing deterministic order.
        seen = set()
        for candidate in candidates:
            key = tuple(np.round(candidate, 12).tolist())
            if key in seen:
                continue
            seen.add(key)
            try:
                hhat = float(self.barrier.h_hat(x, candidate))
            except Exception:
                continue
            if not np.isfinite(hhat):
                continue
            risk = max(0.0, float(candidate[1])) if close_longitudinal else 0.0
            dist = float(np.sum(weight * (candidate - u_ref) ** 2))
            better_h = hhat > best_h + h_tol
            tied_h = abs(hhat - best_h) <= h_tol
            better_tie = tied_h and (
                risk < best_risk - 1e-12
                or (abs(risk - best_risk) <= 1e-12 and dist < best_dist)
            )
            if better_h or better_tie:
                best_u = candidate.copy()
                best_h = hhat
                best_risk = risk
                best_dist = dist

        if not np.isfinite(best_h):
            best_u = np.clip(self.machine.neutral, low, high)
            try:
                best_h = float(self.barrier.h_hat(x, best_u))
            except Exception:
                best_h = float("-inf")
        return best_u, float(best_h)

    def project_action(self, state, nominal_action, q_tilde):
        """Project at the requested margin, then cap to verified headroom.

        If strict projection fails, a bounded max-h search supplies a conservative
        lower estimate of the available margin authority.  The filter subtracts a
        numerical buffer, re-solves at the resulting capped margin, and accepts the
        result only after exact component-wise verification.  A verified capped
        projection is attributable and calibration-valid.  If the capped re-solve
        fails, q=0/max-h restoration remains explicitly invalid.
        """
        x = np.asarray(state, dtype=np.float64).reshape(-1)
        u_pi = np.asarray(nominal_action, dtype=np.float64).reshape(self.action_dim)
        low, high, actuator_set_feasible = (
            self.machine.effective_bounds_with_status(
                self.prev_action, self.prevprev_action
            )
        )

        if not actuator_set_feasible:
            # The true issued-command history leaves no action satisfying the
            # simultaneous absolute, rate, and jerk limits. Do not fabricate a
            # feasible box by rebasing history. Issue a bounded recovery command
            # and mark the transition invalid for calibration.
            u_out = np.clip(
                np.asarray(self.prev_action, dtype=np.float64).reshape(
                    self.action_dim
                ),
                self.machine.low,
                self.machine.high,
            )
            h_cur_components = self._h_components(x)
            requested_rhs_components = (
                (1.0 - float(self.cfg.alpha)) * h_cur_components
                + float(q_tilde)
            )
            try:
                hhat_components = self._hhat_components(x, u_out)
            except Exception:
                hhat_components = np.full_like(
                    h_cur_components, float("-inf")
                )
            active_component, active_value = self._active_barrier_component(x)
            projection_margin = (
                float(np.min(hhat_components - requested_rhs_components))
                if (
                    hhat_components.size == requested_rhs_components.size
                    and np.all(np.isfinite(hhat_components))
                )
                else float("-inf")
            )
            restoration_slack = (
                float(max(
                    0.0,
                    float(np.max(
                        requested_rhs_components - hhat_components
                    )),
                ))
                if np.all(np.isfinite(hhat_components))
                else float("inf")
            )
            info = {
                "h_current": float(np.min(h_cur_components)),
                "barrier_rhs": float(np.min(requested_rhs_components)),
                "hhat_projected": float(np.min(hhat_components)),
                "projection_margin": projection_margin,
                "active_constraints": 0,
                "feasible": False,
                "strict_feasible": False,
                "linear_feasible": False,
                "exact_candidate_hhat": float(np.min(hhat_components)),
                "used_fallback": False,
                "restoration_used": True,
                "restoration_kind": "actuator_history_infeasible",
                "restoration_slack": restoration_slack,
                "effective_q": 0.0,
                "margin_capped": False,
                "headroom_search_succeeded": False,
                "headroom_estimate": float("nan"),
                "active_barrier_component": str(active_component),
                "active_barrier_value": float(active_value),
                "h_current_components": h_cur_components,
                "hhat_projected_components": hhat_components,
                "barrier_rhs_components": requested_rhs_components,
                "requested_barrier_rhs_components": requested_rhs_components,
                "mode": "restore",
                "low": self.machine.low.copy(),
                "high": self.machine.high.copy(),
                "actuator_admissible_set_feasible": False,
            }
            return u_out, info

        u_ref = np.clip(u_pi, low, high)

        strict = self._project_for_margin(x, u_ref, float(q_tilde), low, high)
        active_component, active_value = self._active_barrier_component(x)
        strict_feasible = bool(strict["exact_feasible"])
        final_projection = strict
        final_feasible = strict_feasible
        restoration_used = False
        restoration_kind = "none"
        effective_q = float(q_tilde)
        margin_capped = False
        headroom_search_succeeded = False
        headroom_estimate = float("nan")
        mode = "strict" if strict_feasible else "fallback"
        requested_rhs_components = np.asarray(
            strict["rhs_components"], dtype=np.float64
        )
        requested_h_current_components = np.asarray(
            strict["h_current_components"], dtype=np.float64
        )

        if strict_feasible:
            u_out = np.asarray(strict["action"], dtype=np.float64)
            hhat_out = float(strict["hhat"])
        elif bool(self.cfg.feasibility_restoration):
            # The bounded search action lies in U_t^adm, so its achieved margin
            # is a conservative lower estimate of the headroom supremum.
            headroom_action, headroom_hhat = self._max_h_restoration_action(
                x, u_ref, low, high,
                seeds=(strict["action"],),
                active_component=active_component,
                active_value=active_value,
            )
            try:
                headroom_components = self._hhat_components(x, headroom_action)
                available = headroom_components - (
                    (1.0 - float(self.cfg.alpha))
                    * requested_h_current_components
                )
                headroom_search_succeeded = bool(
                    headroom_components.size == requested_h_current_components.size
                    and np.all(np.isfinite(headroom_components))
                    and np.all(np.isfinite(headroom_action))
                    and np.all(headroom_action >= low - float(self.cfg.projection_tol))
                    and np.all(headroom_action <= high + float(self.cfg.projection_tol))
                )
                if headroom_search_succeeded:
                    headroom_estimate = float(np.min(available))
            except Exception:
                headroom_components = np.full_like(
                    requested_h_current_components, float("-inf")
                )

            q_cap = 0.0
            if headroom_search_succeeded:
                q_cap = float(min(
                    float(q_tilde),
                    max(
                        headroom_estimate - float(self.cfg.headroom_buffer),
                        0.0,
                    ),
                ))

            cap_is_reduction = bool(
                self.cfg.method == "ecocsf"
                and bool(self.cfg.headroom_margin_cap)
                and headroom_search_succeeded
                and q_cap < float(q_tilde)
                    - float(self.cfg.margin_comparison_tolerance)
            )
            capped = None
            if cap_is_reduction:
                capped = self._project_for_margin(
                    x, u_ref, q_cap, low, high
                )
                if bool(capped["exact_feasible"]):
                    final_projection = capped
                    final_feasible = True
                    margin_capped = True
                    effective_q = q_cap
                    u_out = np.asarray(capped["action"], dtype=np.float64)
                    hhat_out = float(capped["hhat"])
                    mode = "capped"

            if not final_feasible:
                # A failed or non-reducing cap is not attributable.  Continue
                # through the domain recovery path, but freeze calibration.
                relaxed = (
                    capped if capped is not None and np.isclose(q_cap, 0.0)
                    else self._project_for_margin(x, u_ref, 0.0, low, high)
                )
                relaxed_action = np.asarray(relaxed["action"], dtype=np.float64)

                # Near an action-insensitive longitudinal boundary, relaxation
                # cannot be treated as permission to accelerate into an actor.
                if (active_component in {"headway", "obstacle"}
                        and np.isfinite(active_value) and active_value < 0.50
                        and self.action_dim >= 2):
                    relaxed_action[1] = float(np.clip(
                        min(float(relaxed_action[1]), 0.0), low[1], high[1]
                    ))
                    try:
                        relaxed_components = self._hhat_components(
                            x, relaxed_action
                        )
                        relaxed_exact = bool(
                            relaxed_components.size
                                == np.asarray(relaxed["rhs_components"]).size
                            and np.all(np.isfinite(relaxed_components))
                            and np.all(
                                relaxed_components + float(self.cfg.projection_tol)
                                >= np.asarray(
                                    relaxed["rhs_components"], dtype=np.float64
                                )
                            )
                        )
                        relaxed_hhat = float(np.min(relaxed_components))
                    except Exception:
                        relaxed_exact = False
                        relaxed_hhat = float("-inf")
                else:
                    relaxed_exact = bool(relaxed["exact_feasible"])
                    relaxed_hhat = float(relaxed["hhat"])

                if relaxed_exact:
                    u_out = np.clip(relaxed_action, low, high)
                    hhat_out = float(relaxed_hhat)
                    restoration_kind = "q_zero"
                    effective_q = 0.0
                else:
                    u_out = np.asarray(headroom_action, dtype=np.float64)
                    hhat_out = float(headroom_hhat)
                    restoration_kind = "max_h"
                    effective_q = float(np.clip(
                        headroom_estimate if np.isfinite(headroom_estimate) else 0.0,
                        0.0,
                        float(q_tilde),
                    ))
                restoration_used = True
                mode = "restore"
        else:
            fallback = (
                self.machine.neutral
                if self.cfg.fallback_mode == "neutral"
                else self.prev_action
            )
            u_out = np.clip(
                np.asarray(fallback, dtype=np.float64).reshape(self.action_dim),
                low,
                high,
            )
            hhat_out = float(self.barrier.h_hat(x, u_out))
            final_feasible = False
            mode = "fallback"

        try:
            hhat_out_components = self._hhat_components(x, u_out)
            hhat_out = float(np.min(hhat_out_components))
        except Exception:
            hhat_out_components = np.full_like(
                np.asarray(strict["rhs_components"], dtype=np.float64),
                float("-inf"),
            )
            hhat_out = float("-inf")

        final_rhs_components = (
            np.asarray(final_projection["rhs_components"], dtype=np.float64)
            if final_feasible else requested_rhs_components
        )

        restoration_slack = float(max(
            0.0,
            float(np.max(
                requested_rhs_components - hhat_out_components
            )),
        )) if np.all(np.isfinite(hhat_out_components)) else float("inf")

        info = {
            "h_current": float(final_projection["h_current"]),
            "barrier_rhs": float(np.min(final_rhs_components)),
            "hhat_projected": float(hhat_out),
            "projection_margin": float(final_projection["margin"]),
            "active_constraints": int(final_projection["active_constraints"]),
            "feasible": bool(final_feasible),
            "strict_feasible": bool(strict_feasible),
            "linear_feasible": bool(final_projection["linear_feasible"]),
            "exact_candidate_hhat": float(final_projection["hhat"]),
            "used_fallback": bool(not final_feasible and not restoration_used),
            "restoration_used": bool(restoration_used),
            "restoration_kind": str(restoration_kind),
            "restoration_slack": float(restoration_slack),
            "effective_q": float(effective_q),
            "margin_capped": bool(margin_capped),
            "headroom_search_succeeded": bool(headroom_search_succeeded),
            "headroom_estimate": float(headroom_estimate),
            "active_barrier_component": str(active_component),
            "active_barrier_value": float(active_value),
            "h_current_components": np.asarray(
                final_projection["h_current_components"], dtype=np.float64
            ),
            "hhat_projected_components": hhat_out_components,
            "barrier_rhs_components": final_rhs_components,
            "requested_barrier_rhs_components": requested_rhs_components,
            "mode": str(mode),
            "low": low,
            "high": high,
            "actuator_admissible_set_feasible": True,
        }
        return u_out, info

    def _select_margin(self) -> float:
        """Select a margin without changing the shared projection machinery."""
        if self.cfg.method == "fixed":
            return float(np.clip(self.cfg.fixed_margin, 0.0, self.cfg.q_max))
        if self.cfg.method == "uncertainty":
            if len(self.residual_window) < int(self.cfg.uncertainty_min_samples):
                return float(self.q)
            quantile = float(np.quantile(
                np.asarray(self.residual_window, dtype=np.float64),
                float(self.cfg.uncertainty_quantile),
            ))
            return float(np.clip(
                float(self.cfg.uncertainty_scale) * quantile,
                0.0,
                float(self.cfg.q_max),
            ))
        return float(self.q)

    def _snapshot_barrier_context(self) -> Dict[str, float]:
        """Capture mutable driving-barrier parameters used for this decision.

        CARLA updates lane width and context-dependent gap parameters online.  The
        snapshot lets the post-transition residual recompute h_hat(x_t,u_exec)
        under the same barrier context that was used when the action was selected.
        """
        out: Dict[str, float] = {}
        for name in (
            "lane_half_width", "heading_limit_rad", "speed_limit_mps",
            "min_speed_mps", "min_headway_m", "time_headway_s",
            "min_obstacle_distance_m",
        ):
            if hasattr(self.barrier, name):
                try:
                    out[name] = float(getattr(self.barrier, name))
                except Exception:
                    pass
        return out

    def _with_barrier_context_hhat(self, state: np.ndarray, action: np.ndarray,
                                   context: Dict[str, float]) -> float:
        """Evaluate h_hat under a saved context and restore current parameters."""
        if not context:
            return float(self.barrier.h_hat(state, action))
        current: Dict[str, float] = {}
        try:
            for name, value in context.items():
                if hasattr(self.barrier, name):
                    current[name] = float(getattr(self.barrier, name))
                    setattr(self.barrier, name, float(value))
            return float(self.barrier.h_hat(state, action))
        finally:
            for name, value in current.items():
                try:
                    setattr(self.barrier, name, float(value))
                except Exception:
                    pass

    def _with_barrier_context_hhat_components(
        self, state: np.ndarray, action: np.ndarray, context: Dict[str, float]
    ) -> np.ndarray:
        """Component form of the context-consistent prediction evaluation."""
        if not context:
            return self._hhat_components(state, action)
        current: Dict[str, float] = {}
        try:
            for name, value in context.items():
                if hasattr(self.barrier, name):
                    current[name] = float(getattr(self.barrier, name))
                    setattr(self.barrier, name, float(value))
            return self._hhat_components(state, action)
        finally:
            for name, value in current.items():
                try:
                    setattr(self.barrier, name, float(value))
                except Exception:
                    pass

    def _with_barrier_context_h_components(
        self, state: np.ndarray, context: Dict[str, float]
    ) -> np.ndarray:
        """Evaluate next-state components under the decision-time semantics."""
        if not context:
            return self._h_components(state)
        current: Dict[str, float] = {}
        try:
            for name, value in context.items():
                if hasattr(self.barrier, name):
                    current[name] = float(getattr(self.barrier, name))
                    setattr(self.barrier, name, float(value))
            return self._h_components(state)
        finally:
            for name, value in current.items():
                try:
                    setattr(self.barrier, name, float(value))
                except Exception:
                    pass

    def _with_barrier_context_component_names(
        self,
        state: np.ndarray,
        count: int,
        context: Dict[str, float],
    ) -> Tuple[str, ...]:
        """Evaluate component identities under the saved decision context."""
        if not context:
            return self._barrier_component_names(state, count)
        current: Dict[str, float] = {}
        try:
            for name, value in context.items():
                if hasattr(self.barrier, name):
                    current[name] = float(getattr(self.barrier, name))
                    setattr(self.barrier, name, float(value))
            return self._barrier_component_names(state, count)
        finally:
            for name, value in current.items():
                try:
                    setattr(self.barrier, name, float(value))
                except Exception:
                    pass

    def _with_barrier_context_h(
        self, state: np.ndarray, context: Dict[str, float]
    ) -> float:
        """Scalar state-barrier evaluation under the saved decision context."""
        values = self._with_barrier_context_h_components(state, context)
        return float(np.min(values))

    # -- step 1-4: choose action ---------------------------------------------
    def select_action(self, state, nominal_action, deterministic_probe: bool = False
                      ) -> FilterDecision:
        x = np.asarray(state, dtype=np.float64).reshape(-1)
        u_pi = np.asarray(nominal_action, dtype=np.float64).reshape(self.action_dim)

        barrier_context = self._snapshot_barrier_context()
        q_nominal = self._select_margin()
        if self.cfg.run_audit and self.cfg.method == "ecocsf":
            # Capture the response diagnostic before drawing the probe and before
            # the transition.  This snapshot is the only audit information that
            # may support the current step or select its gain.
            audit_context = self.audit.current_status(
                q0=q_nominal, infeasible=False
            )
        else:
            audit_context = {
                "status": "audit_off", "certified": False,
                "supported": False, "crossing_test_failure": False,
                "slope": float("nan"), "slope_ub": float("nan"),
                "mu_hat": float("nan"), "Dbar_hat": float("nan"),
                "q_star": float("nan"),
            }
        zeta = self._sample_probe(deterministic=deterministic_probe)
        q_tilde = clip_scalar(q_nominal + zeta, 0.0, self.cfg.q_max)

        u_safe, info = self.project_action(x, u_pi, q_tilde)
        weight = self._weight()
        intervention = weighted_norm(u_safe - u_pi, weight)
        component_names = self._barrier_component_names(
            x, len(info["h_current_components"])
        )

        decision = FilterDecision(
            action=u_safe.copy(), nominal_action=u_pi.copy(),
            q=q_nominal, q_tilde=float(q_tilde), zeta=float(zeta),
            h_current=float(info["h_current"]), hhat_projected=float(info["hhat_projected"]),
            barrier_rhs=float(info["barrier_rhs"]), feasible=bool(info["feasible"]),
            certified_before_execution=bool(info["feasible"] and
                info["hhat_projected"] >= info["barrier_rhs"] - self.cfg.projection_tol),
            intervention_norm=float(intervention),
            projection_margin=float(info["projection_margin"]),
            active_constraints=int(info["active_constraints"]),
            mode=str(info["mode"]),
            restoration_used=bool(info["restoration_used"]),
            restoration_kind=str(info["restoration_kind"]),
            restoration_slack=float(info["restoration_slack"]),
            effective_q=float(info["effective_q"]),
            strict_feasible=bool(info["strict_feasible"]),
            margin_capped=bool(info["margin_capped"]),
            headroom_search_succeeded=bool(info["headroom_search_succeeded"]),
            headroom_estimate=float(info["headroom_estimate"]),
            active_barrier_component=str(info["active_barrier_component"]),
            active_barrier_value=float(info["active_barrier_value"]),
            linear_feasible=bool(info["linear_feasible"]),
            exact_candidate_hhat=float(info["exact_candidate_hhat"]),
            h_current_components=np.asarray(
                info["h_current_components"], dtype=float
            ).tolist(),
            hhat_projected_components=np.asarray(
                info["hhat_projected_components"], dtype=float
            ).tolist(),
            barrier_rhs_components=np.asarray(
                info["barrier_rhs_components"], dtype=float
            ).tolist(),
            requested_barrier_rhs_components=np.asarray(
                info["requested_barrier_rhs_components"], dtype=float
            ).tolist(),
            barrier_context=barrier_context,
            barrier_component_names=list(component_names),
            audit_context=audit_context)

        # Do not advance actuator history here. The environment can still
        # modify this projected action with red-light, collision, turn, or
        # lane-edge guards. History is committed exactly once in
        # update_after_transition() using the action that CARLA actually executed.
        return decision

    # -- step 5-7: update margin after the transition ------------------------

    def update_after_transition(self, state, action, next_state, decision: FilterDecision
                                ) -> TransitionUpdate:
        x = np.asarray(state, dtype=np.float64).reshape(-1)
        xp = np.asarray(next_state, dtype=np.float64).reshape(-1)

        raw_u = np.asarray(action, dtype=np.float64).reshape(self.action_dim)
        finite_u = bool(np.all(np.isfinite(raw_u)))
        u = np.nan_to_num(raw_u, nan=0.0, posinf=0.0, neginf=0.0)
        absolute_feasible = bool(
            finite_u
            and np.all(u >= self.machine.low - float(self.cfg.projection_tol))
            and np.all(u <= self.machine.high + float(self.cfg.projection_tol))
        )
        u = np.clip(u, self.machine.low, self.machine.high)

        # Evaluate rate/jerk feasibility against the history that existed when this
        # action was selected.  The old code checked only the barrier inequality.
        low_eff, high_eff, actuator_set_feasible = (
            self.machine.effective_bounds_with_status(
                self.prev_action, self.prevprev_action
            )
        )
        dynamic_limits_feasible = bool(
            actuator_set_feasible
            and np.all(u >= low_eff - float(self.cfg.projection_tol))
            and np.all(u <= high_eff + float(self.cfg.projection_tol))
        )
        executed_limits_feasible = bool(
            absolute_feasible and dynamic_limits_feasible
        )
        # Always retain the true issued-command history. An external guard
        # override that violates rate or jerk limits makes the current
        # transition invalid; rebasing both history samples would erase that
        # violation and could falsely validate the following transition.
        history_rebased = False

        q_before = float(decision.q)
        q_tilde = float(decision.q_tilde)
        calibration_margin = clip_scalar(
            float(getattr(decision, "effective_q", q_tilde)),
            0.0,
            float(self.cfg.q_max),
        )
        barrier_context = getattr(decision, "barrier_context", {})
        executed_hhat_components = self._with_barrier_context_hhat_components(
            x, u, barrier_context
        )
        # The same decision-time barrier parameters and component ordering must
        # evaluate the post-transition state.  Using the mutable current CARLA
        # barrier here would mix I_t and I_{t+1} semantics in one residual.
        h_next_components = self._with_barrier_context_h_components(
            xp, barrier_context
        )
        component_count_before = int(executed_hhat_components.size)
        component_count_after = int(h_next_components.size)
        count_match = bool(component_count_before == component_count_after)
        saved_names = tuple(str(v) for v in getattr(
            decision, "barrier_component_names", []
        ))
        next_names = self._with_barrier_context_component_names(
            xp, component_count_after, barrier_context
        )
        names_match = bool(
            not saved_names or saved_names == tuple(next_names)
        )
        component_semantics_consistent = bool(count_match and names_match)

        # The calibrated score remains scalar, R=[min_i hhat_i-min_i h_i]_+.
        # If a custom barrier changes its component set, retain a scalar
        # diagnostic but invalidate attribution rather than crashing mid-run.
        if count_match:
            executed_hhat = float(np.min(executed_hhat_components))
            h_next = float(np.min(h_next_components))
        else:
            executed_hhat = self._with_barrier_context_hhat(
                x, u, barrier_context
            )
            h_next = self._with_barrier_context_h(xp, barrier_context)
        residual = float(max(0.0, executed_hhat - h_next))
        rhs_components = np.asarray(
            getattr(decision, "barrier_rhs_components", []), dtype=np.float64
        ).reshape(-1)
        if rhs_components.size == 0:
            rhs_components = np.asarray([decision.barrier_rhs], dtype=np.float64)
        executed_barrier_feasible = bool(
            component_semantics_consistent
            and executed_hhat_components.size == rhs_components.size
            and np.all(np.isfinite(executed_hhat_components))
            and np.all(
                executed_hhat_components + float(self.cfg.projection_tol)
                >= rhs_components
            )
        )
        executed_feasible = bool(executed_barrier_feasible and executed_limits_feasible)

        cap_attempted = bool(getattr(decision, "margin_capped", False))
        tol_margin = max(
            float(self.cfg.projection_tol),
            float(self.cfg.margin_comparison_tolerance),
        )
        margin_semantics_consistent = bool(
            (
                cap_attempted
                and 0.0 <= calibration_margin
                and calibration_margin < q_tilde - tol_margin
            )
            or (
                not cap_attempted
                and abs(calibration_margin - q_tilde) <= tol_margin
            )
        )
        # Strict and deliberately re-solved capped projections are attributable.
        # Restoration, fallback, or a downstream command that fails re-verification
        # freezes calibration even if it is physically conservative.
        calibration_valid = bool(
            executed_feasible
            and bool(decision.feasible)
            and not bool(getattr(decision, "restoration_used", False))
            and margin_semantics_consistent
        )
        # B_t is one only after final-command verification.  Keep the earlier
        # pre-execution cap result separately for attempt diagnostics.
        margin_capped = bool(calibration_valid and cap_attempted)
        direct_enforcement_valid = bool(
            calibration_valid
            and not margin_capped
            and bool(getattr(decision, "strict_feasible", decision.feasible))
        )
        # q_enforced is undefined on invalid transitions.  Do not fabricate an
        # enforced-margin exceedance or feed a numerical loss to the recursion.
        hard_exceed = bool(
            residual > calibration_margin
        ) if calibration_valid else False
        hard_exceed_requested = bool(residual > q_tilde)
        soft_loss = (
            self.soft_exceedance_loss(residual - calibration_margin)
            if calibration_valid else float("nan")
        )

        if (
            (self.cfg.method == "ecocsf" and calibration_valid)
            or self.cfg.method in {"naive_aci", "uncertainty"}
        ):
            self.residual_window.append(residual)

        if self.cfg.run_audit and self.cfg.method == "ecocsf":
            # ``audit_context`` was captured in select_action() before the probe.
            # Falling back to current_status supports older serialized decisions
            # while still using only records from prior transitions.
            audit_before_raw = dict(
                getattr(decision, "audit_context", {}) or
                self.audit.current_status(q0=q_before, infeasible=False)
            )
            audit_before = dict(audit_before_raw)
            if not direct_enforcement_valid:
                audit_before["status"] = (
                    "capped" if calibration_valid and margin_capped
                    else "invalid_execution"
                )
                audit_before["certified"] = False
                audit_before["supported"] = False
            audit_after = self.audit.update(
                q_tilde=calibration_margin,
                residual=residual,
                soft_exceedance=soft_loss,
                # The response model and tracking theorem use direct, uncapped
                # enforcement records only.  Capped records remain valid for the
                # runtime recursion and valid-exceedance accounting.
                infeasible=not direct_enforcement_valid,
            )
        else:
            audit_before_raw = {
                "status": "audit_off", "certified": False,
                "supported": False,
                "crossing_test_failure": False, "slope": float("nan"),
                "slope_ub": float("nan"), "mu_hat": float("nan"),
                "Dbar_hat": float("nan"), "q_star": float("nan"),
            }
            audit_before = dict(audit_before_raw)
            audit_after = dict(audit_before)

        # The optional schedule is prospective: the current residual may update
        # audit_after for t+1, but it cannot select the gain used at time t.
        eta_used = self._scheduled_eta(audit_before_raw)
        self.eta_current = eta_used
        anti_windup_backcalc = 0.0
        if self.cfg.method == "ecocsf" and calibration_valid:
            innovation = float(soft_loss - float(self.cfg.epsilon))
            if margin_capped:
                # The capped margin is deliberately re-solved and finally
                # verified, so its loss is attributable.  Retaining positive
                # capped innovation lets the pooled enforcement-valid loss
                # influence future requests.  A configurable gain preserves the
                # original conditional-integration ablation at gain=0.
                if innovation > 0.0:
                    innovation *= float(
                        self.cfg.capped_positive_integration
                    )
                    # Do not issue a downward back-calculation correction on the
                    # same transition that reports loss above the target.
                    if not bool(
                        self.cfg.freeze_backcalc_on_positive_cap
                    ):
                        anti_windup_backcalc = float(
                            self.cfg.anti_windup_gamma
                            * max(q_before - calibration_margin, 0.0)
                        )
                else:
                    # On a target-satisfying capped transition, ordinary
                    # back-calculation safely removes accumulated demand above
                    # the margin that was actually enforceable.
                    anti_windup_backcalc = float(
                        self.cfg.anti_windup_gamma
                        * max(q_before - calibration_margin, 0.0)
                    )
            q_after = clip_scalar(
                q_before + eta_used * innovation - anti_windup_backcalc,
                0.0,
                float(self.cfg.q_max),
            )
        elif self.cfg.method == "naive_aci":
            # Deliberate ablation: the recursion is updated even when the
            # requested margin was restored/overridden and hence unidentified.
            naive_loss = self.soft_exceedance_loss(residual - q_tilde)
            q_after = clip_scalar(
                q_before + eta_used * (naive_loss - float(self.cfg.epsilon)),
                0.0,
                float(self.cfg.q_max),
            )
        elif self.cfg.method == "fixed":
            q_after = float(np.clip(
                self.cfg.fixed_margin, 0.0, self.cfg.q_max
            ))
        elif self.cfg.method == "uncertainty":
            q_after = self._select_margin()
        else:
            q_after = q_before
        self.q = q_after

        violation = bool(h_next < 0.0)
        infeasible = bool(not calibration_valid)
        q_star = float(audit_before.get("q_star", float("nan")))
        delta = float(self.cfg.certified_tube_delta)
        tube_low = float(q_star - delta) if np.isfinite(q_star) else float("nan")
        tube_high = float(q_star + delta) if np.isfinite(q_star) else float("nan")
        in_certified_tube = False
        if direct_enforcement_valid:
            in_certified_tube = bool(
                np.isfinite(q_star)
                and 0.0 < q_star < float(self.cfg.q_max)
                and tube_low <= q_before <= tube_high
                and tube_low <= calibration_margin <= tube_high
            )

        audit_self_correcting = bool(audit_before_raw.get("supported",
                                                          audit_before_raw.get("certified", False)))
        audit_supported = bool(
            audit_self_correcting
            and in_certified_tube
            and direct_enforcement_valid
        )
        cross_fail = bool(
            audit_before_raw.get("crossing_test_failure", False)
        )
        # Observable unsupported-operation accounting.  The latent root event in
        # the theorem cannot be measured, so this is an operational diagnostic,
        # not an estimate of the exact theoretical P_bad.
        e_out = bool(not audit_supported)

        if calibration_valid:
            execution_mode = "capped" if margin_capped else "strict"
        elif bool(getattr(decision, "restoration_used", False)):
            execution_mode = "restore"
        elif str(getattr(decision, "mode", "")) == "fallback":
            execution_mode = "fallback"
        else:
            execution_mode = "unverified"

        upd = TransitionUpdate(
            residual=float(residual), hard_exceedance=hard_exceed,
            hard_exceedance_requested=hard_exceed_requested,
            soft_exceedance=float(soft_loss), q_before=q_before, q_tilde=q_tilde,
            q_after=q_after, eta_used=float(eta_used), h_next=float(h_next),
            violation=violation, audit_status=str(audit_before["status"]),
            certified=audit_supported, infeasible=infeasible,
            crossing_test_failure=cross_fail, e_out=e_out,
            audit_slope=float(audit_before["slope"]),
            audit_slope_ub=float(audit_before["slope_ub"]),
            mu_hat=float(audit_before["mu_hat"]),
            Dbar_hat=float(audit_before["Dbar_hat"]),
            q_star=q_star,
            certified_tube_low=tube_low,
            certified_tube_high=tube_high,
            in_certified_tube=in_certified_tube,
            executed_hhat=float(executed_hhat),
            executed_feasible=executed_feasible,
            history_rebased=history_rebased,
            calibration_margin=calibration_margin,
            q_enforced=(calibration_margin if calibration_valid else float("nan")),
            calibration_valid=calibration_valid,
            direct_enforcement_valid=direct_enforcement_valid,
            cap_attempted=cap_attempted,
            margin_capped=margin_capped,
            anti_windup_backcalc=anti_windup_backcalc,
            audit_supported=audit_supported,
            execution_mode=execution_mode,
            audit_status_after=str(audit_after["status"]),
            certified_after_update=bool(audit_after.get("certified", False)),
            component_semantics_consistent=component_semantics_consistent,
            component_count_before=component_count_before,
            component_count_after=component_count_after,
        )

        self.log.append({
            "t": len(self.log),
            **{f"x{i}": float(v) for i, v in enumerate(x.tolist())},
            **{f"u{i}": float(v) for i, v in enumerate(u.tolist())},
            **decision.to_dict(), **upd.to_dict(),
            "V_t": int(calibration_valid),
            "B_t": int(calibration_valid and margin_capped),
            "J_t": execution_mode,
            "executed_barrier_feasible": bool(executed_barrier_feasible),
            "executed_limits_feasible": bool(executed_limits_feasible),
        })

        self.prevprev_action = self.prev_action.copy()
        self.prev_action = u.copy()
        return upd
    def _scheduled_eta(self, audit_info: Dict[str, Any]) -> float:
        """eta* = 2 (Dbar^2 / mu)^(1/3), clamped to the stable range eta < 2mu/Lg^2."""
        eta_used = float(self.cfg.eta)
        if self.cfg.use_gain_schedule and audit_info.get("certified"):
            mu_hat = float(audit_info.get("mu_hat", float("nan")))
            Dbar = float(audit_info.get("Dbar_hat", float("nan")))
            if np.isfinite(mu_hat) and mu_hat > 0 and np.isfinite(Dbar) and Dbar > 0:
                eta_star = 2.0 * (Dbar * Dbar / mu_hat) ** (1.0 / 3.0)
                eta_stab = 1.9 * mu_hat / max(self.cfg.L_g ** 2, 1e-9)
                eta_cap = max(1e-12, min(self.cfg.eta_max, eta_stab))
                # Stability takes precedence if its cap lies below eta_min.
                eta_floor = min(self.cfg.eta_min, eta_cap)
                eta_used = clip_scalar(eta_star, eta_floor, eta_cap)
        return eta_used

    # -- calibration-only step (no projection): margin recursion + audit ------

    def step_with_residual(self, residual_fn: Callable[[float], float]) -> Dict[str, Any]:
        """Run one calibration-only recursion step against an endogenous residual."""
        q_before = self._select_margin()
        if self.cfg.run_audit:
            audit_before = self.audit.current_status(
                q0=q_before, infeasible=False
            )
        else:
            audit_before = {
                "status": "audit_off", "certified": False,
                "supported": False, "crossing_test_failure": False,
                "slope": float("nan"), "slope_ub": float("nan"),
                "mu_hat": float("nan"), "Dbar_hat": float("nan"),
                "q_star": float("nan"),
            }
        zeta = self._sample_probe()
        q_tilde = clip_scalar(q_before + zeta, 0.0, self.cfg.q_max)

        residual = float(residual_fn(q_tilde))
        hard_exceed = bool(residual > q_tilde)
        self.residual_window.append(residual)

        soft_loss = self.soft_exceedance_loss(residual - q_tilde)
        if self.cfg.run_audit:
            audit_after = self.audit.update(
                q_tilde=q_tilde,
                residual=residual,
                soft_exceedance=self.soft_exceedance_loss(residual - q_tilde),
                infeasible=False,
            )
        else:
            audit_after = dict(audit_before)
        eta_used = self._scheduled_eta(audit_before)
        self.eta_current = eta_used

        q_after = clip_scalar(
            q_before + eta_used * (soft_loss - float(self.cfg.epsilon)),
            0.0,
            float(self.cfg.q_max),
        )
        self.q = q_after

        q_star = float(audit_before.get("q_star", float("nan")))
        delta = float(self.cfg.certified_tube_delta)
        in_tube = bool(
            np.isfinite(q_star)
            and 0.0 < q_star < self.cfg.q_max
            and abs(q_before - q_star) <= delta
            and abs(q_tilde - q_star) <= delta
        )
        certified = bool(
            audit_before.get("supported", audit_before.get("certified", False))
            and in_tube
        )
        cross_fail = bool(
            audit_before.get("crossing_test_failure", False)
        )
        rec = {
            "t": len(self.log), "residual": residual,
            "hard_exceedance": hard_exceed,
            "hard_exceedance_requested": hard_exceed,
            "soft_exceedance": soft_loss,
            "q": q_before, "q_before": q_before,
            "q_tilde": q_tilde, "q_after": q_after,
            "eta_used": eta_used, "violation": False,
            "certified": certified, "crossing_test_failure": cross_fail,
            "infeasible": False,
            "feasible": True,
            "strict_feasible": True,
            "executed_feasible": True,
            "restoration_used": False,
            "headroom_search_succeeded": False,
            "calibration_valid": True,
            "q_enforced": q_tilde,
            "direct_enforcement_valid": True,
            "cap_attempted": False,
            "calibration_margin": q_tilde,
            "margin_capped": False,
            "anti_windup_backcalc": 0.0,
            "audit_supported": certified,
            "execution_mode": "strict",
            "V_t": 1, "B_t": 0, "J_t": "strict",
            "e_out": bool(not certified),
            "intervention_norm": 0.0,
            "audit_status": str(audit_before["status"]),
            "audit_status_after": str(audit_after["status"]),
            "certified_after_update": bool(audit_after.get("certified", False)),
            "audit_slope": float(audit_before["slope"]),
            "audit_slope_ub": float(audit_before["slope_ub"]),
            "mu_hat": float(audit_before["mu_hat"]),
            "Dbar_hat": float(audit_before["Dbar_hat"]),
        }
        self.log.append(rec)
        return rec

    # -- metrics aligned with Theorem 1 --------------------------------------
    def metrics(self) -> Dict[str, float]:
        if not self.log:
            return {}
        L = self.log
        n = len(L)
        exceed = np.asarray([r["hard_exceedance"] for r in L], dtype=np.float64)
        exceed_requested = np.asarray(
            [r.get("hard_exceedance_requested", r["hard_exceedance"]) for r in L],
            dtype=np.float64,
        )
        support = np.asarray(
            [bool(r.get("audit_supported", r.get("certified", False))) for r in L],
            dtype=bool,
        )
        valid = np.asarray(
            [bool(r.get("calibration_valid", not r.get("infeasible", False))) for r in L],
            dtype=bool,
        )
        direct = np.asarray(
            [bool(r.get("direct_enforcement_valid", valid[i])) for i, r in enumerate(L)],
            dtype=bool,
        )
        capped = np.asarray(
            [bool(r.get("margin_capped", False)) for r in L], dtype=bool
        )
        cap_attempted = np.asarray(
            [bool(r.get("cap_attempted", r.get("margin_capped", False))) for r in L],
            dtype=bool,
        )
        enforced_displacement = np.asarray([
            abs(float(r.get("calibration_margin", r["q_tilde"]))
                - float(r.get("q_before", r.get("q", 0.0))))
            for r in L
        ], dtype=np.float64)
        supported_probe_displacement = np.where(
            support & direct, enforced_displacement, 0.0
        )
        support_mask = support & direct & valid
        exceed_valid = float(
            exceed[valid].sum() / max(1, int(valid.sum()))
        )
        exceed_supported = (
            float(exceed[support_mask].mean())
            if support_mask.any() else float("nan")
        )
        eps = float(self.cfg.epsilon)
        signed_error = float(exceed_valid - eps) if np.isfinite(exceed_valid) else float("nan")
        q_after_values = np.asarray(
            [float(r.get("q_after", r.get("q", 0.0))) for r in L],
            dtype=np.float64,
        )
        operational_unsupported = ~support_mask
        deployment_failure = np.where(support_mask, exceed, 1.0)
        return {
            "steps": float(n),
            "violation_rate": float(np.mean([r["violation"] for r in L])),
            # Valid-transition exceedance includes verified strict and verified
            # capped enforcement; restored/fallback/unverified steps are excluded.
            "exceedance_rate": exceed_valid,
            "valid_exceedance_rate": exceed_valid,
            "exceedance_rate_all_steps_diagnostic": float(exceed_requested.mean()),
            "requested_exceedance_rate_all_steps": float(exceed_requested.mean()),
            "exceedance_certified": exceed_supported,
            "supported_exceedance_rate": exceed_supported,
            "coverage_error_signed": signed_error,
            "coverage_error_abs": float(abs(signed_error)) if np.isfinite(signed_error) else float("nan"),
            "coverage_gap": float(max(0.0, signed_error)) if np.isfinite(signed_error) else float("nan"),
            "coverage_gap_certified": float(max(0.0, (exceed_supported - eps))) if support_mask.any() else float("nan"),
            "calibration_valid_rate": float(valid.mean()),
            "invalid_rate": float((~valid).mean()),
            "direct_enforcement_rate": float(direct.mean()),
            "capping_rate": float(capped.mean()),
            "capped_valid_rate": float((capped & valid).mean()),
            "cap_attempt_rate": float(cap_attempted.mean()),
            "P_out": float(operational_unsupported.mean()),
            "P_bad_operational": float(operational_unsupported.mean()),
            "bad_operation_rate": float(operational_unsupported.mean()),
            "operational_unsupported_rate": float(operational_unsupported.mean()),
            "Z_rate_operational": float(deployment_failure.mean()),
            "deployment_failure_rate": float(deployment_failure.mean()),
            "P_cross": float(np.mean([r["crossing_test_failure"] for r in L])),
            "infeasible_rate": float(np.mean([r["infeasible"] for r in L])),
            "executed_infeasible_rate": float(np.mean([
                not bool(r.get("executed_feasible", False)) for r in L
            ])),
            "strict_projection_infeasible_rate": float(np.mean([
                not bool(r.get("strict_feasible", r.get("feasible", True))) for r in L
            ])),
            "headroom_search_rate": float(np.mean([
                bool(r.get("headroom_search_succeeded", False)) for r in L
            ])),
            "fallback_rate": float(np.mean([
                str(r.get("mode", "")) == "fallback" for r in L
            ])),
            "restoration_rate": float(np.mean([
                bool(r.get("restoration_used", False)) for r in L
            ])),
            "restoration_slack_mean": float(np.mean([
                float(r.get("restoration_slack", 0.0)) for r in L
                if np.isfinite(float(r.get("restoration_slack", 0.0)))
            ])) if any(np.isfinite(float(r.get("restoration_slack", 0.0))) for r in L) else float("nan"),
            "history_rebase_rate": float(np.mean([
                bool(r.get("history_rebased", False)) for r in L
            ])),
            "audit_support_fraction": float(support_mask.mean()),
            "certified_fraction": float(support_mask.mean()),
            "certified_fraction_valid": (
                float(support_mask[valid].mean()) if valid.any() else float("nan")
            ),
            # Observable counterpart of the theorem's supported displacement:
            # unsupported steps contribute zero before the all-step time average.
            "zeta_bar": float(supported_probe_displacement.mean()),
            "requested_probe_displacement_mean": float(np.mean([
                abs(float(r["q_tilde"]) - float(r.get("q_before", r.get("q", 0.0))))
                for r in L
            ])),
            "anti_windup_backcalc_mean": float(np.mean([
                float(r.get("anti_windup_backcalc", 0.0)) for r in L
            ])),
            "intervention_mean": float(np.mean([r["intervention_norm"] for r in L])),
            "q_mean": float(np.mean([r["q_after"] for r in L])),
            "q_final": float(self.q),
            "q_at_zero_rate": float(np.mean(
                q_after_values <= float(self.cfg.projection_tol)
            )),
            "q_at_max_rate": float(np.mean(
                q_after_values >= float(self.cfg.q_max) - float(self.cfg.projection_tol)
            )),
            "eta_mean": float(np.mean([r["eta_used"] for r in L])),
        }

    def save_audit_log(self, out_dir, prefix: str = "eclcs_audit") -> Dict[str, str]:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}
        if not self.log:
            return paths
        keys = sorted({k for rec in self.log for k in rec.keys()})
        csv_path = out_dir / f"{prefix}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for rec in self.log:
                w.writerow(rec)
        paths["csv"] = str(csv_path)
        if self.cfg.save_jsonl:
            jsonl_path = out_dir / f"{prefix}.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for rec in self.log:
                    f.write(json.dumps(rec) + "\n")
            paths["jsonl"] = str(jsonl_path)
        metrics_path = out_dir / f"{prefix}_metrics.json"
        metrics_path.write_text(json.dumps(self.metrics(), indent=2), encoding="utf-8")
        paths["metrics"] = str(metrics_path)
        return paths


# Aliases using paper acronyms.
ECOCSF = EndogenousClosedLoopConformalSafetyFilter
ECLCS = EndogenousClosedLoopConformalSafetyFilter


# =============================================================================
# Policy wrappers
# =============================================================================

class ECLCSAgent:
    def __init__(self, policy: Callable[[ArrayLike], np.ndarray],
                 safety_filter: EndogenousClosedLoopConformalSafetyFilter):
        self.policy = policy
        self.filter = safety_filter

    def act(self, state, deterministic_probe: bool = False) -> FilterDecision:
        u = np.asarray(self.policy(state), dtype=np.float64).reshape(self.filter.action_dim)
        return self.filter.select_action(state, u, deterministic_probe=deterministic_probe)

    def observe(self, state, decision, next_state,
                executed_action: Optional[ArrayLike] = None) -> TransitionUpdate:
        action = decision.action if executed_action is None else executed_action
        return self.filter.update_after_transition(state, action, next_state, decision)


class RandomDrivingPolicy:
    def __init__(self, machine: MachineCard, seed: int = 0):
        self.machine = machine
        self.rng = np.random.default_rng(seed)

    def __call__(self, state) -> np.ndarray:
        return self.rng.uniform(self.machine.low, self.machine.high)


class RuleBasedDrivingPolicy:
    """Lightweight black-box driving policy (placeholder; not the contribution)."""

    def __init__(self, target_speed=18.0, kp_lat=0.25, kp_head=0.80, kp_speed=0.35):
        self.target_speed, self.kp_lat, self.kp_head, self.kp_speed = \
            target_speed, kp_lat, kp_head, kp_speed

    def __call__(self, state) -> np.ndarray:
        x = np.asarray(state, dtype=np.float64).reshape(-1)
        lat, heading, speed = float(x[0]), float(x[1]), float(x[2])
        route_heading = float(x[8]) if x.size > 8 else heading
        lead = float(x[3]) if x.size > 3 else 100.0
        obs = float(x[5]) if x.size > 5 else 100.0
        steering = -self.kp_lat * lat - self.kp_head * heading - 0.35 * route_heading
        desired = self.target_speed
        if lead < 25.0 or obs < 20.0:
            desired = min(desired, 8.0)
        return np.asarray([steering, self.kp_speed * (desired - speed)], dtype=np.float64)


class ACCDriverPolicy:
    """Black-box ACC policy: chase target speed, ignore safety (filter handles it)."""

    def __init__(self, target_speed=30.0, kp=0.6):
        self.target_speed, self.kp = target_speed, kp

    def __call__(self, state) -> np.ndarray:
        v_ego = float(np.asarray(state, dtype=np.float64).reshape(-1)[1])
        return np.asarray([self.kp * (self.target_speed - v_ego)], dtype=np.float64)


# =============================================================================
# Simulation environments
# =============================================================================

class AnalyticACCEnv:
    """
    Analytic ACC companion (paper's controlled-drift setting).

    The lead vehicle accelerates with a slowly varying mean plus noise whose
    amplitude is `drift_scale`. Larger drift_scale -> larger Dbar, used to
    verify the cube-root coverage law. h is exact, so exceedance is measurable.
    """

    def __init__(self, dt=0.05, max_steps=2000, seed=0, drift_scale=0.5,
                 drift_period_s=20.0):
        self.dt, self.max_steps = float(dt), int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.barrier = ACCBarrierModel(dt=dt)
        self.drift_scale = float(drift_scale)
        self.drift_period = float(drift_period_s)
        self.t, self.state = 0, np.zeros(3)

    def reset(self) -> np.ndarray:
        self.t = 0
        v_ego = self.rng.uniform(15, 25); v_lead = self.rng.uniform(15, 25)
        gap = self.barrier.d0 + self.barrier.T * v_ego + self.rng.uniform(5, 20)
        self.state = np.asarray([gap, v_ego, v_lead])
        return self.state.copy()

    def _lead_accel(self) -> float:
        phase = 2.0 * math.pi * (self.t * self.dt) / max(self.drift_period, 1e-6)
        mean = 0.6 * self.drift_scale * math.sin(phase)
        return float(mean + self.drift_scale * self.rng.normal(0, 1.0))

    def step(self, action):
        a_ego = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        gap, v_ego, v_lead = float(self.state[0]), float(self.state[1]), float(self.state[2])
        a_lead = self._lead_accel()
        nv_ego = float(np.clip(v_ego + a_ego * self.dt, 0.0, self.barrier.v_max))
        nv_lead = float(np.clip(v_lead + a_lead * self.dt, 0.0, self.barrier.v_max))
        ngap = gap + (nv_lead - nv_ego) * self.dt + 0.02 * self.rng.normal(0, 1.0)
        self.state = np.asarray([max(ngap, 0.0), nv_ego, nv_lead])
        h = self.barrier.h(self.state)
        reward = nv_ego / self.barrier.v_max - 0.01 * a_ego ** 2 - (5.0 if h < 0 else 0.0)
        self.t += 1
        return self.state.copy(), float(reward), bool(self.t >= self.max_steps), {"h": float(h)}


def _carla_vec_forward(transform) -> Tuple[float, float]:
    """Unit forward (x, y) of a carla.Transform in the ground plane."""
    yaw = math.radians(transform.rotation.yaw)
    return math.cos(yaw), math.sin(yaw)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi



class RouteProgressTracker:
    """Monotone arc-length progress tracker for a precomputed CARLA waypoint route."""

    def __init__(self, waypoints: Sequence[Any]):
        self.route = list(waypoints)
        self.last_idx = 0
        self.last_s = 0.0
        if len(self.route) < 2:
            self.xy = np.zeros((0, 2), dtype=np.float64)
            self.s = np.zeros((0,), dtype=np.float64)
            self.yaw = np.zeros((0,), dtype=np.float64)
            return
        self.xy = np.asarray(
            [[wp.transform.location.x, wp.transform.location.y] for wp in self.route],
            dtype=np.float64,
        )
        self.s = np.zeros((len(self.route),), dtype=np.float64)
        for i in range(1, len(self.route)):
            self.s[i] = self.s[i - 1] + float(np.linalg.norm(self.xy[i] - self.xy[i - 1]))
        self.yaw = np.asarray([math.radians(wp.transform.rotation.yaw) for wp in self.route],
                              dtype=np.float64)

    def reset(self) -> None:
        self.last_idx = 0
        self.last_s = 0.0

    def project(self, loc: Any, back: int = 20, forward: int = 80) -> Tuple[float, float, float]:
        """Return continuity-safe route progress, lateral error, and route yaw.

        Junctions and self-near route geometry can place two different route
        segments at almost the same Cartesian location.  A pure nearest-segment
        search over a large forward window can therefore jump tens of metres to
        a future branch, producing the wrong route heading and steering the ego
        into the wrong lane.  Keep the projection local in route arc length and
        reject implausible one-tick progress jumps.
        """
        if self.xy.shape[0] < 2:
            return 0.0, 0.0, 0.0

        p = np.asarray([float(loc.x), float(loc.y)], dtype=np.float64)
        lo = max(0, int(self.last_idx) - int(back))

        # A 30-segment cap is already ~60 m for the default 2 m route spacing,
        # much larger than one control tick, but small enough to avoid snapping
        # to a geometrically nearby future branch at an intersection.
        local_forward = min(int(forward), 30)
        hi = min(self.xy.shape[0] - 2, int(self.last_idx) + max(local_forward, 5))

        # One 0.05 s CARLA tick moves <1 m even at 18 m/s.  A fixed 12 m
        # allowance was large enough to snap onto crossing/future geometry and
        # corrupt stop-line distance.  Allow roughly 1.5 route segments, with
        # a 3 m floor for spawn/reset numerical tolerance.
        positive_seg = np.diff(self.s)
        positive_seg = positive_seg[positive_seg > 1e-6]
        typical_seg = float(np.median(positive_seg)) if positive_seg.size else 2.0
        max_progress_jump_m = max(3.0, 1.5 * typical_seg)

        best_score = float("inf")
        best_s = float(self.last_s)
        best_idx = int(self.last_idx)
        best_lat = 0.0
        best_yaw = float(self.yaw[min(self.last_idx, len(self.yaw) - 1)])

        for i in range(lo, hi + 1):
            p0, p1 = self.xy[i], self.xy[i + 1]
            v = p1 - p0
            seg_len = float(np.linalg.norm(v))
            if seg_len < 1e-6:
                continue

            t = float(np.clip(np.dot(p - p0, v) / (seg_len * seg_len), 0.0, 1.0))
            s_here = float(self.s[i] + t * seg_len)

            # Allow a little backward geometric recovery, but never select a
            # segment that would make monotone mission progress jump forward by
            # an implausible amount in a single simulator tick.
            if s_here < self.last_s - 3.0:
                continue
            if s_here > self.last_s + max_progress_jump_m:
                continue

            proj = p0 + t * v
            dvec = p - proj
            d = float(np.linalg.norm(dvec))
            tangent = v / seg_len
            # Match CARLA's signed lateral convention used elsewhere in this
            # environment: positive means displacement toward the waypoint's
            # right vector. For tangent=(tx,ty), right=(-ty,tx).
            lat = float(-dvec[0] * tangent[1] + dvec[1] * tangent[0])

            # Prefer spatial proximity, with a small continuity penalty so an
            # equally-close future segment does not win merely due to crossing
            # geometry.
            index_jump = max(0, int(i) - int(self.last_idx))
            score = d + 0.015 * float(index_jump)
            if score < best_score:
                best_score = score
                best_s = s_here
                best_idx = i
                best_lat = lat
                best_yaw = math.atan2(float(v[1]), float(v[0]))

        # Preserve monotone progress while allowing the selected route index to
        # advance only through the continuity-safe candidate above.
        if best_s + 1.0 >= self.last_s:
            self.last_s = max(self.last_s, best_s)
            self.last_idx = max(self.last_idx, best_idx)

        return float(self.last_s), float(best_lat), float(best_yaw)

    def yaw_at_s(self, s_target: float) -> float:
        """Route tangent yaw at a requested arc-length position."""
        if self.yaw.size == 0:
            return 0.0
        idx = int(np.searchsorted(self.s, float(s_target), side="left"))
        idx = min(max(idx, 0), len(self.yaw) - 1)
        return float(self.yaw[idx])

    def pose_at_s(self, s_target: float) -> Tuple[float, float, float]:
        """Interpolate route position and tangent at arc length ``s_target``.

        This is intentionally non-mutating.  It is used by the predictive
        collision guard to sweep the ego footprint along the already-selected
        global route without changing progress bookkeeping.
        """
        if self.xy.shape[0] == 0:
            return 0.0, 0.0, 0.0
        if self.xy.shape[0] == 1 or self.s.size < 2:
            return float(self.xy[0, 0]), float(self.xy[0, 1]), float(self.yaw[0])

        s_clamped = float(np.clip(float(s_target), float(self.s[0]), float(self.s[-1])))
        i = int(np.searchsorted(self.s, s_clamped, side="right") - 1)
        i = min(max(i, 0), len(self.s) - 2)
        seg = float(self.s[i + 1] - self.s[i])
        alpha = 0.0 if seg <= 1e-9 else float(
            np.clip((s_clamped - float(self.s[i])) / seg, 0.0, 1.0)
        )
        p = (1.0 - alpha) * self.xy[i] + alpha * self.xy[i + 1]
        dv = self.xy[i + 1] - self.xy[i]
        yaw = (
            math.atan2(float(dv[1]), float(dv[0]))
            if float(np.linalg.norm(dv)) > 1e-9
            else float(self.yaw[i])
        )
        return float(p[0]), float(p[1]), float(yaw)

    def lookahead_yaw(self, distance_m: float) -> float:
        """Route tangent yaw a fixed distance ahead of current progress."""
        return self.yaw_at_s(float(self.last_s) + max(0.0, float(distance_m)))

    def turn_angle_ahead(self, distance_m: float) -> float:
        """Signed planned heading change over the look-ahead distance."""
        if self.yaw.size == 0:
            return 0.0
        current = self.yaw_at_s(float(self.last_s))
        future = self.lookahead_yaw(float(distance_m))
        return float(_wrap_pi(future - current))

    def query(self, loc: Any, center_idx: Optional[int] = None, back: int = 40,
              forward: int = 120) -> Tuple[float, float, float, int]:
        """Non-mutating route projection.

        Returns (arc_length_s, signed_lateral_error, route_yaw, segment_index).
        This is used for other vehicles. It must NOT update last_s/last_idx,
        otherwise a crossing or adjacent-lane car can corrupt ego progress.
        """
        if self.xy.shape[0] < 2:
            return 0.0, 0.0, 0.0, 0
        p = np.asarray([float(loc.x), float(loc.y)], dtype=np.float64)
        ci = self.last_idx if center_idx is None else int(center_idx)
        lo = max(0, ci - int(back))
        hi = min(self.xy.shape[0] - 2, ci + int(forward))
        best_score = float("inf")
        best_s = 0.0
        best_idx = ci
        best_lat = 0.0
        best_yaw = float(self.yaw[min(max(ci, 0), len(self.yaw) - 1)])
        for i in range(lo, hi + 1):
            p0, p1 = self.xy[i], self.xy[i + 1]
            v = p1 - p0
            seg_len = float(np.linalg.norm(v))
            if seg_len < 1e-6:
                continue
            t = float(np.clip(np.dot(p - p0, v) / (seg_len * seg_len), 0.0, 1.0))
            proj = p0 + t * v
            dvec = p - proj
            d = float(np.linalg.norm(dvec))
            tangent = v / seg_len
            # Match CARLA's signed lateral convention used elsewhere in this
            # environment: positive means displacement toward the waypoint's
            # right vector. For tangent=(tx,ty), right=(-ty,tx).
            lat = float(-dvec[0] * tangent[1] + dvec[1] * tangent[0])
            s_here = float(self.s[i] + t * seg_len)
            # At Town10HD junctions, spatially crossing route branches can be
            # nearly equidistant. Prefer the segment nearest the caller's route
            # index unless a farther-index segment is meaningfully closer in XY.
            # This is non-mutating and therefore safe for actor/light queries.
            score = d + 0.015 * abs(int(i) - int(ci))
            if score < best_score:
                best_score = score
                best_s = s_here
                best_idx = i
                best_lat = lat
                best_yaw = math.atan2(float(v[1]), float(v[0]))
        return float(best_s), float(best_lat), float(best_yaw), int(best_idx)

    def curvature_ahead(self, lookahead_segments: int = 15) -> float:
        if self.xy.shape[0] < 3:
            return 0.0
        i0 = min(max(self.last_idx, 0), len(self.yaw) - 1)
        i1 = min(i0 + max(1, int(lookahead_segments)), len(self.yaw) - 1)
        return float(_wrap_pi(float(self.yaw[i1] - self.yaw[i0])))



class CarlaDrivingEnv:
    """
    Runnable CARLA (>=0.9.15) environment for the E-COCSF driving experiments.

    Exposes the 8-D state expected by AutonomousDrivingBarrierModel:
        [lat_offset, heading_err, speed, lead_gap, rel_vel, obstacle_dist,
         lane_half - lat, lane_half + lat]
    so the SAME barrier h is used for projection (filter), shortfall, reward,
    and termination. The filter's one-step predictor (the simple kinematic
    model in AutonomousDrivingBarrierModel) acts as the nominal certificate;
    the shortfall R = [hhat(x_t,u_t) - h_CARLA(x_{t+1})]_+ therefore measures
    the genuine model-vs-CARLA error (physics, tire friction, weather, latency,
    lead behaviour) -- exactly the endogenous residual the margin calibrates.

    Controlled drift for the scaling study is injected through the lead-vehicle
    speed profile (amplitude ~ drift_scale) and, optionally, weather/friction.

    Usage:
        env = CarlaDrivingEnv(host="localhost", port=2200, town="Town04",
                              dt=0.05, drift_scale=0.5)
        state = env.reset()
        decision = agent.act(state)
        next_state, reward, done, info = env.step(decision.action)
        agent.observe(state, decision, next_state)
        ...
        env.close()
    """

    def __init__(self, host: str = "localhost", port: int = 2000, town: str = "Town04",
                 dt: float = 0.05, max_steps: int = 1200, seed: int = 0,
                 drift_scale: float = 0.5, drift_period_s: float = 20.0,
                 lead_gap0: float = 40.0, spawn_lead_vehicle: bool = True,
                 target_speed: float = 18.0,
                 vary_weather: bool = True, render_follow: bool = True,
                 load_town: bool = True, timeout_s: float = 20.0,
                 throttle_floor: float = 0.35,
                 route_distance: float = 1000.0,
                 success_bonus: float = 100.0,
                 action_smoothing_beta: float = 0.15,
                 terminate_on_headway_violation: bool = False,
                 headway_hard_fail_gap: float = 0.75,
                 route_step_m: float = 2.0,
                 route_search_back: int = 20,
                 route_search_forward: int = 80,
                 route_planner_mode: str = "global",
                 allow_heuristic_route_fallback: bool = False,
                 route_destination_candidates: int = 32,
                 route_turn_lookahead_m: float = 12.0,
                 route_turn_speed: float = 4.5,
                 route_turn_steer_gain: float = 0.85,
                 turn_recovery_speed: float = 0.8,
                 turn_recovery_accel: float = 0.55,
                 turn_recovery_patience_ticks: int = 25,
                 failure_persistence_ticks: int = 10,
                 headway_failure_persistence_ticks: int = 8,
                 use_augmented_state: bool = True,
                 num_traffic_vehicles: int = 0,
                 num_walkers: int = 0,
                 tm_port: int = 8000,
                 traffic_speed_difference: float = 35.0,
                 traffic_min_distance: float = 6.0,
                 traffic_rear_safety_distance: float = 7.0,
                 traffic_auto_lane_change: bool = False,
                 traffic_radius: float = 0.0,
                 walker_speed_min: float = 0.8,
                 walker_speed_max: float = 1.6,
                 traffic_warmup_ticks: int = 40,
                 tm_hybrid_physics: bool = False,
                 protect_cross_town_walkers: bool = True,
                 episode_reset_settle_ticks: int = 1,
                 destroy_all_stale_actors: bool = False,
                 weather_mode: str = "clear",
                 traffic_light_guard: bool = True,
                 red_light_stop_distance: float = 70.0,
                 traffic_light_hysteresis_ticks: int = 5,
                 red_light_reaction_time_s: float = 0.35,
                 red_light_activation_margin: float = 2.0,
                 red_light_blend_distance: float = 2.0,
                 yellow_light_stop: bool = True,
                 traffic_light_use_affected_lanes: bool = True,
                 traffic_light_heading_tolerance_deg: float = 35.0,
                 yellow_reaction_time_s: float = 0.5,
                 yellow_comfort_decel: float = 3.0,
                 yellow_stop_margin: float = 1.0,
                 junction_commit_clear_distance: float = 8.0,
                 junction_commit_clear_ticks: int = 5,
                 stop_line_cross_tolerance: float = 0.20,
                 red_light_stop_buffer: float = 1.2,
                 red_light_virtual_offset: float = 0.0,
                 red_light_creep_speed: float = 0.8,
                 red_light_creep_distance: float = 3.0,
                 red_light_comfort_decel: float = 3.0,
                 red_light_keep_lead_gap: float = 8.0,
                 queue_stop_gap: float = 3.0,
                 queue_detect_distance: float = 25.0,
                 queue_creep_speed: float = 1.5,
                 vehicle_collision_guard: bool = True,
                 vehicle_stop_gap: float = 4.0,
                 vehicle_detect_distance: float = 45.0,
                 vehicle_ttc_soft: float = 3.5,
                 vehicle_ttc_hard: float = 1.4,
                 vehicle_moving_time_headway: float = 1.2,
                 vehicle_soft_extra_gap: float = 1.5,
                 vehicle_queue_speed_threshold: float = 0.7,
                 predictive_collision_guard: bool = True,
                 predictive_horizon_s: float = 3.0,
                 predictive_step_s: float = 0.20,
                 predictive_vehicle_radius_m: float = 45.0,
                 predictive_lateral_margin_m: float = 0.35,
                 predictive_longitudinal_margin_m: float = 1.00,
                 predictive_ttc_soft: float = 3.0,
                 predictive_ttc_hard: float = 1.2,
                 predictive_clear_ticks: int = 5,
                 predictive_junction_lookahead_m: float = 30.0,
                 predictive_junction_preview_speed: float = 3.0,
                 ego_overtake_enabled: bool = True,
                 external_blockage_recovery: bool = False,
                 external_blockage_patience_s: float = 15.0,
                 external_blockage_max_recoveries: int = 2,
                 road_edge_guard: bool = True,
                 lane_edge_soft_margin: float = 0.75,
                 lane_edge_hard_margin: float = 0.30,
                 lane_edge_target_speed: float = 3.0,
                 lane_edge_brake: float = 2.5,
                 lane_edge_steer_gain: float = 0.75,
                 lane_edge_heading_gain: float = 0.60,
                 traffic_light_route_scan: bool = True,
                 traffic_light_route_scan_step: float = 4.0,
                 traffic_light_landmark_fallback: bool = True,
                 vehicle_route_corridor_factor: float = 0.55,
                 vehicle_route_corridor_max: float = 1.00):
        self.host, self.port, self.town = host, int(port), town
        self.dt, self.max_steps = float(dt), int(max_steps)
        self.seed = int(seed)
        self.drift_scale, self.drift_period = float(drift_scale), float(drift_period_s)
        self.lead_gap0, self.target_speed = float(lead_gap0), float(target_speed)
        self.spawn_lead_vehicle = bool(spawn_lead_vehicle)
        self.vary_weather, self.render_follow = bool(vary_weather), bool(render_follow)
        self.load_town, self.timeout_s = bool(load_town), float(timeout_s)
        self.throttle_floor = float(throttle_floor)
        self.route_distance = float(route_distance)
        self._goal_distance_m = float(route_distance)
        self.route_length_m = 0.0
        self._front_vehicle_heading_diff = 0.0
        self.success_bonus = float(success_bonus)
        self.action_smoothing_beta = float(np.clip(action_smoothing_beta, 0.0, 0.30))
        self.terminate_on_headway_violation = bool(terminate_on_headway_violation)
        self.headway_hard_fail_gap = float(headway_hard_fail_gap)

        # Automatically increase max_steps for long routes or slow target speeds
        # so red-light/junction waiting is less likely to terminate by timeout.
        if self.route_distance > 0 and self.max_steps < 3 * self.route_distance / max(self.target_speed * self.dt, 1e-6):
            self.max_steps = int(max(self.max_steps, math.ceil(3.0 * self.route_distance / max(self.target_speed * self.dt, 1e-6))))
        self.route_step_m = float(max(route_step_m, 0.5))
        self.route_search_back = int(max(route_search_back, 1))
        self.route_search_forward = int(max(route_search_forward, 5))
        self.route_planner_mode = str(route_planner_mode or "global").strip().lower()
        if self.route_planner_mode not in {"global", "heuristic"}:
            raise ValueError("route_planner_mode must be 'global' or 'heuristic'.")
        self.allow_heuristic_route_fallback = bool(allow_heuristic_route_fallback)
        self.route_destination_candidates = int(np.clip(route_destination_candidates, 4, 256))
        self._route_planner_used = "none"
        self._route_planner_error = ""
        self._route_candidate_count = 0

        # Turn-aware route following.  The policy observation remains 12-D for
        # checkpoint compatibility; look-ahead route geometry is consumed by a
        # common low-level guard that is shared by training and every evaluation
        # method.
        self.route_turn_lookahead_m = float(np.clip(route_turn_lookahead_m, 4.0, 30.0))
        self.route_turn_speed = float(np.clip(route_turn_speed, 1.5, 10.0))
        self.route_turn_steer_gain = float(np.clip(route_turn_steer_gain, 0.10, 2.50))
        self.turn_recovery_speed = float(np.clip(turn_recovery_speed, 0.20, 2.00))
        self.turn_recovery_accel = float(np.clip(turn_recovery_accel, 0.10, 1.50))
        self.turn_recovery_patience_ticks = int(max(turn_recovery_patience_ticks, 5))

        self.failure_persistence_ticks = int(max(failure_persistence_ticks, 1))
        self.headway_failure_persistence_ticks = int(max(headway_failure_persistence_ticks, 1))
        self.use_augmented_state = bool(use_augmented_state)

        # Background traffic and pedestrians. Keep these low during training;
        # the observation is compact route-corridor state, not full perception.
        self.num_traffic_vehicles = int(max(num_traffic_vehicles, 0))
        self.num_walkers = int(max(num_walkers, 0))
        self.tm_port = int(tm_port)
        self.traffic_speed_difference = float(traffic_speed_difference)
        self.traffic_min_distance = float(max(traffic_min_distance, 0.0))
        # NPC vehicles use this larger distance when following any vehicle,
        # including the ego.  This specifically reduces rear-end impacts when
        # the ego legitimately slows for a turn, queue, pedestrian, or red light.
        self.traffic_rear_safety_distance = float(max(
            traffic_rear_safety_distance, self.traffic_min_distance, 0.0
        ))
        self.traffic_auto_lane_change = bool(traffic_auto_lane_change)
        self.traffic_radius = float(max(traffic_radius, 0.0))
        self.walker_speed_min = float(max(walker_speed_min, 0.0))
        self.walker_speed_max = float(max(walker_speed_max, self.walker_speed_min))
        self.traffic_warmup_ticks = int(max(traffic_warmup_ticks, 0))
        self.tm_hybrid_physics = bool(tm_hybrid_physics)

        # Cross-town lifecycle safety. CARLA 0.9.15 has a documented bug where
        # pedestrian navigation data from the previously loaded town can be
        # retained after client.load_world(). On vulnerable/unknown server
        # versions we therefore suppress AI walkers after an actual map switch
        # unless the caller explicitly opts out of this protection.
        self.protect_cross_town_walkers = bool(protect_cross_town_walkers)
        self.episode_reset_settle_ticks = int(max(episode_reset_settle_ticks, 0))
        # Default to ownership-safe stale cleanup. Set True only on a dedicated
        # CARLA server when every vehicle/walker/sensor belongs to this experiment.
        self.destroy_all_stale_actors = bool(destroy_all_stale_actors)
        self._effective_num_walkers = int(self.num_walkers)
        self._map_changed_on_connect = False
        self._startup_map_name = ""
        self._server_version = "unknown"

        # Weather and traffic-rule options.
        # weather_mode: clear | rain | night | night_rain | fog | morning_rain | random | dynamic
        self.weather_mode = str(weather_mode or "clear").lower()
        self._current_weather_mode = self.weather_mode
        self.traffic_light_guard = bool(traffic_light_guard)
        self.red_light_stop_distance = float(max(red_light_stop_distance, 5.0))
        self.traffic_light_hysteresis_ticks = int(np.clip(traffic_light_hysteresis_ticks, 0, 30))
        self.red_light_reaction_time_s = float(np.clip(red_light_reaction_time_s, 0.0, 2.0))
        self.red_light_activation_margin = float(np.clip(red_light_activation_margin, 0.0, 10.0))
        self.red_light_blend_distance = float(np.clip(red_light_blend_distance, 0.5, 10.0))
        self.yellow_light_stop = bool(yellow_light_stop)

        # Human-like signal semantics.  CARLA exposes both stop waypoints and
        # affected-lane waypoints for each traffic light; use both whenever
        # possible so cross-street/adjacent-lane signals are rejected.
        self.traffic_light_use_affected_lanes = bool(traffic_light_use_affected_lanes)
        self.traffic_light_heading_tolerance_deg = float(np.clip(
            traffic_light_heading_tolerance_deg, 10.0, 80.0))
        self.yellow_reaction_time_s = float(np.clip(yellow_reaction_time_s, 0.0, 2.0))
        self.yellow_comfort_decel = float(np.clip(yellow_comfort_decel, 0.5, 8.0))
        self.yellow_stop_margin = float(np.clip(yellow_stop_margin, 0.0, 5.0))
        self.junction_commit_clear_distance = float(np.clip(
            junction_commit_clear_distance, 3.0, 30.0))
        self.junction_commit_clear_ticks = int(np.clip(
            junction_commit_clear_ticks, 1, 50))
        self.stop_line_cross_tolerance = float(np.clip(
            stop_line_cross_tolerance, 0.05, 1.0))

        # Desired visible stopping distance before the physical CARLA stop line.
        # ``red_light_virtual_offset`` is retained only for CLI/checkpoint
        # compatibility; the controller now uses physical stop-waypoint distance.
        self.red_light_stop_buffer = float(np.clip(red_light_stop_buffer, 0.5, 6.0))
        self.red_light_virtual_offset = float(np.clip(red_light_virtual_offset, 0.0, 10.0))
        self.red_light_creep_speed = float(np.clip(red_light_creep_speed, 0.5, 5.0))
        self.red_light_creep_distance = float(np.clip(red_light_creep_distance, 3.0, 30.0))
        # Braking curve for a red light.  The previous hard-coded 1.45 m/s^2
        # began constraining an 8 m/s ego about 22 m before the target stop pose,
        # which looked like premature crawling.  A configurable urban comfort
        # deceleration keeps normal speed until a physically meaningful braking
        # distance, while the final creep zone still handles precise positioning.
        self.red_light_comfort_decel = float(np.clip(red_light_comfort_decel, 1.0, 6.0))
        # Deprecated compatibility value. Queue behavior is governed by
        # queue_stop_gap / queue_detect_distance and the universal front guard.
        self.red_light_keep_lead_gap = float(np.clip(red_light_keep_lead_gap, 1.0, 25.0))
        # Queue-following controller: when another vehicle is already waiting
        # before the same red light, stop behind that vehicle rather than creeping
        # to the zebra/stop line.  This fixes collisions with queued NPC cars.
        self.queue_stop_gap = float(np.clip(queue_stop_gap, 1.0, 8.0))
        self.queue_detect_distance = float(np.clip(queue_detect_distance, self.queue_stop_gap + 2.0, 60.0))
        self.queue_creep_speed = float(np.clip(queue_creep_speed, 0.5, 4.0))

        # Universal vehicle-collision guard.  Unlike the red-light queue logic,
        # this is active at every step and prevents rear-end collisions with
        # spawned/Traffic-Manager vehicles that are in the same route corridor.
        self.vehicle_collision_guard = bool(vehicle_collision_guard)
        self.vehicle_stop_gap = float(np.clip(vehicle_stop_gap, 1.0, 8.0))
        self.vehicle_detect_distance = float(np.clip(vehicle_detect_distance, self.vehicle_stop_gap + 3.0, 80.0))
        self.vehicle_ttc_soft = float(np.clip(vehicle_ttc_soft, 0.5, 8.0))
        self.vehicle_ttc_hard = float(np.clip(vehicle_ttc_hard, 0.2, self.vehicle_ttc_soft))
        # Moving traffic should not cause early parking.  We keep a small
        # stopped/queued gap but add only a modest speed-dependent headway for
        # moving vehicles, so ego slows around 5--6 m and does not stop at 15--20 m.
        self.vehicle_moving_time_headway = float(np.clip(vehicle_moving_time_headway, 0.0, 1.5))
        self.vehicle_soft_extra_gap = float(np.clip(vehicle_soft_extra_gap, 0.0, 6.0))
        self.vehicle_queue_speed_threshold = float(np.clip(vehicle_queue_speed_threshold, 0.0, 3.0))

        # Checkpoint-compatible dynamic-occupancy shield.  The learned policy
        # still consumes the original 8/12-D observation.  This common
        # environment guard predicts oriented vehicle footprints with measured
        # velocities, so perpendicular junction traffic and adjacent-lane
        # cut-ins are protected without changing the checkpoint state shape.
        self.predictive_collision_guard = bool(predictive_collision_guard)
        self.predictive_horizon_s = float(np.clip(predictive_horizon_s, 1.0, 6.0))
        self.predictive_step_s = float(np.clip(predictive_step_s, self.dt, 0.50))
        self.predictive_vehicle_radius_m = float(np.clip(
            predictive_vehicle_radius_m, 15.0, 100.0
        ))
        self.predictive_lateral_margin_m = float(np.clip(
            predictive_lateral_margin_m, 0.05, 1.00
        ))
        self.predictive_longitudinal_margin_m = float(np.clip(
            predictive_longitudinal_margin_m, 0.20, 3.00
        ))
        self.predictive_ttc_soft = float(np.clip(predictive_ttc_soft, 1.0, 6.0))
        self.predictive_ttc_hard = float(np.clip(
            predictive_ttc_hard, 0.30, self.predictive_ttc_soft
        ))
        self.predictive_clear_ticks = int(np.clip(predictive_clear_ticks, 1, 30))
        self.predictive_junction_lookahead_m = float(np.clip(
            predictive_junction_lookahead_m, 8.0, 60.0
        ))
        self.predictive_junction_preview_speed = float(np.clip(
            predictive_junction_preview_speed, 1.0, 6.0
        ))
        self.ego_overtake_enabled = bool(ego_overtake_enabled)
        # Optional simulator-hygiene recovery for an experiment-owned NPC that
        # has become stationary and physically blocks the ego for a long time.
        # This never forces the ego through an occupied junction.  It removes
        # only a vehicle spawned by this environment, only after a configurable
        # grace period, and logs every intervention for transparent reporting.
        self.external_blockage_recovery = bool(external_blockage_recovery)
        self.external_blockage_patience_s = float(np.clip(
            external_blockage_patience_s, 5.0, 120.0
        ))
        self.external_blockage_max_recoveries = int(np.clip(
            external_blockage_max_recoveries, 0, 20
        ))
        self._predictive_conflict_raw = False
        self._predictive_conflict_active = False
        self._predictive_conflict_clear_ticks = 0
        self._predictive_conflict_actor_id = -1
        self._predictive_conflict_actor_type = "none"
        self._predictive_conflict_kind = "none"
        self._predictive_conflict_ttc = 999.0
        self._predictive_conflict_distance = 999.0
        self._predictive_conflict_accel_cap = 2.5
        self._predictive_conflict_guard_active = False
        self._predictive_junction_context = False
        self._safety_trace: Deque[Dict[str, Any]] = deque(
            maxlen=max(20, int(round(5.0 / max(self.dt, 1e-6))))
        )

        # Lane/road-edge guard.  This keeps the ego inside the drivable lane by
        # steering back to lane center and reducing acceleration near edges.
        self.road_edge_guard = bool(road_edge_guard)
        self.lane_edge_soft_margin = float(np.clip(lane_edge_soft_margin, 0.35, 2.0))
        self.lane_edge_hard_margin = float(np.clip(lane_edge_hard_margin, 0.10, self.lane_edge_soft_margin))
        self.lane_edge_target_speed = float(np.clip(lane_edge_target_speed, 0.5, 8.0))
        self.lane_edge_brake = float(np.clip(lane_edge_brake, 0.5, 4.0))
        self.lane_edge_steer_gain = float(np.clip(lane_edge_steer_gain, 0.0, 2.0))
        self.lane_edge_heading_gain = float(np.clip(lane_edge_heading_gain, 0.0, 2.0))

        self._front_vehicle_gap = 80.0
        self._front_vehicle_speed = 0.0
        self._front_vehicle_kind = "none"
        self._front_vehicle_ttc = 20.0
        # True only for an actor that is genuinely ahead in the ego lane/route
        # corridor. A raw actor rejected by red-light priority must never survive
        # into the universal collision guard and cause an early false stop.
        self._front_vehicle_guard_valid = False
        self._front_vehicle_guard_active = False
        self._front_vehicle_accel_cap = 2.5
        self._lane_edge_guard_active = False
        self._lane_edge_margin = 999.0
        self._lane_edge_accel_cap = 2.5
        self._lane_edge_steer_correction = 0.0
        self._red_light_front_gap = 80.0
        self._red_light_front_speed = 0.0
        self._red_light_queue_active = False
        self._red_light_queue_gap_error = 999.0
        # Robust detection helpers. Town10HD junctions can fail a single
        # get_traffic_lights_from_waypoint() query because the stop line may be
        # attached to a future waypoint/landmark rather than the current ego
        # waypoint.  We therefore scan route waypoints and optionally OpenDRIVE
        # dynamic landmarks as fallbacks.
        self.traffic_light_route_scan = bool(traffic_light_route_scan)
        self.traffic_light_route_scan_step = float(max(traffic_light_route_scan_step, 1.0))
        self.traffic_light_landmark_fallback = bool(traffic_light_landmark_fallback)
        # Vehicle hazard fallback corridor.  A too-wide route corridor can mark
        # an adjacent-lane car as the lead vehicle and make ego stop in the
        # middle of the road.  Require same-lane when possible; otherwise accept
        # only vehicles very close to the planned route centerline.
        self.vehicle_route_corridor_factor = float(np.clip(vehicle_route_corridor_factor, 0.25, 1.50))
        self.vehicle_route_corridor_max = float(np.clip(vehicle_route_corridor_max, 0.50, 2.50))
        self._traffic_light_active = False
        self._traffic_light_state = "none"
        self._traffic_light_distance = 999.0
        self._traffic_light_virtual_gap = 999.0
        self._traffic_light_stop_error = 999.0
        self._traffic_light_accel_cap = 2.5
        self._traffic_light_id = -1
        self._traffic_light_yellow_go = False
        self._yellow_required_stop_distance = 0.0
        self._yellow_go_signal_keys: set = set()
        self._traffic_light_miss_ticks = 0
        self._traffic_light_detection_dropout = False
        self._traffic_light_last_signal_key: Optional[Tuple[Any, ...]] = None
        self._traffic_light_last_stop_s = -1.0
        self._red_light_crossed_on_red = False
        self._red_light_violation_count = 0
        self._red_light_stop_success = False
        self._red_light_stop_success_count = 0
        self._red_light_last_success_key: Optional[Tuple[Any, ...]] = None

        # Stop-line tracking and per-signal junction commitment.  Once the ego
        # front bumper crosses the governing stop line, that same signal is
        # ignored until the vehicle has genuinely cleared the junction.  This
        # prevents the unsafe/unhuman behavior of stopping inside an intersection
        # merely because the light turned red after entry.
        self._tracked_light_id: Optional[int] = None
        self._tracked_signal_key: Optional[Tuple[Any, ...]] = None
        self._tracked_stop_s = -1.0
        self._tracked_stop_gap_prev = 999.0
        self._tracked_light_seen_ahead = False
        self._tracked_light_state = "none"
        self._tracked_yellow_go_latched = False
        self._committed_light_id: Optional[int] = None
        self._committed_signal_key: Optional[Tuple[Any, ...]] = None
        self._committed_stop_s = -1.0
        self._junction_commit_active = False
        self._junction_seen_since_commit = False
        self._junction_exit_clear_ticks = 0

        self._red_light_control_active = False
        self._red_light_front_gap = 80.0
        self._red_light_front_speed = 0.0
        self._red_light_queue_active = False
        self._red_light_queue_gap_error = 999.0
        self._front_vehicle_gap = 80.0
        self._front_vehicle_speed = 0.0
        self._front_vehicle_kind = "none"
        self._front_vehicle_ttc = 20.0
        # True only for an actor that is genuinely ahead in the ego lane/route
        # corridor. A raw actor rejected by red-light priority must never survive
        # into the universal collision guard and cause an early false stop.
        self._front_vehicle_guard_valid = False
        self._front_vehicle_guard_active = False
        self._front_vehicle_accel_cap = 2.5
        self._lane_edge_guard_active = False
        self._lane_edge_margin = 999.0
        self._lane_edge_accel_cap = 2.5
        self._lane_edge_steer_correction = 0.0
        self._last_env_action_applied = np.zeros(2, dtype=np.float64)
        self._traffic_mean_speed = 0.0
        self._traffic_moving_count = 0

        self.traffic_manager = None
        self.traffic_vehicles: List[Any] = []
        self.walker_actors: List[Any] = []
        self.walker_controllers: List[Any] = []
        self._lead_actor_kind = "none"

        self.barrier = AutonomousDrivingBarrierModel(dt=dt)
        self.rng = np.random.default_rng(seed)
        self._carla = None
        self.client = self.world = self.map = None
        self.ego = self.lead = self.collision_sensor = None
        self._connection_lost = False
        self._orig_settings = None
        self.t = 0
        self._collided = False
        self._collision_actor_type = "none"
        self._collision_zone = "none"
        self._collision_impulse = 0.0
        self._safety_exception_count = 0
        self._safety_exception_by_context: Dict[str, int] = {}
        self._last_safety_exception = ""
        self._traffic_light_detector_fault_active = False
        self._traffic_light_detector_fault_ticks = 0
        self._lead_speed_cmd = float(target_speed)
        self.progress_m = 0.0
        self._prev_loc = None
        self.destination_wp = None
        self.destination_loc = None
        self.route_waypoints: List[Any] = []
        self.route_tracker: Optional[RouteProgressTracker] = None
        self._route_lat_error = 0.0
        self._route_yaw_error = 0.0
        self._state_geometry_source = "map"
        self._map_lane_lat_error = 0.0
        self._map_lane_yaw_error = 0.0
        self._route_lookahead_yaw_error = 0.0
        self._route_turn_strength = 0.0
        self._route_curvature = 0.0
        self._route_turn_guard_active = False
        self._turn_recovery_active = False
        self._turn_stuck_count = 0
        self._predictive_conflict_raw = False
        self._predictive_conflict_active = False
        self._predictive_conflict_clear_ticks = 0
        self._predictive_conflict_actor_id = -1
        self._predictive_conflict_actor_type = "none"
        self._predictive_conflict_kind = "none"
        self._predictive_conflict_ttc = 999.0
        self._predictive_conflict_distance = 999.0
        self._predictive_conflict_accel_cap = 2.5
        self._predictive_conflict_guard_active = False
        self._predictive_conflict_guard_active_last_action = False
        self._predictive_conflict_accel_cap_last_action = 2.5
        self._predictive_junction_context = False
        self._external_blockage_actor_id = -1
        self._external_blockage_ticks = 0
        self._external_blockage_recovery_count = 0
        self._external_blockage_recovered_this_step = False
        self._external_blockage_last_reason = "none"
        self._safety_trace.clear()
        self._lateral_fail_count = 0
        self._heading_fail_count = 0
        self._headway_fail_count = 0
        self._obstacle_fail_count = 0
        self._last_steer = 0.0
        self._last_throttle = 0.0
        self._last_brake = 0.0
        self._prev_env_action = np.zeros(2, dtype=np.float64)
        self._lead_is_relevant = False
        self._lead_same_lane = False
        self._lead_route_lat = 999.0
        self._lead_route_ds = 999.0
        self._lead_actor_kind = "none"
        self._traffic_light_active = False
        self._traffic_light_state = "none"
        self._traffic_light_distance = 999.0
        self._traffic_light_virtual_gap = 999.0
        self._traffic_light_stop_error = 999.0
        self._traffic_light_accel_cap = 2.5
        self._traffic_light_id = -1
        self._traffic_light_yellow_go = False
        self._yellow_required_stop_distance = 0.0
        self._yellow_go_signal_keys = set()
        self._traffic_light_miss_ticks = 0
        self._traffic_light_detection_dropout = False
        self._traffic_light_last_signal_key = None
        self._traffic_light_last_stop_s = -1.0
        self._red_light_crossed_on_red = False
        self._red_light_violation_count = 0
        self._red_light_stop_success = False
        self._red_light_stop_success_count = 0
        self._red_light_last_success_key = None
        self._traffic_light_detector_fault_active = False
        self._traffic_light_detector_fault_ticks = 0
        self._tracked_light_id = None
        self._tracked_signal_key = None
        self._tracked_stop_s = -1.0
        self._tracked_stop_gap_prev = 999.0
        self._tracked_light_seen_ahead = False
        self._tracked_light_state = "none"
        self._tracked_yellow_go_latched = False
        self._committed_light_id = None
        self._committed_signal_key = None
        self._committed_stop_s = -1.0
        self._junction_commit_active = False
        self._junction_seen_since_commit = False
        self._junction_exit_clear_ticks = 0
        self._red_light_control_active = False
        self._red_light_front_gap = 80.0
        self._red_light_front_speed = 0.0
        self._red_light_queue_active = False
        self._red_light_queue_gap_error = 999.0
        self._front_vehicle_gap = 80.0
        self._front_vehicle_speed = 0.0
        self._front_vehicle_kind = "none"
        self._front_vehicle_ttc = 20.0
        # True only for an actor that is genuinely ahead in the ego lane/route
        # corridor. A raw actor rejected by red-light priority must never survive
        # into the universal collision guard and cause an early false stop.
        self._front_vehicle_guard_valid = False
        self._front_vehicle_guard_active = False
        self._front_vehicle_accel_cap = 2.5
        self._lane_edge_guard_active = False
        self._lane_edge_margin = 999.0
        self._lane_edge_accel_cap = 2.5
        self._lane_edge_steer_correction = 0.0
        self._last_env_action_applied = np.zeros(2, dtype=np.float64)
        self._traffic_mean_speed = 0.0
        self._traffic_moving_count = 0

    # -- connection / world ---------------------------------------------------
    def _ensure_carla(self):
        if self._carla is None:
            try:
                import carla  # noqa
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Could not import the `carla` module. Install the matching "
                    "PythonAPI for your server, e.g.:\n"
                    "  pip install carla==0.9.15\n"
                    "or add the egg from <CARLA>/PythonAPI/carla/dist to PYTHONPATH."
                ) from e
            self._carla = carla
        return self._carla

    @staticmethod
    def _short_map_name(name: Any) -> str:
        """Return TownXX/TownXXHD from a CARLA asset path or plain town name."""
        text = str(name or "").replace("\\", "/").rstrip("/")
        return text.split("/")[-1] if text else ""

    @classmethod
    def _canonical_map_name(cls, name: Any) -> str:
        """Compare normal and optimized assets as the same CARLA town.

        Town10HD and Town10HD_Opt share the same road network.  Treating them
        as different forced a needless native load_world() transition, which
        is both expensive and fragile in the packaged CARLA 0.9.15 client.
        """
        short = cls._short_map_name(name).strip().lower()
        return short[:-4] if short.endswith("_opt") else short

    @staticmethod
    def _version_tuple(text: Any) -> Tuple[int, ...]:
        """Best-effort numeric version parser, e.g. '0.9.15' -> (0, 9, 15)."""
        nums: List[int] = []
        token = ""
        for ch in str(text or ""):
            if ch.isdigit():
                token += ch
            elif token:
                nums.append(int(token)); token = ""
                if len(nums) >= 3:
                    break
        if token and len(nums) < 3:
            nums.append(int(token))
        return tuple(nums[:3])

    @staticmethod
    def _root_cause_is_import_error(exc: Exception) -> bool:
        """True when the exception chain bottoms out in a missing module.

        A missing Python package is a property of the interpreter environment,
        not of the episode: respawning actors and retrying cannot ever fix it,
        so reset must fail fast with an actionable message instead of burning
        eight full spawn/teardown cycles repeating the identical error.
        """
        seen = set()
        node: Optional[BaseException] = exc
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            if isinstance(node, ImportError):
                return True
            if "ModuleNotFoundError" in str(node) or "ImportError" in str(node):
                return True
            node = node.__cause__ or node.__context__
        return False

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """Classify errors that mean the CARLA server is unreachable/stalled.

        libcarla surfaces these as RuntimeError with a 'time-out of XXXXms'
        message (RPC timeout), or as connection-refused/reset style OSErrors
        after a server crash.  Once one RPC times out, every further RPC will
        stall for the full client timeout too, so callers must stop hammering
        the server and go through reconnect() instead.
        """
        msg = str(exc).lower()
        return any(tok in msg for tok in (
            "time-out", "timeout", "timed out",
            "connection refused", "connection reset",
            "not connected", "connection lost",
        ))

    def reconnect(self) -> bool:
        """Rebuild client/world/TM after a server stall or server restart.

        Drops every reference into the dead session WITHOUT issuing RPCs
        against it (each RPC would stall for the full client timeout), then
        re-runs the normal map-switch-safe _connect().  Ghost actors from the
        aborted episode are cleaned by _destroy_stale_actors() inside the next
        reset(), which runs against the fresh connection.
        """
        print(f"[carla-safe] reconnecting to {self.host}:{self.port} ...", flush=True)
        self.ego = None
        self.lead = None
        self.collision_sensor = None
        self.client = None
        self.world = None
        self.map = None
        self._grp_cache = None
        self.route_tracker = None
        self.route_waypoints = []
        try:
            self._connect()
        except Exception as exc:
            self._record_safety_exception("reconnect", exc)
            self._connection_lost = True
            print(f"[carla-safe] reconnect FAILED: {type(exc).__name__}: {exc}",
                  flush=True)
            return False
        self._connection_lost = False
        print("[carla-safe] reconnect successful", flush=True)
        return True

    def _connect(self):
        """Connect to CARLA with a map-switch-safe world/TM lifecycle.

        Important ordering:
          1) connect and inspect the current world,
          2) disable Traffic Manager synchronous mode,
          3) force the current world asynchronous,
          4) load the requested map only after that cleanup,
          5) configure the new world synchronous, then synchronize TM.

        This avoids changing maps while a previous synchronous world/TM pair is
        still active. It also detects the CARLA 0.9.15 pedestrian-navigation
        map-switch hazard and suppresses cross-town walkers conservatively.
        """
        carla = self._ensure_carla()
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout_s)

        try:
            self._server_version = str(self.client.get_server_version())
        except Exception:
            self._server_version = "unknown"
        try:
            client_version = str(self.client.get_client_version())
        except Exception:
            client_version = "unknown"
        module_path = str(getattr(carla, "__file__", "unknown"))
        print(
            f"[carla-safe] connected: client={client_version} "
            f"server={self._server_version} carla_module={module_path}",
            flush=True,
        )
        if (
            client_version != "unknown"
            and self._server_version != "unknown"
            and self._version_tuple(client_version)
                != self._version_tuple(self._server_version)
        ):
            print(
                "[carla-safe][WARN] CARLA client/server VERSION MISMATCH "
                f"({client_version} vs {self._server_version}). This is a known "
                "cause of client-side segfaults and protocol failures. Use the "
                "PythonAPI (egg or wheel) that matches the running server build.",
                flush=True,
            )

        # First obtain the currently loaded world. Before any load_world() call,
        # turn off TM sync and world sync so a stale previous run cannot leave
        # the server in a fragile synchronous state during the map transition.
        current_world = self.client.get_world()
        try:
            current_map_name = self._short_map_name(current_world.get_map().name)
        except Exception:
            current_map_name = ""
        self._startup_map_name = current_map_name

        try:
            pre_tm = self.client.get_trafficmanager(self.tm_port)
            pre_tm.set_synchronous_mode(False)
        except Exception:
            pass

        try:
            current_settings = current_world.get_settings()
            current_settings.synchronous_mode = False
            current_settings.fixed_delta_seconds = None
            current_world.apply_settings(current_settings)
        except Exception:
            pass

        target_map_name = self._short_map_name(self.town)
        if (
            not self.load_town
            and current_map_name
            and target_map_name
            and self._canonical_map_name(current_map_name)
                != self._canonical_map_name(target_map_name)
        ):
            raise RuntimeError(
                "CARLA map mismatch with load_town disabled: "
                f"server={current_map_name}, requested={target_map_name}. "
                "Start/switch CARLA to the requested town before evaluation."
            )
        self._map_changed_on_connect = bool(
            self.load_town and current_map_name and target_map_name
            and self._canonical_map_name(current_map_name)
                != self._canonical_map_name(target_map_name)
        )

        if self.load_town and self._map_changed_on_connect:
            self.world = self.client.load_world(self.town, True)
        elif self.load_town and not current_map_name:
            self.world = self.client.load_world(self.town, True)
        else:
            self.world = current_world

        self.map = self.world.get_map()
        loaded_map_name = self._short_map_name(getattr(self.map, "name", self.town))
        if (
            target_map_name
            and loaded_map_name
            and self._canonical_map_name(loaded_map_name)
                != self._canonical_map_name(target_map_name)
        ):
            raise RuntimeError(
                "CARLA loaded an unexpected map: "
                f"requested={target_map_name}, loaded={loaded_map_name}."
            )
        self._connection_lost = False

        # CARLA 0.9.15 officially documented retaining the previous town's
        # pedestrian navigation data after a map switch. If the server version
        # is 0.9.15 or older (or unavailable), disable walkers only for that
        # cross-town environment. Vehicles remain enabled.
        version = self._version_tuple(self._server_version)
        vulnerable_or_unknown = (not version) or (version <= (0, 9, 15))
        self._effective_num_walkers = int(self.num_walkers)
        if (self.protect_cross_town_walkers and self._map_changed_on_connect
                and vulnerable_or_unknown and self.num_walkers > 0):
            self._effective_num_walkers = 0
            print(
                f"[carla-safe] server={self._server_version} map_switch="
                f"{current_map_name or 'unknown'}->{loaded_map_name or target_map_name}; "
                f"suppressing {self.num_walkers} walkers to avoid the known "
                f"cross-town pedestrian-navigation hazard. Vehicles remain enabled.",
                flush=True,
            )

        # New world starts from a deterministic async baseline. Destroy stale
        # actors in this new world before enabling synchronous stepping.
        try:
            s0 = self.world.get_settings()
            s0.synchronous_mode = False
            s0.fixed_delta_seconds = None
            self.world.apply_settings(s0)
        except Exception:
            pass
        self._destroy_stale_actors()

        # Synchronous, fixed-step simulation.
        self._orig_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        settings.no_rendering_mode = not self.render_follow

        # Large-map-only actor activation expansion. Avoid carrying Town10HD's
        # large active distance into smaller held-out maps.
        large_map_names = {
            "town10hd", "town10hd_opt", "town11", "town12", "town13", "town15"
        }
        is_large_map = loaded_map_name.lower() in large_map_names
        if is_large_map:
            try:
                settings.actor_active_distance = max(
                    float(getattr(settings, "actor_active_distance", 0.0)), 2000.0
                )
            except Exception:
                pass
        self.world.apply_settings(settings)

        # Traffic Manager must be synchronized only after the new world has been
        # placed in synchronous mode.
        #
        # IMPORTANT ROBUSTNESS RULE:
        # Getting the TM handle is the only operation whose failure should set
        # self.traffic_manager = None. Individual optional/configuration calls are
        # isolated so one unsupported API method cannot disable the whole Traffic
        # Manager and silently skip all later per-vehicle TM configuration.
        try:
            self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        except Exception as exc:
            self.traffic_manager = None
            print(
                f"[carla-safe] Could not obtain Traffic Manager on port "
                f"{self.tm_port}: {exc}",
                flush=True,
            )

        if self.traffic_manager is not None:
            # World and TM synchronous modes must be coordinated.
            try:
                self.traffic_manager.set_synchronous_mode(True)
            except Exception as exc:
                print(
                    f"[carla-safe] TM synchronous-mode warning: {exc}",
                    flush=True,
                )

            try:
                self.traffic_manager.set_random_device_seed(self.seed)
            except Exception as exc:
                print(
                    f"[carla-safe] TM random-seed warning: {exc}",
                    flush=True,
                )

            # CARLA's Python API name is:
            #     set_global_distance_to_leading_vehicle(distance)
            # Do not use the non-existent:
            #     global_distance_to_leading_vehicle(...)
            try:
                self.traffic_manager.set_global_distance_to_leading_vehicle(
                    max(
                        self.traffic_min_distance,
                        self.traffic_rear_safety_distance,
                    )
                )
            except Exception as exc:
                print(
                    f"[carla-safe] TM global-distance warning: {exc}",
                    flush=True,
                )

            try:
                self.traffic_manager.global_percentage_speed_difference(
                    self.traffic_speed_difference
                )
            except Exception as exc:
                print(
                    f"[carla-safe] TM global-speed-difference warning: {exc}",
                    flush=True,
                )

            # Dormant-vehicle respawning is intended for large maps/streamed
            # tiles. Explicitly turn it off for Town02/Town05 and other small maps.
            try:
                self.traffic_manager.set_respawn_dormant_vehicles(bool(is_large_map))
                if is_large_map:
                    self.traffic_manager.set_boundaries_respawn_dormant_vehicles(
                        25.0, 700.0
                    )
            except Exception as exc:
                print(
                    f"[carla-safe] TM dormant-respawn warning: {exc}",
                    flush=True,
                )

            try:
                self.traffic_manager.set_hybrid_physics_mode(
                    bool(self.tm_hybrid_physics)
                )
                if self.tm_hybrid_physics:
                    self.traffic_manager.set_hybrid_physics_radius(200.0)
            except Exception as exc:
                print(
                    f"[carla-safe] TM hybrid-physics warning: {exc}",
                    flush=True,
                )

        self._register_cleanup()

    def _actor_is_alive(self, actor: Any) -> bool:
        """Return True only if the handle still points to a live CARLA actor.

        CARLA actor handles can outlive the simulator actor. Calling destroy()
        on such stale handles prints warnings such as
        "attempting to destroy an actor that is already dead".  We verify both
        actor.is_alive and world.get_actor(actor.id) before sending DestroyActor.
        """
        try:
            if actor is None:
                return False
            if not bool(getattr(actor, "is_alive", False)):
                return False
            if self.world is not None and self.world.get_actor(int(actor.id)) is None:
                return False
            return True
        except Exception:
            return False

    def _safe_stop_walker_controller(self, controller: Any) -> None:
        """Stop a live walker controller without raising on stale handles."""
        try:
            if self._actor_is_alive(controller):
                controller.stop()
        except Exception:
            pass

    def _alive_actor_ids(self, actors: Sequence[Any]) -> List[int]:
        """Unique live actor IDs, preserving order."""
        ids: List[int] = []
        seen = set()
        for actor in actors:
            try:
                if self._actor_is_alive(actor):
                    aid = int(actor.id)
                    if aid not in seen:
                        ids.append(aid)
                        seen.add(aid)
            except Exception:
                pass
        return ids
        
    def _snapshot_alive_ids(
        self,
        actor_ids: Sequence[int],
    ) -> List[int]:
        """Return actor IDs confirmed alive in the latest CARLA world snapshot.

        A post-tick WorldSnapshot is used as the primary authority because actor
        handles can outlive the corresponding simulator actor. If snapshot access
        fails, fall back to World.get_actors(ids), which excludes nonexistent IDs.

        If both verification mechanisms fail, return all queried IDs so teardown
        fails closed rather than spawning a new ego into an unknown world state.
        """

        ids: List[int] = []
        seen = set()

        for raw_id in actor_ids:
            try:
                aid = int(raw_id)
            except Exception:
                continue

            if aid not in seen:
                ids.append(aid)
                seen.add(aid)

        if not ids or self.world is None:
            return []

        # Primary verification: latest simulation snapshot.
        try:
            snapshot = self.world.get_snapshot()
            return [
                aid
                for aid in ids
                if bool(snapshot.has_actor(int(aid)))
            ]

        except Exception as exc:
            self._record_safety_exception(
                "actor_snapshot_verify",
                exc,
            )

        # Secondary authoritative registry query.
        try:
            actors = self.world.get_actors(ids)

            return sorted({
                int(actor.id)
                for actor in actors
            })

        except Exception as exc:
            self._record_safety_exception(
                "actor_registry_verify",
                exc,
            )

        # Fail closed: unknown registry state.
        return ids
    def _destroy_ids_batch(
        self,
        actor_ids: Sequence[int],
        do_tick: bool = True,
    ) -> List[int]:
        """Destroy actors without treating already-absent actors as survivors.

        Important CARLA behavior:
          * DestroyActor may report "not found" if an actor has already disappeared.
          * That is an idempotent-cleanup success, not evidence of a surviving actor.
          * Final survival verification uses a fresh post-tick WorldSnapshot.

        Returns IDs that remain genuinely alive when verification is authoritative.
        """

        ids: List[int] = []
        seen = set()

        for raw_id in actor_ids:
            try:
                aid = int(raw_id)
            except Exception:
                continue

            if aid in seen:
                continue

            seen.add(aid)
            ids.append(aid)

        if not ids:
            return []

        if getattr(self, "_connection_lost", False):
            # Every RPC against a dead server stalls for the full client
            # timeout.  Skip destruction; the next reconnect() + reset() sweeps
            # ghosts via _destroy_stale_actors() on the fresh connection.
            return []

        failed = set(ids)

        try:
            if self._carla is None:
                raise RuntimeError(
                    "CARLA module unavailable during actor destruction."
                )

            if getattr(self, "client", None) is None:
                raise RuntimeError(
                    "CARLA client unavailable during actor destruction."
                )

            commands = [
                self._carla.command.DestroyActor(aid)
                for aid in ids
            ]

            # Same sync-mode rule as the spawn batch: never let batch_sync
            # issue the server-internal tick cue while world+TM are
            # synchronous; tick explicitly from this client instead.
            responses = self.client.apply_batch_sync(
                commands,
                False,
            )
            if bool(do_tick):
                self.world.tick()

            for index, aid in enumerate(ids):
                if index >= len(responses):
                    continue

                response = responses[index]

                try:
                    has_error = bool(response.has_error())
                except Exception:
                    has_error = bool(
                        getattr(response, "error", "")
                    )

                error_text = str(
                    getattr(response, "error", "") or ""
                ).strip()

                error_lower = error_text.lower()

                if not has_error:
                    failed.discard(aid)
                    continue

                # Actor is already absent. This is successful idempotent cleanup.
                already_absent = any(
                    marker in error_lower
                    for marker in (
                        "not found",
                        "already destroyed",
                        "already dead",
                        "does not exist",
                    )
                )

                if already_absent:
                    failed.discard(aid)
                    continue

                print(
                    f"[carla-safe] DestroyActor failed id={aid}: "
                    f"{error_text or 'unknown error'}",
                    flush=True,
                )

        except Exception as exc:
            if self._is_connection_error(exc):
                # Server unreachable: mark it so subsequent destroy/tick paths
                # short-circuit instead of stacking more full-timeout stalls
                # (the observed failure spent minutes re-timing-out destroys
                # against a hung server before finally dying in step()).
                self._connection_lost = True
            self._record_safety_exception(
                "destroy_batch",
                exc,
            )

            failed = set(ids)

        # Retry only genuinely failed entries.
        for aid in list(failed):
            try:
                # Use snapshot first. If actor is already absent, cleanup succeeded.
                if aid not in self._snapshot_alive_ids([aid]):
                    failed.discard(aid)
                    continue

                actor = (
                    self.world.get_actor(aid)
                    if self.world is not None
                    else None
                )

                if actor is None:
                    failed.discard(aid)
                    continue

                try:
                    destroyed = bool(actor.destroy())
                except RuntimeError as exc:
                    text = str(exc).lower()

                    if (
                        "not found" in text
                        or "already destroyed" in text
                        or "already dead" in text
                    ):
                        failed.discard(aid)
                        continue

                    raise

                if destroyed:
                    failed.discard(aid)

            except Exception as exc:
                self._record_safety_exception(
                    f"destroy_actor_{aid}",
                    exc,
                )

        if do_tick:
            try:
                if (
                    self.world is not None
                    and bool(
                        self.world.get_settings().synchronous_mode
                    )
                ):
                    self.world.tick()

            except Exception as exc:
                self._record_safety_exception(
                    "destroy_verify_tick",
                    exc,
                )

            # After a controlled tick, verify actual existence.
            return self._snapshot_alive_ids(ids)

        # Without a tick, return only commands that genuinely failed.
        # The caller is responsible for ticking and final verification.
        return sorted(failed)

    def _destroy_stale_actors(self) -> List[int]:
        """Destroy stale experiment-owned actors and return real survivors.

        This function uses the current world snapshot to reject stale Python actor
        handles before issuing DestroyActor commands.

        Ownership-safe mode removes:
          * eclcs_ego
          * eclcs_lead
          * eclcs_traffic
          * eclcs_walker
          * legacy Tesla Model 3 ego with role_name='hero'
          * attached sensors/controllers belonging to owned actors
        """

        if self.world is None:
            return []

        try:
            actors = list(self.world.get_actors())
            snapshot = self.world.get_snapshot()

        except Exception as exc:
            self._record_safety_exception(
                "stale_actor_scan",
                exc,
            )

            raise RuntimeError(
                "Cannot verify stale CARLA actors; refusing to continue reset."
            ) from exc

        # Remove stale handles that are not actually present in the latest snapshot.
        live_actors: List[Any] = []

        for actor in actors:
            try:
                aid = int(actor.id)

                if bool(snapshot.has_actor(aid)):
                    live_actors.append(actor)

            except Exception:
                continue

        controllers: List[Any] = []
        others: List[Any] = []

        if bool(
            getattr(
                self,
                "destroy_all_stale_actors",
                False,
            )
        ):
            for actor in live_actors:
                try:
                    type_id = str(actor.type_id)

                    if type_id.startswith(
                        "controller.ai.walker"
                    ):
                        controllers.append(actor)

                    elif (
                        type_id.startswith("vehicle.")
                        or type_id.startswith("sensor.")
                        or type_id.startswith("walker.")
                    ):
                        others.append(actor)

                except Exception:
                    continue

        else:
            owned_roles = {
                "eclcs_ego",
                "eclcs_lead",
                "eclcs_traffic",
                "eclcs_walker",
            }

            owned_ids = set()

            for actor in live_actors:
                try:
                    attrs = (
                        getattr(actor, "attributes", {})
                        or {}
                    )

                    role = str(
                        attrs.get("role_name", "")
                    )

                    type_id = str(actor.type_id)

                    owned = role in owned_roles

                    # Backward compatibility with older revisions.
                    legacy_ego = bool(
                        role == "hero"
                        and type_id
                        == "vehicle.tesla.model3"
                    )

                    if owned or legacy_ego:
                        owned_ids.add(int(actor.id))

                except Exception:
                    continue

            # Include descendants such as sensors and walker controllers.
            changed = True

            while changed:
                changed = False

                for actor in live_actors:
                    try:
                        parent = getattr(
                            actor,
                            "parent",
                            None,
                        )

                        if (
                            parent is not None
                            and int(parent.id) in owned_ids
                        ):
                            aid = int(actor.id)

                            if aid not in owned_ids:
                                owned_ids.add(aid)
                                changed = True

                    except Exception:
                        continue

            for actor in live_actors:
                try:
                    if int(actor.id) not in owned_ids:
                        continue

                    if str(actor.type_id).startswith(
                        "controller.ai.walker"
                    ):
                        controllers.append(actor)
                    else:
                        others.append(actor)

                except Exception:
                    continue

        for controller in controllers:
            self._safe_stop_walker_controller(
                controller
            )

        ids = self._alive_actor_ids(
            controllers + others
        )

        if not ids:
            return []

        self._destroy_ids_batch(
            ids,
            do_tick=False,
        )

        # One fresh frame makes snapshot verification authoritative.
        try:
            if bool(
                self.world
                .get_settings()
                .synchronous_mode
            ):
                self.world.tick()

        except Exception as exc:
            self._record_safety_exception(
                "stale_cleanup_tick",
                exc,
            )

        survivors = self._snapshot_alive_ids(ids)

        # One conservative retry for genuine survivors only.
        if survivors:
            self._destroy_ids_batch(
                survivors,
                do_tick=False,
            )

            try:
                if bool(
                    self.world
                    .get_settings()
                    .synchronous_mode
                ):
                    self.world.tick()

            except Exception as exc:
                self._record_safety_exception(
                    "stale_cleanup_retry_tick",
                    exc,
                )

            survivors = self._snapshot_alive_ids(
                survivors
            )

        return survivors

    def _register_cleanup(self):
        if getattr(self, "_cleanup_registered", False):
            return
        import atexit
        atexit.register(self._restore_async_only)
        self._cleanup_registered = True

    def _restore_async_only(self):
        """Best-effort: put the server/TM back in async mode (no exceptions)."""
        try:
            if getattr(self, "traffic_manager", None) is not None:
                self.traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            if self.world is not None:
                s = self.world.get_settings()
                s.synchronous_mode = False
                s.fixed_delta_seconds = None
                self.world.apply_settings(s)
        except Exception:
            pass

    def _destroy_actors(self) -> None:
        """Safely destroy all actors belonging to the current episode.

        Key properties:
          * stops sensor callbacks only when listening;
          * stops walker controllers before destruction;
          * disables TM autopilot before vehicle destruction;
          * destroys all current-episode actors in one batch;
          * performs one synchronization tick;
          * verifies survival with WorldSnapshot.has_actor();
          * retries only genuine survivors;
          * does NOT redundantly call _destroy_stale_actors() internally because
            reset() already does that immediately afterward.
        """
        if getattr(self, "_connection_lost", False):
            # The server is unreachable: drop local references without RPCs.
            # reconnect() + the next reset()'s stale-actor sweep will clean the
            # server side once it is reachable again.
            self.walker_controllers = []
            self.walker_actors = []
            self.traffic_vehicles = []
            self.collision_sensor = None
            self.lead = None
            self.ego = None
            return


        # --------------------------------------------------------------
        # Stop collision sensor callback.
        # --------------------------------------------------------------
        try:
            if self._actor_is_alive(
                self.collision_sensor
            ):
                is_listening = bool(
                    getattr(
                        self.collision_sensor,
                        "is_listening",
                        True,
                    )
                )

                if is_listening:
                    self.collision_sensor.stop()

        except Exception as exc:
            self._record_safety_exception(
                "collision_sensor_stop",
                exc,
            )

        controllers = list(
            getattr(
                self,
                "walker_controllers",
                [],
            )
        )

        walkers = list(
            getattr(
                self,
                "walker_actors",
                [],
            )
        )

        traffic = list(
            getattr(
                self,
                "traffic_vehicles",
                [],
            )
        )

        tail = [
            self.collision_sensor,
            self.lead,
            self.ego,
        ]

        # --------------------------------------------------------------
        # Stop walker AI first.
        # --------------------------------------------------------------
        for controller in controllers:
            self._safe_stop_walker_controller(
                controller
            )

        # --------------------------------------------------------------
        # Disable TM autopilot before destroying vehicles.
        # --------------------------------------------------------------
        for vehicle in traffic:
            try:
                if self._actor_is_alive(vehicle):
                    try:
                        vehicle.set_autopilot(
                            False,
                            self.tm_port,
                        )
                    except TypeError:
                        vehicle.set_autopilot(False)

            except Exception as exc:
                self._record_safety_exception(
                    "disable_traffic_autopilot",
                    exc,
                )

        actor_ids = self._alive_actor_ids(
            controllers
            + walkers
            + traffic
            + tail
        )

        if actor_ids:
            self._destroy_ids_batch(
                actor_ids,
                do_tick=False,
            )

        # --------------------------------------------------------------
        # Advance exactly one frame before authoritative verification.
        # --------------------------------------------------------------
        try:
            if (
                self.world is not None
                and bool(
                    self.world
                    .get_settings()
                    .synchronous_mode
                )
            ):
                self.world.tick()

        except Exception as exc:
            self._record_safety_exception(
                "episode_destroy_tick",
                exc,
            )

        remaining = self._snapshot_alive_ids(
            actor_ids
        )

        # --------------------------------------------------------------
        # One retry for genuine snapshot-confirmed survivors.
        # --------------------------------------------------------------
        if remaining:
            self._destroy_ids_batch(
                remaining,
                do_tick=False,
            )

            try:
                if (
                    self.world is not None
                    and bool(
                        self.world
                        .get_settings()
                        .synchronous_mode
                    )
                ):
                    self.world.tick()

            except Exception as exc:
                self._record_safety_exception(
                    "episode_destroy_retry_tick",
                    exc,
                )

            remaining = self._snapshot_alive_ids(
                remaining
            )

        # Optional additional settle ticks.
        for _ in range(
            max(
                0,
                int(
                    getattr(
                        self,
                        "episode_reset_settle_ticks",
                        1,
                    )
                )
                - 1,
            )
        ):
            try:
                if (
                    self.world is not None
                    and bool(
                        self.world
                        .get_settings()
                        .synchronous_mode
                    )
                ):
                    self.world.tick()

            except Exception:
                break

        if remaining:
            raise RuntimeError(
                "Episode teardown failed; "
                "owned CARLA actors are still alive after "
                "snapshot verification and retry: "
                f"{remaining}. Refusing to spawn a new ego."
            )

        # Only clear handles after server-side absence is confirmed.
        self.collision_sensor = None
        self.lead = None
        self.ego = None

        self.traffic_vehicles = []
        self.walker_actors = []
        self.walker_controllers = []

        self._traffic_mean_speed = 0.0
        self._traffic_moving_count = 0


    def _route_query_for_location(self, loc: Any, center_idx: int = 0,
                                  forward_extra: int = 40) -> Tuple[float, float, float, int]:
        """Non-mutating projection helper for spawn filtering and hazard sensing."""
        if self.route_tracker is None:
            return 0.0, 999.0, 0.0, 0
        fwd = max(self.route_search_forward + forward_extra, len(self.route_waypoints) + 5)
        return self.route_tracker.query(loc, center_idx=center_idx, back=0, forward=fwd)

    def _spawn_background_vehicles(self) -> None:
        """Spawn Traffic-Manager-controlled vehicles near the ego route.

        Robust CARLA/Town10HD version:
        * uses apply_batch_sync with SetAutopilot(FutureActor, True, tm_port),
          matching CARLA's recommended Traffic Manager registration pattern;
        * keeps vehicles on drivable spawn points near the hero/ego route;
        * disables hybrid physics by default so nearby NPCs are simulated rather
          than visually present but dormant;
        * warms the Traffic Manager for several ticks while holding the ego, so
          NPCs start with non-zero motion before the RL episode begins.
        """
        if self.num_traffic_vehicles <= 0 or self.world is None:
            return
        carla = self._carla
        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = []
        for bp in list(bp_lib.filter("vehicle.*")):
            try:
                if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) != 4:
                    continue
                # Avoid very large/special vehicles that often block routes,
                # matching CARLA's own generate_traffic --safe exclusions.
                tid = str(bp.id).lower()
                if any(x in tid for x in ("carlamotors", "firetruck", "ambulance",
                                          "cybertruck", "trailer", "microlino",
                                          "t2", "sprinter", "fusorosa")):
                    continue
            except Exception:
                pass
            vehicle_bps.append(bp)
        if not vehicle_bps:
            return

        ego_loc = self.ego.get_location() if self.ego is not None else None
        lead_loc = self.lead.get_location() if self.lead is not None else None
        spawn_points = list(self.map.get_spawn_points())
        self.rng.shuffle(spawn_points)

        candidates = []
        used_xy = []
        for sp in spawn_points:
            if ego_loc is not None:
                d_ego = sp.location.distance(ego_loc)
                if d_ego < 18.0:
                    continue
                if self.traffic_radius > 0.0 and d_ego > self.traffic_radius:
                    continue
            if lead_loc is not None and sp.location.distance(lead_loc) < 12.0:
                continue
            # Avoid spawning multiple NPCs on nearly identical spawn points.
            too_close = False
            for ux, uy in used_xy:
                dx = float(sp.location.x) - ux
                dy = float(sp.location.y) - uy
                if dx * dx + dy * dy < 12.0 * 12.0:
                    too_close = True
                    break
            if too_close:
                continue
            if self.route_tracker is not None:
                s, lat, _, _ = self._route_query_for_location(sp.location, forward_extra=220)
                # Put traffic around and ahead of the route, but not exactly on
                # top of the ego. Town10HD has many sparse spawn points, so the
                # lateral corridor is intentionally wider than the ego lane.
                if not (15.0 <= s <= self.route_distance + 220.0 and abs(lat) <= 28.0):
                    continue
            candidates.append(sp)
            used_xy.append((float(sp.location.x), float(sp.location.y)))
            if len(candidates) >= max(self.num_traffic_vehicles * 3, self.num_traffic_vehicles):
                break

        # Fallback: if the route corridor is too sparse, use nearby map spawns.
        if len(candidates) < self.num_traffic_vehicles:
            for sp in spawn_points:
                if sp in candidates:
                    continue
                if ego_loc is not None:
                    d_ego = sp.location.distance(ego_loc)
                    if d_ego < 25.0:
                        continue
                    if self.traffic_radius > 0.0 and d_ego > self.traffic_radius:
                        continue
                candidates.append(sp)
                if len(candidates) >= self.num_traffic_vehicles:
                    break

        if not candidates:
            return

        tm_port = self.traffic_manager.get_port() if self.traffic_manager is not None else self.tm_port
        batch = []
        for sp in candidates[: self.num_traffic_vehicles]:
            bp = self.rng.choice(vehicle_bps)
            try:
                if bp.has_attribute("color"):
                    bp.set_attribute("color", self.rng.choice(bp.get_attribute("color").recommended_values))
                if bp.has_attribute("role_name"):
                    bp.set_attribute("role_name", "eclcs_traffic")
            except Exception:
                pass
            # Batch spawn + SetAutopilot(FutureActor, True, tm_port) is more
            # reliable than spawning first and calling set_autopilot later.
            batch.append(carla.command.SpawnActor(bp, sp).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)))

        try:
            # CRITICAL sync-mode rule: apply_batch_sync(batch, True) makes the
            # SERVER issue an internal tick cue.  With both the world and the
            # Traffic Manager in synchronous mode, TM's tick barrier expects
            # the tick to come from this client via world.tick(); the internal
            # cue can deadlock the server's game thread.  The wedge is
            # timing-dependent, which is why it reproduced deterministically
            # at the same reset on Town02 while never firing on Town10HD.
            responses = self.client.apply_batch_sync(batch, False)
            self.world.tick()
        except Exception as exc:
            if self._is_connection_error(exc):
                self._connection_lost = True
                self._record_safety_exception("traffic_spawn_connection", exc)
                raise RuntimeError(
                    f"carla_connection_lost during traffic spawn: {exc}"
                ) from exc
            self._record_safety_exception("traffic_spawn_batch", exc)
            responses = []

        actor_ids = []
        for resp in responses:
            try:
                if not resp.error:
                    actor_ids.append(resp.actor_id)
            except Exception:
                pass
        if actor_ids:
            actors = self.world.get_actors(actor_ids)
            self.traffic_vehicles = [a for a in actors if a is not None]

        # Per-vehicle Traffic Manager configuration and physics release.
        for actor in list(self.traffic_vehicles):
            try:
                actor.set_simulate_physics(True)
            except Exception:
                pass
            try:
                actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0,
                                                        hand_brake=False, manual_gear_shift=False))
            except Exception:
                pass
            if self.traffic_manager is None:
                continue
            try:
                actor.set_autopilot(True, tm_port)
                self.traffic_manager.distance_to_leading_vehicle(
                    actor, max(self.traffic_min_distance, self.traffic_rear_safety_distance)
                )
                jitter = float(self.rng.uniform(-5.0, 5.0))
                self.traffic_manager.vehicle_percentage_speed_difference(
                    actor, self.traffic_speed_difference + jitter)
                self.traffic_manager.auto_lane_change(actor, self.traffic_auto_lane_change)
                self.traffic_manager.ignore_lights_percentage(actor, 0.0)
                self.traffic_manager.ignore_vehicles_percentage(actor, 0.0)
                self.traffic_manager.ignore_walkers_percentage(actor, 0.0)
                try:
                    self.traffic_manager.update_vehicle_lights(actor, True)
                except Exception:
                    pass
            except Exception:
                pass

        # Let TM build path buffers and issue first controls. Hold ego stationary
        # during this warmup; otherwise the first episode often begins before TM
        # actors have moved, making them look frozen.
        try:
            hold = carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False,
                                        manual_gear_shift=False)
            for _ in range(int(self.traffic_warmup_ticks)):
                if self.ego is not None and self.ego.is_alive:
                    self.ego.apply_control(hold)
                self._drive_lead()
                self.world.tick()
        except Exception as exc:
            # A hung server used to be swallowed here, letting the reset limp
            # through several more full-timeout RPC stalls before failing far
            # away from the true wedge point.  Fail fast and marked instead.
            if self._is_connection_error(exc):
                self._connection_lost = True
                self._record_safety_exception("traffic_warmup_connection", exc)
                raise RuntimeError(
                    f"carla_connection_lost during traffic warmup: {exc}"
                ) from exc
            self._record_safety_exception("traffic_warmup", exc)


    def _spawn_walkers(self) -> None:
        """Spawn owned AI pedestrians near the route corridor."""
        n_walkers = int(getattr(self, "_effective_num_walkers", self.num_walkers))
        if n_walkers <= 0 or self.world is None:
            return
        carla = self._carla
        bp_lib = self.world.get_blueprint_library()
        walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
        if not walker_bps:
            return

        spawn_points = []
        attempts = max(80, n_walkers * 20)
        for _ in range(attempts):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            if self.route_tracker is not None:
                s, lat, _, _ = self._route_query_for_location(loc)
                if not (30.0 <= s <= self.route_distance + 80.0 and abs(lat) <= 18.0):
                    continue
            spawn_points.append(carla.Transform(loc))
            if len(spawn_points) >= n_walkers:
                break

        while len(spawn_points) < n_walkers:
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                break
            spawn_points.append(carla.Transform(loc))

        for sp in spawn_points[:n_walkers]:
            bp = self.rng.choice(walker_bps)
            try:
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                if bp.has_attribute("speed"):
                    bp.set_attribute("speed", str(self.walker_speed_max))
                if bp.has_attribute("role_name"):
                    bp.set_attribute("role_name", "eclcs_walker")
            except Exception:
                pass
            try:
                actor = self.world.try_spawn_actor(bp, sp)
            except Exception as exc:
                self._record_safety_exception("walker_spawn", exc)
                actor = None
            if actor is not None:
                self.walker_actors.append(actor)

        if not self.walker_actors:
            return
        controller_bp = bp_lib.find("controller.ai.walker")
        for walker in list(self.walker_actors):
            try:
                con = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                self.walker_controllers.append(con)
            except Exception as exc:
                self._record_safety_exception("walker_controller_spawn", exc)

        try:
            self.world.tick()
        except Exception as exc:
            self._record_safety_exception("walker_spawn_tick", exc)

        live_controllers = []
        for con in list(self.walker_controllers):
            try:
                if not self._actor_is_alive(con):
                    continue
                con.start()
                dest = self.world.get_random_location_from_navigation()
                if dest is not None:
                    con.go_to_location(dest)
                spd = float(self.rng.uniform(self.walker_speed_min, self.walker_speed_max))
                con.set_max_speed(spd)
                live_controllers.append(con)
            except Exception as exc:
                self._record_safety_exception("walker_controller_start", exc)
        self.walker_controllers = live_controllers
        self.walker_actors = [
            w for w in list(self.walker_actors) if self._actor_is_alive(w)
        ]


    def _on_collision(self, event: Any) -> None:
        """Record collision counterpart and whether it came from front/rear/side."""
        self._collided = True
        other = None
        try:
            other = getattr(event, "other_actor", None)
            self._collision_actor_type = str(getattr(other, "type_id", "unknown"))
        except Exception:
            self._collision_actor_type = "unknown"

        try:
            imp = getattr(event, "normal_impulse", None)
            self._collision_impulse = float(math.sqrt(
                float(imp.x) ** 2 + float(imp.y) ** 2 + float(imp.z) ** 2
            ))
        except Exception:
            self._collision_impulse = 0.0

        self._collision_zone = "unknown"
        try:
            if self.ego is None or other is None:
                return
            ego_tf = self.ego.get_transform()
            ego_loc = ego_tf.location
            other_loc = other.get_transform().location
            fwd = _carla_vec_forward(ego_tf)
            # CARLA/UE left-handed convention: right = (-sin(yaw), cos(yaw)).
            # The previous (fwd[1], -fwd[0]) was the LEFT vector and swapped
            # the "left"/"right" collision-zone labels.
            right = (-fwd[1], fwd[0])
            dx = float(other_loc.x - ego_loc.x)
            dy = float(other_loc.y - ego_loc.y)
            along = dx * fwd[0] + dy * fwd[1]
            lateral = dx * right[0] + dy * right[1]
            if abs(along) >= abs(lateral):
                self._collision_zone = "front" if along >= 0.0 else "rear"
            else:
                self._collision_zone = "right" if lateral >= 0.0 else "left"
        except Exception:
            self._collision_zone = "unknown"

    # -- spawning -------------------------------------------------------------
    def _spawn(self):
        carla = self._carla
        bp_lib = self.world.get_blueprint_library()
        ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        # Critical for Town10HD/large maps: Traffic Manager hybrid physics uses
        # vehicles with role_name='hero' as the center of the active radius.
        try:
            if ego_bp.has_attribute("role_name"):
                ego_bp.set_attribute("role_name", "hero" if self.tm_hybrid_physics else "eclcs_ego")
        except Exception:
            pass
        lead_bp = bp_lib.filter("vehicle.audi.tt")[0] if bp_lib.filter("vehicle.audi.tt") \
            else bp_lib.filter("vehicle.*")[0]
        try:
            if lead_bp.has_attribute("role_name"):
                lead_bp.set_attribute("role_name", "eclcs_lead")
        except Exception:
            pass

        spawn_points = self.map.get_spawn_points()
        self.rng.shuffle(spawn_points)
        self.ego = None
        for sp in spawn_points:
            self.ego = self.world.try_spawn_actor(ego_bp, sp)
            if self.ego is not None:
                ego_sp = sp
                break
        if self.ego is None:
            raise RuntimeError("Failed to spawn ego vehicle (no free spawn point).")

        # Optional manual lead vehicle ahead in the same lane.
        # For dense-traffic / traffic-light experiments, disable this with
        # --no_manual_lead.  The hand-coded lead controller does not obey
        # junction traffic lights like Traffic Manager vehicles do, and it can
        # make the ego stop far before the red-light zebra/stop line.
        wp = self.map.get_waypoint(ego_sp.location,
                                   lane_type=carla.LaneType.Driving)
        self.lead = None
        if self.spawn_lead_vehicle:
            nexts = wp.next(self.lead_gap0)
            if nexts:
                lead_tf = nexts[0].transform
                lead_tf.location.z += 0.3
                self.lead = self.world.try_spawn_actor(lead_bp, lead_tf)
            # If lead spawn failed, retry a few alternative distances.
            if self.lead is None:
                for d in (15.0, 35.0, 45.0):
                    nx = wp.next(d)
                    if nx:
                        tf = nx[0].transform; tf.location.z += 0.3
                        self.lead = self.world.try_spawn_actor(lead_bp, tf)
                        if self.lead is not None:
                            break

        # Collision sensor on ego.
        try:
            cs_bp = bp_lib.find("sensor.other.collision")
            self.collision_sensor = self.world.spawn_actor(
                cs_bp, carla.Transform(), attach_to=self.ego)
            self.collision_sensor.listen(self._on_collision)
        except Exception:
            self.collision_sensor = None

        # Explicitly enable physics and release the handbrake. A freshly spawned
        # actor in synchronous mode can have physics dormant or the handbrake set,
        # either of which leaves the car frozen no matter what throttle is sent.
        for v in (self.ego, self.lead):
            if v is not None and v.is_alive:
                try:
                    v.set_simulate_physics(True)
                except Exception:
                    pass
                v.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=0.0, hand_brake=False, manual_gear_shift=False))

    # -- weather drift --------------------------------------------------------
    def _select_episode_weather_mode(self) -> str:
        """Choose the concrete weather preset for this episode.

        ``dynamic`` remains time-varying within an episode. ``random`` samples a
        fixed preset at reset time so each episode has a clear, auditable weather
        label in ``info["weather_mode"]``.
        """
        mode = str(self.weather_mode).lower()
        if mode in {"random", "random_weather", "mixed"}:
            return str(self.rng.choice([
                "clear", "morning_rain", "fog", "night", "night_rain", "rain"
            ]))
        return mode

    def _apply_weather(self):
        """Apply reproducible weather/lighting.

        Modes:
          clear       : clear daytime.
          rain        : rainy daytime.
          morning_rain: lower-sun rainy morning with wet road.
          fog         : cloudy/foggy daytime.
          night       : dry night.
          night_rain  : rainy night with wet road, clouds, fog, and wind.
          random      : episode-level random draw from the above presets.
          dynamic     : slowly varying drift weather used for the scaling study.
        """
        if not self.vary_weather and self.weather_mode in {"clear", "none", "off"}:
            return
        carla = self._carla
        mode = str(getattr(self, "_current_weather_mode", self.weather_mode)).lower()
        if mode in {"night_rain", "rainy_night", "night-rain"}:
            self.world.set_weather(carla.WeatherParameters(
                cloudiness=100.0, precipitation=85.0, precipitation_deposits=85.0,
                wetness=95.0, fog_density=18.0, fog_distance=25.0,
                wind_intensity=45.0, sun_azimuth_angle=0.0, sun_altitude_angle=-35.0))
        elif mode in {"morning_rain", "rain_morning", "morning-rain"}:
            self.world.set_weather(carla.WeatherParameters(
                cloudiness=90.0, precipitation=70.0, precipitation_deposits=70.0,
                wetness=85.0, fog_density=10.0, fog_distance=45.0,
                wind_intensity=25.0, sun_azimuth_angle=90.0, sun_altitude_angle=12.0))
        elif mode in {"fog", "foggy", "heavy_fog"}:
            self.world.set_weather(carla.WeatherParameters(
                cloudiness=85.0, precipitation=0.0, precipitation_deposits=10.0,
                wetness=30.0, fog_density=55.0, fog_distance=18.0,
                wind_intensity=15.0, sun_azimuth_angle=180.0, sun_altitude_angle=25.0))
        elif mode in {"rain", "rainy", "heavy_rain"}:
            self.world.set_weather(carla.WeatherParameters(
                cloudiness=95.0, precipitation=80.0, precipitation_deposits=80.0,
                wetness=90.0, fog_density=8.0, fog_distance=40.0,
                wind_intensity=35.0, sun_azimuth_angle=180.0, sun_altitude_angle=25.0))
        elif mode in {"night", "dark"}:
            self.world.set_weather(carla.WeatherParameters(
                cloudiness=20.0, precipitation=0.0, precipitation_deposits=0.0,
                wetness=0.0, fog_density=5.0, sun_azimuth_angle=0.0,
                sun_altitude_angle=-35.0))
        elif mode in {"dynamic", "drift"}:
            # Slowly varying wetness/fog scaled by drift_scale -> visual shift.
            phase = 2.0 * math.pi * (self.t * self.dt) / max(self.drift_period, 1e-6)
            wet = float(np.clip(40.0 * self.drift_scale * (0.5 + 0.5 * math.sin(phase)), 0, 100))
            fog = float(np.clip(20.0 * self.drift_scale * (0.5 + 0.5 * math.cos(phase)), 0, 100))
            self.world.set_weather(carla.WeatherParameters(
                cloudiness=min(80.0, 30.0 + 50.0 * self.drift_scale),
                precipitation=wet, precipitation_deposits=wet,
                wetness=wet, fog_density=fog, sun_altitude_angle=45.0))
        else:
            self.world.set_weather(carla.WeatherParameters(
                cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0,
                wetness=0.0, fog_density=0.0, sun_azimuth_angle=180.0,
                sun_altitude_angle=45.0))

    # -- state extraction -----------------------------------------------------
    def _vehicle_speed(self, actor) -> float:
        v = actor.get_velocity()
        fwd = _carla_vec_forward(actor.get_transform())
        return float(v.x * fwd[0] + v.y * fwd[1])  # forward speed (m/s)

    def _update_traffic_speed_stats(self) -> None:
        """Track whether Traffic Manager vehicles are actually moving."""
        speeds = []
        for v in list(getattr(self, "traffic_vehicles", [])):
            try:
                if v is not None and v.is_alive:
                    vel = v.get_velocity()
                    speeds.append(float(math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)))
            except Exception:
                pass
        self._traffic_mean_speed = float(np.mean(speeds)) if speeds else 0.0
        self._traffic_moving_count = int(sum(1 for spd in speeds if spd > 0.20))

    @staticmethod
    def _lane_key(wp: Any) -> Tuple[int, int, int]:
        return (
            int(getattr(wp, "road_id", 0)),
            int(getattr(wp, "section_id", 0)),
            int(getattr(wp, "lane_id", 0)),
        )

    def _record_safety_exception(self, context: str, exc: BaseException) -> None:
        """Record and surface safety-critical runtime errors without silently failing open."""
        name = str(context or "unknown")
        self._safety_exception_count = int(getattr(self, "_safety_exception_count", 0)) + 1
        by = dict(getattr(self, "_safety_exception_by_context", {}))
        by[name] = int(by.get(name, 0)) + 1
        self._safety_exception_by_context = by
        self._last_safety_exception = f"{name}: {type(exc).__name__}: {exc}"
        # Print first occurrence and then sparsely to avoid log flooding.
        count = by[name]
        if count == 1 or count in {2, 5, 10} or count % 50 == 0:
            print(f"[carla-safe][FAIL-SAFE] {self._last_safety_exception} count={count}", flush=True)

    def _fail_safe_brake(self, accel: float, ego_speed: float,
                         moving_decel: float = 4.0, hold_decel: float = 2.0) -> float:
        """Conservative mutually monotone longitudinal fallback: never increases acceleration."""
        speed = max(0.0, float(ego_speed))
        cap = -abs(float(moving_decel)) if speed > 0.15 else -abs(float(hold_decel))
        return float(min(float(accel), cap))

    def _snapshot_traffic_light_state(self) -> Dict[str, Any]:
        return {
            "active": bool(getattr(self, "_traffic_light_active", False)),
            "state": str(getattr(self, "_traffic_light_state", "none")),
            "distance": float(getattr(self, "_traffic_light_distance", 999.0)),
            "id": int(getattr(self, "_traffic_light_id", -1)),
            "signal_key": getattr(self, "_traffic_light_last_signal_key", None),
            "stop_s": float(getattr(self, "_traffic_light_last_stop_s", -1.0)),
        }

    def _handle_traffic_light_detector_exception(
        self, previous: Dict[str, Any], speed: float, ego_half: float, exc: BaseException
    ) -> Optional[Tuple[float, float, float, str, float]]:
        """Handle a traffic-light sensing exception without inventing a red light.

        A previously confirmed red/yellow stop signal is retained through the
        normal hysteresis path.  When there is no confirmed previous signal, we
        record the detector fault but leave the road state unconfirmed instead of
        turning one transient API exception into immediate hard braking.  The
        actuator-stage controller may coast after repeated consecutive faults.
        """
        self._record_safety_exception("traffic_light_hazard", exc)
        self._traffic_light_detector_fault_active = True
        self._traffic_light_detector_fault_ticks = int(
            getattr(self, "_traffic_light_detector_fault_ticks", 0)
        ) + 1

        held = self._hold_previous_traffic_light_if_needed(
            previous,
            speed=float(speed),
            ego_half=float(ego_half),
            explicit_release_ids=set(),
        )
        if held is not None:
            return held

        # No confirmed signal exists. Do not manufacture a red-light hazard from
        # an exception alone. Keep the explicit fault flags for diagnostics and
        # for the repeated-fault coast policy in _apply_red_light_close_stop().
        self._traffic_light_active = False
        self._traffic_light_state = "none"
        self._traffic_light_distance = 999.0
        self._traffic_light_virtual_gap = 999.0
        return None

    @staticmethod
    def _traffic_light_candidate_score(signed_gap: float, lane_priority: int,
                                       lateral_score: float) -> Tuple[float, int, float]:
        """Nearest valid governing stop line first; lane quality breaks ties."""
        return (float(signed_gap), int(lane_priority), float(lateral_score))

    @staticmethod
    def _traffic_light_tracking_score(signed_gap: float, lane_priority: int,
                                      lateral_score: float,
                                      cross_tolerance: float) -> Tuple[int, float, int, float]:
        return (
            0 if float(signed_gap) > float(cross_tolerance) else 1,
            abs(float(signed_gap)),
            int(lane_priority),
            float(lateral_score),
        )

    @staticmethod
    def _smoothstep01(x: float) -> float:
        t = float(np.clip(float(x), 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    def _red_light_activation_distance(self, speed: float) -> float:
        v = max(0.0, float(speed))
        a = max(float(self.red_light_comfort_decel), 1e-3)
        return float(
            v * float(self.red_light_reaction_time_s)
            + (v * v) / (2.0 * a)
            + float(self.red_light_stop_buffer)
            + float(self.red_light_activation_margin)
        )

    def _red_light_desired_speed(self, stop_error: float) -> float:
        """Continuous speed target from braking curve into final creep zone."""
        e = max(0.0, float(stop_error))
        creep_zone = float(self.red_light_creep_distance)
        blend = float(self.red_light_blend_distance)
        creep_v = float(self.red_light_creep_speed)
        curve_v = min(
            float(self.target_speed),
            math.sqrt(max(0.0, 2.0 * float(self.red_light_comfort_decel) * e)),
        )
        near_v = min(creep_v, max(0.0, 0.35 * e))
        if e <= creep_zone:
            return float(near_v)
        if e >= creep_zone + blend:
            return float(curve_v)
        lam = self._smoothstep01((e - creep_zone) / max(blend, 1e-6))
        return float((1.0 - lam) * creep_v + lam * curve_v)

    def _red_light_queue_desired_speed(self, queue_error: float,
                                       front_speed: float) -> float:
        """Human-like queue approach speed without far-distance crawling.

        The previous queue controller forced ``desired <= 3 m/s`` as soon as a
        stopped vehicle was detected within ``queue_detect_distance`` (normally
        25 m).  With background traffic this made the ego crawl long before the
        traffic light.  Use the same physics-based braking curve as the physical
        stop-line controller instead:

            v_rel_safe = sqrt(2 * a_comfort * queue_error)
            v_des      = min(v_target, v_front + v_rel_safe)

        A continuous blend enters precision creep only in the final few metres.
        """
        e = max(0.0, float(queue_error))
        vf = max(0.0, float(front_speed))
        creep_zone = float(self.red_light_creep_distance)
        blend = float(self.red_light_blend_distance)
        creep_v = float(self.queue_creep_speed)
        a = max(float(self.red_light_comfort_decel), 1e-3)

        curve_v = min(
            float(self.target_speed),
            vf + math.sqrt(max(0.0, 2.0 * a * e)),
        )
        near_v = min(creep_v, max(0.0, 0.35 * e))
        if e <= creep_zone:
            return float(near_v)
        if e >= creep_zone + blend:
            return float(curve_v)
        lam = self._smoothstep01((e - creep_zone) / max(blend, 1e-6))
        return float((1.0 - lam) * creep_v + lam * curve_v)

    def _hold_previous_traffic_light_if_needed(
        self,
        previous: Dict[str, Any],
        speed: float,
        ego_half: float,
        explicit_release_ids: Optional[set] = None,
    ) -> Optional[Tuple[float, float, float, str, float]]:
        """Short grace period for one-frame detector dropouts.

        A previous red/yellow hazard is retained only while its stop line is
        still ahead, it was not explicitly observed green/off, it is not the
        committed signal already being crossed, and the miss budget is not
        exhausted.  This prevents brake-release-brake oscillation without
        creating stale multi-second stops.
        """
        explicit_release_ids = explicit_release_ids or set()
        if not bool(previous.get("active", False)):
            self._traffic_light_miss_ticks = 0
            return None
        state = str(previous.get("state", "none")).lower()
        stop_state = state.startswith("red") or (
            self.yellow_light_stop and state.startswith("yellow")
        )
        if not stop_state:
            self._traffic_light_miss_ticks = 0
            return None
        light_id = int(previous.get("id", -1))
        if light_id in explicit_release_ids:
            self._traffic_light_miss_ticks = 0
            return None
        if self._junction_commit_active:
            same_committed = bool(
                (self._committed_signal_key is not None
                 and self._committed_signal_key == previous.get("signal_key"))
                or (self._committed_signal_key is None
                    and self._committed_light_id is not None
                    and int(self._committed_light_id) == light_id)
            )
            if same_committed:
                self._traffic_light_miss_ticks = 0
                return None
        if self._traffic_light_miss_ticks >= self.traffic_light_hysteresis_ticks:
            self._traffic_light_miss_ticks = 0
            return None

        stop_s = float(previous.get("stop_s", -1.0))
        if self.route_tracker is not None and stop_s >= 0.0:
            gap = float(stop_s - self.progress_m - ego_half)
        else:
            gap = float(previous.get("distance", 999.0)) - max(0.0, float(speed)) * float(self.dt)
        if not np.isfinite(gap) or gap <= self.stop_line_cross_tolerance:
            self._traffic_light_miss_ticks = 0
            return None

        self._traffic_light_miss_ticks += 1
        self._traffic_light_detection_dropout = True
        self._traffic_light_active = True
        self._traffic_light_state = str(previous.get("state", "red"))
        self._traffic_light_distance = float(gap)
        self._traffic_light_virtual_gap = float(gap)
        self._traffic_light_stop_error = float(gap - self.red_light_stop_buffer)
        self._traffic_light_id = light_id
        self._traffic_light_last_signal_key = previous.get("signal_key")
        self._traffic_light_last_stop_s = stop_s
        return (float(gap), 0.0, float(gap), self._traffic_light_state, float(gap))

    def _traffic_light_hazard(self, ego_wp, lane_half: float) -> Optional[Tuple[float, float, float, str, float]]:
        """Return the nearest route-relevant red/yellow stop-line hazard.

        Human-like rules implemented here:
          * obey only a signal that governs the ego/planned-route lane;
          * stop for red only while its stop line is still ahead;
          * for yellow, stop only when a comfortable stop is still possible;
          * once the ego front bumper crosses the governing stop line, latch a
            per-signal junction commitment and ignore that same signal until the
            junction has been cleared;
          * never ignore all lights globally merely because ego is in a junction.

        CARLA's stop waypoints provide the physical stopping positions, while
        affected-lane waypoints identify the lanes controlled by a signal.  The
        route/heading tests remain as robust fallbacks for complex Town10HD
        junction topology.
        """
        previous_signal = {
            "active": bool(getattr(self, "_traffic_light_active", False)),
            "state": str(getattr(self, "_traffic_light_state", "none")),
            "distance": float(getattr(self, "_traffic_light_distance", 999.0)),
            "id": int(getattr(self, "_traffic_light_id", -1)),
            "signal_key": getattr(self, "_traffic_light_last_signal_key", None),
            "stop_s": float(getattr(self, "_traffic_light_last_stop_s", -1.0)),
        }
        self._traffic_light_active = False
        self._traffic_light_state = "none"
        self._traffic_light_distance = 999.0
        self._traffic_light_virtual_gap = 999.0
        self._traffic_light_stop_error = 999.0
        self._traffic_light_id = -1
        self._traffic_light_yellow_go = False
        self._yellow_required_stop_distance = 0.0
        self._traffic_light_detection_dropout = False
        self._red_light_crossed_on_red = False
        self._red_light_stop_success = False
        if not self.traffic_light_guard:
            return None

        carla = self._carla
        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        ego_half = float(self.ego.bounding_box.extent.x)
        fwd = _carla_vec_forward(ego_tf)
        # CARLA right vector: (-sin(yaw), cos(yaw)). lat_ego is consumed via
        # abs() below, but keep the sign convention consistent with the route
        # tracker and waypoint get_right_vector() usage.
        right_x, right_y = -fwd[1], fwd[0]
        ego_heading = math.radians(ego_tf.rotation.yaw)
        speed = max(0.0, self._vehicle_speed(self.ego))
        ego_in_junction = bool(getattr(ego_wp, "is_junction", False))

        def _signal_key(tl: Any, swp: Any, stop_s: float) -> Tuple[Any, ...]:
            """Stable key for one physical signal/stop-line/lane combination."""
            try:
                od_id = str(tl.get_opendrive_id())
            except Exception:
                od_id = ""
            try:
                pole_idx = int(tl.get_pole_index())
            except Exception:
                pole_idx = -1
            return (
                int(getattr(tl, "id", -1)),
                od_id,
                pole_idx,
                int(getattr(swp, "road_id", 0)),
                int(getattr(swp, "section_id", 0)),
                int(getattr(swp, "lane_id", 0)),
                round(float(stop_s) * 2.0) / 2.0,
            )

        # --------------------------------------------------------------
        # A) Maintain/clear an existing junction commitment.
        # --------------------------------------------------------------
        if bool(getattr(self, "_junction_commit_active", False)):
            if ego_in_junction:
                self._junction_seen_since_commit = True

            committed_stop_s = float(getattr(self, "_committed_stop_s", -1.0))
            sufficiently_past = bool(
                float(self.progress_m)
                > committed_stop_s + float(self.junction_commit_clear_distance)
            )
            # Primary release requires proof that the ego entered and then exited
            # the junction. A conservative far-past fallback avoids a permanent
            # latch when map waypoint metadata never marks the junction correctly.
            far_past_distance = max(
                3.0 * float(self.junction_commit_clear_distance), 25.0
            )
            far_past = bool(
                float(self.progress_m) > committed_stop_s + far_past_distance
            )
            clear_ready = bool(
                (self._junction_seen_since_commit and not ego_in_junction and sufficiently_past)
                or (not ego_in_junction and far_past)
            )

            if clear_ready:
                self._junction_exit_clear_ticks += 1
            else:
                self._junction_exit_clear_ticks = 0

            if self._junction_exit_clear_ticks >= self.junction_commit_clear_ticks:
                self._committed_light_id = None
                self._committed_signal_key = None
                self._committed_stop_s = -1.0
                self._junction_commit_active = False
                self._junction_seen_since_commit = False
                self._junction_exit_clear_ticks = 0

                # The tracked stop line that created this commitment is already
                # behind the ego. Clear it as well; otherwise the fallback
                # crossing detector below would immediately re-latch the old
                # signal on the next tick.
                self._tracked_light_id = None
                self._tracked_signal_key = None
                self._tracked_stop_s = -1.0
                self._tracked_stop_gap_prev = 999.0
                self._tracked_light_seen_ahead = False
                self._tracked_light_state = "none"
                self._tracked_yellow_go_latched = False

        # If a previously tracked governing stop line has just been crossed,
        # commit even when that traffic-light actor disappears from this tick's
        # search results.  The track was created only while the line was ahead.
        if (
            not self._junction_commit_active
            and getattr(self, "_tracked_light_id", None) is not None
            and bool(getattr(self, "_tracked_light_seen_ahead", False))
            and float(getattr(self, "_tracked_stop_s", -1.0)) >= 0.0
            and self.route_tracker is not None
        ):
            ego_front_s = float(self.progress_m) + ego_half
            if ego_front_s >= float(self._tracked_stop_s) - self.stop_line_cross_tolerance:
                crossed_on_red = bool(
                    str(getattr(self, "_tracked_light_state", "none")).lower().startswith("red")
                    and not bool(getattr(self, "_tracked_yellow_go_latched", False))
                )
                self._red_light_crossed_on_red = crossed_on_red
                if crossed_on_red:
                    self._red_light_violation_count += 1
                self._committed_light_id = int(self._tracked_light_id)
                self._committed_signal_key = getattr(self, "_tracked_signal_key", None)
                self._committed_stop_s = float(self._tracked_stop_s)
                self._junction_commit_active = True
                self._junction_seen_since_commit = bool(ego_in_junction)
                self._junction_exit_clear_ticks = 0
                # Mark this crossing as consumed. If the same tracked signal
                # also appears in this tick's candidate scan, the crossing
                # detector in section D would otherwise see gap_prev > tol and
                # signed_gap <= tol and count the SAME crossing (and the same
                # red-light violation) a second time.
                self._tracked_stop_gap_prev = -999.0

        # --------------------------------------------------------------
        # B) Collect candidate traffic lights robustly.
        # --------------------------------------------------------------
        lights: List[Any] = []
        seen = set()
        tl_query_successes = 0
        tl_query_failures = 0
        tl_critical_failures = 0

        def _add_light(tl: Any) -> None:
            try:
                if tl is None:
                    return
                tid = int(tl.id)
                if tid not in seen:
                    lights.append(tl)
                    seen.add(tid)
            except Exception:
                pass

        def _query_lights_from_wp(wp: Any, dist: float) -> None:
            nonlocal tl_query_successes, tl_query_failures
            try:
                found = list(self.world.get_traffic_lights_from_waypoint(wp, float(dist)))
                tl_query_successes += 1
                for tl in found:
                    _add_light(tl)
            except Exception as exc:
                tl_query_failures += 1
                self._record_safety_exception("traffic_light_waypoint_query", exc)

        _query_lights_from_wp(ego_wp, self.red_light_stop_distance)

        if bool(getattr(self, "traffic_light_route_scan", True)) and self.route_tracker is not None:
            try:
                step_idx = max(1, int(round(
                    self.traffic_light_route_scan_step / max(self.route_step_m, 0.5))))
                max_ahead_idx = int(math.ceil(
                    self.red_light_stop_distance / max(self.route_step_m, 0.5))) + 25
                i0 = int(max(0, self.route_tracker.last_idx))
                i1 = int(min(len(self.route_waypoints) - 1, i0 + max_ahead_idx))
                for i in range(i0, i1 + 1, step_idx):
                    _query_lights_from_wp(
                        self.route_waypoints[i],
                        max(8.0, 2.0 * self.traffic_light_route_scan_step))
            except Exception:
                pass
        else:
            try:
                for d in np.arange(
                    4.0, self.red_light_stop_distance + 1e-6,
                    self.traffic_light_route_scan_step
                ):
                    nxt = ego_wp.next(float(d))
                    if nxt:
                        _query_lights_from_wp(
                            nxt[0], max(8.0, 2.0 * self.traffic_light_route_scan_step))
            except Exception:
                pass

        if bool(getattr(self, "traffic_light_landmark_fallback", True)):
            try:
                landmark_lists = []
                try:
                    landmark_lists.append(list(
                        ego_wp.get_landmarks(self.red_light_stop_distance, False)))
                except Exception:
                    pass
                try:
                    landmark_lists.append(list(ego_wp.get_landmarks_of_type(
                        self.red_light_stop_distance, "1000001", False)))
                except Exception:
                    pass
                for lms in landmark_lists:
                    for lm in lms:
                        try:
                            if (
                                not bool(getattr(lm, "is_dynamic", False))
                                and str(getattr(lm, "type", "")) != "1000001"
                            ):
                                continue
                            _add_light(self.world.get_traffic_light(lm))
                        except Exception:
                            continue
            except Exception:
                pass

        # This is useful precisely around the trigger boundary.  Commitment
        # memory below prevents the associated light from commanding a stop once
        # its own stop line has been crossed.
        try:
            if self.ego is not None and self.ego.is_at_traffic_light():
                _add_light(self.ego.get_traffic_light())
        except Exception:
            pass

        if not lights:
            if tl_query_failures > 0 and tl_query_successes == 0:
                return self._handle_traffic_light_detector_exception(
                    previous_signal, speed, ego_half,
                    RuntimeError("all traffic-light waypoint discovery queries failed"),
                )
            return self._hold_previous_traffic_light_if_needed(
                previous_signal, speed, ego_half, set()
            )

        # Route-lane keys ahead: used with TrafficLight.get_affected_lane_waypoints
        # to reject cross-street and adjacent-lane signals.
        route_lane_keys = set()
        if self.route_tracker is not None:
            try:
                i0 = int(max(0, self.route_tracker.last_idx - 2))
                n_ahead = int(math.ceil(
                    self.red_light_stop_distance / max(self.route_step_m, 0.5))) + 20
                i1 = int(min(len(self.route_waypoints) - 1, i0 + n_ahead))
                for rwp in self.route_waypoints[i0:i1 + 1]:
                    route_lane_keys.add((
                        int(getattr(rwp, "road_id", 0)),
                        int(getattr(rwp, "section_id", 0)),
                        int(getattr(rwp, "lane_id", 0)),
                    ))
            except Exception:
                route_lane_keys = set()

        best: Optional[Tuple[float, float, float, str, float]] = None
        best_score: Optional[Tuple[float, int, float]] = None
        best_light_id = -1
        best_signal_key: Optional[Tuple[Any, ...]] = None
        best_stop_s = -1.0
        explicit_release_ids: set = set()
        tracking_best = None
        tracking_score = None
        heading_tol = math.radians(self.traffic_light_heading_tolerance_deg)

        # --------------------------------------------------------------
        # C) Evaluate stop waypoints and signal relevance.
        # --------------------------------------------------------------
        for tl in lights:
            try:
                light_id = int(tl.id)
                state = tl.get_state()
                state_name = str(state).split(".")[-1]
                is_red = state == carla.TrafficLightState.Red
                is_yellow = state == carla.TrafficLightState.Yellow
                if not is_red and not is_yellow:
                    explicit_release_ids.add(light_id)
            except Exception as exc:
                tl_critical_failures += 1
                self._record_safety_exception("traffic_light_state_read", exc)
                continue

            try:
                stop_wps = list(tl.get_stop_waypoints())
            except Exception as exc:
                tl_critical_failures += 1
                self._record_safety_exception("traffic_light_stop_waypoints", exc)
                stop_wps = []
            if not stop_wps:
                continue

            affected_keys = set()
            if self.traffic_light_use_affected_lanes:
                try:
                    for awp in list(tl.get_affected_lane_waypoints()):
                        affected_keys.add((
                            int(getattr(awp, "road_id", 0)),
                            int(getattr(awp, "section_id", 0)),
                            int(getattr(awp, "lane_id", 0)),
                        ))
                except Exception:
                    affected_keys = set()

            for swp in stop_wps:
                try:
                    swp_loc = swp.transform.location
                    dx = float(swp_loc.x - ego_loc.x)
                    dy = float(swp_loc.y - ego_loc.y)
                    along_ego = dx * fwd[0] + dy * fwd[1]
                    lat_ego = dx * right_x + dy * right_y
                    stop_heading = math.radians(swp.transform.rotation.yaw)
                    ego_heading_ok = abs(_wrap_pi(stop_heading - ego_heading)) < heading_tol

                    same_lane = bool(
                        swp.road_id == ego_wp.road_id
                        and swp.section_id == ego_wp.section_id
                        and swp.lane_id == ego_wp.lane_id
                    )
                    swp_lane_key = self._lane_key(swp)
                    route_stop_key = swp_lane_key

                    if self.route_tracker is not None:
                        # Search only as far as this traffic-light horizon can
                        # physically matter. The old +260-segment window could
                        # span hundreds of metres and snap a Town10HD stop line
                        # to a geometrically nearby future/crossing route branch.
                        tl_forward_segments = int(math.ceil(
                            (self.red_light_stop_distance + 15.0) /
                            max(self.route_step_m, 0.5)
                        )) + 8
                        tl_forward_segments = int(np.clip(tl_forward_segments, 12, 80))
                        tl_back_segments = int(np.clip(self.route_search_back + 5, 5, 25))
                        s_stop, lat_stop, route_yaw, route_idx = self.route_tracker.query(
                            swp_loc,
                            center_idx=self.route_tracker.last_idx,
                            back=tl_back_segments,
                            forward=tl_forward_segments,
                        )
                        route_ds = float(s_stop - self.progress_m)
                        if 0 <= int(route_idx) < len(self.route_waypoints):
                            route_stop_key = self._lane_key(self.route_waypoints[int(route_idx)])
                        heading_ok = abs(_wrap_pi(stop_heading - route_yaw)) < heading_tol
                        corridor = abs(lat_stop) <= max(0.55 * lane_half, 1.05)
                        route_relevant = bool(
                            -3.0 < route_ds <= self.red_light_stop_distance + 5.0
                            and corridor and heading_ok
                        )
                        ego_frame_relevant = bool(
                            along_ego > -3.0
                            and abs(lat_ego) <= max(0.75 * lane_half, 1.25)
                            and ego_heading_ok
                        )

                        # Match affected-lane metadata locally at this physical
                        # stop waypoint / projected route position.  A light that
                        # affects some other lane elsewhere in the lookahead must
                        # not make this stop line relevant.
                        local_affected_match = bool(
                            swp_lane_key in affected_keys
                            or route_stop_key in affected_keys
                        )
                        if self.traffic_light_use_affected_lanes and affected_keys:
                            relevant = bool(
                                same_lane
                                or (local_affected_match and route_relevant)
                            )
                        else:
                            relevant = bool(same_lane or route_relevant or ego_frame_relevant)

                        if route_ds > -3.0:
                            dist_along = route_ds
                            stop_s_track = float(s_stop)
                        else:
                            dist_along = along_ego
                            stop_s_track = float(self.progress_m + along_ego)
                    else:
                        lat_stop = float(lat_ego)
                        route_ds = float(along_ego)
                        local_affected_match = bool(
                            swp_lane_key in affected_keys
                        )
                        relevant = bool(same_lane and along_ego > -3.0 and ego_heading_ok)
                        dist_along = along_ego
                        stop_s_track = float(self.progress_m + along_ego)

                    if not relevant:
                        continue

                    # Signed front-bumper gap. Positive => stop line ahead;
                    # negative => already crossed.
                    signed_gap = float(dist_along) - ego_half
                    if not (-3.0 <= signed_gap <= self.red_light_stop_distance):
                        continue

                    lane_priority = 0 if same_lane else (1 if local_affected_match else 2)
                    lateral_score = min(abs(float(lat_stop)), abs(float(lat_ego)))
                    tscore = self._traffic_light_tracking_score(
                        signed_gap, lane_priority, lateral_score,
                        self.stop_line_cross_tolerance,
                    )
                    signal_key = _signal_key(tl, swp, stop_s_track)
                    tcand = {
                        "light_id": light_id,
                        "signal_key": signal_key,
                        "stop_s": float(stop_s_track),
                        "signed_gap": float(signed_gap),
                        "state_name": state_name,
                        "is_red": bool(is_red),
                        "yellow_go_latched": bool(signal_key in getattr(self, "_yellow_go_signal_keys", set())),
                    }
                    if tracking_best is None or tscore < tracking_score:
                        tracking_best = tcand
                        tracking_score = tscore

                    # A committed signal cannot ask the vehicle to stop after
                    # its stop line has been crossed. Other valid signals remain
                    # fully active, including another signal inside a compound
                    # junction.
                    committed_same_signal = bool(
                        self._junction_commit_active
                        and (
                            (
                                self._committed_signal_key is not None
                                and self._committed_signal_key == signal_key
                            )
                            or (
                                self._committed_signal_key is None
                                and self._committed_light_id is not None
                                and int(self._committed_light_id) == light_id
                            )
                        )
                    )
                    if committed_same_signal:
                        continue

                    if signed_gap <= self.stop_line_cross_tolerance:
                        continue
                    if signed_gap > self.red_light_stop_distance:
                        continue

                    yellow_go_keys = getattr(self, "_yellow_go_signal_keys", None)
                    if yellow_go_keys is None:
                        yellow_go_keys = set()
                        self._yellow_go_signal_keys = yellow_go_keys

                    should_stop = bool(is_red)
                    if is_yellow and self.yellow_light_stop:
                        required_stop_distance = float(
                            speed * self.yellow_reaction_time_s
                            + (speed * speed) / (2.0 * self.yellow_comfort_decel)
                            + self.yellow_stop_margin
                        )
                        self._yellow_required_stop_distance = max(
                            self._yellow_required_stop_distance,
                            required_stop_distance,
                        )
                        # Human-like dilemma-zone decision: if too close to stop
                        # comfortably, continue. A nearly stopped ego remains
                        # stopped rather than accelerating through late yellow.
                        should_stop = bool(
                            speed <= 0.5 or signed_gap >= required_stop_distance)
                        if not should_stop:
                            self._traffic_light_yellow_go = True
                            # Latch: this signal was legally committed through
                            # its dilemma zone on yellow.  A subsequent flip to
                            # red must not demand a physically impossible stop.
                            yellow_go_keys.add(signal_key)
                        else:
                            yellow_go_keys.discard(signal_key)
                    elif is_red and signal_key in yellow_go_keys:
                        # Yellow flipped to red mid-dilemma.  If stopping from
                        # the current state needs more than the actuator can
                        # deliver, continue through; the crossing detector
                        # below latches the junction commitment.  Otherwise
                        # drop the latch and apply the normal red stop.
                        hard_stop_needed = (speed * speed) / (2.0 * max(signed_gap, 0.30))
                        if hard_stop_needed > 3.8 and speed > 1.0:
                            continue
                        yellow_go_keys.discard(signal_key)
                    elif not is_red and not is_yellow:
                        yellow_go_keys.discard(signal_key)

                    if not should_stop:
                        continue

                    cand = (
                        float(signed_gap),
                        float(lat_stop),
                        float(route_ds),
                        state_name,
                        float(signed_gap),
                    )
                    # Once validity is established, the nearest governing
                    # stop line must win; lane authority is only a tie-break.
                    score = self._traffic_light_candidate_score(
                        signed_gap, lane_priority, lateral_score
                    )
                    if best is None or score < best_score:
                        best = cand
                        best_score = score
                        best_light_id = light_id
                        best_signal_key = signal_key
                        best_stop_s = float(stop_s_track)
                except Exception as exc:
                    tl_critical_failures += 1
                    self._record_safety_exception(
                        "traffic_light_candidate_processing",
                        exc,
                    )
                    continue

        # --------------------------------------------------------------
        # D) Update governing-stop tracking and latch crossing commitment.
        # --------------------------------------------------------------
        if tracking_best is not None:
            light_id = int(tracking_best["light_id"])
            signal_key = tracking_best.get("signal_key")
            stop_s = float(tracking_best["stop_s"])
            signed_gap = float(tracking_best["signed_gap"])

            same_track = (
                self._tracked_light_id is not None
                and int(self._tracked_light_id) == light_id
                and (
                    getattr(self, "_tracked_signal_key", None) is None
                    or signal_key is None
                    or self._tracked_signal_key == signal_key
                )
            )
            crossed_now = bool(
                same_track
                and self._tracked_light_seen_ahead
                and self._tracked_stop_gap_prev > self.stop_line_cross_tolerance
                and signed_gap <= self.stop_line_cross_tolerance
            )

            # If the junction commitment already belongs to this very signal
            # (latched by the progress-based fallback in section A, either this
            # tick or a previous one), this crossing was already consumed and
            # any red-light violation was already counted.  Only a *different*
            # signal crossing while a commitment is active may still count.
            already_committed_same_signal = bool(
                self._junction_commit_active
                and (
                    (
                        self._committed_signal_key is not None
                        and signal_key is not None
                        and self._committed_signal_key == signal_key
                    )
                    or (
                        self._committed_signal_key is None
                        and self._committed_light_id is not None
                        and int(self._committed_light_id) == light_id
                    )
                )
            )

            if crossed_now and not already_committed_same_signal:
                crossed_on_red = bool(
                    tracking_best.get("is_red", False)
                    and not tracking_best.get("yellow_go_latched", False)
                )
                self._red_light_crossed_on_red = crossed_on_red
                if crossed_on_red:
                    self._red_light_violation_count += 1

            if crossed_now and not self._junction_commit_active:
                self._committed_light_id = light_id
                self._committed_signal_key = signal_key
                self._committed_stop_s = stop_s
                self._junction_commit_active = True
                self._junction_seen_since_commit = bool(ego_in_junction)
                self._junction_exit_clear_ticks = 0

            if signed_gap > self.stop_line_cross_tolerance:
                self._tracked_light_id = light_id
                self._tracked_signal_key = signal_key
                self._tracked_stop_s = stop_s
                self._tracked_stop_gap_prev = signed_gap
                self._tracked_light_seen_ahead = True
                self._tracked_light_state = str(tracking_best.get("state_name", "none"))
                self._tracked_yellow_go_latched = bool(
                    tracking_best.get("yellow_go_latched", False)
                )
            elif same_track:
                self._tracked_stop_gap_prev = signed_gap

        if best is not None:
            # A crossing can be latched above in the same tick. Do not activate
            # braking for the just-committed signal.
            same_committed_best = bool(
                self._junction_commit_active
                and (
                    (
                        self._committed_signal_key is not None
                        and best_signal_key is not None
                        and self._committed_signal_key == best_signal_key
                    )
                    or (
                        self._committed_signal_key is None
                        and self._committed_light_id is not None
                        and int(self._committed_light_id) == int(best_light_id)
                    )
                )
            )
            if same_committed_best:
                self._traffic_light_miss_ticks = 0
                return None

            self._traffic_light_active = True
            self._traffic_light_state = best[3]
            self._traffic_light_distance = best[4]
            self._traffic_light_id = int(best_light_id)
            self._traffic_light_stop_error = float(best[4] - self.red_light_stop_buffer)
            self._traffic_light_virtual_gap = float(best[4])
            self._traffic_light_last_signal_key = best_signal_key
            self._traffic_light_last_stop_s = float(best_stop_s)
            self._traffic_light_miss_ticks = 0
            self._traffic_light_detection_dropout = False
            return best

        if best is None and tracking_best is None and tl_critical_failures > 0:
            return self._handle_traffic_light_detector_exception(
                previous_signal, speed, ego_half,
                RuntimeError("traffic-light candidate evaluation failed before a reliable result"),
            )
        return self._hold_previous_traffic_light_if_needed(
            previous_signal, speed, ego_half, explicit_release_ids
        )

    def _collect_dynamic_hazard_candidates(
        self,
    ) -> List[Tuple[Any, str]]:
        """Return all live world vehicles/walkers except the current ego.

        The CARLA world actor registry is treated as the authoritative source for
        physically collidable dynamic actors. Internal Python actor lists remain
        only a secondary fallback.

        If the authoritative world scan fails, set an explicit safety-fault flag.
        The longitudinal collision guard consumes this flag and applies conservative
        braking instead of interpreting an empty candidate list as a clear road.
        """

        candidates: List[Tuple[Any, str]] = []
        seen = set()

        # Lazily defined for backward compatibility with older checkpoints/code.
        self._world_hazard_scan_fault_active = False

        ego_id = (
            int(self.ego.id)
            if self._actor_is_alive(self.ego)
            else -1
        )

        lead_id = (
            int(self.lead.id)
            if self._actor_is_alive(self.lead)
            else -1
        )

        def add(actor: Any, kind: str) -> None:
            """Add one unique live non-ego actor."""
            try:
                if actor is None:
                    return

                if not bool(getattr(actor, "is_alive", False)):
                    return

                aid = int(actor.id)

                if aid == ego_id:
                    return

                if aid in seen:
                    return

                seen.add(aid)

                candidates.append(
                    (
                        actor,
                        str(kind),
                    )
                )

            except Exception:
                return

        # ------------------------------------------------------------------
        # 1. Authoritative CARLA world-registry scan.
        # ------------------------------------------------------------------
        world_scan_ok = False

        try:
            if self.world is None:
                raise RuntimeError(
                    "CARLA world is unavailable during dynamic hazard scan."
                )

            all_actors = self.world.get_actors()

            if all_actors is None:
                raise RuntimeError(
                    "CARLA world.get_actors() returned None."
                )

            for actor in all_actors.filter("vehicle.*"):
                kind = (
                    "lead"
                    if int(actor.id) == lead_id
                    else "vehicle"
                )
                add(actor, kind)

            for actor in all_actors.filter("walker.*"):
                add(actor, "walker")

            world_scan_ok = True

        except Exception as exc:
            self._record_safety_exception(
                "world_hazard_scan",
                exc,
            )

            self._world_hazard_scan_fault_active = True

            self._world_hazard_scan_fault_ticks = (
                int(
                    getattr(
                        self,
                        "_world_hazard_scan_fault_ticks",
                        0,
                    )
                )
                + 1
            )

        # Successful authoritative scan clears the fault immediately.
        if world_scan_ok:
            self._world_hazard_scan_fault_active = False
            self._world_hazard_scan_fault_ticks = 0

        # ------------------------------------------------------------------
        # 2. Internal-list fallback.
        #
        # These lists improve continuity but do NOT clear a world-scan fault.
        # A stale/orphan actor can exist outside these lists.
        # ------------------------------------------------------------------
        add(self.lead, "lead")

        for actor in list(
            getattr(
                self,
                "traffic_vehicles",
                [],
            )
        ):
            add(actor, "vehicle")

        for actor in list(
            getattr(
                self,
                "walker_actors",
                [],
            )
        ):
            add(actor, "walker")

        return candidates


    def _route_geometry_is_sane(self, route_lat: float, route_heading_error: float,
                                lane_half: float) -> bool:
        """Reject route projections that are finite but physically impossible."""
        if not (np.isfinite(route_lat) and np.isfinite(route_heading_error)):
            return False
        max_lat = max(2.25 * float(lane_half), 4.5)
        max_heading = math.radians(80.0)
        return bool(abs(float(route_lat)) <= max_lat and
                    abs(float(route_heading_error)) <= max_heading)


    def _route_trace_start_is_sane(self, waypoints: Sequence[Any], start_wp: Any) -> bool:
        """Validate that a traced route really starts near and along the ego heading."""
        if len(waypoints) < 2:
            return False
        # Yaw tolerance must sit strictly INSIDE the reset initial-heading gate
        # (heading_limit_rad + 0.25) and the per-tick failure threshold
        # (heading_limit_rad + 0.35).  With the previous 65 deg tolerance, a
        # route misaligned by 49-65 deg passed candidate selection, the env
        # then spawned all background traffic and walkers, ticked, and only the
        # reset gate rejected it, wasting a full attempt cycle.  Rejecting the
        # candidate here instead lets the builder try other destinations and,
        # if every raw-origin trace misaligns, routes the spawn into the
        # nudged-origin rescue pass, both far cheaper than a reset retry.
        barrier = getattr(self, "barrier", None)
        heading_limit = float(getattr(barrier, "heading_limit_rad", 0.60)) if barrier is not None else 0.60
        yaw_tol = heading_limit + 0.15
        try:
            start_loc = start_wp.transform.location
            first_loc = waypoints[0].transform.location
            dx = float(first_loc.x - start_loc.x)
            dy = float(first_loc.y - start_loc.y)
            first_dist = math.hypot(dx, dy)
            if first_dist > max(6.0, 3.0 * float(self.route_step_m)):
                return False

            p0 = waypoints[0].transform.location
            # Find the first non-degenerate route segment.
            p1 = None
            for wp in waypoints[1:min(len(waypoints), 8)]:
                q = wp.transform.location
                if math.hypot(float(q.x - p0.x), float(q.y - p0.y)) > 0.25:
                    p1 = q
                    break
            if p1 is None:
                return False
            route_yaw = math.atan2(float(p1.y - p0.y), float(p1.x - p0.x))
            ego_yaw = math.radians(float(start_wp.transform.rotation.yaw))
            if abs(_wrap_pi(route_yaw - ego_yaw)) > yaw_tol:
                return False

            tracker = RouteProgressTracker(waypoints)
            s0, lat0, yaw0, _ = tracker.query(start_loc, center_idx=0, back=0, forward=20)
            lane_half = max(0.5 * float(start_wp.lane_width), 1.0)
            if s0 > max(8.0, 4.0 * float(self.route_step_m)):
                return False
            if abs(float(lat0)) > max(1.5 * lane_half, 3.0):
                return False
            if abs(_wrap_pi(ego_yaw - float(yaw0))) > yaw_tol:
                return False
            return True
        except Exception:
            return False



    def _route_has_junction_ahead(self, distance_m: Optional[float] = None) -> bool:
        """Return whether the selected route enters a junction soon."""
        if self.route_tracker is None or not self.route_waypoints:
            return False
        horizon = float(
            self.predictive_junction_lookahead_m
            if distance_m is None else max(0.0, float(distance_m))
        )
        try:
            s0 = float(self.route_tracker.last_s)
            i0 = int(max(0, self.route_tracker.last_idx - 1))
            for i in range(i0, len(self.route_waypoints)):
                if i < len(self.route_tracker.s):
                    ds = float(self.route_tracker.s[i] - s0)
                    if ds > horizon:
                        break
                    if ds >= -2.0 and bool(getattr(self.route_waypoints[i], "is_junction", False)):
                        return True
        except Exception:
            return False
        return False

    @staticmethod
    def _obb_overlap_2d(center_a: Tuple[float, float], yaw_a: float,
                        half_a: Tuple[float, float],
                        center_b: Tuple[float, float], yaw_b: float,
                        half_b: Tuple[float, float]) -> bool:
        """Separating-axis test for two oriented rectangles in the XY plane."""
        ca, sa = math.cos(float(yaw_a)), math.sin(float(yaw_a))
        cb, sb = math.cos(float(yaw_b)), math.sin(float(yaw_b))
        axes_a = ((ca, sa), (-sa, ca))
        axes_b = ((cb, sb), (-sb, cb))
        delta = (float(center_b[0] - center_a[0]), float(center_b[1] - center_a[1]))

        for ax in (axes_a[0], axes_a[1], axes_b[0], axes_b[1]):
            center_sep = abs(delta[0] * ax[0] + delta[1] * ax[1])
            radius_a = (
                float(half_a[0]) * abs(axes_a[0][0] * ax[0] + axes_a[0][1] * ax[1])
                + float(half_a[1]) * abs(axes_a[1][0] * ax[0] + axes_a[1][1] * ax[1])
            )
            radius_b = (
                float(half_b[0]) * abs(axes_b[0][0] * ax[0] + axes_b[0][1] * ax[1])
                + float(half_b[1]) * abs(axes_b[1][0] * ax[0] + axes_b[1][1] * ax[1])
            )
            if center_sep > radius_a + radius_b:
                return False
        return True

    def _predict_ego_footprint_pose(self, t_future: float, planned_speed: float,
                                    route_sweep: bool,
                                    ego_xy: Tuple[float, float], ego_yaw: float,
                                    ego_velocity: Tuple[float, float]
                                    ) -> Tuple[Tuple[float, float], float]:
        """Predict an ego pose without mutating the route tracker."""
        t_future = max(0.0, float(t_future))
        if route_sweep and self.route_tracker is not None:
            try:
                s = float(self.progress_m) + max(0.0, float(planned_speed)) * t_future
                x, y, yaw = self.route_tracker.pose_at_s(s)
                # Preserve the measured route offset initially, then converge
                # smoothly to the selected route centerline.
                lat = float(getattr(self, "_route_lat_error", 0.0))
                decay = math.exp(-t_future / 1.2)
                x += (-math.sin(yaw)) * lat * decay
                y += math.cos(yaw) * lat * decay
                return (float(x), float(y)), float(yaw)
            except Exception:
                pass
        return (
            float(ego_xy[0] + ego_velocity[0] * t_future),
            float(ego_xy[1] + ego_velocity[1] * t_future),
        ), float(ego_yaw)

    def _scan_predictive_vehicle_conflicts(self, ego_speed: float,
                                           accel_hint: float = 0.0) -> bool:
        """Predict cut-in and junction conflicts using velocity-swept OBBs.

        The scan intentionally excludes ordinary aligned same-lane leaders;
        those are handled by the more accurate front-gap guard.  It includes
        adjacent-lane, diagonal, and perpendicular vehicles and therefore
        closes the two gaps left by lane-membership-only filtering.
        """
        self._predictive_conflict_guard_active = False
        self._predictive_conflict_accel_cap = 2.5
        if not self.predictive_collision_guard or self.ego is None:
            self._predictive_conflict_raw = False
            self._predictive_conflict_active = False
            return False

        try:
            ego_tf = self.ego.get_transform()
            ego_loc = ego_tf.location
            ego_yaw = math.radians(float(ego_tf.rotation.yaw))
            ego_vel = self.ego.get_velocity()
            ego_velocity = (float(ego_vel.x), float(ego_vel.y))
            if math.hypot(*ego_velocity) < 0.10:
                ego_velocity = (
                    max(0.0, float(ego_speed)) * math.cos(ego_yaw),
                    max(0.0, float(ego_speed)) * math.sin(ego_yaw),
                )
            ego_extent = self.ego.bounding_box.extent
            ego_half = (
                float(ego_extent.x) + self.predictive_longitudinal_margin_m,
                float(ego_extent.y) + self.predictive_lateral_margin_m,
            )
            ego_wp = self.map.get_waypoint(
                ego_loc, project_to_road=True,
                lane_type=self._carla.LaneType.Driving,
            )
            lane_half = max(1.0, 0.5 * float(getattr(ego_wp, "lane_width", 3.5)))
            at_junction = bool(getattr(ego_wp, "is_junction", False))
        except Exception as exc:
            self._record_safety_exception("predictive_conflict_ego", exc)
            return bool(getattr(self, "_predictive_conflict_active", False))

        junction_context = bool(
            at_junction
            or self._route_has_junction_ahead(self.predictive_junction_lookahead_m)
            or float(getattr(self, "_route_turn_strength", 0.0)) > 0.08
        )
        self._predictive_junction_context = junction_context
        overtake_active = str(getattr(self, "_overtake_mode", "idle")) != "idle"
        route_sweep = bool(junction_context and not overtake_active)

        planned_speed = max(0.0, float(ego_speed))
        deliberate_stop = bool(
            bool(getattr(self, "_red_light_queue_active", False))
            or (
                bool(getattr(self, "_traffic_light_active", False))
                and float(getattr(self, "_traffic_light_stop_error", 999.0)) < 8.0
            )
        )
        if junction_context and not deliberate_stop:
            planned_speed = max(planned_speed, self.predictive_junction_preview_speed)

        fwd = (math.cos(ego_yaw), math.sin(ego_yaw))
        right = (-fwd[1], fwd[0])
        best: Optional[Dict[str, Any]] = None
        try:
            vehicles = list(self.world.get_actors().filter("vehicle.*"))
        except Exception as exc:
            self._record_safety_exception("predictive_conflict_scan", exc)
            return bool(getattr(self, "_predictive_conflict_active", False))

        times = np.arange(
            self.predictive_step_s,
            self.predictive_horizon_s + 0.5 * self.predictive_step_s,
            self.predictive_step_s,
            dtype=np.float64,
        )
        ego_id = int(self.ego.id)
        for actor in vehicles:
            try:
                if int(actor.id) == ego_id or not bool(actor.is_alive):
                    continue
                a_tf = actor.get_transform()
                a_loc = a_tf.location
                dx = float(a_loc.x - ego_loc.x)
                dy = float(a_loc.y - ego_loc.y)
                distance = math.hypot(dx, dy)
                if distance > self.predictive_vehicle_radius_m:
                    continue

                along0 = dx * fwd[0] + dy * fwd[1]
                lat0 = dx * right[0] + dy * right[1]
                a_yaw = math.radians(float(a_tf.rotation.yaw))
                heading_diff = abs(_wrap_pi(a_yaw - ego_yaw))
                a_vel = actor.get_velocity()
                av = (float(a_vel.x), float(a_vel.y))
                a_lat_vel = av[0] * right[0] + av[1] * right[1]
                a_extent = actor.bounding_box.extent
                a_half = (
                    float(a_extent.x) + self.predictive_longitudinal_margin_m,
                    float(a_extent.y) + self.predictive_lateral_margin_m,
                )

                same_lane = False
                try:
                    a_wp = self.map.get_waypoint(
                        a_loc, project_to_road=True,
                        lane_type=self._carla.LaneType.Driving,
                    )
                    same_lane = bool(
                        a_wp is not None and ego_wp is not None
                        and self._lane_key(a_wp) == self._lane_key(ego_wp)
                    )
                except Exception:
                    same_lane = False

                # Ordinary aligned leaders are already protected by the ACC and
                # front guard.  Keep actors with lateral motion because a TM
                # vehicle can begin a cut-in while still mapping to its old lane.
                if same_lane and heading_diff <= math.radians(25.0) and abs(a_lat_vel) < 0.25:
                    continue

                candidate = bool(
                    junction_context
                    or overtake_active
                    or heading_diff > math.radians(25.0)
                    or abs(lat0) > 0.45 * lane_half
                    or abs(a_lat_vel) >= 0.25
                )
                if not candidate:
                    continue

                conflict_t = None
                conflict_relative = (999.0, 999.0)
                for tau in times:
                    ego_center, ego_pred_yaw = self._predict_ego_footprint_pose(
                        float(tau), planned_speed, route_sweep,
                        (float(ego_loc.x), float(ego_loc.y)), ego_yaw,
                        ego_velocity,
                    )
                    actor_center = (
                        float(a_loc.x + av[0] * float(tau)),
                        float(a_loc.y + av[1] * float(tau)),
                    )
                    if self._obb_overlap_2d(
                        ego_center, ego_pred_yaw, ego_half,
                        actor_center, a_yaw, a_half,
                    ):
                        rdx = actor_center[0] - ego_center[0]
                        rdy = actor_center[1] - ego_center[1]
                        conflict_relative = (
                            rdx * math.cos(ego_pred_yaw) + rdy * math.sin(ego_pred_yaw),
                            rdx * (-math.sin(ego_pred_yaw)) + rdy * math.cos(ego_pred_yaw),
                        )
                        conflict_t = float(tau)
                        break

                if conflict_t is None:
                    continue

                # Pure rear approach: braking increases the closing speed and
                # must not be the response.  Side/crossing motion is retained.
                rear_only = bool(
                    along0 < -max(2.0, float(ego_extent.x))
                    and heading_diff <= math.radians(25.0)
                    and abs(a_lat_vel) < 0.25
                    and conflict_relative[0] < 0.0
                )
                if rear_only:
                    continue

                if heading_diff > math.radians(35.0):
                    conflict_kind = "junction_crossing" if junction_context else "crossing"
                elif abs(lat0) > 0.70 * lane_half or abs(a_lat_vel) >= 0.25:
                    conflict_kind = "cut_in"
                elif overtake_active:
                    conflict_kind = "lane_change"
                else:
                    conflict_kind = "dynamic"

                record = {
                    "ttc": float(conflict_t),
                    "distance": float(distance),
                    "actor_id": int(actor.id),
                    "actor_type": str(getattr(actor, "type_id", "vehicle.unknown")),
                    "kind": conflict_kind,
                }
                if best is None or (record["ttc"], record["distance"]) < (
                    best["ttc"], best["distance"]
                ):
                    best = record
            except Exception:
                continue

        raw = best is not None
        self._predictive_conflict_raw = bool(raw)
        if raw:
            self._predictive_conflict_active = True
            self._predictive_conflict_clear_ticks = 0
            self._predictive_conflict_actor_id = int(best["actor_id"])
            self._predictive_conflict_actor_type = str(best["actor_type"])
            self._predictive_conflict_kind = str(best["kind"])
            self._predictive_conflict_ttc = float(best["ttc"])
            self._predictive_conflict_distance = float(best["distance"])
        elif self._predictive_conflict_active:
            self._predictive_conflict_clear_ticks += 1
            if self._predictive_conflict_clear_ticks >= self.predictive_clear_ticks:
                self._predictive_conflict_active = False
        else:
            self._predictive_conflict_clear_ticks = min(
                self.predictive_clear_ticks,
                self._predictive_conflict_clear_ticks + 1,
            )

        if not self._predictive_conflict_active:
            self._predictive_conflict_actor_id = -1
            self._predictive_conflict_actor_type = "none"
            self._predictive_conflict_kind = "none"
            self._predictive_conflict_ttc = 999.0
            self._predictive_conflict_distance = 999.0
        return bool(self._predictive_conflict_active)

    def _apply_predictive_collision_guard(self, accel: float,
                                           ego_speed: float) -> float:
        """Monotone brake-only response to a predicted dynamic conflict."""
        active = self._scan_predictive_vehicle_conflicts(ego_speed, accel)
        if not active:
            return float(accel)

        ttc = float(getattr(self, "_predictive_conflict_ttc", 999.0))
        speed = max(0.0, float(ego_speed))
        if ttc <= self.predictive_ttc_hard:
            cap = -4.0 if speed > 0.15 else -1.5
        elif ttc <= self.predictive_ttc_soft:
            travel_to_conflict = max(
                0.50,
                speed * ttc - self.predictive_longitudinal_margin_m,
            )
            required = speed * speed / (2.0 * travel_to_conflict)
            cap = -float(np.clip(required + 0.25, 0.45, 4.0))
            if speed < 0.50:
                cap = min(cap, -0.60)
        else:
            cap = min(0.0, float(accel))

        self._predictive_conflict_guard_active = True
        self._predictive_conflict_accel_cap = float(cap)
        if str(getattr(self, "_overtake_mode", "idle")) != "idle":
            self._overtake_abort_requested = True
        return min(float(accel), float(cap))

    def _ensure_overtake_state(self) -> None:
        """Lazily initialize overtaking state without changing constructor APIs."""
        defaults = {
            "_overtake_mode": "idle",
            "_overtake_side": "none",
            "_overtake_source_lane_key": None,
            "_overtake_target_lane_key": None,
            "_overtake_blocked_actor_id": -1,
            "_overtake_blocked_ticks": 0,
            "_overtake_mode_ticks": 0,
            "_overtake_centered_ticks": 0,
            "_overtake_start_progress": 0.0,
            "_overtake_guard_active": False,
            "_overtake_abort_requested": False,
            "_overtake_target_lane_clear": False,
            "_overtake_corridor_valid": False,
            "_overtake_target_front_gap": 999.0,
            "_overtake_target_rear_gap": 999.0,
            "_overtake_target_rear_ttc": 999.0,
            "_overtake_target_front_ttc": 999.0,
            "_overtake_target_side_blocked": False,
            "_overtake_target_front_speed": 0.0,
            "_overtake_target_front_kind": "none",
            "_overtake_target_front_actor_id": -1,
            "_route_projection_rejected": False,
            "_route_projection_reject_reason": "",
            "_front_vehicle_actor_id": -1,
            "_barrier_geometry_source": "lane",
            "_reward_lane_lat_error": 0.0,
            "_reward_lane_heading_error": 0.0,
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)

        config_defaults = {
            "ego_overtake_enabled": True,
            "ego_overtake_wait_ticks": 40,
            "ego_overtake_trigger_gap": 16.0,
            "ego_overtake_front_speed_max": 0.7,
            "ego_overtake_min_front_gap": 18.0,
            "ego_overtake_min_rear_gap": 12.0,
            "ego_overtake_min_rear_ttc": 4.0,
            "ego_overtake_min_front_ttc": 3.5,
            "ego_overtake_min_side_gap": 4.0,
            "ego_overtake_pass_clearance": 10.0,
            "ego_overtake_lane_change_speed": 6.0,
            "ego_overtake_max_mode_ticks": 240,
            "ego_overtake_active_turn_threshold": 0.18,
        }
        for name, value in config_defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)



    def _reset_overtake_state(self) -> None:
        self._ensure_overtake_state()
        self._overtake_mode = "idle"
        self._overtake_side = "none"
        self._overtake_source_lane_key = None
        self._overtake_target_lane_key = None
        self._overtake_blocked_actor_id = -1
        self._overtake_blocked_ticks = 0
        self._overtake_mode_ticks = 0
        self._overtake_centered_ticks = 0
        self._overtake_start_progress = float(getattr(self, "progress_m", 0.0))
        self._overtake_guard_active = False
        self._overtake_abort_requested = False
        self._overtake_target_lane_clear = False
        self._overtake_corridor_valid = False
        self._overtake_target_front_gap = 999.0
        self._overtake_target_rear_gap = 999.0
        self._overtake_target_rear_ttc = 999.0
        self._overtake_target_front_ttc = 999.0
        self._overtake_target_side_blocked = False
        self._overtake_target_front_speed = 0.0
        self._overtake_target_front_kind = "none"
        self._overtake_target_front_actor_id = -1


    def _adjacent_lane_for_side(self, wp: Any, side: str) -> Optional[Any]:
        try:
            side = str(side).lower()
            lane_change = getattr(wp, "lane_change", self._carla.LaneChange.NONE)
            wanted = (
                self._carla.LaneChange.Left
                if side == "left" else self._carla.LaneChange.Right
            )
            both = self._carla.LaneChange.Both
            if lane_change not in (wanted, both):
                return None

            marking = (
                getattr(wp, "left_lane_marking", None)
                if side == "left" else getattr(wp, "right_lane_marking", None)
            )
            if marking is None:
                return None
            marking_change = getattr(marking, "lane_change", self._carla.LaneChange.NONE)
            if marking_change not in (wanted, both):
                return None
            solid_types = {
                self._carla.LaneMarkingType.Solid,
                self._carla.LaneMarkingType.SolidSolid,
                self._carla.LaneMarkingType.Curb,
                self._carla.LaneMarkingType.Grass,
            }
            if getattr(marking, "type", None) in solid_types:
                return None

            adj = wp.get_left_lane() if side == "left" else wp.get_right_lane()
            if adj is None:
                return None
            if adj.lane_type != self._carla.LaneType.Driving:
                return None
            # Same travel direction: yaw difference is more robust across road/section IDs.
            yaw0 = math.radians(float(wp.transform.rotation.yaw))
            yaw1 = math.radians(float(adj.transform.rotation.yaw))
            if abs(_wrap_pi(yaw1 - yaw0)) > math.radians(35.0):
                return None
            return adj
        except Exception:
            return None


    def _find_lane_waypoint_by_key(self, current_wp: Any,
                                   lane_key: Optional[Tuple[int, int, int]]) -> Optional[Any]:
        if current_wp is None or lane_key is None:
            return None
        frontier = [current_wp]
        seen = set()
        for _ in range(3):
            next_frontier = []
            for wp in frontier:
                if wp is None:
                    continue
                try:
                    key = self._lane_key(wp)
                    if key == lane_key:
                        return wp
                    marker = (int(wp.road_id), int(wp.section_id), int(wp.lane_id))
                    if marker in seen:
                        continue
                    seen.add(marker)
                    for adj in (wp.get_left_lane(), wp.get_right_lane()):
                        if adj is not None:
                            next_frontier.append(adj)
                except Exception:
                    continue
            frontier = next_frontier
        return None



    def _target_lane_clear(self, target_wp: Any,
                           ignore_actor_id: int = -1) -> Tuple[bool, float, float, float]:
        """Check target-lane front/rear gaps, rear TTC, and pedestrians.

        Vehicles use CARLA lane membership plus direction consistency. Pedestrians
        are checked geometrically in the target-lane frame because they do not have
        a driving-lane identity. Rear TTC is computed per actor before taking the
        minimum, avoiding the old cross-actor gap/TTC mix-up.
        """
        self._ensure_overtake_state()
        if target_wp is None or self.ego is None:
            return False, 0.0, 0.0, 0.0
        try:
            ego_loc = self.ego.get_location()
            ego_speed = max(0.0, self._vehicle_speed(self.ego))
            target_tf = target_wp.transform
            yaw = math.radians(float(target_tf.rotation.yaw))
            fwd = (math.cos(yaw), math.sin(yaw))
            right = (-fwd[1], fwd[0])
            target_key = self._lane_key(target_wp)
            lane_half = max(0.5 * float(target_wp.lane_width), 1.0)
            target_center = target_tf.location
            ego_half = float(getattr(self.ego.bounding_box.extent, "x", 2.0))
        except Exception as exc:
            self._record_safety_exception("overtake_target_lane_geometry", exc)
            return False, 0.0, 0.0, 0.0

        front_gap = 999.0
        rear_gap = 999.0
        rear_ttc = 999.0
        front_ttc = 999.0
        side_blocked = False
        nearest_front_speed = 0.0
        nearest_front_kind = "none"
        nearest_front_actor_id = -1
        nearest_front_open_vel = 0.0
        ego_id = int(self.ego.id)

        try:
            actors = list(self.world.get_actors())
        except Exception as exc:
            self._record_safety_exception("overtake_target_lane_scan", exc)
            return False, 0.0, 0.0, 0.0

        for actor in actors:
            try:
                aid = int(actor.id)
                if aid == ego_id or aid == int(ignore_actor_id) or not bool(actor.is_alive):
                    continue
                tid = str(getattr(actor, "type_id", ""))
                is_vehicle = tid.startswith("vehicle.")
                is_walker = tid.startswith("walker.")
                if not (is_vehicle or is_walker):
                    continue

                a_loc = actor.get_location()
                dx = float(a_loc.x - ego_loc.x)
                dy = float(a_loc.y - ego_loc.y)
                along = dx * fwd[0] + dy * fwd[1]
                lateral_to_target = (
                    (float(a_loc.x) - float(target_center.x)) * right[0]
                    + (float(a_loc.y) - float(target_center.y)) * right[1]
                )

                # Signed velocity along the TARGET lane direction.  Using the
                # actor's own forward speed here hid oncoming (wrong-way or
                # junction-diagonal) traffic: its forward speed is positive but
                # its along-lane velocity is negative, i.e. closing fast.
                try:
                    vel = actor.get_velocity()
                    along_vel = float(vel.x) * fwd[0] + float(vel.y) * fwd[1]
                except Exception:
                    along_vel = 0.0

                if is_vehicle:
                    a_wp = self.map.get_waypoint(
                        a_loc, project_to_road=True,
                        lane_type=self._carla.LaneType.Driving,
                    )
                    if a_wp is None:
                        continue
                    same_target_lane = self._lane_key(a_wp) == target_key
                    if not same_target_lane:
                        same_target_lane = bool(
                            int(a_wp.road_id) == int(target_wp.road_id)
                            and int(a_wp.lane_id) == int(target_wp.lane_id)
                            and abs(_wrap_pi(
                                math.radians(float(a_wp.transform.rotation.yaw)) - yaw
                            )) <= math.radians(25.0)
                        )
                    # Geometric occupancy: a vehicle mid-lane-change (e.g. one
                    # currently PASSING the ego through this lane) or near a
                    # junction can project onto another lane's waypoint while
                    # its body physically occupies the target-lane corridor
                    # right beside the ego.  Membership alone made exactly
                    # those passers invisible.  Curvature makes this local
                    # frame unreliable far away, so apply it only nearby,
                    # which is where the passing hazard lives.
                    geometric_overlap = bool(
                        abs(float(lateral_to_target)) <= lane_half + 0.9
                        and abs(float(along)) <= 15.0
                    )
                    if not same_target_lane and not geometric_overlap:
                        continue
                    actor_half = float(getattr(actor.bounding_box.extent, "x", 2.0))
                    actor_speed = max(0.0, float(along_vel))
                    kind = "vehicle"
                else:
                    # Pedestrians close to the target lane are genuine lane-change
                    # hazards even though map.get_waypoint may project them elsewhere.
                    if abs(float(lateral_to_target)) > lane_half + 0.80:
                        continue
                    actor_half = float(getattr(actor.bounding_box.extent, "x", 0.4))
                    vel = actor.get_velocity()
                    actor_speed = float(math.sqrt(
                        float(vel.x) ** 2 + float(vel.y) ** 2 + float(vel.z) ** 2
                    ))
                    kind = "walker"

                bumper_gap = abs(float(along)) - ego_half - actor_half
                # Hard no-go zone (user rule): while ANY vehicle occupies the
                # target lane alongside the ego or within min_side_gap (4 m)
                # fore/aft bumper-to-bumper, the lane change is forbidden.
                # Only once the passer has fully crossed and pulled at least
                # this far away may the change begin.
                if is_vehicle and bumper_gap < float(self.ego_overtake_min_side_gap):
                    side_blocked = True
                if along >= 0.0:
                    actor_front_gap = max(0.0, float(bumper_gap))
                    # Per-actor front TTC: how long until the ego, at its
                    # current speed, reaches an actor ahead in the target lane.
                    # A stopped car 19 m ahead or an oncoming car 30 m ahead
                    # both passed the old static gap threshold while being
                    # seconds from impact at lane-change speed.
                    closing_front = ego_speed - float(along_vel)
                    if closing_front > 1e-3:
                        front_ttc = min(front_ttc, actor_front_gap / closing_front)
                    if actor_front_gap < front_gap:
                        front_gap = actor_front_gap
                        nearest_front_speed = float(actor_speed)
                        nearest_front_kind = kind
                        nearest_front_actor_id = aid
                        nearest_front_open_vel = float(along_vel) - ego_speed
                elif is_vehicle:
                    actor_rear_gap = max(0.0, float(bumper_gap))
                    rear_gap = min(rear_gap, actor_rear_gap)
                    closing = float(along_vel) - ego_speed
                    if closing > 1e-3:
                        rear_ttc = min(rear_ttc, actor_rear_gap / closing)
            except Exception:
                continue

        # Pass-release rule: a vehicle that has just overtaken the ego is
        # ahead and pulling away.  Demanding the full static front gap (18 m)
        # would make the ego trail every passer for many extra seconds; the
        # requested behaviour is to allow the change once the passer has
        # crossed and is at least min_side_gap (4 m) clear.  The release only
        # applies while the lead is genuinely OPENING; if it brakes, the
        # opening velocity collapses, the full gap requirement returns, and
        # the per-tick re-check during the maneuver aborts or brakes.
        front_required = float(self.ego_overtake_min_front_gap)
        if nearest_front_kind == "vehicle" and nearest_front_open_vel >= 1.0:
            front_required = max(float(self.ego_overtake_min_side_gap), 4.0)

        clear = bool(
            front_gap >= front_required
            and front_ttc >= float(self.ego_overtake_min_front_ttc)
            and rear_gap >= float(self.ego_overtake_min_rear_gap)
            and rear_ttc >= float(self.ego_overtake_min_rear_ttc)
            and not side_blocked
        )
        self._overtake_target_lane_clear = clear
        self._overtake_target_front_gap = float(front_gap)
        self._overtake_target_rear_gap = float(rear_gap)
        self._overtake_target_rear_ttc = float(rear_ttc)
        self._overtake_target_front_ttc = float(front_ttc)
        self._overtake_target_side_blocked = bool(side_blocked)
        self._overtake_target_front_speed = float(nearest_front_speed)
        self._overtake_target_front_kind = str(nearest_front_kind)
        self._overtake_target_front_actor_id = int(nearest_front_actor_id)
        return clear, float(front_gap), float(rear_gap), float(rear_ttc)



    def _update_overtake_state(self, ego_wp: Any, ego_speed: float) -> None:
        """Conservative overtaking state machine for a persistent stopped vehicle."""
        self._ensure_overtake_state()
        if not bool(self.ego_overtake_enabled):
            self._reset_overtake_state()
            return

        mode = str(self._overtake_mode)
        if mode != "idle":
            self._overtake_mode_ticks = int(self._overtake_mode_ticks) + 1

        try:
            at_junction = bool(getattr(ego_wp, "is_junction", False))
        except Exception:
            at_junction = False
        active_conflict = bool(
            at_junction
            or self._route_has_junction_ahead(self.predictive_junction_lookahead_m)
            or float(getattr(self, "_route_turn_strength", 0.0))
                > float(self.ego_overtake_active_turn_threshold)
            or bool(getattr(self, "_traffic_light_active", False))
            or bool(getattr(self, "_red_light_queue_active", False))
            or self._distance_to_destination() < 35.0
        )
        current_key = self._lane_key(ego_wp) if ego_wp is not None else None

        if mode == "idle":
            kind = str(getattr(self, "_front_vehicle_kind", "none")).lower()
            front_gap = float(getattr(self, "_front_vehicle_gap", 80.0))
            front_speed = float(getattr(self, "_front_vehicle_speed", 0.0))
            valid = bool(getattr(self, "_front_vehicle_guard_valid", False))
            is_vehicle = not (
                "walker" in kind or "pedestrian" in kind
                or kind in {"none", "red_light_priority"}
            )
            blocked = bool(
                valid and is_vehicle and not active_conflict
                and 2.0 < front_gap <= float(self.ego_overtake_trigger_gap)
                and front_speed <= float(self.ego_overtake_front_speed_max)
            )
            self._overtake_blocked_ticks = (
                self._overtake_blocked_ticks + 1 if blocked else 0
            )
            if self._overtake_blocked_ticks < int(self.ego_overtake_wait_ticks):
                return

            for side in ("left", "right"):
                target_wp = self._adjacent_lane_for_side(ego_wp, side)
                if target_wp is None:
                    continue
                clear, _, _, _ = self._target_lane_clear(target_wp)
                if not clear:
                    continue
                self._overtake_mode = "changing_out"
                self._overtake_side = side
                self._overtake_source_lane_key = current_key
                self._overtake_target_lane_key = self._lane_key(target_wp)
                self._overtake_blocked_actor_id = int(
                    getattr(self, "_front_vehicle_actor_id", -1)
                )
                self._overtake_start_progress = float(self.progress_m)
                self._overtake_mode_ticks = 0
                self._overtake_centered_ticks = 0
                self._overtake_abort_requested = False
                self._overtake_guard_active = True
                print(
                    f"[carla-safe] overtake=start side={side} "
                    f"actor={self._overtake_blocked_actor_id} gap={front_gap:.1f}m",
                    flush=True,
                )
                return
            return

        if self._overtake_mode_ticks > int(self.ego_overtake_max_mode_ticks):
            self._overtake_abort_requested = True
        if active_conflict:
            self._overtake_abort_requested = True
            if mode == "changing_out" and current_key == self._overtake_source_lane_key:
                self._reset_overtake_state()
                return

        if mode == "changing_out":
            target_wp = self._find_lane_waypoint_by_key(
                ego_wp, self._overtake_target_lane_key
            )
            clear, _, _, _ = (
                self._target_lane_clear(target_wp)
                if target_wp is not None else (False, 0.0, 0.0, 0.0)
            )
            if not clear and current_key == self._overtake_source_lane_key:
                self._reset_overtake_state()
                return
            if current_key == self._overtake_target_lane_key:
                self._overtake_centered_ticks += 1
            else:
                self._overtake_centered_ticks = 0
            if self._overtake_centered_ticks >= 5:
                self._overtake_mode = "passing"
                self._overtake_mode_ticks = 0
                self._overtake_centered_ticks = 0
            return

        if mode == "passing":
            passed = False
            actor = None
            try:
                if self._overtake_blocked_actor_id >= 0:
                    actor = self.world.get_actor(int(self._overtake_blocked_actor_id))
            except Exception:
                actor = None
            if actor is None:
                passed = self._overtake_mode_ticks >= 25
            else:
                try:
                    ego_tf = self.ego.get_transform()
                    ego_loc = ego_tf.location
                    a_loc = actor.get_location()
                    fwd = _carla_vec_forward(ego_tf)
                    along = (
                        (a_loc.x - ego_loc.x) * fwd[0]
                        + (a_loc.y - ego_loc.y) * fwd[1]
                    )
                    passed = bool(
                        along <= -float(self.ego_overtake_pass_clearance)
                    )
                except Exception:
                    passed = False

            if passed or bool(self._overtake_abort_requested):
                source_wp = self._find_lane_waypoint_by_key(
                    ego_wp, self._overtake_source_lane_key
                )
                # Never ignore the overtaken vehicle here: after passing it becomes
                # a rear-approach hazard and must contribute to rear TTC.
                clear, _, _, _ = (
                    self._target_lane_clear(source_wp)
                    if source_wp is not None else (False, 0.0, 0.0, 0.0)
                )
                if clear:
                    self._overtake_mode = "returning"
                    self._overtake_mode_ticks = 0
                    self._overtake_centered_ticks = 0
            return

        if mode == "returning":
            source_wp = self._find_lane_waypoint_by_key(
                ego_wp, self._overtake_source_lane_key
            )
            clear, _, _, _ = (
                self._target_lane_clear(source_wp)
                if source_wp is not None else (False, 0.0, 0.0, 0.0)
            )
            if not clear and current_key != self._overtake_source_lane_key:
                # Stay centered in the passing lane until the source lane is safe.
                self._overtake_mode = "passing"
                self._overtake_abort_requested = True
                self._overtake_mode_ticks = 0
                self._overtake_centered_ticks = 0
                return
            if current_key == self._overtake_source_lane_key:
                self._overtake_centered_ticks += 1
            else:
                self._overtake_centered_ticks = 0
            if self._overtake_centered_ticks >= 5:
                print("[carla-safe] overtake=complete", flush=True)
                self._reset_overtake_state()



    def _overtake_corridor_geometry(self, ego_wp: Any, ego_loc: Any,
                                     ego_yaw_rad: float) -> Optional[Tuple[float, float, float]]:
        """Return legal two-lane outer-boundary geometry for an approved overtake."""
        self._ensure_overtake_state()
        if str(self._overtake_mode) == "idle" or ego_wp is None:
            return None
        source_wp = self._find_lane_waypoint_by_key(
            ego_wp, self._overtake_source_lane_key
        )
        target_wp = self._find_lane_waypoint_by_key(
            ego_wp, self._overtake_target_lane_key
        )
        if source_wp is None or target_wp is None:
            return None
        try:
            s_loc = source_wp.transform.location
            t_loc = target_wp.transform.location
            yaw_s = math.radians(float(source_wp.transform.rotation.yaw))
            yaw_t = math.radians(float(target_wp.transform.rotation.yaw))
            if abs(_wrap_pi(yaw_t - yaw_s)) > math.radians(35.0):
                return None
            center_sep = math.hypot(
                float(t_loc.x - s_loc.x), float(t_loc.y - s_loc.y)
            )
            max_adjacent_sep = 0.75 * (
                float(source_wp.lane_width) + float(target_wp.lane_width)
            )
            if center_sep > max(max_adjacent_sep, 5.5):
                return None
            cx = 0.5 * (float(s_loc.x) + float(t_loc.x))
            cy = 0.5 * (float(s_loc.y) + float(t_loc.y))
            right = source_wp.transform.get_right_vector()
            lat = (
                (float(ego_loc.x) - cx) * float(right.x)
                + (float(ego_loc.y) - cy) * float(right.y)
            )
            heading = _wrap_pi(float(ego_yaw_rad) - yaw_s)
            corridor_half = 0.5 * (
                float(source_wp.lane_width) + float(target_wp.lane_width)
            )
            corridor_half = max(corridor_half, 2.5)
            return float(lat), float(heading), float(corridor_half)
        except Exception:
            return None



    def _apply_overtake_guard(self, steer_env: float, accel: float,
                              ego_speed: float) -> Tuple[float, float]:
        """Steer an approved lane change without ever releasing safety braking."""
        self._ensure_overtake_state()
        self._overtake_guard_active = False
        mode = str(self._overtake_mode)
        if mode == "idle" or self.ego is None:
            return float(steer_env), float(accel)
        try:
            carla = self._carla
            ego_tf = self.ego.get_transform()
            ego_loc = ego_tf.location
            current_wp = self.map.get_waypoint(
                ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if current_wp is None:
                return float(steer_env), float(min(accel, -1.0))

            desired_key = (
                self._overtake_source_lane_key
                if mode == "returning" else self._overtake_target_lane_key
            )
            desired_wp = self._find_lane_waypoint_by_key(current_wp, desired_key)
            if desired_wp is None:
                self._overtake_corridor_valid = False
                return float(steer_env), float(min(accel, -1.0))

            lane_clear, _, _, _ = self._target_lane_clear(desired_wp)
            current_key = self._lane_key(current_wp)
            if mode == "changing_out" and not lane_clear:
                if current_key == self._overtake_source_lane_key:
                    self._reset_overtake_state()
                    return float(steer_env), float(min(accel, -1.0 if ego_speed > 0.5 else accel))
                # Already committed laterally: continue centering conservatively
                # but brake in proportion to the ACTUAL threat in the
                # destination lane.  The previous fixed -1.5 m/s^2 cap was far
                # too weak against a close or fast-closing vehicle; scale up to
                # full braking using the front gap/TTC just measured by
                # _target_lane_clear on this tick.  Rear-only threats (a fast
                # passer approaching from behind) must NOT trigger braking:
                # slowing down in front of a closing vehicle worsens it, and
                # completing the centering is the fastest way out.
                front_gap_t = float(getattr(self, "_overtake_target_front_gap", 999.0))
                front_ttc_t = float(getattr(self, "_overtake_target_front_ttc", 999.0))
                side_blocked_t = bool(getattr(self, "_overtake_target_side_blocked", False))
                front_threat = bool(
                    front_gap_t < float(self.ego_overtake_min_front_gap)
                    or front_ttc_t < float(self.ego_overtake_min_front_ttc)
                )
                if front_threat or side_blocked_t:
                    if front_ttc_t < 2.0 or front_gap_t < 5.0:
                        brake_cap = -4.0
                    elif front_ttc_t < 3.5 or front_gap_t < 9.0:
                        brake_cap = -2.8
                    else:
                        brake_cap = -1.5
                    accel = min(float(accel), brake_cap if ego_speed > 0.5 else -0.3)
            elif mode == "returning" and not lane_clear:
                self._overtake_mode = "passing"
                self._overtake_abort_requested = True
                desired_key = self._overtake_target_lane_key
                desired_wp = self._find_lane_waypoint_by_key(current_wp, desired_key)
                if desired_wp is None:
                    return float(steer_env), float(min(accel, -1.0))
                accel = min(float(accel), -1.0 if ego_speed > 0.5 else -0.3)

            lookahead = list(desired_wp.next(8.0))
            if lookahead:
                ego_yaw = math.radians(float(ego_tf.rotation.yaw))
                desired_wp = min(
                    lookahead,
                    key=lambda w: abs(_wrap_pi(
                        math.radians(float(w.transform.rotation.yaw)) - ego_yaw
                    )),
                )
            dloc = desired_wp.transform.location
            right = desired_wp.transform.get_right_vector()
            lat_err = (
                (float(ego_loc.x) - float(dloc.x)) * float(right.x)
                + (float(ego_loc.y) - float(dloc.y)) * float(right.y)
            )
            heading_err = _wrap_pi(
                math.radians(
                    float(ego_tf.rotation.yaw)
                    - float(desired_wp.transform.rotation.yaw)
                )
            )
            steer_target = float(np.clip(
                -0.42 * lat_err - 0.95 * heading_err, -0.60, 0.60
            ))
            blend = 0.80 if mode in {"changing_out", "returning"} else 0.55
            steer_out = float(np.clip(
                (1.0 - blend) * float(steer_env) + blend * steer_target,
                -0.60, 0.60,
            ))

            speed_cap = float(self.ego_overtake_lane_change_speed)
            accel_cap = 0.75 * (speed_cap - max(0.0, float(ego_speed)))
            # Monotone safety rule: overtaking may only preserve or reduce the
            # incoming acceleration. It can never turn a brake command positive.
            accel_out = min(float(accel), float(accel_cap))

            active_conflict = bool(
                getattr(current_wp, "is_junction", False)
                or self._route_has_junction_ahead(self.predictive_junction_lookahead_m)
                or float(getattr(self, "_route_turn_strength", 0.0))
                    > float(self.ego_overtake_active_turn_threshold)
                or bool(getattr(self, "_traffic_light_active", False))
                or bool(getattr(self, "_red_light_queue_active", False))
            )
            if active_conflict:
                self._overtake_abort_requested = True
                # The old fixed negative cap could park the ego in the passing
                # lane while the source lane remained occupied.  Request a
                # return, but preserve motion at a junction-safe speed; the
                # red-light/front/predictive guards above still have priority
                # and retain any genuine brake command.
                conflict_speed = min(float(self.route_turn_speed), 4.0)
                speed_cap_accel = float(np.clip(
                    0.65 * (conflict_speed - max(0.0, float(ego_speed))),
                    -2.0, 0.8,
                ))
                accel_out = min(accel_out, speed_cap_accel)

            self._overtake_guard_active = True
            return steer_out, float(accel_out)
        except Exception as exc:
            self._record_safety_exception("overtake_guard", exc)
            return float(steer_env), float(min(
                accel, -1.0 if ego_speed > 0.5 else -0.3
            ))

    def _read_state(self) -> np.ndarray:
        carla = self._carla
        tf = self.ego.get_transform()
        loc = tf.location
        wp = self.map.get_waypoint(loc, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        wp_loc = wp.transform.location
        right = wp.transform.get_right_vector()
        lat = float((loc.x - wp_loc.x) * right.x + (loc.y - wp_loc.y) * right.y)
        heading = _wrap_pi(math.radians(tf.rotation.yaw - wp.transform.rotation.yaw))
        speed = max(0.0, self._vehicle_speed(self.ego))
        lane_half = max(0.5 * float(wp.lane_width), 1.0)

        # Update route progress BEFORE route-based hazard checks. Previously,
        # traffic-light and lead distances used the previous tick's progress,
        # which could make the ego stop visibly too far from the stop line.
        # Keep map-projected geometry as the default. Around Town10HD
        # junctions CARLA's nearest-lane projection can select an adjacent or
        # crossing lane, so route-centerline geometry may replace it below.
        route_lat_now = float(lat)
        route_heading_error = float(heading)
        route_curvature = 0.0
        progress_frac = 0.0
        remaining_frac = 1.0
        if self.route_tracker is not None:
            if int(getattr(self, "_progress_tick_t", -1)) == int(self.t):
                # _update_progress already projected on this tick: reuse its
                # result instead of opening a second forward-snap window.
                route_lat_now = float(self._route_lat_error)
                route_yaw_now = float(getattr(
                    self, "_route_yaw_now",
                    math.radians(tf.rotation.yaw) - float(self._route_yaw_error)))
            else:
                s_now, route_lat_now, route_yaw_now = self.route_tracker.project(
                    loc, self.route_search_back, self.route_search_forward)
                self.progress_m = max(self.progress_m, float(s_now))
                self._route_lat_error = route_lat_now
            ego_yaw_rad = math.radians(tf.rotation.yaw)
            route_heading_error = _wrap_pi(ego_yaw_rad - route_yaw_now)
            self._route_yaw_error = route_heading_error

            lookahead_yaw = self.route_tracker.lookahead_yaw(self.route_turn_lookahead_m)
            self._route_lookahead_yaw_error = _wrap_pi(ego_yaw_rad - lookahead_yaw)
            self._route_turn_strength = abs(_wrap_pi(lookahead_yaw - route_yaw_now))

            route_curvature = self.route_tracker.curvature_ahead(lookahead_segments=15)
            self._route_curvature = route_curvature
            if self.route_distance > 0:
                goal_distance = max(float(getattr(self, "_goal_distance_m", self.route_distance)), 1e-6)
                progress_frac = float(np.clip(self.progress_m / goal_distance, 0.0, 1.5))
                remaining_frac = float(np.clip((goal_distance - self.progress_m) /
                                               goal_distance, 0.0, 1.5))

        # Junction-aware observation geometry.
        #
        # map.get_waypoint(..., project_to_road=True) returns the nearest
        # driving-lane center. In a complex junction that nearest lane may be an
        # adjacent/crossing branch rather than the branch of our precomputed
        # route. The controller, barrier diagnostics, and termination logic all
        # consume state[0:2], so they must use the same continuity-safe route
        # geometry during a planned turn.
        try:
            at_junction = bool(getattr(wp, "is_junction", False))
        except Exception:
            at_junction = False

        planned_turn = float(getattr(self, "_route_turn_strength", 0.0))
        route_sane = self._route_geometry_is_sane(
            route_lat_now, route_heading_error, lane_half
        )
        use_route_geometry = bool(
            self.route_tracker is not None
            and route_sane
            and (at_junction or planned_turn > 0.08)
        )
        self._route_projection_rejected = bool(
            self.route_tracker is not None and not route_sane
        )
        self._route_projection_reject_reason = (
            f"lat={route_lat_now:.2f},heading={route_heading_error:.2f}"
            if self._route_projection_rejected else ""
        )

        if use_route_geometry:
            state_lat = float(route_lat_now)
            state_heading = float(route_heading_error)
            self._state_geometry_source = "route"
        else:
            state_lat = float(lat)
            state_heading = float(heading)
            self._state_geometry_source = "map"

        self._map_lane_lat_error = float(lat)
        self._map_lane_yaw_error = float(heading)

        # Lane-aware route-corridor hazard detection.
        # We consider the designated lead car, Traffic Manager vehicles, and
        # walkers. Only actors ahead along the ego route and close to the route
        # corridor become headway/obstacle hazards.
        self._lead_is_relevant = False
        self._lead_same_lane = False
        self._lead_route_lat = 999.0
        self._lead_route_ds = 999.0
        self._lead_actor_kind = "none"
        self._traffic_light_active = False
        self._traffic_light_state = "none"
        self._traffic_light_distance = 999.0
        self._traffic_light_virtual_gap = 999.0
        gap, rel_vel, obstacle = 80.0, 0.0, 80.0

        ego_half = float(self.ego.bounding_box.extent.x)
        candidates = self._collect_dynamic_hazard_candidates()

        # During an active lane change the ego is about to occupy (or leave
        # through) a specific adjacent lane.  Vehicles in that DESTINATION lane
        # must be braking-relevant leads: the same-lane test still points at
        # the source lane, the route corridor still follows the original route
        # (a destination-lane car sits ~3.5 m route-lateral), and the tight
        # ego-frame corridor only admits the car once the ego is nearly on top
        # of it.  All three filters rejected exactly the vehicle the ego was
        # steering into, which is why lane changes collided while same-lane
        # following worked.
        overtake_mode = str(getattr(self, "_overtake_mode", "idle"))
        if overtake_mode in ("changing_out", "passing"):
            maneuver_lane_key = getattr(self, "_overtake_target_lane_key", None)
        elif overtake_mode == "returning":
            maneuver_lane_key = getattr(self, "_overtake_source_lane_key", None)
        else:
            maneuver_lane_key = None

        best = None
        for actor, kind in candidates:
            try:
                a_tf = actor.get_transform()
                a_loc = a_tf.location
            except Exception:
                continue

            same_lane = False
            in_maneuver_lane = False
            if kind != "walker":
                try:
                    a_wp = self.map.get_waypoint(a_loc, project_to_road=True,
                                                 lane_type=carla.LaneType.Driving)
                    same_lane = bool(
                        a_wp.road_id == wp.road_id and
                        a_wp.section_id == wp.section_id and
                        a_wp.lane_id == wp.lane_id
                    )
                    if maneuver_lane_key is not None and a_wp is not None:
                        in_maneuver_lane = bool(
                            self._lane_key(a_wp) == maneuver_lane_key
                        )
                except Exception:
                    same_lane = False
                    in_maneuver_lane = False

            try:
                actor_half = float(actor.bounding_box.extent.x)
            except Exception:
                actor_half = 0.4 if kind == "walker" else 2.0

            relevant = False
            raw_gap = 999.0
            route_lat = 999.0
            route_ds = 999.0

            # Ego-frame geometry is a second safety net: it catches a car directly
            # in front even if route projection or CARLA lane ids are unreliable
            # around Town10HD junctions.
            fwd_ego = _carla_vec_forward(tf)
            # CARLA right vector: (-sin(yaw), cos(yaw)); consistent with the
            # route tracker and _target_lane_clear.
            right_ego = (-fwd_ego[1], fwd_ego[0])
            dx_ego = float(a_loc.x - loc.x)
            dy_ego = float(a_loc.y - loc.y)
            along_ego = dx_ego * fwd_ego[0] + dy_ego * fwd_ego[1]
            lat_ego = dx_ego * right_ego[0] + dy_ego * right_ego[1]
            ego_gap = float(along_ego) - ego_half - actor_half

            if self.route_tracker is not None:
                actor_s, route_lat, _, _ = self.route_tracker.query(
                    a_loc, center_idx=self.route_tracker.last_idx,
                    back=self.route_search_back + 20,
                    forward=self.route_search_forward + 160)
                route_ds = float(actor_s - self.progress_m)
                route_gap = route_ds - ego_half - actor_half
                if kind == "walker":
                    corridor_width = max(lane_half + 1.2, 2.5)
                    route_corridor = abs(route_lat) <= corridor_width
                    ego_corridor = abs(lat_ego) <= max(lane_half + 0.8, 2.2)
                    relevant = bool(
                        (route_ds > 0.0 and route_corridor and route_gap > 0.0) or
                        (along_ego > 0.0 and ego_corridor and ego_gap > 0.0 and along_ego < 30.0)
                    )
                    raw_gap = min(route_gap if route_gap > 0.0 else 999.0,
                                  ego_gap if ego_gap > 0.0 else 999.0)
                else:
                    # Vehicle hazards should be in ego's lane. Same-lane CARLA
                    # IDs are trusted when available; near junctions, accept only
                    # a tight route or ego-frame corridor. This rejects adjacent
                    # lane cars but keeps stopped/queued cars directly ahead.
                    # During an active lane change, vehicles ahead in the
                    # maneuver's destination lane are additionally relevant.
                    strict_corridor = abs(route_lat) <= min(
                        max(float(self.vehicle_route_corridor_factor) * lane_half, 0.65),
                        float(self.vehicle_route_corridor_max))
                    ego_corridor = abs(lat_ego) <= min(max(0.45 * lane_half, 0.65),
                                                       float(self.vehicle_route_corridor_max))
                    maneuver_relevant = bool(
                        in_maneuver_lane and along_ego > 0.0 and ego_gap > 0.0
                        and along_ego < 45.0
                        and abs(lat_ego) <= 2.0 * lane_half + 0.8
                    )
                    if same_lane:
                        relevant = bool(route_ds > 0.0 and route_gap > 0.0)
                        raw_gap = route_gap
                    elif maneuver_relevant:
                        relevant = True
                        raw_gap = min(route_gap if route_gap > 0.0 else 999.0,
                                      ego_gap if ego_gap > 0.0 else 999.0)
                    else:
                        relevant = bool(
                            (route_ds > 0.0 and strict_corridor and route_gap > 0.0) or
                            (along_ego > 0.0 and ego_corridor and ego_gap > 0.0 and along_ego < 35.0)
                        )
                        raw_gap = min(route_gap if route_gap > 0.0 else 999.0,
                                      ego_gap if ego_gap > 0.0 else 999.0)
            elif same_lane or in_maneuver_lane:
                raw_gap = ego_gap
                relevant = bool(
                    raw_gap > 0.0
                    and (same_lane
                         or (along_ego > 0.0
                             and abs(lat_ego) <= 2.0 * lane_half + 0.8))
                )

            if not relevant:
                continue

            # --------------------------------------------------------------
            # CRITICAL FIX: validate each candidate BEFORE selecting the
            # nearest actor. Previously, an invalid cross-street/adjacent actor
            # could win the nearest-candidate race, then fail the final guard
            # validity check, while still leaving state[3] with a short gap.
            # That short gap could trigger the outer ACC prior and create a
            # phantom stop on an otherwise clear ego lane.
            # --------------------------------------------------------------
            try:
                a_yaw = math.radians(actor.get_transform().rotation.yaw)
                e_yaw = math.radians(tf.rotation.yaw)
                heading_diff = abs(_wrap_pi(a_yaw - e_yaw))
            except Exception:
                heading_diff = 0.0

            kind_lower = str(kind).lower()
            if "walker" in kind_lower or "pedestrian" in kind_lower:
                route_ok = bool(
                    float(route_ds) > 0.0
                    and abs(float(route_lat)) <= max(lane_half + 1.2, 2.5)
                )
                ego_ok = bool(
                    float(along_ego) > 0.0
                    and abs(float(lat_ego)) <= max(lane_half + 0.8, 2.2)
                )
                candidate_guard_valid = bool(route_ok or ego_ok)
            else:
                strict_corridor = min(
                    max(float(self.vehicle_route_corridor_factor) * lane_half, 0.65),
                    float(self.vehicle_route_corridor_max),
                )
                ego_corridor = min(
                    max(0.45 * lane_half, 0.65),
                    float(self.vehicle_route_corridor_max),
                )
                same_direction = bool(heading_diff <= math.radians(30.0))
                route_ok = bool(
                    float(route_ds) > 0.0
                    and abs(float(route_lat)) <= strict_corridor
                )
                ego_ok = bool(
                    0.0 < float(along_ego) < 35.0
                    and abs(float(lat_ego)) <= ego_corridor
                )
                # A vehicle ahead in the lane the ego is actively moving into
                # is braking-relevant irrespective of its heading: oncoming or
                # junction-diagonal traffic in the destination lane closes
                # fastest of all.
                maneuver_ok = bool(
                    in_maneuver_lane
                    and 0.0 < float(along_ego) < 45.0
                    and abs(float(lat_ego)) <= 2.0 * lane_half + 0.8
                )
                candidate_guard_valid = bool(
                    (same_direction and (bool(same_lane) or route_ok or ego_ok))
                    or maneuver_ok
                )

            if not candidate_guard_valid:
                continue

            if best is None or raw_gap < best[0]:
                try:
                    if kind == "walker":
                        vv = actor.get_velocity()
                        actor_speed = float(math.sqrt(vv.x * vv.x + vv.y * vv.y + vv.z * vv.z))
                    else:
                        actor_speed = max(0.0, self._vehicle_speed(actor))
                except Exception:
                    actor_speed = 0.0

                best = (
                    float(raw_gap), actor_speed, same_lane, float(route_lat),
                    float(route_ds), kind, float(heading_diff),
                    float(lat_ego), float(along_ego), int(actor.id),
                    bool(in_maneuver_lane),
                )

        if best is not None:
            gap = float(min(best[0], 80.0))
            lead_speed = float(best[1])
            rel_vel = lead_speed - speed
            obstacle = gap
            self._lead_is_relevant = True
            self._lead_same_lane = bool(best[2])
            self._lead_route_lat = float(best[3])
            self._lead_route_ds = float(best[4])
            self._lead_actor_kind = str(best[5])
            self._front_vehicle_heading_diff = float(best[6])
            self._front_vehicle_actor_id = int(best[9])

            # Validate the raw actor independently of the policy-state lead flag.
            # Vehicles must be same-lane or tightly aligned with the route/ego
            # corridor and travel in approximately the same direction. Walkers
            # may be directionless but must still be geometrically ahead.
            best_kind = str(best[5]).lower()
            best_heading_diff = float(best[6])
            best_lat_ego = float(best[7])
            best_along_ego = float(best[8])
            if "walker" in best_kind or "pedestrian" in best_kind:
                route_ok = bool(
                    float(best[4]) > 0.0
                    and abs(float(best[3])) <= max(lane_half + 1.2, 2.5)
                )
                ego_ok = bool(
                    best_along_ego > 0.0
                    and abs(best_lat_ego) <= max(lane_half + 0.8, 2.2)
                )
                self._front_vehicle_guard_valid = bool(route_ok or ego_ok)
            else:
                strict_corridor = min(
                    max(float(self.vehicle_route_corridor_factor) * lane_half, 0.65),
                    float(self.vehicle_route_corridor_max),
                )
                ego_corridor = min(
                    max(0.45 * lane_half, 0.65),
                    float(self.vehicle_route_corridor_max),
                )
                same_direction = best_heading_diff <= math.radians(30.0)
                route_ok = bool(float(best[4]) > 0.0 and abs(float(best[3])) <= strict_corridor)
                ego_ok = bool(
                    0.0 < best_along_ego < 35.0
                    and abs(best_lat_ego) <= ego_corridor
                )
                maneuver_ok = bool(
                    bool(best[10])
                    and 0.0 < best_along_ego < 45.0
                    and abs(best_lat_ego) <= 2.0 * lane_half + 0.8
                )
                self._front_vehicle_guard_valid = bool(
                    (same_direction and (bool(best[2]) or route_ok or ego_ok))
                    or maneuver_ok
                )
        else:
            self._front_vehicle_heading_diff = 0.0
            self._front_vehicle_guard_valid = False
            self._front_vehicle_actor_id = -1

        # --------------------------------------------------------------
        # FINAL INVARIANT: no actor that failed the explicit front-guard
        # geometry validation may leak into policy state[3:6], reward shaping,
        # barrier headway, or the outer ACC prior.  Keep this safety net even
        # though candidates are now validated before nearest selection.
        # --------------------------------------------------------------
        if best is not None and not bool(
            getattr(self, "_front_vehicle_guard_valid", False)
        ):
            gap, rel_vel, obstacle = 80.0, 0.0, 80.0
            self._lead_is_relevant = False
            self._lead_same_lane = False
            self._lead_route_lat = 999.0
            self._lead_route_ds = 999.0
            self._lead_actor_kind = "none"
            self._front_vehicle_heading_diff = 0.0
            self._front_vehicle_actor_id = -1

        # Store the raw same-route front actor before any red-light priority
        # suppresses state[3]. The low-level collision shield uses this to avoid
        # rear-ending spawned/NPC vehicles even when red-light logic is active.
        self._front_vehicle_gap = float(gap)
        self._front_vehicle_speed = float(max(0.0, speed + rel_vel))
        self._front_vehicle_kind = str(self._lead_actor_kind)
        closing_for_guard = max(0.0, speed - self._front_vehicle_speed)
        self._front_vehicle_ttc = 20.0 if closing_for_guard <= 1e-3 else float(
            np.clip(self._front_vehicle_gap / closing_for_guard, 0.0, 20.0)
        )
        # Do not clear *_guard_active here. It describes the action that was
        # executed on the previous tick and is snapshotted by step() before this
        # sensing pass. _apply_front_vehicle_collision_guard resets it at the
        # beginning of the next action.

        # Red/yellow traffic light detection. Do NOT inject the traffic
        # light as a fake lead obstacle in state[3]; otherwise the generic ACC
        # time-headway guard stops far before the zebra/stop line. The actual
        # traffic-light stop is handled in _apply_ego_action using the physical
        # distance to CARLA's stop waypoint.
        previous_signal = self._snapshot_traffic_light_state()
        self._traffic_light_detector_fault_active = False
        try:
            self._traffic_light_hazard(wp, lane_half)
            if not bool(
                getattr(
                    self,
                    "_traffic_light_detector_fault_active",
                    False,
                )
            ):
                self._traffic_light_detector_fault_ticks = 0
        except Exception as exc:
            ego_half_for_fault = float(self.ego.bounding_box.extent.x)
            self._handle_traffic_light_detector_exception(
                previous_signal, speed=speed, ego_half=ego_half_for_fault, exc=exc
            )

        # Red-light priority with queue awareness.
        # If another car is already waiting ahead at the red light, keep it as
        # the lead hazard and let the queue-following controller stop about
        # queue_stop_gap metres behind it.  Only when the front actor is farther
        # than queue_detect_distance do we ignore it and prioritize the stop line.
        self._red_light_front_gap = float(gap)
        self._red_light_front_speed = float(max(0.0, speed + rel_vel))
        self._red_light_queue_active = False
        self._red_light_queue_gap_error = 999.0
        if self._traffic_light_active and self._lead_is_relevant:
            # Treat only stopped vehicles as red-light queue targets.  Walkers
            # are still hazards, but they are not part of a vehicle queue and
            # must not use the tight 2 m queue-following gap.
            front_before_stop_line = bool(float(gap) <= float(self._traffic_light_distance) + 3.0)
            lead_kind_before = str(self._lead_actor_kind).lower()
            is_walker_ahead = ("walker" in lead_kind_before) or ("pedestrian" in lead_kind_before)

            # Queue target must be a same-lane/same-route vehicle before the
            # stop line.  This rejects adjacent-lane or cross-street cars that
            # made the ego stop far from the zebra line in Town10HD screenshots.
            heading_diff = float(getattr(self, "_front_vehicle_heading_diff", 0.0))
            same_direction = heading_diff <= math.radians(30.0)
            same_route_queue = bool(
                self._lead_same_lane or
                (abs(float(self._lead_route_lat)) <= 0.55 and same_direction)
            )
            front_is_slow = bool(
                float(self._red_light_front_speed)
                <= float(self.vehicle_queue_speed_threshold) + 0.05
            )
            is_vehicle_queue_candidate = bool(
                (not is_walker_ahead)
                and front_before_stop_line
                and same_route_queue
                and front_is_slow
            )
            if (is_vehicle_queue_candidate and
                    float(gap) <= float(self.queue_detect_distance)):
                self._red_light_queue_active = True
                self._red_light_queue_gap_error = float(gap - self.queue_stop_gap)
                self._lead_actor_kind = "queue_" + str(self._lead_actor_kind)
                self._front_vehicle_kind = str(self._lead_actor_kind)
                self._front_vehicle_guard_valid = True
            elif is_walker_ahead and float(gap) <= float(self.vehicle_detect_distance):
                # Keep a geometrically valid pedestrian/walker as a normal
                # obstacle; never reinterpret it as a tight vehicle queue.
                self._red_light_queue_active = False
                self._red_light_queue_gap_error = 999.0
                self._front_vehicle_kind = str(self._lead_actor_kind)
            elif (
                (not is_walker_ahead)
                and front_before_stop_line
                and same_route_queue
                and float(gap) <= float(self.vehicle_detect_distance)
            ):
                # A moving same-route vehicle before the stop line is not a
                # queue target. Keep normal lead/collision protection instead
                # of suppressing it merely because a red light also exists.
                self._red_light_queue_active = False
                self._red_light_queue_gap_error = 999.0
                self._front_vehicle_kind = str(self._lead_actor_kind)
                self._front_vehicle_guard_valid = True
            else:
                # No valid close same-route actor before the stop line: let the
                # physical stop-line controller govern. Explicitly invalidate
                # the preserved raw actor so cross-street/beyond-line traffic
                # cannot leak into the universal front guard and stop ego early.
                gap, rel_vel, obstacle = 80.0, 0.0, 80.0
                self._lead_is_relevant = False
                self._lead_same_lane = False
                self._lead_route_lat = 999.0
                self._lead_route_ds = 999.0
                self._lead_actor_kind = "red_light_priority"
                self._front_vehicle_kind = "red_light_priority"
                self._front_vehicle_guard_valid = False

        # Update the overtaking state after traffic-light/queue semantics are
        # known. Keep policy observation semantics route/map relative; only the
        # explicit left/right barrier margins x[6]/x[7] widen to the legal
        # two-lane corridor. This avoids rewarding the lane divider or hiding a
        # mode-dependent meaning change in state[0:2].
        self._update_overtake_state(wp, speed)
        self._ensure_overtake_state()
        if str(getattr(self, "_overtake_mode", "idle")) != "idle" and route_sane:
            # Keep the policy state continuously route-relative during a lane
            # change. CARLA's nearest-lane projection can jump from source to
            # target lane center mid-maneuver, creating a discontinuous state.
            state_lat = float(route_lat_now)
            state_heading = float(route_heading_error)
            self._state_geometry_source = "route_overtake"
        self._reward_lane_lat_error = float(state_lat)
        self._reward_lane_heading_error = float(state_heading)
        self._barrier_geometry_source = "lane"
        right_margin = float(lane_half - state_lat)
        left_margin = float(lane_half + state_lat)

        overtake_geom = self._overtake_corridor_geometry(
            wp, loc, math.radians(float(tf.rotation.yaw))
        )
        self._overtake_corridor_valid = overtake_geom is not None
        if overtake_geom is not None:
            corridor_lat, _, corridor_half = overtake_geom
            right_margin = float(corridor_half - corridor_lat)
            left_margin = float(corridor_half + corridor_lat)
            self._barrier_geometry_source = "overtake_corridor"

            mode = str(getattr(self, "_overtake_mode", "idle"))
            desired_key = (
                self._overtake_source_lane_key
                if mode == "returning" else self._overtake_target_lane_key
            )
            desired_wp = self._find_lane_waypoint_by_key(wp, desired_key)
            if desired_wp is not None:
                self._target_lane_clear(desired_wp)
                # Reward centering follows the maneuver's intended lane, not the
                # center line between two lanes.
                try:
                    dloc = desired_wp.transform.location
                    dright = desired_wp.transform.get_right_vector()
                    self._reward_lane_lat_error = float(
                        (loc.x - dloc.x) * dright.x + (loc.y - dloc.y) * dright.y
                    )
                    self._reward_lane_heading_error = _wrap_pi(
                        math.radians(tf.rotation.yaw - desired_wp.transform.rotation.yaw)
                    )
                except Exception:
                    pass

                # Use the desired lane's nearest front hazard for headway and
                # obstacle barriers. The original stopped blocker remains
                # relevant only while dangerously close before lateral escape.
                source_emergency = bool(
                    mode == "changing_out"
                    and int(getattr(self, "_front_vehicle_actor_id", -1))
                        == int(getattr(self, "_overtake_blocked_actor_id", -2))
                    and float(getattr(self, "_front_vehicle_gap", 80.0))
                        <= max(float(self.vehicle_stop_gap) + 0.75, 3.0)
                )
                if not source_emergency:
                    target_gap = float(getattr(self, "_overtake_target_front_gap", 999.0))
                    target_speed = float(getattr(self, "_overtake_target_front_speed", 0.0))
                    target_kind = str(getattr(self, "_overtake_target_front_kind", "none"))
                    target_actor_id = int(getattr(self, "_overtake_target_front_actor_id", -1))
                    gap = float(min(target_gap, 80.0))
                    if gap < 79.0:
                        rel_vel = float(target_speed - speed)
                        obstacle = gap
                        self._lead_is_relevant = True
                        self._lead_actor_kind = target_kind
                        self._front_vehicle_gap = gap
                        self._front_vehicle_speed = target_speed
                        self._front_vehicle_kind = target_kind
                        self._front_vehicle_actor_id = target_actor_id
                        self._front_vehicle_guard_valid = True
                        closing_target = max(0.0, speed - target_speed)
                        self._front_vehicle_ttc = (
                            20.0 if closing_target <= 1e-3
                            else float(np.clip(gap / closing_target, 0.0, 20.0))
                        )
                    else:
                        gap, rel_vel, obstacle = 80.0, 0.0, 80.0
                        self._lead_is_relevant = False
                        self._lead_actor_kind = "none"
                        self._front_vehicle_gap = 80.0
                        self._front_vehicle_speed = 0.0
                        self._front_vehicle_kind = "none"
                        self._front_vehicle_actor_id = -1
                        self._front_vehicle_guard_valid = False
                        self._front_vehicle_ttc = 20.0

        closing_speed = max(0.0, -rel_vel)
        ttc_front = 20.0 if closing_speed <= 1e-3 else float(
            np.clip(gap / closing_speed, 0.0, 20.0)
        )

        base = np.asarray([state_lat, state_heading, speed, gap, rel_vel, obstacle,
                           right_margin, left_margin], dtype=np.float64)
        if self.use_augmented_state:
            extra = np.asarray([route_heading_error, progress_frac, remaining_frac, ttc_front],
                               dtype=np.float64)
            state = np.concatenate([base, extra], axis=0)
        else:
            state = base
        state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64, copy=False)
        self.barrier.lane_half_width = lane_half
        return state

    # -- lead control (controlled drift source) -------------------------------
    def _drive_lead(self):
        if self.lead is None or not self.lead.is_alive:
            return
        carla = self._carla
        phase = 2.0 * math.pi * (self.t * self.dt) / max(self.drift_period, 1e-6)
        target = self.target_speed + self.drift_scale * (
            3.0 * math.sin(phase) + float(self.rng.normal(0, 1.0)))
        target = float(np.clip(target, 0.0, self.barrier.speed_limit_mps))
        lead_speed = self._vehicle_speed(self.lead)
        accel_cmd = 0.6 * (target - lead_speed)
        # Keep the lead vehicle in its own lane. The old lead controller used
        # steer=0, which lets the lead cut across lanes/curves and then falsely
        # trigger headway failures for the ego.
        steer = 0.0
        try:
            tf = self.lead.get_transform()
            loc = tf.location
            wp = self.map.get_waypoint(loc, project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
            right = wp.transform.get_right_vector()
            wp_loc = wp.transform.location
            lat = float((loc.x - wp_loc.x) * right.x + (loc.y - wp_loc.y) * right.y)
            head = _wrap_pi(math.radians(tf.rotation.yaw - wp.transform.rotation.yaw))
            steer = float(np.clip(-0.25 * lat - 0.85 * head, -0.45, 0.45))
        except Exception:
            steer = 0.0

        if accel_cmd >= 0:
            ctrl = carla.VehicleControl(
                throttle=max(float(np.clip(accel_cmd / 3.0, 0, 1)), 0.30),
                brake=0.0, steer=steer, hand_brake=False)
        else:
            ctrl = carla.VehicleControl(throttle=0.0,
                                        brake=float(np.clip(-accel_cmd / 5.0, 0, 1)),
                                        steer=steer, hand_brake=False)
        self.lead.apply_control(ctrl)

    def _build_destination(self, start_wp) -> None:
        """Build an explicit destination route using CARLA's GlobalRoutePlanner.

        Default behavior is fail-fast global routing.  The old local ``Waypoint.next``
        branch heuristic remains available only when explicitly requested or when
        ``allow_heuristic_route_fallback`` is set. This prevents silent route changes
        at complex junctions and makes traffic-light relevance refer to one fixed,
        destination-conditioned route.
        """
        self.route_waypoints = [start_wp]
        self.destination_wp = start_wp
        self.destination_loc = start_wp.transform.location
        self.route_tracker = None
        self.route_length_m = 0.0
        self._route_planner_used = "none"
        self._route_planner_error = ""
        self._route_candidate_count = 0
        if self.route_distance <= 0:
            return

        if self.route_planner_mode == "global":
            try:
                self._build_destination_global(start_wp)
                return
            except Exception as exc:
                self._route_planner_error = f"{type(exc).__name__}: {exc}"
                self._record_safety_exception("global_route_planner", exc)
                if not self.allow_heuristic_route_fallback:
                    if isinstance(exc, ImportError):
                        hint = (
                            "Ensure CARLA PythonAPI agents are on PYTHONPATH (module "
                            "agents.navigation.global_route_planner or carla_agents.navigation."
                            "global_route_planner). "
                        )
                    else:
                        hint = (
                            "The planner imported correctly but this spawn produced no "
                            "acceptable route; the reset loop will retry with a new spawn. "
                        )
                    raise RuntimeError(
                        "Global route planning failed and heuristic fallback is disabled. "
                        + hint +
                        "To explicitly permit the legacy route heuristic, "
                        "pass --allow_heuristic_route_fallback. Original error: "
                        f"{self._route_planner_error}"
                    ) from exc
                print(
                    f"[route][WARN] global planner failed; explicit heuristic fallback enabled: "
                    f"{self._route_planner_error}", flush=True
                )

        self._build_destination_heuristic(start_wp)


    def _import_global_route_planner(self):
        """Import CARLA's GlobalRoutePlanner from common 0.9.15 layouts.

        The pip ``carla`` wheel may expose the simulator API without the
        ``agents`` package.  When needed, also search ``CARLA_ROOT`` and a few
        standard local installation paths for ``PythonAPI/carla/agents``.
        """
        errors = []
        module_names = (
            "agents.navigation.global_route_planner",
            "carla_agents.navigation.global_route_planner",
        )

        def _try_import():
            for module_name in module_names:
                try:
                    module = __import__(module_name, fromlist=["GlobalRoutePlanner"])
                    return getattr(module, "GlobalRoutePlanner")
                except Exception as exc:
                    errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            return None

        planner = _try_import()
        if planner is not None:
            return planner

        candidates: List[Path] = []
        carla_root = str(os.environ.get("CARLA_ROOT", "")).strip()
        if carla_root:
            candidates.extend([
                Path(carla_root) / "PythonAPI" / "carla",
                Path(carla_root) / "PythonAPI",
            ])
        # Derive from wherever the `carla` module itself was imported from: an
        # egg under <root>/PythonAPI/carla/dist places the `agents` package two
        # levels up.  This makes the planner import independent of which conda
        # environment is active, as long as `import carla` resolves to a full
        # simulator installation rather than the bare pip wheel.
        try:
            carla_mod = self._ensure_carla()
            mod_path = Path(str(getattr(carla_mod, "__file__", "") or "")).resolve()
            for parent in list(mod_path.parents)[:8]:
                try:
                    if (parent / "agents").is_dir():
                        candidates.append(parent)
                except Exception:
                    continue
        except Exception:
            pass
        candidates.extend([
            Path.home() / "CARLA_0.9.15" / "PythonAPI" / "carla",
            Path.home() / "CARLA" / "PythonAPI" / "carla",
            Path("/opt/carla/PythonAPI/carla"),
        ])
        # Wildcard installs (CARLA unpacked under a nonstandard name/location).
        try:
            import glob as _glob
            patterns = [
                str(Path.home() / "CARLA_*" / "PythonAPI" / "carla"),
                str(Path.home() / "carla*" / "PythonAPI" / "carla"),
                str(Path.home() / "Desktop" / "CARLA_*" / "PythonAPI" / "carla"),
                "/opt/carla*/PythonAPI/carla",
            ]
            for pat in patterns:
                for hit in sorted(_glob.glob(pat)):
                    candidates.append(Path(hit))
        except Exception:
            pass

        for candidate in candidates:
            try:
                if candidate.is_dir() and str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
            except Exception:
                continue

        planner = _try_import()
        if planner is not None:
            return planner
        raise ImportError("; ".join(errors))


    def _route_length_from_waypoints(self, waypoints: Sequence[Any]) -> float:
        total = 0.0
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            try:
                la = a.transform.location
                lb = b.transform.location
                dx = float(lb.x - la.x)
                dy = float(lb.y - la.y)
                dz = float(lb.z - la.z)
                total += math.sqrt(dx * dx + dy * dy + dz * dz)
            except Exception:
                continue
        return float(total)



    def _get_global_route_planner(self, GlobalRoutePlanner):
        """Return a cached GlobalRoutePlanner for the current map.

        Building the planner constructs the full road-topology graph, which
        costs seconds on Town10HD-class maps.  The graph only depends on the
        map object and the sampling resolution, so rebuilding it on every
        reset (and on every retry within one reset) is pure waste.  The cache
        key includes ``id(self.map)`` so a reconnect/world reload naturally
        invalidates it; the cached planner keeps the old map object alive,
        which guarantees CPython cannot reuse its id for a new map.
        """
        key = (str(getattr(self.map, "name", "")), float(self.route_step_m), id(self.map))
        cached = getattr(self, "_grp_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            planner = GlobalRoutePlanner(
                self.map, sampling_resolution=float(self.route_step_m)
            )
        except TypeError:
            planner = GlobalRoutePlanner(self.map, float(self.route_step_m))
        self._grp_cache = (key, planner)
        return planner


    def _build_destination_global(self, start_wp) -> None:
        """Choose a globally planned route long enough to guarantee target distance."""
        GlobalRoutePlanner = self._import_global_route_planner()
        planner = self._get_global_route_planner(GlobalRoutePlanner)

        start_loc = start_wp.transform.location
        spawn_points = list(self.map.get_spawn_points())
        if not spawn_points:
            raise RuntimeError("CARLA map has no spawn points for destination selection.")

        target = max(float(self.route_distance), 10.0)
        ranked = []
        for idx, tf in enumerate(spawn_points):
            try:
                dx = float(tf.location.x - start_loc.x)
                dy = float(tf.location.y - start_loc.y)
                dz = float(tf.location.z - start_loc.z)
                aerial = math.sqrt(dx * dx + dy * dy + dz * dz)
                if aerial < max(15.0, 0.05 * target):
                    continue
                ranked.append((abs(aerial - 0.65 * target), idx, tf))
            except Exception:
                continue
        ranked.sort(key=lambda x: (x[0], x[1]))
        if not ranked:
            raise RuntimeError("No sufficiently distant destination spawn point was available.")

        max_candidates = min(int(self.route_destination_candidates), len(ranked))
        primary_pool = ranked[:max_candidates]
        extended_pool = ranked[max_candidates:]
        self._route_candidate_count = len(primary_pool)

        best = None
        best_score = float("inf")
        longest_sane = None
        longest_len = 0.0
        errors = []
        required_trace_len = target + max(5.0, float(self.route_step_m))
        # Relaxed floor: episodes shorter than this fraction of the target are
        # rejected even in rescue mode.
        min_relaxed_len = max(0.60 * target, 60.0)

        def _trace_pool(pool, origin_loc, prefix_wps) -> None:
            nonlocal best, best_score, longest_sane, longest_len
            prefix = list(prefix_wps)
            for _, _, tf in pool:
                try:
                    traced = list(planner.trace_route(origin_loc, tf.location))
                    waypoints = [
                        pair[0] if isinstance(pair, (tuple, list)) else pair
                        for pair in traced
                    ]
                    waypoints = prefix + [wp for wp in waypoints if wp is not None]
                    if len(waypoints) < 2:
                        continue
                    if not self._route_trace_start_is_sane(waypoints, start_wp):
                        errors.append("rejected route with invalid start alignment/projection")
                        continue
                    route_len = self._route_length_from_waypoints(waypoints)
                    if route_len > longest_len:
                        longest_len = route_len
                        longest_sane = (waypoints, route_len)
                    if route_len < required_trace_len:
                        errors.append(
                            f"route too short: {route_len:.1f}m < required {required_trace_len:.1f}m"
                        )
                        continue
                    score = float(route_len - target)
                    if score < best_score:
                        best_score = score
                        best = (waypoints, route_len)
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")

        # Pass 1: the aerial-distance ranking prefers destinations roughly
        # 0.65*target away.  On compact maps (Town10HD) there are spawn points
        # from which every candidate in that band yields a shortest road route
        # well below the target, while candidates that WOULD satisfy it rank
        # outside the top-N cutoff and were previously never traced.  That is
        # exactly the failure that aborted whole training runs with
        # "route too short" after 5 reset attempts.
        _trace_pool(primary_pool, start_loc, [])

        # Pass 2: full-length search over every remaining destination spawn
        # point.  trace_route is a per-pair graph search on the cached planner,
        # so scanning the whole list is cheap relative to a failed reset.
        if best is None and extended_pool:
            _trace_pool(extended_pool, start_loc, [])
            self._route_candidate_count = len(ranked)

        # Pass 2b (nudged origin): if NOT A SINGLE candidate produced a
        # start-sane route, the trace origin itself is the problem: near
        # junction entrances GlobalRoutePlanner localizes the origin onto the
        # nearest topology edge, which can belong to a crossing road, so every
        # traced route begins misaligned with the ego and fails the sanity
        # check regardless of destination.  Re-trace from a point a few meters
        # forward along the ego's OWN lane (Waypoint.next follows the lane) and
        # prepend the true start waypoint, so the executed route still begins
        # at the ego with a short straight same-lane connector.
        nudged = False
        if best is None and longest_sane is None:
            try:
                nudge_dist = max(6.0, 3.0 * float(self.route_step_m))
                branches = list(start_wp.next(nudge_dist) or [])
            except Exception:
                branches = []
            for branch_wp in branches[:2]:
                try:
                    _trace_pool(ranked, branch_wp.transform.location, [start_wp])
                except Exception:
                    continue
                if best is not None or longest_sane is not None:
                    nudged = True
                    self._route_candidate_count = len(ranked)
                    break

        # Pass 3 (rescue): no destination reaches the required length from this
        # spawn.  Accept the longest start-sane route instead of aborting the
        # episode, provided it is not degenerately short.  The goal is moved to
        # the end of the traced route so success remains achievable; the
        # requested target is unchanged for all other episodes.
        relaxed = False
        if best is None and longest_sane is not None and longest_len >= min_relaxed_len:
            best = longest_sane
            relaxed = True

        if best is None:
            detail = errors[-1] if errors else "no valid trace_route output"
            raise RuntimeError(
                "GlobalRoutePlanner could not produce a start-sane route long enough "
                f"for the requested {target:.1f}m target: {detail}"
            )

        waypoints, route_len = best
        self.route_waypoints = list(waypoints)
        self.route_length_m = float(route_len)
        if relaxed:
            effective_goal = min(
                target, max(10.0, float(route_len) - max(5.0, float(self.route_step_m)))
            )
        else:
            effective_goal = float(target)
        self._goal_distance_m = float(effective_goal)
        self.route_tracker = RouteProgressTracker(self.route_waypoints)
        goal_idx = int(np.searchsorted(self.route_tracker.s, effective_goal, side="left"))
        goal_idx = min(max(goal_idx, 0), len(self.route_waypoints) - 1)
        self.destination_wp = self.route_waypoints[goal_idx]
        self.destination_loc = self.destination_wp.transform.location
        base_tag = "global_nudged" if nudged else "global"
        self._route_planner_used = f"{base_tag}_relaxed" if relaxed else base_tag
        if relaxed:
            print(
                f"[route][WARN] relaxed route accepted: no start-sane route reached the "
                f"requested {target:.1f}m from this spawn; using longest sane route "
                f"{route_len:.1f}m with goal={effective_goal:.1f}m",
                flush=True,
            )
        print(
            f"[route] planner={self._route_planner_used} candidates={self._route_candidate_count} "
            f"requested={target:.1f}m traced={route_len:.1f}m goal={self._goal_distance_m:.1f}m",
            flush=True,
        )


    def _build_destination_heuristic(self, start_wp) -> None:
        """Legacy deterministic local-branch route builder, used only by explicit opt-in."""
        self.route_waypoints = [start_wp]
        self.destination_wp = start_wp
        self.destination_loc = start_wp.transform.location
        self.route_tracker = None
        if self.route_distance <= 0:
            return

        cur = start_wp
        travelled = 0.0
        recent_keys: Deque[Tuple[int, int, int]] = deque(maxlen=80)

        def _wp_key(wp: Any) -> Tuple[int, int, int]:
            try:
                return (int(wp.road_id), int(wp.lane_id), int(round(float(wp.s) / 2.0)))
            except Exception:
                return (id(wp), 0, 0)

        recent_keys.append(_wp_key(cur))
        while travelled < float(self.route_distance) + 30.0:
            nxts = list(cur.next(self.route_step_m))
            if not nxts:
                break
            if len(nxts) == 1:
                chosen = nxts[0]
            else:
                cur_yaw = math.radians(float(cur.transform.rotation.yaw))
                cur_keyset = set(recent_keys)

                def _branch_score(wp: Any) -> float:
                    score = 0.0
                    first_yaw = math.radians(float(wp.transform.rotation.yaw))
                    score += 0.45 * abs(_wrap_pi(first_yaw - cur_yaw))
                    try:
                        if int(wp.lane_id) != int(cur.lane_id):
                            score += 0.35
                        if int(wp.road_id) != int(cur.road_id):
                            score += 0.05
                    except Exception:
                        pass
                    rollout = wp
                    prev_yaw = first_yaw
                    for _ in range(max(2, int(round(12.0 / max(self.route_step_m, 0.5))))):
                        if _wp_key(rollout) in cur_keyset:
                            score += 8.0
                        nx = list(rollout.next(self.route_step_m))
                        if not nx:
                            score += 2.0
                            break
                        rollout = min(
                            nx,
                            key=lambda cand: abs(_wrap_pi(
                                math.radians(float(cand.transform.rotation.yaw)) - prev_yaw
                            )),
                        )
                        yaw = math.radians(float(rollout.transform.rotation.yaw))
                        score += 0.20 * abs(_wrap_pi(yaw - prev_yaw))
                        prev_yaw = yaw
                    score += 1.5 * max(
                        0.0, abs(_wrap_pi(prev_yaw - cur_yaw)) - math.radians(100.0)
                    )
                    return float(score)

                chosen = min(nxts, key=_branch_score)
            cur = chosen
            self.route_waypoints.append(cur)
            recent_keys.append(_wp_key(cur))
            travelled += self.route_step_m

        self.route_length_m = float(travelled)
        self._goal_distance_m = float(min(self.route_distance, max(10.0, travelled - 5.0)))
        if self.route_waypoints:
            idx = min(
                len(self.route_waypoints) - 1,
                max(0, int(round(float(self._goal_distance_m) / self.route_step_m))),
            )
            self.destination_wp = self.route_waypoints[idx]
            self.destination_loc = self.destination_wp.transform.location
        self.route_tracker = RouteProgressTracker(self.route_waypoints)
        self._route_planner_used = "heuristic"
        print(
            f"[route] planner=heuristic requested={self.route_distance:.1f}m "
            f"built={self.route_length_m:.1f}m goal={self._goal_distance_m:.1f}m",
            flush=True,
        )


    def _update_progress(self) -> float:
        """Update monotone arc-length progress along the precomputed road route."""
        loc = self.ego.get_location()
        old = float(self.progress_m)
        if self.route_tracker is not None:
            s_now, route_lat, route_yaw = self.route_tracker.project(
                loc, self.route_search_back, self.route_search_forward)
            self.progress_m = max(self.progress_m, float(s_now))
            self._route_lat_error = route_lat
            ego_yaw_rad = math.radians(self.ego.get_transform().rotation.yaw)
            self._route_yaw_error = _wrap_pi(ego_yaw_rad - route_yaw)
            lookahead_yaw = self.route_tracker.lookahead_yaw(self.route_turn_lookahead_m)
            self._route_lookahead_yaw_error = _wrap_pi(ego_yaw_rad - lookahead_yaw)
            self._route_turn_strength = abs(_wrap_pi(lookahead_yaw - route_yaw))
            self._route_curvature = self.route_tracker.curvature_ahead(lookahead_segments=15)
            # Record this tick's projection so _read_state can reuse it instead
            # of projecting a second time.  Two projections per tick double the
            # forward-snap allowance, and progress_m is monotone, so one bad
            # snap at crossing Town10HD geometry permanently inflates progress,
            # shrinks route_ds to the stop line (early stops far before the
            # zebra line), and can falsely trigger the progress-based
            # stop-line-crossing commit (red light then ignored).
            self._route_yaw_now = float(route_yaw)
            self._progress_tick_t = int(self.t)
        else:
            if self._prev_loc is not None:
                dx = float(loc.x - self._prev_loc.x)
                dy = float(loc.y - self._prev_loc.y)
                fwd = _carla_vec_forward(self.ego.get_transform())
                delta = max(0.0, dx * fwd[0] + dy * fwd[1])
                if delta < 10.0:
                    self.progress_m += delta
        self._prev_loc = loc
        return float(max(0.0, self.progress_m - old))

    def _distance_to_destination(self) -> float:
        if self.destination_loc is None:
            return float("nan")
        loc = self.ego.get_location()
        dx = float(loc.x - self.destination_loc.x)
        dy = float(loc.y - self.destination_loc.y)
        dz = float(loc.z - self.destination_loc.z)
        return float(math.sqrt(dx * dx + dy * dy + dz * dz))

    def _front_actor_is_queue_like(self, gap: Optional[float] = None,
                                   front_speed: Optional[float] = None) -> bool:
        """Return True only for a stopped *vehicle* queue.

        Human-like traffic-rule priority is:
          * stopped vehicle at red/yellow: queue target, usually 2 m gap;
          * moving vehicle: moving-following target, usually 4--6 m gap;
          * walker/pedestrian: obstacle target, not a vehicle queue.

        The previous implementation returned True for any ``queue_*`` label,
        including ``queue_walker``.  That made the ego crawl behind pedestrians
        with a car-like 2 m queue rule and could cause collisions.  This version
        never classifies walkers as queue vehicles.
        """
        try:
            g = float(self._front_vehicle_gap if gap is None else gap)
            fs = float(self._front_vehicle_speed if front_speed is None else front_speed)
            kind = str(getattr(self, "_front_vehicle_kind", self._lead_actor_kind)).lower()
            if "walker" in kind or "pedestrian" in kind:
                return False
            if not bool(getattr(self, "_lead_is_relevant", False)) and not kind.startswith("queue_"):
                return False
            stopped = fs <= float(getattr(self, "vehicle_queue_speed_threshold", 0.7))
            if kind.startswith("queue_"):
                stopped = True
            if not stopped:
                return False
            # A genuine queue vehicle points the same way we do and sits in our
            # lane/route corridor. Parked cars at the curb, angled cars, and
            # cross/oncoming traffic must NOT trigger the 2 m queue standoff --
            # that is what made the ego stop metres before the stop line.
            heading_diff = float(getattr(self, "_front_vehicle_heading_diff", 0.0))
            if heading_diff > math.radians(40.0):
                return False
            if not bool(getattr(self, "_lead_same_lane", False)):
                # Non-same-lane queue classification is allowed only for very
                # tight route-corridor matches near junctions.  Adjacent-lane
                # stopped vehicles must not make ego wait in the middle of its lane.
                if abs(float(getattr(self, "_lead_route_lat", 999.0))) > 0.55:
                    return False
            if bool(getattr(self, "_traffic_light_active", False)):
                tl_gap = float(getattr(self, "_traffic_light_distance", 999.0))
                # A queue car must be between ego and the stop line. A vehicle
                # far beyond the line is junction/cross traffic, not our queue.
                return bool(g <= min(float(getattr(self, "queue_detect_distance", 25.0)), tl_gap + 3.0))
            return bool(g <= float(getattr(self, "vehicle_detect_distance", 45.0)))
        except Exception:
            return False

    def _desired_front_gap(self, ego_speed: float, gap: Optional[float] = None,
                           front_speed: Optional[float] = None) -> float:
        """Traffic-rule gap used for reward, diagnostics, and termination.

        * stopped vehicle before a red light: queue_stop_gap, normally 2 m;
        * walker/pedestrian: at least ~3.5 m, never the 2 m vehicle-queue gap;
        * moving vehicle: vehicle_stop_gap + vehicle_moving_time_headway * speed,
          normally about 4--6 m at these speeds.
        """
        try:
            speed = max(0.0, float(ego_speed))
            kind = str(getattr(self, "_front_vehicle_kind", self._lead_actor_kind)).lower()
            if "walker" in kind or "pedestrian" in kind:
                return float(max(getattr(self, "vehicle_stop_gap", 3.5), 3.5))
            if self._front_actor_is_queue_like(gap, front_speed):
                return float(getattr(self, "queue_stop_gap", 2.0))
            base = float(getattr(self, "vehicle_stop_gap", 3.5))
            th = float(getattr(self, "vehicle_moving_time_headway", 0.25))
            return float(np.clip(base + th * speed, 1.0, 25.0))
        except Exception:
            return 5.0

    def _runtime_barrier_values(self, state: ArrayLike) -> np.ndarray:
        """Barrier values from the same model object used by the filter.

        Context-dependent following gaps are represented by updating the shared
        barrier parameters before evaluation.  When the filtered evaluator passes
        ``env.barrier`` into E-COCSF, projection, residual prediction, reward and
        termination use the same barrier object and lane width.
        """
        try:
            x = np.asarray(state, dtype=np.float64).reshape(-1)
            speed = max(0.0, float(x[2])) if x.size > 2 else 0.0
            gap = max(0.0, float(x[3])) if x.size > 3 else 80.0
            front_speed = float(getattr(self, "_front_vehicle_speed", 0.0))

            if bool(getattr(self, "_lead_is_relevant", False)) and gap < 79.0:
                kind = str(getattr(self, "_front_vehicle_kind", self._lead_actor_kind)).lower()
                if "walker" in kind or "pedestrian" in kind:
                    self.barrier.min_headway_m = float(max(self.vehicle_stop_gap, 3.5))
                    self.barrier.time_headway_s = 0.0
                elif self._front_actor_is_queue_like(gap, front_speed):
                    self.barrier.min_headway_m = float(self.queue_stop_gap)
                    self.barrier.time_headway_s = 0.0
                else:
                    self.barrier.min_headway_m = float(self.vehicle_stop_gap)
                    self.barrier.time_headway_s = float(self.vehicle_moving_time_headway)
                desired_gap = self._desired_front_gap(speed, gap, front_speed)
                self.barrier.min_obstacle_distance_m = float(
                    min(2.0, max(0.8, 0.75 * desired_gap))
                )
            else:
                # No relevant front actor: these components are far from active
                # because state[3]/state[5] are set to ~80 m.
                self.barrier.min_headway_m = 2.0
                self.barrier.time_headway_s = 0.45
                self.barrier.min_obstacle_distance_m = 3.0
        except Exception:
            pass
        return np.asarray(self.barrier.barrier_values(state), dtype=np.float64).copy()
    def _desired_speed_from_headway(self, state: np.ndarray, comps: np.ndarray) -> float:
        """Traffic-rule desired speed used only for reward shaping.

        Earlier code used ``safe_gap + 8 m`` with a conservative headway model;
        that trained the policy to park long before normal traffic.  The new
        logic slows only near the desired following gap: about 2 m for stopped
        red-light queues and about 4--6 m for moving vehicles.
        """
        speed = float(state[2])
        gap = float(state[3])
        rel_vel = float(state[4])  # lead_speed - ego_speed; negative means closing
        if not bool(getattr(self, "_lead_is_relevant", False)) or gap >= 79.0:
            return float(self.target_speed)

        front_speed = max(0.0, speed + rel_vel)
        desired_gap = self._desired_front_gap(speed, gap, front_speed)
        closing_adjust = max(0.0, -rel_vel)
        soft_extra = float(getattr(self, "vehicle_soft_extra_gap", 1.5))
        desired = float(self.target_speed)

        kind = str(getattr(self, "_front_vehicle_kind", self._lead_actor_kind)).lower()
        if "walker" in kind or "pedestrian" in kind:
            # Pedestrians/walkers are obstacles, not traffic queues. Slow to a
            # cautious crawl and stop with a larger gap than vehicle queues.
            err = gap - desired_gap
            if err <= 0.30:
                desired = 0.0
            elif err < 8.0:
                desired = min(1.0, 0.25 * err)
            else:
                desired = min(2.0, 0.20 * err)
        elif self._front_actor_is_queue_like(gap, front_speed):
            # Reward shaping must match the physical queue controller.  The old
            # far-queue rule capped desired speed at 3 m/s for any stopped car
            # more than 8 m beyond the target gap; with background traffic that
            # explicitly taught SAC to crawl long before a red light.  Use the
            # same continuous physics-based queue target as the low-level guard.
            err = float(gap - desired_gap)
            desired = self._red_light_queue_desired_speed(err, front_speed)
        elif gap < desired_gap + soft_extra:
            # Moving traffic: follow front speed and bleed off closing speed.
            desired = min(desired, max(0.0, front_speed + 0.8 * (gap - desired_gap) - 0.5 * closing_adjust))

        return float(np.clip(desired, 0.0, self.target_speed))

    def _apply_red_light_close_stop(self, accel: float, ego_speed: float) -> float:
        """Human-like red/yellow stop and queue governor with fault debouncing.

        Key invariants:
          * detection and braking activation are separate;
          * a detector exception alone never invents a red light;
          * the first two unconfirmed detector faults do not command braking;
          * repeated unconfirmed faults only prevent positive acceleration;
          * a previously confirmed red/yellow light is still retained by the
            detector hysteresis and receives the normal physical stop control;
          * queue and stop-line creep never override deliberate braking.
        """
        self._red_light_control_active = False
        self._red_light_braking_active = False
        self._red_light_control_reason = "none"
        self._traffic_light_accel_cap = 2.5

        try:
            if not self.traffic_light_guard:
                return float(accel)

            detector_fault = bool(
                getattr(self, "_traffic_light_detector_fault_active", False)
            )
            detector_fault_ticks = int(
                getattr(self, "_traffic_light_detector_fault_ticks", 0)
            )
            light_active = bool(
                getattr(self, "_traffic_light_active", False)
            )

            # No confirmed light exists. A one- or two-frame sensor/API dropout
            # must not become a phantom hard brake on an otherwise clear road.
            if detector_fault and not light_active:
                if detector_fault_ticks <= 2:
                    self._red_light_control_reason = "detector_fault_transient"
                    self._traffic_light_accel_cap = float(accel)
                    return float(accel)

                # Persistent uncertainty: do not add positive acceleration, but
                # also do not invent an emergency stop without a confirmed light.
                safe = min(float(accel), 0.0)
                self._red_light_control_active = bool(safe < float(accel) - 1e-9)
                self._red_light_control_reason = "detector_fault_coast"
                self._traffic_light_accel_cap = float(safe)
                return float(safe)

            if not light_active:
                return float(accel)

            tl_state = str(self._traffic_light_state).lower()
            is_stop_light = tl_state.startswith("red") or (
                self.yellow_light_stop and tl_state.startswith("yellow")
            )
            if not is_stop_light:
                return float(accel)

            physical_gap = float(self._traffic_light_distance)
            if (
                not np.isfinite(physical_gap)
                or physical_gap > float(self.red_light_stop_distance)
            ):
                return float(accel)

            speed = max(0.0, float(ego_speed))
            physical_gap = max(0.0, physical_gap - speed * float(self.dt))
            stop_error = float(physical_gap - self.red_light_stop_buffer)
            self._traffic_light_virtual_gap = float(physical_gap)
            self._traffic_light_stop_error = stop_error

            front_gap = float(getattr(self, "_red_light_front_gap", 80.0))
            front_speed = max(
                0.0, float(getattr(self, "_red_light_front_speed", 0.0))
            )
            queue_active = bool(
                getattr(self, "_red_light_queue_active", False)
            )
            front_kind = str(
                getattr(self, "_front_vehicle_kind", "none")
            ).lower()
            front_is_walker = (
                "walker" in front_kind or "pedestrian" in front_kind
            )

            # 1) Same-route stopped vehicle queue.
            if (
                queue_active
                and not front_is_walker
                and np.isfinite(front_gap)
                and front_gap <= float(self.queue_detect_distance)
                and front_gap <= physical_gap + 3.0
            ):
                closing = max(0.0, speed - front_speed)
                front_gap = max(0.0, front_gap - closing * float(self.dt))
                queue_error = float(front_gap - self.queue_stop_gap)
                self._red_light_queue_gap_error = queue_error

                if queue_error <= 0.15:
                    cap = -4.0 if speed > 0.12 else -2.0
                    self._red_light_control_active = True
                    self._red_light_braking_active = True
                    self._red_light_control_reason = "queue_hold"
                    self._traffic_light_accel_cap = float(cap)
                    return min(float(accel), float(cap))

                a_comfort = max(float(self.red_light_comfort_decel), 1e-3)
                queue_activation_gap = float(
                    self.queue_stop_gap
                    + closing * float(self.red_light_reaction_time_s)
                    + (closing * closing) / (2.0 * a_comfort)
                    + float(self.red_light_activation_margin)
                )
                queue_ttc = (
                    20.0
                    if closing <= 1e-3
                    else float(np.clip(front_gap / closing, 0.0, 20.0))
                )
                if (
                    front_gap > queue_activation_gap
                    and queue_ttc >= float(self.vehicle_ttc_soft)
                ):
                    self._red_light_control_reason = "queue_track_only"
                    self._traffic_light_accel_cap = float(accel)
                    return float(accel)

                desired = self._red_light_queue_desired_speed(
                    queue_error, front_speed
                )
                req_stop = (closing * closing) / (
                    2.0 * max(queue_error, 0.30)
                )

                if speed > desired + 0.15 or req_stop >= a_comfort:
                    required = max(
                        req_stop,
                        max(0.0, speed * speed - desired * desired)
                        / (2.0 * max(queue_error, 0.30)),
                    )
                    cap = -float(np.clip(required + 0.10, 0.10, 4.0))
                    if queue_error < 1.0 and speed > 0.12:
                        cap = min(cap, -3.0)
                    self._red_light_control_active = True
                    self._red_light_braking_active = True
                    self._red_light_control_reason = "queue_brake"
                    self._traffic_light_accel_cap = float(cap)
                    return min(float(accel), float(cap))

                if (
                    0.60 < queue_error <= float(self.red_light_creep_distance)
                    and speed < desired - 0.10
                    and float(accel) >= -0.02
                ):
                    crawl_acc = float(
                        np.clip(0.50 * (desired - speed), 0.01, 0.25)
                    )
                    out = max(float(accel), crawl_acc)
                    self._red_light_control_active = True
                    self._red_light_control_reason = "queue_creep"
                    self._traffic_light_accel_cap = float(out)
                    return float(out)

                self._red_light_control_reason = "queue_monitor"
                self._traffic_light_accel_cap = float(accel)
                return float(accel)

            # 2) No queue: physical CARLA stop waypoint.
            if stop_error <= 0.10:
                cap = -4.0 if speed > 0.10 else -2.0
                self._red_light_control_active = True
                self._red_light_braking_active = True
                self._red_light_control_reason = "stop_line_hold"
                self._traffic_light_accel_cap = float(cap)
                return min(float(accel), float(cap))

            activation_distance = self._red_light_activation_distance(speed)
            if physical_gap > activation_distance:
                self._red_light_control_reason = "track_only"
                self._traffic_light_accel_cap = float(accel)
                return float(accel)

            desired_speed = self._red_light_desired_speed(stop_error)
            req_stop = (speed * speed) / (2.0 * max(stop_error, 0.30))
            comfortable_decel = float(self.red_light_comfort_decel)

            if speed > desired_speed + 0.15 or req_stop >= comfortable_decel:
                required = max(
                    req_stop,
                    max(0.0, speed * speed - desired_speed * desired_speed)
                    / (2.0 * max(stop_error, 0.30)),
                )
                cap = -float(np.clip(required + 0.10, 0.10, 4.0))
                self._red_light_control_active = True
                self._red_light_braking_active = True
                self._red_light_control_reason = "stop_line_brake"
                self._traffic_light_accel_cap = float(cap)
                return min(float(accel), float(cap))

            if (
                0.60 < stop_error <= float(self.red_light_creep_distance)
                and speed < desired_speed - 0.10
                and float(accel) >= -0.02
            ):
                crawl_acc = float(
                    np.clip(0.55 * (desired_speed - speed), 0.02, 0.35)
                )
                out = max(float(accel), crawl_acc)
                self._red_light_control_active = True
                self._red_light_control_reason = "stop_line_creep"
                self._traffic_light_accel_cap = float(out)
                return float(out)

            self._red_light_control_reason = "stop_line_monitor"
            self._traffic_light_accel_cap = float(accel)
            return float(accel)

        except Exception as exc:
            self._record_safety_exception("red_light_controller", exc)

            # A confirmed active stop signal remains safety-critical.
            if bool(getattr(self, "_traffic_light_active", False)):
                safe = self._fail_safe_brake(
                    accel, ego_speed, moving_decel=4.0, hold_decel=2.0
                )
                self._red_light_control_active = True
                self._red_light_braking_active = True
                self._red_light_control_reason = "confirmed_light_controller_fault"
                self._traffic_light_accel_cap = float(safe)
                return float(safe)

            # No confirmed light: only coast after repeated detector faults.
            detector_fault = bool(
                getattr(self, "_traffic_light_detector_fault_active", False)
            )
            detector_fault_ticks = int(
                getattr(self, "_traffic_light_detector_fault_ticks", 0)
            )
            if detector_fault and detector_fault_ticks >= 3:
                safe = min(float(accel), 0.0)
                self._red_light_control_active = bool(
                    safe < float(accel) - 1e-9
                )
                self._red_light_control_reason = "detector_fault_exception_coast"
                self._traffic_light_accel_cap = float(safe)
                return float(safe)

            return float(accel)
    def _apply_front_vehicle_collision_guard(self, accel: float, ego_speed: float) -> float:
        """Universal same-route front-actor guard with debounced scan faults.

        Fixes phantom stops caused by treating a single world.get_actors() failure
        as if a physical obstacle had been confirmed. A confirmed close actor still
        receives immediate fail-safe braking; one or two transient registry faults
        do not invent a hazard; persistent faults only suppress positive acceleration.
        """
        self._front_vehicle_guard_active = False
        self._front_vehicle_accel_cap = 2.5

        try:
            if not bool(getattr(self, "vehicle_collision_guard", True)):
                return float(accel)

            gap = float(getattr(self, "_front_vehicle_gap", 80.0))
            front_speed = max(
                0.0, float(getattr(self, "_front_vehicle_speed", 0.0))
            )
            kind = str(
                getattr(self, "_front_vehicle_kind", "none")
            ).lower()
            guard_valid = bool(
                getattr(self, "_front_vehicle_guard_valid", False)
            )

            scan_fault = bool(
                getattr(self, "_world_hazard_scan_fault_active", False)
            )
            scan_fault_ticks = int(
                getattr(self, "_world_hazard_scan_fault_ticks", 0)
            )

            if scan_fault:
                # Only a real, geometrically validated, close actor justifies
                # immediate fail-safe braking during a registry failure.
                confirmed_close_hazard = bool(
                    guard_valid
                    and np.isfinite(gap)
                    and gap < 79.0
                    and gap <= min(
                        float(getattr(self, "vehicle_detect_distance", 30.0)),
                        15.0,
                    )
                )

                if confirmed_close_hazard:
                    safe = self._fail_safe_brake(
                        accel=float(accel),
                        ego_speed=float(ego_speed),
                        moving_decel=3.5,
                        hold_decel=1.5,
                    )
                    self._front_vehicle_guard_active = True
                    self._front_vehicle_accel_cap = float(safe)
                    return float(safe)

                # The first two faults are treated as transient. The candidate
                # list still contains the environment's internal fallback actors,
                # so continue normal geometry-based processing below.
                if scan_fault_ticks >= 3 and not guard_valid:
                    safe = min(float(accel), 0.0)
                    self._front_vehicle_guard_active = bool(
                        safe < float(accel) - 1e-9
                    )
                    self._front_vehicle_accel_cap = float(safe)
                    return float(safe)

            if (
                (not guard_valid)
                or kind == "red_light_priority"
                or (not np.isfinite(gap))
                or gap >= 79.0
            ):
                return float(accel)

            speed = max(0.0, float(ego_speed))
            closing = max(0.0, speed - front_speed)
            is_walker = (
                "walker" in kind or "pedestrian" in kind
            )
            is_queue_vehicle = bool(
                (not is_walker)
                and self._front_actor_is_queue_like(gap, front_speed)
            )

            if is_walker:
                active_range = max(float(self.vehicle_detect_distance), 18.0)
            elif is_queue_vehicle:
                active_range = max(
                    float(self.queue_detect_distance),
                    float(self.vehicle_detect_distance),
                )
            else:
                active_range = float(self.vehicle_detect_distance)

            if gap > active_range:
                return float(accel)

            if is_walker:
                desired_gap = float(
                    max(getattr(self, "vehicle_stop_gap", 3.5), 3.5)
                )
            elif is_queue_vehicle:
                desired_gap = float(self.queue_stop_gap)
            else:
                desired_gap = float(self.vehicle_stop_gap) + float(
                    self.vehicle_moving_time_headway
                ) * speed
            desired_gap = float(np.clip(desired_gap, 1.0, 25.0))

            gap_error = float(gap - desired_gap)
            ttc = (
                20.0
                if closing <= 1e-3
                else float(np.clip(gap / closing, 0.0, 20.0))
            )
            self._front_vehicle_ttc = ttc

            if gap <= max(0.55, 0.70 * desired_gap):
                cap = -4.0 if speed > 0.10 else -2.0
            elif ttc < float(self.vehicle_ttc_hard):
                cap = -4.0
            elif gap_error < 0.0 or ttc < float(self.vehicle_ttc_soft):
                required = max(
                    0.0, speed * speed - front_speed * front_speed
                ) / (2.0 * max(gap - desired_gap, 0.50))
                cap = -float(
                    np.clip(required + 0.35 * closing + 0.20, 0.20, 4.0)
                )
            elif is_walker and gap < desired_gap + 3.0:
                cap = min(float(accel), -0.45 * max(closing, speed - 0.8))
            elif (
                gap < desired_gap + float(self.vehicle_soft_extra_gap)
                and closing > 0.15
            ):
                cap = min(float(accel), -0.25 * closing)
            else:
                return float(accel)

            self._front_vehicle_guard_active = True
            self._front_vehicle_accel_cap = float(cap)
            return min(float(accel), float(cap))

        except Exception as exc:
            self._record_safety_exception("front_vehicle_guard", exc)
            try:
                gap = float(getattr(self, "_front_vehicle_gap", 80.0))
            except Exception:
                gap = 80.0

            valid = bool(
                getattr(self, "_front_vehicle_guard_valid", False)
            )
            if (
                valid
                and np.isfinite(gap)
                and gap <= float(
                    getattr(self, "vehicle_detect_distance", 30.0)
                )
            ):
                safe = self._fail_safe_brake(
                    accel, ego_speed, moving_decel=4.0, hold_decel=2.0
                )
                self._front_vehicle_guard_active = True
                self._front_vehicle_accel_cap = float(safe)
                return float(safe)

            return float(accel)

    def _apply_route_turn_guard(self, steer_env: float, accel: float,
                                ego_speed: float) -> Tuple[float, float]:
        """Anticipate route turns and recover from safe low-speed turn stalls.

        The nominal controller historically used only the current lane tangent.
        At a sharp junction the correct future branch may not influence steering
        until the ego is already at the corner, causing overrun, oscillation, or
        a zero-speed stall.  This guard uses the precomputed route's look-ahead
        tangent while respecting front-actor and red-light blocking conditions.
        """
        self._route_turn_guard_active = False
        self._turn_recovery_active = False
        try:
            if self.route_tracker is None:
                self._turn_stuck_count = 0
                return float(steer_env), float(accel)

            route_lat = float(getattr(self, "_route_lat_error", 0.0))
            local_head = float(getattr(self, "_route_yaw_error", 0.0))
            look_head = float(getattr(self, "_route_lookahead_yaw_error", local_head))
            planned_turn = float(getattr(self, "_route_turn_strength", 0.0))

            # Defensive sanitization: the route builder intentionally rejects
            # U-turn-like branches. Therefore a >110-degree planned heading
            # change inside one short lookahead is treated as a route/topology
            # discontinuity rather than a steering target. Fall back to the
            # current route tangent for that tick instead of saturating steering.
            if not np.isfinite(route_lat):
                route_lat = 0.0
            if not np.isfinite(local_head):
                local_head = 0.0
            if not np.isfinite(look_head):
                look_head = local_head
            if not np.isfinite(planned_turn):
                planned_turn = 0.0

            max_plausible_turn = math.radians(110.0)
            if abs(planned_turn) > max_plausible_turn:
                look_head = local_head
                planned_turn = 0.0

            turn_severity = float(np.clip(
                max(abs(planned_turn), 0.65 * abs(look_head)) / 0.70, 0.0, 1.0
            ))
            off_center = abs(route_lat) > 0.45
            needs_turn_help = bool(turn_severity > 0.08 or off_center or abs(local_head) > 0.20)
            if not needs_turn_help:
                self._turn_stuck_count = 0
                return float(steer_env), float(accel)

            # Blend current-route and future-route heading.  The future term is
            # stronger in a real turn, while current heading damps oscillation.
            head_for_control = (1.0 - 0.65 * turn_severity) * local_head +                                (0.65 * turn_severity) * look_head
            correction = (
                -0.55 * route_lat
                -float(self.route_turn_steer_gain) * head_for_control
            )
            correction = float(np.clip(correction, -0.60, 0.60))

            blend = float(np.clip(0.22 + 0.58 * turn_severity + 0.15 * min(abs(route_lat), 1.0),
                                  0.20, 0.90))
            steer_out = float(np.clip(
                (1.0 - blend) * float(steer_env) + blend * correction,
                -0.60, 0.60,
            ))

            # Slow before sharp turns instead of arriving fast and then braking
            # after the vehicle has already left the route centerline.
            # Speed cap ONLY for genuine turns. off_center/heading alone must
            # give steering help without a hidden speed governor: previously a
            # 45 cm offset on a straight forced braking toward ~6 m/s.
            if turn_severity > 0.12:
                target_speed = float(self.route_turn_speed + 1.5 * (1.0 - turn_severity))
                acc_cap = float(np.clip(0.90 * (target_speed - max(0.0, float(ego_speed))),
                                        -4.0, 1.2))
                accel_out = min(float(accel), acc_cap)
            else:
                accel_out = float(accel)

            # A recovery pulse is allowed only when there is no close lead,
            # queue, pedestrian, or red-light stop demand.  Never override a
            # deliberate braking command from the filter/ACC.
            front_gap = float(getattr(self, "_front_vehicle_gap", 80.0))
            front_speed = float(getattr(self, "_front_vehicle_speed", 0.0))
            desired_gap = self._desired_front_gap(max(0.0, float(ego_speed)),
                                                  front_gap, front_speed)
            blocked_front = bool(front_gap < max(6.0, desired_gap + 1.5))
            blocked_queue = bool(getattr(self, "_red_light_queue_active", False))
            red_close = bool(
                getattr(self, "_traffic_light_active", False)
                and float(getattr(self, "_traffic_light_stop_error", 999.0)) < 7.0
            )
            blocked = bool(blocked_front or blocked_queue or red_close)

            if (max(0.0, float(ego_speed)) < self.turn_recovery_speed
                    and not blocked
                    and float(accel) > -0.10
                    and (turn_severity > 0.10 or off_center)):
                self._turn_stuck_count += 1
            else:
                self._turn_stuck_count = 0

            if self._turn_stuck_count >= self.turn_recovery_patience_ticks:
                # A radial "any moving actor within 15 m" test deadlocked beside
                # parallel traffic.  Recovery now depends on the time-aligned
                # route/velocity conflict predictor run earlier in this action.
                pulse_ok = not bool(getattr(
                    self, "_predictive_conflict_active", False
                ))
                if pulse_ok:
                    accel_out = max(accel_out, float(self.turn_recovery_accel))
                    steer_out = correction
                    self._turn_recovery_active = True
                else:
                    self._turn_stuck_count = self.turn_recovery_patience_ticks  # hold, re-check next tick

            self._route_turn_guard_active = True
            return float(steer_out), float(accel_out)
        except Exception as exc:
            self._record_safety_exception("route_turn_guard", exc)
            self._turn_stuck_count = 0
            try:
                turn_strength = abs(float(getattr(self, "_route_turn_strength", 0.0)))
            except Exception:
                turn_strength = 1.0
            safe_accel = float(accel)
            if turn_strength > 0.12 and float(ego_speed) > float(getattr(self, "route_turn_speed", 4.5)):
                safe_accel = min(float(accel), -1.5)
            return float(np.clip(steer_env, -0.60, 0.60)), float(safe_accel)


    def _apply_lane_edge_guard(self, steer_env: float, accel: float,
                               ego_speed: float) -> Tuple[float, float]:
        """Protect legal outer road boundaries in normal driving and overtaking."""
        previous_lane_edge_margin = float(getattr(self, "_lane_edge_margin", 999.0))
        self._lane_edge_guard_active = False
        self._lane_edge_accel_cap = 2.5
        self._lane_edge_steer_correction = 0.0
        self._lane_edge_margin = 999.0
        try:
            if not bool(getattr(self, "road_edge_guard", True)):
                return float(steer_env), float(accel)
            carla = self._carla
            tf = self.ego.get_transform()
            loc = tf.location
            wp = self.map.get_waypoint(
                loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if wp is None:
                return float(steer_env), float(min(accel, -1.0))
            wp_loc = wp.transform.location
            right = wp.transform.get_right_vector()
            lat = float(
                (loc.x - wp_loc.x) * right.x + (loc.y - wp_loc.y) * right.y
            )
            heading = _wrap_pi(
                math.radians(tf.rotation.yaw - wp.transform.rotation.yaw)
            )
            lane_half = max(0.5 * float(wp.lane_width), 1.0)

            soft = float(self.lane_edge_soft_margin)
            hard = float(self.lane_edge_hard_margin)
            route_head = float(getattr(self, "_route_yaw_error", heading))
            route_look = float(getattr(self, "_route_lookahead_yaw_error", route_head))
            route_lat = float(getattr(self, "_route_lat_error", lat))
            planned_turn = float(getattr(self, "_route_turn_strength", 0.0))
            at_junction = bool(getattr(wp, "is_junction", False))

            corridor = self._overtake_corridor_geometry(
                wp, loc, math.radians(float(tf.rotation.yaw))
            )
            if str(getattr(self, "_overtake_mode", "idle")) != "idle" and corridor is not None:
                lat_control, heading_control, lane_half = corridor
                self._overtake_corridor_valid = True
            else:
                if str(getattr(self, "_overtake_mode", "idle")) != "idle":
                    self._overtake_corridor_valid = False
                use_route_geometry = bool(at_junction or planned_turn > 0.08)
                lat_control = route_lat if use_route_geometry and np.isfinite(route_lat) else lat
                heading_control = (
                    0.35 * heading + 0.65 * route_look
                    if use_route_geometry and np.isfinite(route_look) else heading
                )

            margin = float(lane_half - abs(float(lat_control)))
            self._lane_edge_margin = margin
            edge_risk = bool(margin <= soft)
            heading_risk = bool(abs(float(heading_control)) >= 0.45)
            if not edge_risk and not heading_risk:
                return float(steer_env), float(accel)

            correction = (
                -float(self.lane_edge_steer_gain) * float(lat_control)
                -float(self.lane_edge_heading_gain) * float(heading_control)
                -0.25 * route_head
            )
            correction = float(np.clip(correction, -0.60, 0.60))
            severity = (
                float(np.clip((soft - margin) / max(soft - hard, 1e-3), 0.0, 1.0))
                if edge_risk else 0.0
            )
            blend = (0.35 + 0.55 * severity) if edge_risk else 0.35
            steer_out = float(np.clip(
                (1.0 - blend) * float(steer_env) + blend * correction,
                -0.60, 0.60,
            ))

            if edge_risk:
                target_speed = float(self.lane_edge_target_speed) + 1.5 * (1.0 - severity)
                acc_cap = 0.90 * (target_speed - max(0.0, float(ego_speed)))
                if margin < hard and ego_speed > target_speed:
                    acc_cap = min(acc_cap, -float(self.lane_edge_brake))
                acc_cap = float(np.clip(acc_cap, -4.0, 1.0))
                accel_out = min(float(accel), acc_cap)
            else:
                accel_out = float(accel)

            self._lane_edge_guard_active = True
            self._lane_edge_accel_cap = float(accel_out)
            self._lane_edge_steer_correction = correction
            return steer_out, float(accel_out)
        except Exception as exc:
            self._record_safety_exception("lane_edge_guard", exc)
            known_margin = float(getattr(self, "_lane_edge_margin", previous_lane_edge_margin))
            safe_accel = float(accel)
            if known_margin <= float(getattr(self, "lane_edge_soft_margin", 0.75)):
                safe_accel = min(
                    float(accel), -float(getattr(self, "lane_edge_brake", 2.5))
                )
            return float(np.clip(steer_env, -0.60, 0.60)), float(safe_accel)


    def _reset_external_blockage_candidate(self) -> None:
        self._external_blockage_actor_id = -1
        self._external_blockage_ticks = 0


    def _maybe_recover_external_blockage(
        self,
        ego_speed: float,
        delta_progress: float,
    ) -> bool:
        """Remove one persistently stuck, experiment-owned NPC when enabled.

        A collided Traffic-Manager vehicle can remain across a junction for the
        rest of an episode.  Accelerating through that vehicle is unsafe, while
        waiting until max_steps only converts a simulator artifact into a policy
        timeout.  This bounded recovery therefore removes the blocker itself,
        but only when all of the following hold:

        * the option is explicitly enabled;
        * ego has remained nearly stationary;
        * the same nearby actor is reported by the front or predictive guard;
        * the actor belongs to ``self.traffic_vehicles`` and is stationary;
        * neither ego nor the candidate is legitimately waiting at a red/yellow
          signal; and
        * the configurable grace period has elapsed.

        Every intervention is logged and exposed in ``info`` so evaluation can
        report simulator recoveries separately from policy performance.
        """
        self._external_blockage_recovered_this_step = False
        if (
            not bool(self.external_blockage_recovery)
            or int(self._external_blockage_recovery_count)
                >= int(self.external_blockage_max_recoveries)
        ):
            self._reset_external_blockage_candidate()
            return False

        if max(0.0, float(ego_speed)) > 0.35 or float(delta_progress) > 0.03:
            self._reset_external_blockage_candidate()
            return False

        light_state = str(getattr(self, "_traffic_light_state", "none")).lower()
        ego_signal_hold = bool(
            getattr(self, "_traffic_light_active", False)
            and (light_state.startswith("red") or light_state.startswith("yellow"))
            and float(getattr(self, "_traffic_light_stop_error", 999.0)) < 12.0
        )
        if ego_signal_hold:
            self._reset_external_blockage_candidate()
            return False

        candidate_id = -1
        candidate_reason = "none"
        if (
            bool(getattr(self, "_front_vehicle_guard_valid", False))
            and int(getattr(self, "_front_vehicle_actor_id", -1)) >= 0
            and float(getattr(self, "_front_vehicle_gap", 999.0)) <= 20.0
            and float(getattr(self, "_front_vehicle_speed", 999.0)) <= 0.25
        ):
            candidate_id = int(self._front_vehicle_actor_id)
            candidate_reason = "front_blocker"
        elif (
            bool(getattr(self, "_predictive_conflict_active", False))
            and int(getattr(self, "_predictive_conflict_actor_id", -1)) >= 0
            and float(getattr(self, "_predictive_conflict_distance", 999.0)) <= 25.0
        ):
            candidate_id = int(self._predictive_conflict_actor_id)
            candidate_reason = str(
                getattr(self, "_predictive_conflict_kind", "predictive_blocker")
            )

        owned = {}
        for actor in list(getattr(self, "traffic_vehicles", [])):
            try:
                if actor is not None and bool(actor.is_alive):
                    owned[int(actor.id)] = actor
            except Exception:
                continue
        actor = owned.get(candidate_id)
        if actor is None:
            self._reset_external_blockage_candidate()
            return False

        try:
            actor_speed = max(0.0, self._vehicle_speed(actor))
            if actor_speed > 0.25:
                self._reset_external_blockage_candidate()
                return False

            # Never delete a normal Traffic-Manager queue member waiting for its
            # own red/yellow signal.
            actor_signal_hold = False
            try:
                if bool(actor.is_at_traffic_light()):
                    actor_light = str(actor.get_traffic_light_state()).lower()
                    actor_signal_hold = (
                        "red" in actor_light or "yellow" in actor_light
                    )
            except Exception:
                actor_signal_hold = False
            if actor_signal_hold:
                self._reset_external_blockage_candidate()
                return False

            if int(self._external_blockage_actor_id) == int(candidate_id):
                self._external_blockage_ticks += 1
            else:
                self._external_blockage_actor_id = int(candidate_id)
                self._external_blockage_ticks = 1

            required_ticks = max(
                1,
                int(math.ceil(
                    float(self.external_blockage_patience_s)
                    / max(float(self.dt), 1e-6)
                )),
            )
            if int(self._external_blockage_ticks) < required_ticks:
                return False

            destroyed = bool(actor.destroy())
            if not destroyed:
                self._record_safety_exception(
                    "external_blockage_destroy",
                    RuntimeError(f"CARLA refused to destroy blocker {candidate_id}"),
                )
                self._reset_external_blockage_candidate()
                return False

            self.traffic_vehicles = [
                item for item in self.traffic_vehicles
                if int(getattr(item, "id", -1)) != int(candidate_id)
            ]
            self._external_blockage_recovery_count += 1
            self._external_blockage_recovered_this_step = True
            self._external_blockage_last_reason = str(candidate_reason)
            waited_s = float(self._external_blockage_ticks) * float(self.dt)
            print(
                "[carla-safe] external_blockage_recovery "
                f"actor={candidate_id} reason={candidate_reason} "
                f"waited={waited_s:.1f}s "
                f"count={self._external_blockage_recovery_count}",
                flush=True,
            )

            if int(getattr(self, "_front_vehicle_actor_id", -1)) == candidate_id:
                self._front_vehicle_actor_id = -1
                self._front_vehicle_gap = 80.0
                self._front_vehicle_speed = 0.0
                self._front_vehicle_kind = "none"
                self._front_vehicle_guard_valid = False
                self._lead_is_relevant = False
            if int(getattr(self, "_predictive_conflict_actor_id", -1)) == candidate_id:
                self._predictive_conflict_raw = False
                self._predictive_conflict_active = False
                self._predictive_conflict_actor_id = -1
                self._predictive_conflict_actor_type = "none"
                self._predictive_conflict_kind = "none"
                self._predictive_conflict_ttc = 999.0
                self._predictive_conflict_distance = 999.0
            self._reset_external_blockage_candidate()
            return True
        except Exception as exc:
            self._record_safety_exception("external_blockage_recovery", exc)
            self._reset_external_blockage_candidate()
            return False


    def _accel_to_actuator_commands(self, accel: float, ego_speed: float,
                                    precision_creep: bool) -> Tuple[float, float]:
        """Map physical acceleration command to mutually exclusive throttle/brake.

        Negative acceleration can never become throttle.  A small deadband is
        true coasting; precision creep requires an explicitly positive command.
        """
        a = float(np.clip(accel, -4.0, 2.5))
        speed = max(0.0, float(ego_speed))
        deadband = 0.03

        if a > deadband:
            throttle = float(np.clip(a / 2.5, 0.0, 1.0))
            if precision_creep:
                creep_cap = 0.18 if speed < 1.0 else 0.12
                throttle = float(np.clip(throttle, 0.03, creep_cap))
            elif speed < 1.0:
                throttle = max(throttle, max(self.throttle_floor, 0.45))
            elif speed < 3.0:
                throttle = max(throttle, max(min(self.throttle_floor, 0.45), 0.35))
            elif a > 0.05:
                throttle = max(throttle, 0.12)
            return float(throttle), 0.0

        if a < -deadband:
            return 0.0, float(np.clip(-a / 4.0, 0.0, 1.0))

        return 0.0, 0.0

    def _apply_ego_action(self, action: np.ndarray):
        """Apply a smooth but movement-preserving low-level CARLA control.

        Why this version is less likely to freeze:
        * CARLA cars often need a non-trivial throttle to break static friction.
        * During early SAC training, normalized acceleration can hover near zero.
        * If near-zero acceleration maps to near-zero throttle, the replay buffer
          fills with static transitions and the policy learns that standing still
          is the safest behavior.

        The logic below gives a stronger throttle only when the car is nearly
        stopped or the policy asks for positive acceleration. When the policy
        asks to brake, braking still wins and throttle is suppressed.
        """
        carla = self._carla
        u = np.asarray(action, dtype=np.float64).reshape(2)
        steer_env = float(np.clip(u[0], -0.60, 0.60))
        accel = float(np.clip(u[1], -4.0, 2.5))                    # m/s^2 command
        ego_speed = max(0.0, self._vehicle_speed(self.ego))

        # Priority: traffic rules may command a red-light crawl, but the
        # universal front-vehicle guard can still override that crawl with
        # braking if a spawned/NPC vehicle is actually in front.  Lane-edge guard
        # then corrects steering and slows down near road boundaries.
        accel = self._apply_red_light_close_stop(accel, ego_speed)
        accel = self._apply_front_vehicle_collision_guard(accel, ego_speed)
        # This guard covers hazards deliberately excluded by the same-direction
        # front filter: junction crossing traffic and adjacent-lane cut-ins.
        # It is brake-only, so no later helper may turn its command positive.
        accel = self._apply_predictive_collision_guard(accel, ego_speed)
        steer_env, accel = self._apply_route_turn_guard(steer_env, accel, ego_speed)
        steer_env, accel = self._apply_overtake_guard(steer_env, accel, ego_speed)
        # The lane-edge guard is corridor-aware: during a legal overtake it uses
        # the combined two-lane outer boundaries, so it never needs to be disabled.
        steer_env, accel = self._apply_lane_edge_guard(steer_env, accel, ego_speed)

        # Snapshot action-time guard decisions before _read_state performs the
        # next sensing pass. These are the values step() must log.
        self._front_vehicle_guard_active_last_action = bool(self._front_vehicle_guard_active)
        self._front_vehicle_accel_cap_last_action = float(self._front_vehicle_accel_cap)
        self._predictive_conflict_guard_active_last_action = bool(
            self._predictive_conflict_guard_active
        )
        self._predictive_conflict_accel_cap_last_action = float(
            self._predictive_conflict_accel_cap
        )
        self._lane_edge_guard_active_last_action = bool(self._lane_edge_guard_active)
        self._route_turn_guard_active_last_action = bool(self._route_turn_guard_active)
        self._overtake_guard_active_last_action = bool(self._overtake_guard_active)
        self._overtake_mode_last_action = str(getattr(self, "_overtake_mode", "idle"))

        steer_cmd = float(np.clip(steer_env / 0.6, -1.0, 1.0))     # 0.6 rad ~ full lock
        self._last_env_action_applied = np.asarray([steer_env, accel], dtype=np.float64)

        # Precision creep is a special actuator regime. A tiny +0.02 m/s^2
        # red-light command must not be promoted to the generic 35--45% throttle
        # floor used to break static friction during ordinary policy motion.
        red_stop_error = float(getattr(self, "_traffic_light_stop_error", 999.0))
        queue_gap_error = float(getattr(self, "_red_light_queue_gap_error", 999.0))
        near_red_creep = bool(
            getattr(self, "_traffic_light_active", False)
            and getattr(self, "_red_light_control_active", False)
            and 0.10 < red_stop_error <= float(self.red_light_creep_distance) + 0.5
        )
        near_queue_creep = bool(
            getattr(self, "_red_light_queue_active", False)
            and -0.25 < queue_gap_error <= 4.5
        )
        precision_creep = bool(near_red_creep or near_queue_creep)

        throttle_cmd, brake_cmd = self._accel_to_actuator_commands(
            accel, ego_speed, precision_creep
        )

        # Reduce smoothing when breaking a standstill; otherwise smoothing can
        # keep old brake commands alive and prevent the first movement.
        beta = self.action_smoothing_beta
        if ego_speed < 0.5 and throttle_cmd > 0.30:
            beta_eff = min(beta, 0.05)
        else:
            beta_eff = beta

        steer = beta_eff * self._last_steer + (1.0 - beta_eff) * steer_cmd
        throttle = beta_eff * self._last_throttle + (1.0 - beta_eff) * throttle_cmd
        brake = beta_eff * self._last_brake + (1.0 - beta_eff) * brake_cmd

        # Avoid fighting commands after smoothing.  Most importantly, a
        # negative acceleration command can never retain positive throttle from
        # the previous tick through the low-pass state.
        if accel < -0.03 or brake_cmd > 0.0:
            throttle = 0.0
        elif accel > 0.03 or throttle_cmd > 0.0:
            brake = 0.0
        elif throttle > 0.0 and brake > 0.0:
            # True coasting deadband: keep only the larger residual actuator.
            if throttle >= brake:
                brake = 0.0
            else:
                throttle = 0.0

        self._last_steer = float(np.clip(steer, -1.0, 1.0))
        self._last_throttle = float(np.clip(throttle, 0.0, 1.0))
        self._last_brake = float(np.clip(brake, 0.0, 1.0))
        self.ego.apply_control(carla.VehicleControl(
            throttle=self._last_throttle,
            steer=self._last_steer,
            brake=self._last_brake,
            hand_brake=False,
            manual_gear_shift=False,
            reverse=False))
        return self._last_env_action_applied.copy()

    def _follow_spectator(self):
        if not self.render_follow:
            return
        try:
            carla = self._carla
            spec = self.world.get_spectator()
            tf = self.ego.get_transform()
            fwd = _carla_vec_forward(tf)
            cam = carla.Transform(
                carla.Location(x=tf.location.x - 8 * fwd[0], y=tf.location.y - 8 * fwd[1],
                               z=tf.location.z + 5.0),
                carla.Rotation(pitch=-15.0, yaw=tf.rotation.yaw))
            spec.set_transform(cam)
        except Exception:
            pass

    # -- gym-like API ---------------------------------------------------------
    def _reset_episode_runtime_state(self) -> None:
        self._collided = False
        self._collision_actor_type = "none"
        self._collision_zone = "none"
        self._collision_impulse = 0.0
        self.t = 0
        self.progress_m = 0.0
        self._lateral_fail_count = 0
        self._heading_fail_count = 0
        self._headway_fail_count = 0
        self._obstacle_fail_count = 0
        self._route_lat_error = 0.0
        self._route_yaw_error = 0.0
        self._state_geometry_source = "map"
        self._map_lane_lat_error = 0.0
        self._map_lane_yaw_error = 0.0
        self._route_lookahead_yaw_error = 0.0
        self._route_turn_strength = 0.0
        self._route_curvature = 0.0
        self._route_turn_guard_active = False
        self._turn_recovery_active = False
        self._turn_stuck_count = 0
        self._predictive_conflict_raw = False
        self._predictive_conflict_active = False
        self._predictive_conflict_clear_ticks = 0
        self._predictive_conflict_actor_id = -1
        self._predictive_conflict_actor_type = "none"
        self._predictive_conflict_kind = "none"
        self._predictive_conflict_ttc = 999.0
        self._predictive_conflict_distance = 999.0
        self._predictive_conflict_accel_cap = 2.5
        self._predictive_conflict_guard_active = False
        self._predictive_conflict_guard_active_last_action = False
        self._predictive_conflict_accel_cap_last_action = 2.5
        self._predictive_junction_context = False
        self._external_blockage_actor_id = -1
        self._external_blockage_ticks = 0
        self._external_blockage_recovery_count = 0
        self._external_blockage_recovered_this_step = False
        self._external_blockage_last_reason = "none"
        self._safety_trace.clear()
        # Per-tick route-projection cache used by _read_state to reuse the
        # projection already computed by _update_progress on the same tick.
        # Must be invalidated at episode start so a stale tick index from a
        # previous episode can never alias the new episode's t counter.
        self._progress_tick_t = -1
        self._route_yaw_now = 0.0
        self._prev_env_action = np.zeros(2, dtype=np.float64)
        self._lead_is_relevant = False
        self._lead_same_lane = False
        self._lead_route_lat = 999.0
        self._lead_route_ds = 999.0
        self._lead_actor_kind = "none"
        self._traffic_light_active = False
        self._traffic_light_state = "none"
        self._traffic_light_distance = 999.0
        self._traffic_light_virtual_gap = 999.0
        self._traffic_light_stop_error = 999.0
        self._traffic_light_accel_cap = 2.5
        self._traffic_light_id = -1
        self._traffic_light_yellow_go = False
        self._yellow_required_stop_distance = 0.0
        self._yellow_go_signal_keys = set()
        self._traffic_light_miss_ticks = 0
        self._traffic_light_detection_dropout = False
        self._traffic_light_last_signal_key = None
        self._traffic_light_last_stop_s = -1.0
        self._red_light_crossed_on_red = False
        self._red_light_violation_count = 0
        self._red_light_stop_success = False
        self._red_light_stop_success_count = 0
        self._red_light_last_success_key = None
        self._traffic_light_detector_fault_active = False
        self._traffic_light_detector_fault_ticks = 0
        self._tracked_light_id = None
        self._tracked_signal_key = None
        self._tracked_stop_s = -1.0
        self._tracked_stop_gap_prev = 999.0
        self._tracked_light_seen_ahead = False
        self._tracked_light_state = "none"
        self._tracked_yellow_go_latched = False
        self._committed_light_id = None
        self._committed_signal_key = None
        self._committed_stop_s = -1.0
        self._junction_commit_active = False
        self._junction_seen_since_commit = False
        self._junction_exit_clear_ticks = 0
        self._red_light_control_active = False
        self._red_light_front_gap = 80.0
        self._red_light_front_speed = 0.0
        self._red_light_queue_active = False
        self._red_light_queue_gap_error = 999.0
        self._front_vehicle_gap = 80.0
        self._front_vehicle_speed = 0.0
        self._front_vehicle_kind = "none"
        self._front_vehicle_ttc = 20.0
        self._front_vehicle_actor_id = -1
        self._front_vehicle_guard_valid = False
        self._front_vehicle_guard_active = False
        self._front_vehicle_accel_cap = 2.5
        self._front_vehicle_guard_active_last_action = False
        self._front_vehicle_accel_cap_last_action = 2.5
        self._lane_edge_guard_active = False
        self._lane_edge_margin = 999.0
        self._lane_edge_accel_cap = 2.5
        self._lane_edge_steer_correction = 0.0
        self._lane_edge_guard_active_last_action = False
        self._route_turn_guard_active_last_action = False
        self._last_env_action_applied = np.zeros(2, dtype=np.float64)
        self._last_steer = 0.0
        self._last_throttle = 0.0
        self._last_brake = 0.0
        self._route_projection_rejected = False
        self._route_projection_reject_reason = ""
        self._reset_overtake_state()

    def reset(self) -> np.ndarray:
        """Reset one CARLA episode with verified teardown and safe initialization."""
        if getattr(self, "_connection_lost", False):
            # The previous episode ended with the server unreachable.  Every
            # RPC against the dead session stalls for the full client timeout,
            # so rebuild the connection first; stale actors from the aborted
            # episode are swept by _destroy_stale_actors() below.
            if not self.reconnect():
                raise RuntimeError(
                    "carla_connection_lost: the CARLA server is unreachable and "
                    "reconnect failed. Check/restart the CARLA server on "
                    f"{self.host}:{self.port}."
                )
        elif self.world is None:
            self._connect()
        else:
            self._destroy_actors()

        # Catch orphan actors from older/crashed runs before any new ego is spawned.
        survivors = self._destroy_stale_actors()
        if survivors:
            raise RuntimeError(
                f"Refusing reset because stale experiment actors survived cleanup: {survivors}"
            )

        carla = self._carla
        last_error: Optional[Exception] = None
        max_attempts = 8

        for attempt in range(1, max_attempts + 1):
            self._reset_episode_runtime_state()
            try:
                self._spawn()
                self._current_weather_mode = self._select_episode_weather_mode()
                self._apply_weather()

                # Hold the ego stationary while the route and surrounding traffic are
                # prepared. The old 12-tick blind throttle burst could hit a ghost or
                # newly spawned actor before hazard sensing existed.
                brake_hold = carla.VehicleControl(
                    throttle=0.0, brake=1.0, steer=0.0,
                    hand_brake=False, manual_gear_shift=False, reverse=False,
                )
                self.ego.apply_control(brake_hold)
                if self.lead is not None and self.lead.is_alive:
                    self.lead.apply_control(carla.VehicleControl(
                        throttle=0.0, brake=1.0, steer=0.0,
                        hand_brake=False, manual_gear_shift=False, reverse=False,
                    ))
                self.world.tick()

                start_wp = self.map.get_waypoint(
                    self.ego.get_location(), project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if start_wp is None:
                    raise RuntimeError("Spawned ego has no valid driving-lane waypoint.")
                self._build_destination(start_wp)
                if self.route_tracker is not None:
                    self.route_tracker.reset()

                self._spawn_background_vehicles()
                self._spawn_walkers()
                self._update_traffic_speed_stats()

                # Reassert brake after traffic/walker warmup and let one clean tick
                # settle collision callbacks.
                if self.ego is None or not self.ego.is_alive:
                    raise RuntimeError("Ego died during episode initialization.")
                self.ego.apply_control(brake_hold)
                self.world.tick()
                self._update_traffic_speed_stats()

                if self._collided:
                    raise RuntimeError(
                        "Collision occurred during reset/warmup; rejecting this episode start."
                    )

                self.progress_m = 0.0
                self._prev_loc = self.ego.get_location()
                self._last_steer = self._last_throttle = self._last_brake = 0.0
                self._follow_spectator()
                state = self._read_state()
                self._runtime_barrier_values(state)

                # Initial state must be geometrically sane.  This rejects the
                # observed -16 m/-29 m route-lateral projection failures before they
                # can poison replay or trigger a false 10-tick termination.
                #
                # The gates must be strictly TIGHTER than the persistent-failure
                # thresholds used in step(): lateral_bad triggers at
                # |lat| > lane_half + 0.50 and heading_bad at
                # |heading| > heading_limit_rad + 0.35.  A brake-held ego at zero
                # speed cannot reduce either error within failure_persistence_ticks,
                # so any start inside those regions deterministically terminates
                # with progress=0 (previous gates of 2.25*lane_half and 80 deg let
                # exactly such episodes through).
                lane_half = max(float(self.barrier.lane_half_width), 1.0)
                lat_gate = lane_half + 0.35
                heading_gate = float(self.barrier.heading_limit_rad) + 0.25
                if abs(float(state[0])) > lat_gate:
                    raise RuntimeError(
                        f"Invalid initial lateral state: {float(state[0]):.2f} m "
                        f"(gate {lat_gate:.2f} m)"
                    )
                if abs(float(state[1])) > heading_gate:
                    raise RuntimeError(
                        f"Invalid initial heading state: {float(state[1]):.2f} rad "
                        f"(gate {heading_gate:.2f} rad)"
                    )

                return state
            except Exception as exc:
                last_error = exc
                self._record_safety_exception(f"reset_attempt_{attempt}", exc)
                if self._root_cause_is_import_error(exc):
                    raise RuntimeError(
                        "Route planning failed due to a missing Python module and "
                        "retrying cannot fix an interpreter environment. The active "
                        "environment cannot import CARLA's `agents` package "
                        "(GlobalRoutePlanner). Activate the environment your earlier "
                        "runs used, or point this one at the simulator install:\n"
                        "  export CARLA_ROOT=/path/to/CARLA_0.9.15\n"
                        "  export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH\n"
                        f"Original error: {exc}"
                    ) from exc
                if getattr(self, "_connection_lost", False) or self._is_connection_error(exc):
                    # The server is unreachable: further attempts would each
                    # burn multiple full-timeout RPC stalls.  Raise a marked
                    # error immediately; the next reset() call goes through
                    # reconnect() (or the training driver does).
                    self._connection_lost = True
                    raise RuntimeError(
                        f"carla_connection_lost during reset attempt {attempt}: {exc}"
                    ) from exc
                try:
                    self._destroy_actors()
                except Exception as destroy_exc:
                    self._record_safety_exception("reset_retry_destroy", destroy_exc)
                    raise
                survivors = self._destroy_stale_actors()
                if survivors:
                    raise RuntimeError(
                        f"Reset retry aborted because stale actors remain alive: {survivors}"
                    ) from exc

        raise RuntimeError(
            f"Failed to create a safe CARLA episode after {max_attempts} attempts: {last_error}"
        )

    def step(self, action):
        action = np.nan_to_num(np.asarray(action, dtype=np.float64).reshape(2),
                               nan=0.0, posinf=2.5, neginf=-4.0)
        action[0] = float(np.clip(action[0], -0.60, 0.60))
        action[1] = float(np.clip(action[1], -4.00, 2.50))
        prev_action = self._prev_env_action.copy()

        actual_action = self._apply_ego_action(action)
        # Stable action-time snapshots: _read_state() senses the next state and
        # must not retroactively rewrite what the just-executed guard actually did.
        front_guard_active_last = bool(getattr(
            self, "_front_vehicle_guard_active_last_action",
            getattr(self, "_front_vehicle_guard_active", False),
        ))
        front_guard_cap_last = float(getattr(
            self, "_front_vehicle_accel_cap_last_action",
            getattr(self, "_front_vehicle_accel_cap", 2.5),
        ))
        predictive_guard_active_last = bool(getattr(
            self, "_predictive_conflict_guard_active_last_action",
            getattr(self, "_predictive_conflict_guard_active", False),
        ))
        predictive_guard_cap_last = float(getattr(
            self, "_predictive_conflict_accel_cap_last_action",
            getattr(self, "_predictive_conflict_accel_cap", 2.5),
        ))
        lane_guard_active_last = bool(getattr(
            self, "_lane_edge_guard_active_last_action",
            getattr(self, "_lane_edge_guard_active", False),
        ))
        turn_guard_active_last = bool(getattr(
            self, "_route_turn_guard_active_last_action",
            getattr(self, "_route_turn_guard_active", False),
        ))
        overtake_guard_active_last = bool(getattr(
            self, "_overtake_guard_active_last_action", False
        ))
        overtake_mode_last = str(getattr(
            self, "_overtake_mode_last_action", getattr(self, "_overtake_mode", "idle")
        ))
        self._drive_lead()
        if self.vary_weather and (self.t % 10 == 0):
            self._apply_weather()
        try:
            self.world.tick()
        except Exception as exc:
            if self._is_connection_error(exc):
                # Server stall/crash mid-step.  Mark the connection dead so no
                # further RPC (each a full client-timeout stall) is attempted
                # until reconnect(), and raise a marked error the training
                # driver can recover from instead of core-dumping the run.
                self._connection_lost = True
                self._record_safety_exception("step_tick_connection", exc)
                raise RuntimeError(
                    f"carla_connection_lost during world.tick(): {exc}"
                ) from exc
            raise
        self._update_traffic_speed_stats()
        self._follow_spectator()
        self.t += 1

        delta_progress = self._update_progress()
        state = self._read_state()
        if self._maybe_recover_external_blockage(
            ego_speed=float(state[2]),
            delta_progress=float(delta_progress),
        ):
            # Actor.destroy() is synchronous in CARLA. Re-sense immediately so
            # the removed blocker cannot remain in the next state/barrier data.
            state = self._read_state()
        comps = self._runtime_barrier_values(state)
        h = float(np.min(comps))
        speed = float(state[2])

        lane_m, heading_m, speed_max_m, speed_min_m, headway_m, obstacle_m = [
            float(x) for x in comps
        ]
        desired_speed = self._desired_speed_from_headway(state, comps)

        nominal_delta = max(self.target_speed * self.dt, 1e-6)
        progress_reward = 4.5 * float(np.clip(delta_progress / nominal_delta, 0.0, 2.0))
        speed_sigma = max(2.0, 0.25 * self.target_speed)
        speed_reward = 0.50 * math.exp(-((speed - desired_speed) / speed_sigma) ** 2)

        lane_half = max(self.barrier.lane_half_width, 1.0)
        reward_lat = float(getattr(self, "_reward_lane_lat_error", state[0]))
        reward_heading = float(getattr(self, "_reward_lane_heading_error", state[1]))
        lane_center_pen = -0.15 * (reward_lat / lane_half) ** 2
        heading_center_pen = -0.12 * (reward_heading / max(self.barrier.heading_limit_rad, 1e-3)) ** 2
        route_heading_pen = -0.08 * (float(self._route_yaw_error) / 0.6) ** 2

        gap = float(state[3])
        rel_vel = float(state[4])
        front_speed_for_gap = max(0.0, speed + rel_vel)
        safe_gap = self._desired_front_gap(speed, gap, front_speed_for_gap) if self._lead_is_relevant else 80.0
        closing_speed = max(0.0, -rel_vel)
        ttc_front = 20.0 if closing_speed <= 1e-3 else float(np.clip(gap / closing_speed, 0.0, 20.0))

        # A stationary ego should be penalized strongly only when it is truly
        # stuck for no legitimate traffic reason. Legal red-light waits, queued
        # traffic, a protected front actor, and a relevant close lead are valid
        # holds and must not receive the same -1.80/tick anti-idle penalty.
        blocked_by_lead = bool(
            self._lead_is_relevant and
            gap < max(2.0, 0.75 * safe_gap)
        )
        red_stop_error = float(getattr(self, "_traffic_light_stop_error", 999.0))
        blocked_by_red_light = bool(
            getattr(self, "_traffic_light_active", False)
            and getattr(self, "_red_light_control_active", False)
            and -float(self.stop_line_cross_tolerance) <= red_stop_error <= 1.5
        )
        queue_gap_error = float(getattr(self, "_red_light_queue_gap_error", 999.0))
        blocked_by_queue = bool(
            getattr(self, "_red_light_queue_active", False)
            and queue_gap_error <= 1.5
        )
        raw_front_gap = float(getattr(self, "_front_vehicle_gap", 80.0))
        raw_front_speed = float(getattr(self, "_front_vehicle_speed", 0.0))
        raw_front_desired_gap = self._desired_front_gap(
            speed, raw_front_gap, raw_front_speed
        ) if bool(getattr(self, "_front_vehicle_guard_valid", False)) else 80.0
        blocked_by_front_guard = bool(
            front_guard_active_last
            and raw_front_gap <= raw_front_desired_gap + 1.5
        )
        is_legitimately_stopped = bool(
            blocked_by_lead or blocked_by_red_light or
            blocked_by_queue or blocked_by_front_guard
        )
        idle_pen = -0.20 if (
            self.t > 30 and speed < 0.7 and is_legitimately_stopped
        ) else (-1.80 if (self.t > 30 and speed < 0.7) else 0.0)

        lane_pen = -1.00 * (max(0.0, -lane_m) / max(lane_half, 1.0))
        heading_pen = -0.80 * (max(0.0, -heading_m) / max(self.barrier.heading_limit_rad, 1e-3))
        headway_pen = -1.60 * max(0.0, -headway_m) / max(safe_gap, 1.0)
        obstacle_pen = -2.00 * max(0.0, -obstacle_m)
        ttc_pen = -0.50 * max(0.0, 1.0 - ttc_front) if closing_speed > 0.5 else 0.0

        # Normalize asymmetric physical acceleration limits before squaring.
        # This prevents -4.0 m/s^2 emergency braking from being penalized 2.56x
        # more than +2.5 m/s^2 maximum acceleration solely because the physical
        # ranges are asymmetric. Preserve the old worst-case longitudinal scale
        # (-0.04) and old full-steer scale (-0.0036).
        steer_norm = float(actual_action[0]) / 0.60
        accel_val = float(actual_action[1])
        accel_norm = accel_val / (2.5 if accel_val >= 0.0 else 4.0)
        control_pen = -0.040 * float(accel_norm ** 2) - 0.0036 * float(steer_norm ** 2)

        # Keep the existing jerk term unchanged in this patch so the five bug
        # fixes do not also introduce an unrelated reward redesign.
        jerk_pen = -0.010 * float(np.sum((actual_action - prev_action) ** 2))
        smoothness = control_pen + jerk_pen

        goal_d = float(getattr(self, "_goal_distance_m", self.route_distance))
        reached_goal = bool(self.progress_m >= goal_d) if self.route_distance > 0 else False

        # The previous -0.90 margin thresholds effectively tolerated about
        # 86 degrees of heading error (0.60 - |heading| < -0.90).  Use tighter
        # persistent thresholds while still allowing transient junction motion.
        lateral_bad = bool(lane_m < -0.50)
        heading_bad = bool(heading_m < -0.35)
        self._lateral_fail_count = self._lateral_fail_count + 1 if lateral_bad else 0
        self._heading_fail_count = self._heading_fail_count + 1 if heading_bad else 0

        severe_lateral_failure = bool(
            self._lateral_fail_count >= self.failure_persistence_ticks or
            self._heading_fail_count >= self.failure_persistence_ticks
        )
        severe_speed_failure = bool(speed_max_m < -7.0)
        # Headway failures are persistent, not single-frame.  Town10HD can
        # produce one-tick projection/cross-traffic spikes; terminating on the
        # first spike creates many false hard_headway_failure episodes.  Collision
        # events still terminate immediately via self._collided.
        hard_headway_bad = bool(
            self._lead_is_relevant and (
                (gap < 0.25 and speed > 0.8) or
                (gap < self.headway_hard_fail_gap and closing_speed > 1.2 and ttc_front < 0.35)
            )
        )
        severe_obstacle_bad = bool(
            self._lead_is_relevant and float(state[5]) < 0.25 and speed > 0.8
        )
        self._headway_fail_count = self._headway_fail_count + 1 if hard_headway_bad else 0
        self._obstacle_fail_count = self._obstacle_fail_count + 1 if severe_obstacle_bad else 0
        hard_headway_failure = bool(self._headway_fail_count >= self.headway_failure_persistence_ticks)
        severe_obstacle_failure = bool(self._obstacle_fail_count >= self.headway_failure_persistence_ticks)
        headway_timeout_failure = bool(self.terminate_on_headway_violation and headway_m < -4.0)

        # Explicit traffic-rule metrics are separate from the generic safety
        # barrier. A legal red-light stop is success even if another barrier
        # component is negative; crossing a governing red on red is logged
        # independently for paper tables and debugging.
        stop_success_condition = bool(
            self._traffic_light_active
            and str(self._traffic_light_state).lower().startswith("red")
            and speed <= 0.30
            and -float(self.stop_line_cross_tolerance)
                <= float(self._traffic_light_stop_error) <= 1.50
        )
        success_key = (
            getattr(self, "_traffic_light_last_signal_key", None)
            if getattr(self, "_traffic_light_last_signal_key", None) is not None
            else (int(getattr(self, "_traffic_light_id", -1)),)
        )
        self._red_light_stop_success = bool(
            stop_success_condition
            and success_key != getattr(self, "_red_light_last_success_key", None)
        )
        if self._red_light_stop_success:
            self._red_light_stop_success_count += 1
            self._red_light_last_success_key = success_key
        red_light_violation = bool(getattr(self, "_red_light_crossed_on_red", False))

        failure = bool(self._collided or severe_lateral_failure or severe_speed_failure
                       or hard_headway_failure or severe_obstacle_failure
                       or headway_timeout_failure)

        # Separate true MDP termination from an external max-step truncation.
        # The rollout still ends in both cases, but SAC should bootstrap through
        # a pure time-limit truncation because the physical state itself is not
        # terminal. Reaching the route goal or any safety failure is terminal.
        terminated = bool(reached_goal or failure)
        timeout = bool(self.t >= self.max_steps and not terminated)
        truncated = bool(timeout)
        done = bool(terminated or truncated)

        terminal_bonus = self.success_bonus if reached_goal else 0.0
        terminal_pen = -60.0 if self._collided else (-25.0 if failure else 0.0)
        reward = (progress_reward + speed_reward + idle_pen + lane_center_pen
                  + heading_center_pen + route_heading_pen + lane_pen + heading_pen
                  + headway_pen + obstacle_pen + ttc_pen + smoothness
                  + terminal_bonus + terminal_pen)

        if reached_goal:
            termination = "reached_goal"
        elif self._collided:
            termination = "collision"
        elif severe_lateral_failure:
            termination = "lane_or_heading_failure"
        elif severe_speed_failure:
            termination = "speed_failure"
        elif hard_headway_failure:
            termination = "hard_headway_failure"
        elif severe_obstacle_failure:
            termination = "obstacle_failure"
        elif headway_timeout_failure:
            termination = "headway_margin_failure"
        elif timeout:
            termination = "timeout"
        else:
            termination = "running"

        self._prev_env_action = actual_action.copy()

        info = {
            "h": h,
            "speed": speed,
            "desired_speed": desired_speed,
            "collided": self._collided,
            "collision_actor_type": str(getattr(self, "_collision_actor_type", "none")),
            "collision_zone": str(getattr(self, "_collision_zone", "none")),
            "collision_impulse": float(getattr(self, "_collision_impulse", 0.0)),
            "progress_m": float(self.progress_m),
            "delta_progress_m": float(delta_progress),
            "route_distance": float(goal_d),
            "requested_route_distance": float(self.route_distance),
            "remaining_m": float(max(0.0, goal_d - self.progress_m)),
            "distance_to_destination": self._distance_to_destination(),
            "route_lat_error": float(self._route_lat_error),
            "route_yaw_error": float(self._route_yaw_error),
            "state_geometry_source": str(getattr(self, "_state_geometry_source", "map")),
            "barrier_geometry_source": str(getattr(self, "_barrier_geometry_source", "lane")),
            "map_lane_lat_error": float(getattr(self, "_map_lane_lat_error", 0.0)),
            "map_lane_yaw_error": float(getattr(self, "_map_lane_yaw_error", 0.0)),
            "route_lookahead_yaw_error": float(getattr(self, "_route_lookahead_yaw_error", 0.0)),
            "route_turn_strength": float(getattr(self, "_route_turn_strength", 0.0)),
            "route_curvature": float(self._route_curvature),
            "route_turn_guard_active": bool(turn_guard_active_last),
            "turn_recovery_active": bool(getattr(self, "_turn_recovery_active", False)),
            "turn_stuck_count": int(getattr(self, "_turn_stuck_count", 0)),
            "ttc_front": float(ttc_front),
            "blocked_by_lead": bool(blocked_by_lead),
            "blocked_by_red_light": bool(blocked_by_red_light),
            "blocked_by_queue": bool(blocked_by_queue),
            "blocked_by_front_guard": bool(blocked_by_front_guard),
            "is_legitimately_stopped": bool(is_legitimately_stopped),
            "lead_is_relevant": bool(self._lead_is_relevant),
            "lead_same_lane": bool(self._lead_same_lane),
            "lead_route_lat": float(self._lead_route_lat),
            "lead_route_ds": float(self._lead_route_ds),
            "lead_actor_kind": str(self._lead_actor_kind),
            "traffic_light_active": bool(self._traffic_light_active),
            "traffic_light_state": str(self._traffic_light_state),
            "traffic_light_distance": float(self._traffic_light_distance),
            "traffic_light_virtual_gap": float(self._traffic_light_virtual_gap),
            "traffic_light_id": int(getattr(self, "_traffic_light_id", -1)),
            "traffic_light_yellow_go": bool(getattr(self, "_traffic_light_yellow_go", False)),
            "traffic_light_detection_dropout": bool(getattr(self, "_traffic_light_detection_dropout", False)),
            "traffic_light_miss_ticks": int(getattr(self, "_traffic_light_miss_ticks", 0)),
            "red_light_crossed_on_red": bool(red_light_violation),
            "red_light_violation_count": int(getattr(self, "_red_light_violation_count", 0)),
            "red_light_stop_success": bool(getattr(self, "_red_light_stop_success", False)),
            "red_light_stop_success_count": int(getattr(self, "_red_light_stop_success_count", 0)),
            "traffic_light_detector_fault_active": bool(getattr(self, "_traffic_light_detector_fault_active", False)),
            "traffic_light_detector_fault_ticks": int(getattr(self, "_traffic_light_detector_fault_ticks", 0)),
            "safety_exception_count": int(getattr(self, "_safety_exception_count", 0)),
            "last_safety_exception": str(getattr(self, "_last_safety_exception", "")),
            "route_planner_mode": str(getattr(self, "route_planner_mode", "global")),
            "route_planner_used": str(getattr(self, "_route_planner_used", "none")),
            "route_planner_error": str(getattr(self, "_route_planner_error", "")),
            "route_length_m": float(getattr(self, "route_length_m", 0.0)),
            "yellow_required_stop_distance": float(getattr(self, "_yellow_required_stop_distance", 0.0)),
            "junction_commit_active": bool(getattr(self, "_junction_commit_active", False)),
            "junction_seen_since_commit": bool(getattr(self, "_junction_seen_since_commit", False)),
            "committed_light_id": int(getattr(self, "_committed_light_id", -1) if getattr(self, "_committed_light_id", None) is not None else -1),
            "committed_stop_s": float(getattr(self, "_committed_stop_s", -1.0)),
            "tracked_light_id": int(getattr(self, "_tracked_light_id", -1) if getattr(self, "_tracked_light_id", None) is not None else -1),
            "red_light_stop_buffer": float(self.red_light_stop_buffer),
            "red_light_virtual_offset": float(self.red_light_virtual_offset),
            "red_light_creep_speed": float(self.red_light_creep_speed),
            "red_light_creep_distance": float(self.red_light_creep_distance),
            "red_light_keep_lead_gap": float(self.red_light_keep_lead_gap),
            "queue_stop_gap": float(self.queue_stop_gap),
            "queue_detect_distance": float(self.queue_detect_distance),
            "queue_creep_speed": float(self.queue_creep_speed),
            "queue_front_gap": float(getattr(self, "_red_light_front_gap", 80.0)),
            "queue_gap_error": float(getattr(self, "_red_light_queue_gap_error", 999.0)),
            "queue_active": bool(getattr(self, "_red_light_queue_active", False)),
            "traffic_light_stop_error": float(self._traffic_light_stop_error),
            "red_light_control_active": bool(self._red_light_control_active),
            "red_light_braking_active": bool(getattr(self, "_red_light_braking_active", False)),
            "red_light_control_reason": str(getattr(self, "_red_light_control_reason", "none")),
            "red_light_accel_cap": float(self._traffic_light_accel_cap),
            "front_vehicle_gap": float(getattr(self, "_front_vehicle_gap", 80.0)),
            "front_vehicle_speed": float(getattr(self, "_front_vehicle_speed", 0.0)),
            "front_vehicle_kind": str(getattr(self, "_front_vehicle_kind", "none")),
            "front_vehicle_ttc": float(getattr(self, "_front_vehicle_ttc", 20.0)),
            "vehicle_moving_time_headway": float(getattr(self, "vehicle_moving_time_headway", 0.25)),
            "vehicle_soft_extra_gap": float(getattr(self, "vehicle_soft_extra_gap", 1.5)),
            "headway_fail_count": int(getattr(self, "_headway_fail_count", 0)),
            "front_vehicle_guard_active": bool(front_guard_active_last),
            "front_vehicle_guard_valid": bool(getattr(self, "_front_vehicle_guard_valid", False)),
            "front_vehicle_accel_cap": float(front_guard_cap_last),
            "predictive_conflict_guard_active": bool(predictive_guard_active_last),
            "predictive_conflict_raw": bool(getattr(self, "_predictive_conflict_raw", False)),
            "predictive_conflict_active": bool(getattr(self, "_predictive_conflict_active", False)),
            "predictive_conflict_clear_ticks": int(getattr(self, "_predictive_conflict_clear_ticks", 0)),
            "predictive_conflict_actor_id": int(getattr(self, "_predictive_conflict_actor_id", -1)),
            "predictive_conflict_actor_type": str(getattr(self, "_predictive_conflict_actor_type", "none")),
            "predictive_conflict_kind": str(getattr(self, "_predictive_conflict_kind", "none")),
            "predictive_conflict_ttc": float(getattr(self, "_predictive_conflict_ttc", 999.0)),
            "predictive_conflict_distance": float(getattr(self, "_predictive_conflict_distance", 999.0)),
            "predictive_conflict_accel_cap": float(predictive_guard_cap_last),
            "predictive_junction_context": bool(getattr(self, "_predictive_junction_context", False)),
            "external_blockage_recovery_enabled": bool(self.external_blockage_recovery),
            "external_blockage_actor_id": int(getattr(self, "_external_blockage_actor_id", -1)),
            "external_blockage_ticks": int(getattr(self, "_external_blockage_ticks", 0)),
            "external_blockage_recovered_this_step": bool(getattr(self, "_external_blockage_recovered_this_step", False)),
            "external_blockage_recovery_count": int(getattr(self, "_external_blockage_recovery_count", 0)),
            "external_blockage_last_reason": str(getattr(self, "_external_blockage_last_reason", "none")),
            "lane_edge_guard_active": bool(lane_guard_active_last),
            "lane_edge_margin": float(getattr(self, "_lane_edge_margin", 999.0)),
            "lane_edge_accel_cap": float(getattr(self, "_lane_edge_accel_cap", 2.5)),
            "lane_edge_steer_correction": float(getattr(self, "_lane_edge_steer_correction", 0.0)),
            "front_vehicle_actor_id": int(getattr(self, "_front_vehicle_actor_id", -1)),
            "route_projection_rejected": bool(getattr(self, "_route_projection_rejected", False)),
            "route_projection_reject_reason": str(getattr(self, "_route_projection_reject_reason", "")),
            "overtake_mode": str(getattr(self, "_overtake_mode", "idle")),
            "overtake_mode_last_action": str(overtake_mode_last),
            "overtake_guard_active": bool(overtake_guard_active_last),
            "overtake_side": str(getattr(self, "_overtake_side", "none")),
            "overtake_blocked_ticks": int(getattr(self, "_overtake_blocked_ticks", 0)),
            "overtake_target_front_gap": float(getattr(self, "_overtake_target_front_gap", 999.0)),
            "overtake_target_rear_gap": float(getattr(self, "_overtake_target_rear_gap", 999.0)),
            "overtake_target_rear_ttc": float(getattr(self, "_overtake_target_rear_ttc", 999.0)),
            "overtake_target_front_ttc": float(getattr(self, "_overtake_target_front_ttc", 999.0)),
            "overtake_target_side_blocked": bool(getattr(self, "_overtake_target_side_blocked", False)),
            "executed_action_env": [float(actual_action[0]), float(actual_action[1])],
            "executed_vehicle_control": {
                "steer": float(getattr(self, "_last_steer", 0.0)),
                "throttle": float(getattr(self, "_last_throttle", 0.0)),
                "brake": float(getattr(self, "_last_brake", 0.0)),
            },
            "weather_mode": str(getattr(self, "_current_weather_mode", self.weather_mode)),
            "weather_mode_request": str(self.weather_mode),
            "num_traffic_vehicles": int(len(getattr(self, "traffic_vehicles", []))),
            "traffic_moving_count": int(self._traffic_moving_count),
            "traffic_mean_speed": float(self._traffic_mean_speed),
            "num_walkers": int(len(getattr(self, "walker_actors", []))),
            "lateral_fail_count": int(self._lateral_fail_count),
            "heading_fail_count": int(self._heading_fail_count),
            "reached_goal": reached_goal,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "timeout": bool(timeout),
            "termination": termination,
            "h_components": {name: float(c) for name, c
                             in zip(self.barrier.COMPONENT_NAMES, comps)}
        }
        trace_row = {
            "tick": int(self.t),
            "speed": float(speed),
            "progress_m": float(self.progress_m),
            "route_turn_strength": float(getattr(self, "_route_turn_strength", 0.0)),
            "junction_context": bool(getattr(self, "_predictive_junction_context", False)),
            "front_actor_id": int(getattr(self, "_front_vehicle_actor_id", -1)),
            "front_gap": float(getattr(self, "_front_vehicle_gap", 80.0)),
            "front_ttc": float(getattr(self, "_front_vehicle_ttc", 20.0)),
            "conflict_active": bool(getattr(self, "_predictive_conflict_active", False)),
            "conflict_actor_id": int(getattr(self, "_predictive_conflict_actor_id", -1)),
            "conflict_actor_type": str(getattr(self, "_predictive_conflict_actor_type", "none")),
            "conflict_kind": str(getattr(self, "_predictive_conflict_kind", "none")),
            "conflict_ttc": float(getattr(self, "_predictive_conflict_ttc", 999.0)),
            "overtake_mode": str(overtake_mode_last),
            "steer": float(actual_action[0]),
            "accel": float(actual_action[1]),
        }
        self._safety_trace.append(trace_row)
        if self._collided:
            info["pre_collision_trace"] = list(self._safety_trace)
        return state, float(reward), done, info

    def close(self):
        # Episode actors are removed while world/TM are still synchronously
        # paired; then TM sync is disabled before the world returns to async.
        # Each stage is independent so a dead server cannot block later cleanup.
        if getattr(self, "_connection_lost", False):
            # Server unreachable: every stage below is an RPC that would stall
            # for the full client timeout.  Drop references and return.
            self.walker_controllers = []
            self.walker_actors = []
            self.traffic_vehicles = []
            self.collision_sensor = None
            self.lead = None
            self.ego = None
            self.traffic_manager = None
            self.client = None
            self.world = None
            self.map = None
            return
        try:
            self._destroy_actors()
        except Exception:
            pass
        try:
            if getattr(self, "traffic_manager", None) is not None:
                self.traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
        except Exception:
            pass
        self.traffic_manager = None


# Backwards-compatible alias.
CarlaEnvAdapter = CarlaDrivingEnv


# =============================================================================
# Drivers
# =============================================================================

def run_episode(agent: ECLCSAgent, env, max_steps: int,
                deterministic_probe: bool = False) -> float:
    # Episodes are independent trials: reset episode-local filter state
    # (actuator history, q, residual window, audit) while preserving the
    # aggregate log, exactly as documented in
    # EndogenousClosedLoopConformalSafetyFilter.reset(). Without this, the
    # final action of episode k (typically a -4 m/s^2 brake) rate/jerk-limits
    # the first projection of episode k+1 and creates artificial startup
    # infeasibility that depends on episode order.
    agent.filter.reset()
    state = env.reset(); ep_return = 0.0
    episode_limit = max(int(max_steps), int(getattr(env, "max_steps", max_steps)))
    for _ in range(episode_limit):
        decision = agent.act(state, deterministic_probe=deterministic_probe)
        next_state, reward, done, info = env.step(decision.action)
        actual_action = np.asarray(info.get("executed_action_env", decision.action), dtype=np.float64)
        agent.filter.update_after_transition(state, actual_action, next_state, decision)
        ep_return += reward; state = next_state
        if done:
            break
    return ep_return


def build_filter(cfg: ECLCSConfig, env_kind: str, barrier: Optional[Any] = None):
    if env_kind == "acc":
        machine = MachineCard(action_low=(-4.0,), action_high=(2.5,),
                              rate_limit=(1.5,), jerk_limit=(1.0,),
                              neutral_action=(0.0,), action_names=("accel",))
        barrier = ACCBarrierModel(dt=cfg.dt)
    else:
        machine = MachineCard(action_low=cfg.action_low, action_high=cfg.action_high,
                              rate_limit=cfg.rate_limit, jerk_limit=cfg.jerk_limit,
                              neutral_action=cfg.neutral_action)
        barrier = barrier if barrier is not None else AutonomousDrivingBarrierModel(dt=cfg.dt)
    return EndogenousClosedLoopConformalSafetyFilter(cfg, machine, barrier)


def run_carla(args) -> Dict[str, Any]:
    """Primary CARLA experiment: run the E-COCSF filter on a real CARLA client."""
    cfg = ECLCSConfig(dt=args.dt, epsilon=args.epsilon, eta=args.eta,
                      q_init=args.q_init, q_max=args.q_max, zeta_max=args.zeta_max,
                      probe_probability=args.probe_probability, seed=args.seed,
                      ramp_tau=args.ramp_tau,
                      restoration_grid_points=args.restoration_grid_points,
                      headroom_buffer=args.headroom_buffer,
                      headroom_margin_cap=not bool(args.no_headroom_cap),
                      margin_comparison_tolerance=args.margin_comparison_tolerance,
                      anti_windup_gamma=args.anti_windup_gamma,
                      audit_window=args.audit_window, audit_min_samples=args.audit_min_samples,
                      audit_min_q_range=args.audit_min_q_range,
                      audit_bandwidth=args.audit_bandwidth,
                      audit_min_weight_mass=args.audit_min_weight_mass,
                      audit_min_abs_slope=args.audit_min_abs_slope,
                      audit_ridge_lambda=args.audit_ridge_lambda,
                      audit_intercept_ridge_scale=args.audit_intercept_ridge_scale,
                      certified_tube_delta=getattr(args, "certified_tube_delta", 0.10),
                      use_gain_schedule=args.gain_schedule)

    env = CarlaDrivingEnv(host=args.carla_host, port=args.carla_port, town=args.carla_town,
                          dt=args.dt, max_steps=args.max_steps, seed=args.seed,
                          drift_scale=args.drift_scale, target_speed=args.target_speed,
                          route_distance=args.route_distance,
                          success_bonus=args.success_bonus,
                          action_smoothing_beta=args.action_smoothing_beta,
                          weather_mode=getattr(args, "weather_mode", "clear"),
                          traffic_light_guard=not getattr(args, "no_traffic_light_guard", False),
                          red_light_stop_distance=getattr(args, "red_light_stop_distance", 70.0),
                          traffic_light_hysteresis_ticks=getattr(args, "traffic_light_hysteresis_ticks", 5),
                          red_light_reaction_time_s=getattr(args, "red_light_reaction_time_s", 0.35),
                          red_light_activation_margin=getattr(args, "red_light_activation_margin", 2.0),
                          red_light_blend_distance=getattr(args, "red_light_blend_distance", 2.0),
                          yellow_light_stop=not getattr(args, "no_yellow_light_stop", False),
                          red_light_stop_buffer=getattr(args, "red_light_stop_buffer", 1.2),
                          red_light_virtual_offset=getattr(args, "red_light_virtual_offset", 0.0),
                          red_light_creep_speed=getattr(args, "red_light_creep_speed", 0.8),
                          red_light_creep_distance=getattr(args, "red_light_creep_distance", 3.0),
                          red_light_comfort_decel=getattr(args, "red_light_comfort_decel", 3.0),
                          red_light_keep_lead_gap=getattr(args, "red_light_keep_lead_gap", 8.0),
                          queue_stop_gap=getattr(args, "queue_stop_gap", 2.0),
                          queue_detect_distance=getattr(args, "queue_detect_distance", 25.0),
                          queue_creep_speed=getattr(args, "queue_creep_speed", 1.5),
                          traffic_light_route_scan=not getattr(args, "no_traffic_light_route_scan", False),
                          traffic_light_route_scan_step=getattr(args, "traffic_light_route_scan_step", 4.0),
                          traffic_light_landmark_fallback=not getattr(args, "no_traffic_light_landmark_fallback", False),
                          vehicle_route_corridor_factor=getattr(args, "vehicle_route_corridor_factor", 0.55),
                          vehicle_route_corridor_max=getattr(args, "vehicle_route_corridor_max", 1.00),
                          predictive_collision_guard=not getattr(args, "no_predictive_collision_guard", False),
                          predictive_horizon_s=getattr(args, "predictive_horizon_s", 3.0),
                          predictive_step_s=getattr(args, "predictive_step_s", 0.20),
                          predictive_vehicle_radius_m=getattr(args, "predictive_vehicle_radius_m", 45.0),
                          predictive_lateral_margin_m=getattr(args, "predictive_lateral_margin_m", 0.35),
                          predictive_longitudinal_margin_m=getattr(args, "predictive_longitudinal_margin_m", 1.0),
                          predictive_ttc_soft=getattr(args, "predictive_ttc_soft", 3.0),
                          predictive_ttc_hard=getattr(args, "predictive_ttc_hard", 1.2),
                          predictive_clear_ticks=getattr(args, "predictive_clear_ticks", 5),
                          predictive_junction_lookahead_m=getattr(args, "predictive_junction_lookahead_m", 30.0),
                          predictive_junction_preview_speed=getattr(args, "predictive_junction_preview_speed", 3.0),
                          ego_overtake_enabled=not getattr(args, "no_ego_overtake", False),
                          terminate_on_headway_violation=args.terminate_on_headway_violation,
                          headway_hard_fail_gap=args.headway_hard_fail_gap,
                          route_step_m=getattr(args, "route_step_m", 2.0),
                          route_planner_mode=getattr(args, "route_planner_mode", "global"),
                          allow_heuristic_route_fallback=getattr(args, "allow_heuristic_route_fallback", False),
                          route_destination_candidates=getattr(args, "route_destination_candidates", 32),
                          failure_persistence_ticks=getattr(args, "failure_persistence_ticks", 10),
                          num_traffic_vehicles=getattr(args, "num_traffic_vehicles", 0),
                          num_walkers=getattr(args, "num_walkers", 0),
                          use_augmented_state=not getattr(args, "compact_obs", False),
                          vary_weather=not args.no_weather, render_follow=not args.no_render,
                          load_town=not args.no_load_town)
    sf = build_filter(cfg, "car", barrier=env.barrier)
    policy = (RandomDrivingPolicy(sf.machine, seed=args.seed) if args.random_policy
              else RuleBasedDrivingPolicy(target_speed=args.target_speed))
    agent = ECLCSAgent(policy, sf)

    returns: List[float] = []
    try:
        for ep in range(int(args.episodes)):
            returns.append(run_episode(agent, env, args.max_steps, args.deterministic_probe))
            print(f"[CARLA] episode {ep + 1}/{args.episodes} return={returns[-1]:.2f}")
    finally:
        env.close()

    metrics = sf.metrics()
    metrics.update({"episodes": float(args.episodes),
                    "return_mean": float(np.mean(returns)) if returns else float("nan"),
                    "return_std": float(np.std(returns)) if returns else float("nan"),
                    "town": args.carla_town, "drift_scale": float(args.drift_scale)})
    paths = sf.save_audit_log(args.out_dir, prefix="ecocsf_carla")
    metrics["out_dir"] = str(args.out_dir)
    for k, v in paths.items():
        metrics[f"{k}_path"] = v
    return metrics


def run_carla_drift_sweep(args) -> Dict[str, Any]:
    """
    CARLA realization of the drift study: sweep the lead/weather drift_scale and
    report exceedance gap and Dbar so the realistic closed loop can be compared
    against the controlled cube-root prediction. (Slope here is noisier than the
    analytic study; CARLA is the high-fidelity demonstration, not the clean fit.)
    """
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    drift_scales = [float(x) for x in args.carla_drift_scales.split(",")]
    rows: List[Dict[str, float]] = []
    for ds in drift_scales:
        cfg = ECLCSConfig(dt=args.dt, epsilon=args.epsilon, eta=args.eta,
                          q_init=args.q_init, q_max=args.q_max, zeta_max=args.zeta_max,
                          probe_probability=args.probe_probability, seed=args.seed,
                          ramp_tau=args.ramp_tau,
                          restoration_grid_points=args.restoration_grid_points,
                          headroom_buffer=args.headroom_buffer,
                          headroom_margin_cap=not bool(args.no_headroom_cap),
                          margin_comparison_tolerance=args.margin_comparison_tolerance,
                          anti_windup_gamma=args.anti_windup_gamma,
                          audit_window=args.audit_window, audit_min_samples=args.audit_min_samples,
                          audit_min_q_range=args.audit_min_q_range,
                          audit_bandwidth=args.audit_bandwidth,
                          audit_min_weight_mass=args.audit_min_weight_mass,
                          audit_min_abs_slope=args.audit_min_abs_slope,
                          audit_ridge_lambda=args.audit_ridge_lambda,
                          audit_intercept_ridge_scale=args.audit_intercept_ridge_scale,
                          use_gain_schedule=args.gain_schedule)
        env = CarlaDrivingEnv(host=args.carla_host, port=args.carla_port, town=args.carla_town,
                              dt=args.dt, max_steps=args.max_steps, seed=args.seed,
                              drift_scale=ds, target_speed=args.target_speed,
                              route_distance=args.route_distance,
                              success_bonus=args.success_bonus,
                              action_smoothing_beta=args.action_smoothing_beta,
                              terminate_on_headway_violation=args.terminate_on_headway_violation,
                              headway_hard_fail_gap=args.headway_hard_fail_gap,
                              num_traffic_vehicles=getattr(args, "num_traffic_vehicles", 0),
                              num_walkers=getattr(args, "num_walkers", 0),
                              vary_weather=not args.no_weather, render_follow=not args.no_render,
                              load_town=not args.no_load_town)
        # Share the environment's barrier object with the filter (same as
        # run_carla). CARLA updates lane_half_width and context-dependent gap
        # parameters online on env.barrier; a separate default barrier would
        # make projection and residuals disagree with the h used for reward,
        # termination, and violation counting.
        sf = build_filter(cfg, "car", barrier=env.barrier)
        agent = ECLCSAgent(RuleBasedDrivingPolicy(target_speed=args.target_speed), sf)
        try:
            for _ in range(int(args.episodes)):
                run_episode(agent, env, args.max_steps)
        finally:
            env.close()
        m = sf.metrics()
        rows.append({"drift_scale": ds, "exceedance_rate": m.get("exceedance_rate", float("nan")),
                     "coverage_gap": m.get("coverage_gap", float("nan")),
                     "violation_rate": m.get("violation_rate", float("nan")),
                     "certified_fraction": m.get("certified_fraction", float("nan")),
                     "P_out": m.get("P_out", float("nan")),
                     "infeasible_rate": m.get("infeasible_rate", float("nan"))})
        sf.save_audit_log(out_dir, prefix=f"eclcs_carla_drift_{ds:.2f}")
        print(f"[CARLA sweep] drift={ds:.2f} exceed={rows[-1]['exceedance_rate']:.3f} "
              f"viol={rows[-1]['violation_rate']:.3f}")
    (out_dir / "carla_drift_sweep.json").write_text(
        json.dumps({"rows": rows, "target_eps": args.epsilon}, indent=2), encoding="utf-8")
    return {"rows": rows, "out_dir": str(out_dir)}


# -----------------------------------------------------------------------------
# Controlled scaling study: self-correcting residual process with constant mu
# -----------------------------------------------------------------------------
class SelfCorrectingResidualProcess:
    """
    Synthetic closed-loop residual with a constant local response slope and a
    controllable root-drift rate.

        R_t(q) = c0 + m(t) + sigma * N(0,1) - beta * (q - c0).

    The runtime recursion uses the ramp loss, so the analytic response used by
    this controlled study is exactly

        g_t^tau(q) = E[ell_tau(R_t(q)-q)],

    rather than the hard-indicator approximation.  Because the Gaussian shift
    enters only through m(t)-(1+beta)(q-c0), the root still translates by
    m(t)/(1+beta), making Dbar independent of the ramp correction, while the
    exact local slope is computed from the ramp-smoothed response.
    """

    def __init__(self, sigma: float, beta: float, c0: float, drift_amp: float,
                 drift_period: int, seed: int = 0, ramp_tau: float = 0.05):
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.c0 = float(c0)
        self.drift_amp = float(drift_amp)
        self.drift_period = int(drift_period)
        self.ramp_tau = float(ramp_tau)
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        if self.ramp_tau <= 0.0:
            raise ValueError("ramp_tau must be positive.")
        if self.beta <= -1.0:
            raise ValueError("beta must exceed -1 so the response is decreasing in q.")
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self._eps = 0.10
        self.last_root = float(c0)

    def set_eps(self, eps: float) -> None:
        eps = float(eps)
        if not 0.0 < eps < 1.0:
            raise ValueError("eps must lie in (0, 1).")
        self._eps = eps

    def _m(self, t: int) -> float:
        # Slow sinusoid: peak root velocity ~ amp*omega, RMS sets Dbar.
        omega = 2.0 * math.pi / max(self.drift_period, 2)
        return self.drift_amp * math.sin(omega * t)

    def _root_mean(self, eps: Optional[float] = None) -> float:
        return _gaussian_ramp_root_mean(
            self._eps if eps is None else float(eps),
            self.sigma,
            self.ramp_tau,
        )

    def analytic_root(self, t: int, eps: Optional[float] = None) -> float:
        root_mean = self._root_mean(eps)
        return self.c0 + (self._m(int(t)) - root_mean) / (1.0 + self.beta)

    def analytic_mu(self, eps: float) -> float:
        root_mean = self._root_mean(eps)
        a = (-self.ramp_tau - root_mean) / self.sigma
        b = -root_mean / self.sigma
        ramp_mass = max(_norm_cdf(b) - _norm_cdf(a), 0.0)
        return (1.0 + self.beta) * ramp_mass / self.ramp_tau

    def analytic_Dbar(self) -> float:
        # Exact discrete-time RMS drift for
        # q*_t = const + drift_amp*sin(omega*t)/(1+beta).
        omega = 2.0 * math.pi / max(self.drift_period, 2)
        return (
            math.sqrt(2.0)
            * abs(self.drift_amp)
            * abs(math.sin(0.5 * omega))
            / (1.0 + self.beta)
        )

    def __call__(self, q_tilde: float) -> float:
        m_now = self._m(self.t)
        self.last_root = self.analytic_root(self.t)
        r = self.c0 + m_now + self.sigma * float(self.rng.standard_normal()) \
            - self.beta * (q_tilde - self.c0)
        self.t += 1
        return float(r)


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    x = float(x)
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _gaussian_ramp_expectation(mean: float, sigma: float, tau: float) -> float:
    """Exact E[ell_tau(Z)] for Z ~ Normal(mean, sigma^2)."""
    mean, sigma, tau = float(mean), float(sigma), float(tau)
    if sigma <= 0.0 or tau <= 0.0:
        raise ValueError("sigma and tau must be positive.")
    a = (-tau - mean) / sigma
    b = -mean / sigma
    cdf_a = _norm_cdf(a)
    cdf_b = _norm_cdf(b)
    interval_mass = max(cdf_b - cdf_a, 0.0)
    truncated_first_moment = mean * interval_mass + sigma * (
        _norm_pdf(a) - _norm_pdf(b)
    )
    value = (1.0 - cdf_b) + interval_mass + truncated_first_moment / tau
    return float(np.clip(value, 0.0, 1.0))


def _gaussian_ramp_root_mean(eps: float, sigma: float, tau: float) -> float:
    """Mean of Z for which E[ell_tau(Z)] equals eps."""
    eps, sigma, tau = float(eps), float(sigma), float(tau)
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must lie in (0, 1).")
    if sigma <= 0.0 or tau <= 0.0:
        raise ValueError("sigma and tau must be positive.")
    span = max(12.0 * sigma + tau, 1.0)
    lo, hi = -span, span
    while _gaussian_ramp_expectation(lo, sigma, tau) > eps:
        lo *= 2.0
    while _gaussian_ramp_expectation(hi, sigma, tau) < eps:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _gaussian_ramp_expectation(mid, sigma, tau) < eps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _run_synthetic(eps: float, eta: float, sigma: float, beta: float, c0: float,
                   drift_amp: float, drift_period: int, steps: int, zeta_max: float,
                   probe_prob: float, seed: int, use_schedule: bool,
                   burn_in: float = 0.3, run_audit: bool = False,
                   ramp_tau: float = 0.05) -> Dict[str, float]:
    cfg = ECLCSConfig(action_dim=1, epsilon=eps, eta=eta,
                      q_init=c0, q_max=12.0, ramp_tau=float(ramp_tau),
                      zeta_max=zeta_max,
                      probe_probability=probe_prob, audit_window=160,
                      audit_min_samples=50, use_gain_schedule=use_schedule,
                      run_audit=run_audit, seed=seed, save_jsonl=False)
    sf = EndogenousClosedLoopConformalSafetyFilter(
        cfg, MachineCard(action_low=(-1.0,), action_high=(1.0,)), ACCBarrierModel())
    proc = SelfCorrectingResidualProcess(
        sigma, beta, c0, drift_amp, drift_period, seed=seed,
        ramp_tau=float(ramp_tau),
    )
    proc.set_eps(eps)
    exceed = np.zeros(steps, dtype=np.float64)
    y2 = np.zeros(steps, dtype=np.float64)
    for k in range(steps):
        rec = sf.step_with_residual(proc)
        exceed[k] = float(rec["hard_exceedance"])
        y2[k] = (float(rec["q"]) - float(proc.last_root)) ** 2
    b = int(burn_in * steps)
    rate = float(exceed[b:].mean())
    rms_margin = float(math.sqrt(float(y2[b:].mean())))
    return {"exceedance_rate": rate,
            "coverage_gap": float(max(0.0, rate - eps)),
            "rms_margin": rms_margin,
            "mu_analytic": proc.analytic_mu(eps),
            "Dbar_analytic": proc.analytic_Dbar()}


def run_scaling_study(args) -> Dict[str, Any]:
    """
    Validate the cube-root RMS tracking law and U-shaped gain trade-off on the
    self-correcting residual process with constant mu.  Each drift level uses
    the analytic eta* and we fit log(RMS margin error) against log(Dbar); the
    predicted slope is 1/3.
    """
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    eps = float(args.epsilon)
    # c0=5 keeps the analytic root strictly inside [0, q_max] for every drift
    # amplitude; the previous c0=2 clipped the largest-amplitude root at zero
    # and corrupted the scaling exponent.
    sigma, beta, c0 = 1.0, 0.7, 5.0
    steps = int(args.sweep_steps)
    seeds = int(args.sweep_seeds)
    if steps < 5000:
        raise ValueError("--scaling requires --sweep_steps >= 5000.")
    # Observe two complete drift cycles after adequate tracking burn-in.  The
    # previous fixed period of 40000 with only 1500 steps observed a tiny arc
    # and produced a misleading negative fitted exponent.
    drift_period = max(2500, steps // 2)
    ramp_tau = float(args.ramp_tau)
    process_template = SelfCorrectingResidualProcess(
        sigma, beta, c0, 0.0, drift_period, ramp_tau=ramp_tau
    )
    mu = process_template.analytic_mu(eps)

    # Cube-root law: amplitudes give ~1 decade of Dbar with bounded root (no clip).
    drift_amps = [0.3, 0.6, 1.2, 2.4, 4.8]
    cube: List[Dict[str, float]] = []
    for amp in drift_amps:
        proc0 = SelfCorrectingResidualProcess(
            sigma, beta, c0, amp, drift_period, ramp_tau=ramp_tau
        )
        Dbar = proc0.analytic_Dbar()
        eta_star = clip_scalar(2.0 * (Dbar * Dbar / mu) ** (1.0 / 3.0), 1e-4, 0.5)
        rmss = []
        for s in range(seeds):
            r = _run_synthetic(eps, eta_star, sigma, beta, c0, amp, drift_period,
                               steps, zeta_max=0.0, probe_prob=0.0, seed=7 + s,
                               use_schedule=False, run_audit=False,
                               ramp_tau=ramp_tau)
            rmss.append(r["rms_margin"])
        cube.append({"drift_amp": amp, "Dbar": Dbar, "eta_star": eta_star,
                     "rms_margin": float(np.mean(rmss)), "rms_std": float(np.std(rmss))})

    xs = np.log(np.asarray([e["Dbar"] for e in cube]))
    ys = np.log(np.asarray([max(e["rms_margin"], 1e-12) for e in cube]))
    coeffs = np.polyfit(xs, ys, 1)
    slope = float(coeffs[0])
    yhat = np.polyval(coeffs, xs)
    r2 = 1.0 - float(np.sum((ys - yhat) ** 2)) / (float(np.sum((ys - ys.mean()) ** 2)) + 1e-12)

    # U-shaped gain at a mid drift level: sweep eta, expect a minimum near eta*.
    u_amp = drift_amps[len(drift_amps) // 2]
    procu = SelfCorrectingResidualProcess(
        sigma, beta, c0, u_amp, drift_period, ramp_tau=ramp_tau
    )
    Dbar_u = procu.analytic_Dbar()
    eta_star_u = 2.0 * (Dbar_u * Dbar_u / mu) ** (1.0 / 3.0)
    eta_grid = sorted(set([eta_star_u * f for f in (0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0)]))
    ushape = []
    for eta in eta_grid:
        rmss = [ _run_synthetic(eps, eta, sigma, beta, c0, u_amp, drift_period, steps,
                                0.0, 0.0, seed=7 + s, use_schedule=False,
                                run_audit=False, ramp_tau=ramp_tau)["rms_margin"]
                 for s in range(seeds) ]
        ushape.append({"eta": eta, "rms_margin": float(np.mean(rmss))})
    eta_emp = min(ushape, key=lambda d: d["rms_margin"])["eta"]

    result = {"cube_root": cube, "fitted_log_log_slope": slope, "predicted_slope": 1.0 / 3.0,
              "r2": r2, "mu": mu, "ramp_tau": ramp_tau,
              "drift_period": drift_period,
              "ushape": ushape, "ushape_meta": {"drift_amp": u_amp, "Dbar": Dbar_u,
                                                 "eta_star_pred": eta_star_u, "eta_star_emp": eta_emp},
              "sigma": sigma, "beta": beta, "c0": c0, "steps": steps, "seeds": seeds,
              "out_dir": str(out_dir)}
    (out_dir / "scaling_study.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_acc_drift_sweep(args) -> Dict[str, Any]:
    """Empirical ACC drift stress test (not the analytic cube-root proof).

    The controlled synthetic study above validates the theoretical exponent.
    This companion sweep reports safety and absolute calibration degradation as
    lead-vehicle drift increases, without pretending eta_mean is root drift.
    """
    drift_scales = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    rows: List[Dict[str, float]] = []
    for ds in drift_scales:
        abs_errors: List[float] = []
        signed_errors: List[float] = []
        violations: List[float] = []
        certified: List[float] = []
        valid_rates: List[float] = []
        for seed in range(int(args.sweep_seeds)):
            cfg = ECLCSConfig(action_dim=1, dt=args.dt,
                              epsilon=args.epsilon, eta=args.eta,
                              q_init=0.2, q_max=8.0,
                              zeta_max=0.05, probe_probability=0.2, seed=seed,
                              audit_window=150, audit_min_samples=40,
                              use_gain_schedule=args.gain_schedule)
            sf = build_filter(cfg, "acc")
            agent = ECLCSAgent(ACCDriverPolicy(target_speed=30.0), sf)
            env = AnalyticACCEnv(dt=args.dt, max_steps=args.sweep_steps,
                                 seed=1000 + seed, drift_scale=ds)
            run_episode(agent, env, args.sweep_steps)
            m = sf.metrics()
            values = (
                (abs_errors, m.get("coverage_error_abs")),
                (signed_errors, m.get("coverage_error_signed")),
                (violations, m.get("violation_rate")),
                (certified, m.get("certified_fraction")),
                (valid_rates, m.get("calibration_valid_rate")),
            )
            for target, value in values:
                if value is not None and np.isfinite(value):
                    target.append(float(value))

        def mean_or_nan(values: Sequence[float]) -> float:
            return float(np.mean(values)) if values else float("nan")

        rows.append({
            "drift_scale": float(ds),
            "absolute_calibration_error_mean": mean_or_nan(abs_errors),
            "absolute_calibration_error_std": (
                float(np.std(abs_errors)) if abs_errors else float("nan")
            ),
            "signed_calibration_error_mean": mean_or_nan(signed_errors),
            "violation_rate_mean": mean_or_nan(violations),
            "certified_fraction_mean": mean_or_nan(certified),
            "calibration_valid_rate_mean": mean_or_nan(valid_rates),
            "n_seeds": float(len(abs_errors)),
        })

    xs = np.log(np.asarray([r["drift_scale"] for r in rows], dtype=np.float64))
    errors = np.asarray(
        [r["absolute_calibration_error_mean"] for r in rows], dtype=np.float64
    )
    ok = np.isfinite(xs) & np.isfinite(errors) & (errors > 0.0)
    trend_slope = (
        float(np.polyfit(xs[ok], np.log(errors[ok]), 1)[0])
        if int(ok.sum()) >= 2 else float("nan")
    )

    result = {
        "rows": rows,
        "fitted_absolute_error_trend_slope": trend_slope,
        "note": "Empirical ACC stress trend; theoretical 1/3 law is tested by --scaling.",
        "out_dir": str(args.out_dir),
    }
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "acc_drift_sweep.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result

# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Endogenous Closed-Loop Conformal Safety Filtering (E-COCSF)")
    p.add_argument("--sweep", action="store_true",
                   help="Run the empirical ACC drift stress sweep.")
    p.add_argument("--scaling", action="store_true",
                   help="Run the controlled cube-root + U-shape scaling study.")
    p.add_argument("--carla", action="store_true", help="Run the E-COCSF filter on CARLA.")
    p.add_argument("--carla_sweep", action="store_true",
                   help="Sweep drift_scale on CARLA (realistic closed-loop drift study).")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./eclcs_outputs")

    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=0.10)
    p.add_argument("--eta", type=float, default=0.03)
    p.add_argument("--q_init", type=float, default=0.10)
    p.add_argument("--q_max", type=float, default=5.0)
    p.add_argument("--ramp_tau", type=float, default=0.05)
    p.add_argument("--zeta_max", type=float, default=0.02)
    p.add_argument("--probe_probability", type=float, default=0.10)
    p.add_argument("--restoration_grid_points", type=int, default=5)
    p.add_argument(
        "--headroom_buffer", "--headroom_cap_delta", dest="headroom_buffer",
        type=float, default=1e-6,
        help="Positive buffer subtracted from the conservative headroom estimate.",
    )
    p.add_argument(
        "--no_headroom_cap", action="store_true",
        help="Disable the verified headroom-capping branch (ablation only).",
    )
    p.add_argument("--margin_comparison_tolerance", type=float, default=1e-8)
    p.add_argument("--anti_windup_gamma", type=float, default=0.25)
    p.add_argument("--certified_tube_delta", type=float, default=0.10,
                   help="Half-width of the local response-support tube around estimated q_star.")
    p.add_argument("--gain_schedule", action="store_true",
                   help="Enable eta* = 2(Dbar^2/mu)^(1/3) schedule.")
    p.add_argument("--audit_window", type=int, default=120)
    p.add_argument("--audit_min_samples", type=int, default=30)
    p.add_argument("--audit_min_q_range", type=float, default=1e-3)
    p.add_argument("--audit_bandwidth", type=float, default=0.05)
    p.add_argument("--audit_min_weight_mass", type=float, default=5.0)
    p.add_argument("--audit_min_abs_slope", type=float, default=1e-6)
    p.add_argument("--audit_ridge_lambda", type=float, default=1e-6)
    p.add_argument("--audit_intercept_ridge_scale", type=float, default=1e-3)

    p.add_argument("--target_speed", type=float, default=18.0)
    p.add_argument("--route_distance", type=float, default=1000.0,
                   help="Destination/route distance in meters; success when progress reaches this value.")
    p.add_argument("--success_bonus", type=float, default=100.0)
    p.add_argument("--action_smoothing_beta", type=float, default=0.15,
                   help="Low-pass smoothing for CARLA throttle/steer/brake commands.")
    p.add_argument("--terminate_on_headway_violation", action="store_true",
                   help="If set, terminate on moderate headway violation; default keeps episode alive so the car can slow/follow.")
    p.add_argument("--headway_hard_fail_gap", type=float, default=0.75,
                   help="Only terminate for headway when bumper gap is below this hard-failure threshold.")
    p.add_argument("--route_step_m", type=float, default=2.0,
                   help="Waypoint spacing / GlobalRoutePlanner sampling resolution.")
    p.add_argument("--route_planner_mode", type=str, default="global",
                   choices=["global", "heuristic"],
                   help="Use explicit destination-based GlobalRoutePlanner by default.")
    p.add_argument("--allow_heuristic_route_fallback", action="store_true",
                   help="Explicitly allow legacy local Waypoint.next branch routing if global planning fails.")
    p.add_argument("--route_destination_candidates", type=int, default=32,
                   help="Number of deterministic map spawn destinations to evaluate with GlobalRoutePlanner.")
    p.add_argument("--failure_persistence_ticks", type=int, default=10,
                   help="Consecutive severe lane/heading ticks before a lateral reset.")
    p.add_argument("--compact_obs", action="store_true",
                   help="Use only the original 8-D observation instead of the 12-D route-aware observation.")
    p.add_argument("--random_policy", action="store_true")
    p.add_argument("--deterministic_probe", action="store_true")

    p.add_argument("--sweep_steps", type=int, default=20000)
    p.add_argument("--sweep_seeds", type=int, default=3)

    # --- CARLA options -------------------------------------------------------
    p.add_argument("--carla_host", type=str, default="localhost")
    p.add_argument("--carla_port", type=int, default=2000,
                   help="CARLA RPC port (use 2200 to match your launch command).")
    p.add_argument("--carla_town", type=str, default="Town04")
    p.add_argument("--num_traffic_vehicles", type=int, default=0)
    p.add_argument("--num_walkers", type=int, default=0)
    p.add_argument("--drift_scale", type=float, default=0.5,
                   help="Lead/weather drift amplitude for CARLA.")
    p.add_argument("--no_weather", action="store_true", help="Disable weather drift in CARLA.")
    p.add_argument("--weather_mode", type=str, default="clear",
                   choices=["clear", "rain", "night", "night_rain", "fog", "morning_rain", "random", "dynamic"],
                   help="CARLA weather/lighting preset. Use random for per-episode random weather.")
    p.add_argument("--no_traffic_light_guard", action="store_true",
                   help="Disable red/yellow traffic-light stopping guard.")
    p.add_argument("--red_light_stop_distance", type=float, default=70.0,
                   help="Detection horizon in meters; braking activation remains speed-dependent.")
    p.add_argument("--traffic_light_hysteresis_ticks", type=int, default=5,
                   help="Grace ticks for one-frame traffic-light detector dropouts.")
    p.add_argument("--red_light_reaction_time_s", type=float, default=0.35)
    p.add_argument("--red_light_activation_margin", type=float, default=2.0)
    p.add_argument("--red_light_blend_distance", type=float, default=2.0,
                   help="Smooth transition length from braking curve into creep speed.")
    p.add_argument("--red_light_stop_buffer", type=float, default=1.2,
                   help="Desired front-bumper distance before the physical stop line.")
    p.add_argument("--red_light_comfort_decel", type=float, default=3.0,
                   help="Comfortable red-light braking deceleration in m/s^2.")
    p.add_argument("--red_light_virtual_offset", type=float, default=0.0,
                   help="Deprecated compatibility flag; physical stop-line distance is used directly.")
    p.add_argument("--red_light_creep_speed", type=float, default=0.8)
    p.add_argument("--red_light_creep_distance", type=float, default=3.0)
    p.add_argument("--no_yellow_light_stop", action="store_true",
                   help="If set, ego only stops for red lights, not yellow lights.")
    p.add_argument("--no_predictive_collision_guard", action="store_true")
    p.add_argument("--predictive_horizon_s", type=float, default=3.0)
    p.add_argument("--predictive_step_s", type=float, default=0.20)
    p.add_argument("--predictive_vehicle_radius_m", type=float, default=45.0)
    p.add_argument("--predictive_lateral_margin_m", type=float, default=0.35)
    p.add_argument("--predictive_longitudinal_margin_m", type=float, default=1.0)
    p.add_argument("--predictive_ttc_soft", type=float, default=3.0)
    p.add_argument("--predictive_ttc_hard", type=float, default=1.2)
    p.add_argument("--predictive_clear_ticks", type=int, default=5)
    p.add_argument("--predictive_junction_lookahead_m", type=float, default=30.0)
    p.add_argument("--predictive_junction_preview_speed", type=float, default=3.0)
    p.add_argument("--no_ego_overtake", action="store_true")
    p.add_argument("--no_render", action="store_true",
                   help="Disable spectator follow / enable no-rendering mode.")
    p.add_argument("--no_load_town", action="store_true",
                   help="Use the currently loaded world instead of loading --carla_town.")
    p.add_argument("--carla_drift_scales", type=str, default="0.2,0.5,1.0,2.0",
                   help="Comma-separated drift scales for --carla_sweep.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    if args.carla:
        print(json.dumps(run_carla(args), indent=2))
    elif args.carla_sweep:
        print(json.dumps(run_carla_drift_sweep(args), indent=2))
    elif args.scaling:
        res = run_scaling_study(args)
        print(json.dumps({k: v for k, v in res.items()
                          if k in ("fitted_log_log_slope", "predicted_slope", "r2",
                                   "ushape_meta", "out_dir")}, indent=2))
    elif args.sweep:
        print(json.dumps(run_acc_drift_sweep(args), indent=2))
    else:
        print("Modes: --carla (real CARLA), --carla_sweep, --sweep (ACC), --scaling (cube-root study).")
        print("Import EndogenousClosedLoopConformalSafetyFilter for external use.")


if __name__ == "__main__":
    main()
