#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
carla_train_eval.py -- Train a black-box driving policy on CARLA and evaluate
it across towns with the E-COCSF safety filter (paper pipeline).

Role in the paper
-----------------
The AAAI manuscript's contribution is the E-COCSF filter (see ECLCS.py), which
is *policy-agnostic*: it treats the driving policy as an opaque black box.
This file provides that black box. It trains a compact SAC policy on the default
12-D route-aware observation (the 8-D barrier state plus four route features),
so the checkpoint plugs directly into `ECLCSAgent` for held-out-town evaluation.
Baseline methods are intentionally maintained in separate scripts.

The network design reuses the trainable core of hcrl_agent.py (the author's
training concept): spectral-norm Lipschitz MLPs, twin critics, soft target
updates, and a uniform replay buffer. The flow-based actor / causal encoder /
latent-barrier machinery of that file is intentionally NOT used here: the paper
treats the policy as a black box, so policy internals are not a contribution,
and the compact learner trains reliably on a 6 GB GPU.

Usage
-----
# 1) train on CARLA (Town04), server started with -carla-rpc-port=2200
python carla_train_eval.py --train --carla_port 2200 --train_town Town04 \
    --total_steps 150000 --out_dir ./runs/sac_town04

# 2) evaluate E-COCSF across towns
python carla_train_eval.py --eval --carla_port 2200 \
    --checkpoint ./runs/sac_town04/policy_final.pt \
    --eval_towns Town03,Town04,Town05 \
    --episodes 10 --out_dir ./runs/eval

Notes
-----
* Training is done without E-COCSF so the policy remains an independent black
  box; the filter is applied only at evaluation. (Filtered training is possible but
  would entangle the policy with the filter and weaken the policy-agnostic
  claim.)
* CARLA physics cannot be verified outside a machine with a CARLA server; the
* The pipeline is CARLA-only; validate with a short --train run (few thousand
  steps) before committing to a full 150k-step run.

Bring-up findings baked into the defaults (validated during bring-up, where they
took SAC from non-learning to rule-based-level driving):
  1. Observation normalization: the raw 8-D state mixes scales from ~0.5 rad to
     ~50 m; networks receive per-dimension normalized states (SACConfig.obs_scale,
     applied inside SACAgent.normalize/act and when storing to replay).
  2. No spectral norm on critics: Lipschitz-capped critics cannot represent the
     value scale; spectral norm is kept only on the actor.
  3. Initial temperature alpha = 0.1 (not 1.0): at this reward scale, alpha = 1
     keeps behaviour near-random long enough to stall learning.
  4. Feasible starts: --lead_gap defaults to 40 m because the headway barrier
     requires d0 + T*v (= 26.6 m at 18 m/s); spawning the lead at 25 m starts
     episodes already in violation and poisons both training and evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import ECLCS as ec


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _canonical_town_name(name: Any) -> str:
    """Normalize CARLA asset paths and treat ``TownXX_Opt`` as ``TownXX``."""
    text = str(name or "").replace("\\", "/").rstrip("/").split("/")[-1]
    text = text.strip().lower()
    return text[:-4] if text.endswith("_opt") else text


def _prepare_carla_town_isolated(args, town: str) -> None:
    """Load a requested evaluation town in a crash-contained subprocess.

    Packaged CARLA 0.9.15 can terminate the Python interpreter inside native
    ``client.load_world`` during a cross-town transition.  Python exceptions
    cannot catch SIGSEGV, so evaluation must never perform that transition in
    the process holding the checkpoint and result buffers.  A tiny helper
    process disables TM/world synchronous mode, loads the map, and verifies the
    resulting world.  The main process then connects with ``load_town=False``.

    If libcarla or the simulator crashes, the parent remains alive and reports
    a clear restart instruction instead of losing the whole evaluation run.
    """
    helper = r'''
import sys, time
import carla

host, port_s, town, tm_port_s, timeout_s = sys.argv[1:6]
port, tm_port, timeout = int(port_s), int(tm_port_s), float(timeout_s)

def short(name):
    value = str(name or "").replace("\\", "/").rstrip("/")
    return value.split("/")[-1] if value else ""

def canonical(name):
    value = short(name).strip().lower()
    return value[:-4] if value.endswith("_opt") else value

client = carla.Client(host, port)
client.set_timeout(timeout)
world = client.get_world()
current = short(world.get_map().name)
if canonical(current) == canonical(town):
    print("MAP_READY=" + current, flush=True)
    raise SystemExit(0)

try:
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(False)
except Exception:
    pass

try:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
except Exception:
    pass

world = client.load_world(town, True)
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    world = client.get_world()
    loaded = short(world.get_map().name)
    if canonical(loaded) == canonical(town):
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass
        print("MAP_READY=" + loaded, flush=True)
        raise SystemExit(0)
    time.sleep(0.25)
raise RuntimeError("map did not become ready: requested=" + town)
'''
    timeout_s = max(30.0, float(args.map_switch_timeout_s))
    cmd = [
        sys.executable, "-c", helper,
        str(args.carla_host), str(int(args.carla_port)), str(town),
        str(int(args.tm_port)), str(timeout_s),
    ]
    print(
        f"[carla-safe] preparing town={town} in isolated map-switch process ...",
        flush=True,
    )
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s + 15.0,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Isolated CARLA map switch to {town} timed out. Restart CARLA "
            f"directly on {town}, wait until the server is ready, then rerun."
        ) from exc

    stdout = str(result.stdout or "").strip()
    stderr = str(result.stderr or "").strip()
    if result.returncode != 0 or "MAP_READY=" not in stdout:
        # A native client can die after the server has already completed the
        # transition. Probe from a second clean process before declaring the
        # switch failed; this probe never calls load_world.
        probe_code = r'''
import sys
import carla
client = carla.Client(sys.argv[1], int(sys.argv[2]))
client.set_timeout(float(sys.argv[3]))
name = str(client.get_world().get_map().name).replace("\\", "/").rstrip("/").split("/")[-1]
print("MAP_READY=" + name, flush=True)
'''
        probe_stdout = ""
        try:
            probe = subprocess.run(
                [sys.executable, "-c", probe_code, str(args.carla_host),
                 str(int(args.carla_port)), "15.0"],
                capture_output=True, text=True, timeout=20.0, check=False,
            )
            probe_stdout = str(probe.stdout or "").strip()
            if probe.returncode == 0 and "MAP_READY=" in probe_stdout:
                probed = probe_stdout.split("MAP_READY=")[-1].splitlines()[0].strip()
                if _canonical_town_name(probed) == _canonical_town_name(town):
                    print(
                        f"[carla-safe] town ready after helper fault: {probed}",
                        flush=True,
                    )
                    return
        except Exception:
            probe_stdout = ""
        signal_hint = (
            " (native SIGSEGV detected)" if int(result.returncode) in {-11, 139}
            else ""
        )
        details = stderr or stdout or "no helper output"
        raise RuntimeError(
            f"CARLA could not safely switch to {town}{signal_hint}. "
            f"Restart CarlaUE4 directly on {town}, wait for RPC port "
            f"{args.carla_port}, and rerun the same command. Helper output: "
            f"{details[-1000:]}"
        )
    ready = stdout.split("MAP_READY=")[-1].splitlines()[0].strip()
    if _canonical_town_name(ready) != _canonical_town_name(town):
        raise RuntimeError(
            f"CARLA map verification failed: requested={town}, loaded={ready}."
        )
    print(f"[carla-safe] town ready: {ready}", flush=True)


# =============================================================================
# Networks (reused concept from hcrl_agent.py: Lipschitz MLPs + twin critics)
# =============================================================================

def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> nn.Module:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module


def spectral_linear(in_f: int, out_f: int, use_sn: bool, gain: float = math.sqrt(2.0)) -> nn.Module:
    layer = orthogonal_init(nn.Linear(in_f, out_f), gain=gain)
    return nn.utils.spectral_norm(layer) if use_sn else layer


class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy on [-1, 1]^A."""

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, state_dim: int, action_dim: int, hidden: int, use_sn: bool):
        super().__init__()
        self.body = nn.Sequential(
            spectral_linear(state_dim, hidden, use_sn), nn.SiLU(),
            spectral_linear(hidden, hidden, use_sn), nn.SiLU(),
        )
        self.mu = orthogonal_init(nn.Linear(hidden, action_dim), gain=0.5)
        self.log_std = orthogonal_init(nn.Linear(hidden, action_dim), gain=0.5)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.body(s)
        mu = self.mu(z)
        log_std = torch.clamp(self.log_std(z), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self(s)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)
        logp = dist.log_prob(u).sum(-1) - torch.log(1.0 - a.pow(2) + 1e-6).sum(-1)
        return a, logp, torch.tanh(mu)


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int, use_sn: bool):
        super().__init__()
        def q_net() -> nn.Sequential:
            return nn.Sequential(
                spectral_linear(state_dim + action_dim, hidden, use_sn), nn.SiLU(),
                spectral_linear(hidden, hidden, use_sn), nn.SiLU(),
                orthogonal_init(nn.Linear(hidden, 1), gain=1.0),
            )
        self.q1, self.q2 = q_net(), q_net()

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([s, a], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


# =============================================================================
# Replay buffer
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.float32)
        self.idx, self.full = 0, False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(self, s, a, r, s2, d) -> None:
        # Keep replay numerically clean. A single NaN/Inf transition can poison
        # SAC targets and make alpha/Q losses explode several thousand steps later.
        i = self.idx
        s = np.nan_to_num(np.asarray(s, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        a = np.clip(np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        s2 = np.nan_to_num(np.asarray(s2, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        r = float(np.nan_to_num(float(r), nan=0.0, posinf=1e4, neginf=-1e4))
        self.s[i] = s; self.a[i] = a; self.r[i] = r; self.s2[i] = s2; self.d[i] = float(d)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch: int, device: torch.device) -> Dict[str, torch.Tensor]:
        n = len(self)
        if n == 0:
            raise RuntimeError("Replay buffer is empty.")
        j = np.random.randint(0, n, size=batch)
        to = lambda x: torch.as_tensor(x[j], device=device)
        return {"s": to(self.s), "a": to(self.a), "r": to(self.r),
                "s2": to(self.s2), "d": to(self.d)}


# =============================================================================
# SAC agent
# =============================================================================

@dataclass
class SACConfig:
    state_dim: int = 12
    action_dim: int = 2
    hidden: int = 256
    use_spectral_norm: bool = True        # actor body (Lipschitz policy)
    critic_spectral_norm: bool = False    # critics need unconstrained value scale
    # Per-dimension observation scale for the route-aware driving state.
    # First 8 entries are the original barrier state:
    # [lat, heading, speed, lead_gap, rel_vel, obstacle, laneL, laneR].
    # Extra entries are: [route_heading_error, progress_frac, remaining_frac, ttc_front].
    obs_scale: Tuple[float, ...] = (
        # Keep every neural-network input roughly O(1).
        # rel_vel uses 15 m/s rather than 10 m/s because dense CARLA traffic
        # can produce larger closing-speed spikes; this avoids over-weighting
        # one dimension while preserving sensitivity to car-following.
        2.0, 0.6, 25.0, 80.0, 15.0, 80.0, 2.0, 2.0,
        0.6, 1.0, 1.0, 20.0,
    )
    obs_clip: float = 5.0
    obs_warn_threshold: float = 4.5
    obs_warn_every: int = 1000
    strict_state_dim: bool = False
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.1
    alpha_min: float = 0.01
    alpha_max: float = 1.00
    batch_size: int = 256
    replay_size: int = 300_000
    warmup_steps: int = 2_000
    updates_per_step: int = 1
    grad_clip: float = 5.0
    seed: int = 42


class SACAgent:
    def __init__(self, cfg: SACConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device if device is not None
                                   else ("cuda" if torch.cuda.is_available() else "cpu"))
        set_seed(cfg.seed)
        self.actor = GaussianPolicy(cfg.state_dim, cfg.action_dim, cfg.hidden,
                                    cfg.use_spectral_norm).to(self.device)
        self.critic = TwinQ(cfg.state_dim, cfg.action_dim, cfg.hidden,
                            cfg.critic_spectral_norm).to(self.device)
        self.critic_t = TwinQ(cfg.state_dim, cfg.action_dim, cfg.hidden,
                              cfg.critic_spectral_norm).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        for p in self.critic_t.parameters():
            p.requires_grad_(False)

        self._obs_scale = np.asarray(cfg.obs_scale, dtype=np.float32).reshape(-1)
        if self._obs_scale.size < cfg.state_dim:
            self._obs_scale = np.pad(self._obs_scale, (0, cfg.state_dim - self._obs_scale.size),
                                     constant_values=1.0)
        elif self._obs_scale.size > cfg.state_dim:
            self._obs_scale = self._obs_scale[:cfg.state_dim]
        # A zero/NaN/Inf scale would silently create NaN/Inf observations.
        # Replace invalid scales with 1.0 and keep a deterministic warning.
        bad_scale = (~np.isfinite(self._obs_scale)) | (np.abs(self._obs_scale) < 1e-6)
        if np.any(bad_scale):
            print(f"[warn] invalid obs_scale entries at {np.where(bad_scale)[0].tolist()}; replacing with 1.0")
            self._obs_scale = self._obs_scale.copy()
            self._obs_scale[bad_scale] = 1.0
        self._norm_calls = 0

        self.alpha_min = float(max(getattr(cfg, "alpha_min", 0.01), 1e-6))
        self.alpha_max = float(max(getattr(cfg, "alpha_max", 1.0), self.alpha_min))
        init_alpha = float(np.clip(cfg.init_alpha, self.alpha_min, self.alpha_max))
        self.log_alpha = torch.tensor([math.log(init_alpha)],
                                      requires_grad=True, device=self.device)
        self.target_entropy = -float(cfg.action_dim)

        self.actor_opt = Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.alpha_opt = Adam([self.log_alpha], lr=cfg.alpha_lr)

        self.replay = ReplayBuffer(cfg.replay_size, cfg.state_dim, cfg.action_dim)
        self.total_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        # Hard clamp prevents entropy-temperature explosion in long CARLA runs.
        # This keeps exploration useful while avoiding the observed alpha >> 1
        # regime where the policy becomes nearly random again.
        return self.log_alpha.exp().clamp(self.alpha_min, self.alpha_max)

    # -- acting ----------------------------------------------------------------
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Scale raw driving state to O(1) per dimension for SAC only.

        Important separation:
        * SAC actor/critic receive this normalized observation.
        * E-COCSF/barrier functions still receive the raw physical state.

        By default this function remains checkpoint-compatible by slicing/padding
        state vectors.  For final debugging runs, pass --strict_state_dim to make
        any state-dimension mismatch fail loudly instead of being hidden.
        """
        raw = np.asarray(state, dtype=np.float32).reshape(-1)
        original_dim = int(raw.size)
        if original_dim != int(self.cfg.state_dim) and bool(getattr(self.cfg, "strict_state_dim", False)):
            raise ValueError(f"Expected state_dim={self.cfg.state_dim}, got {original_dim}. "
                             "Use the matching --state_dim/--compact_obs setting or disable --strict_state_dim.")

        if raw.size > self.cfg.state_dim:
            s = raw[:self.cfg.state_dim]
        elif raw.size < self.cfg.state_dim:
            s = np.pad(raw, (0, self.cfg.state_dim - raw.size), constant_values=0.0)
        else:
            s = raw

        s = np.nan_to_num(s, nan=0.0, posinf=1e6, neginf=-1e6)
        z = s / self._obs_scale
        z = np.nan_to_num(z, nan=0.0, posinf=float(getattr(self.cfg, "obs_clip", 5.0)),
                          neginf=-float(getattr(self.cfg, "obs_clip", 5.0)))

        warn_thr = float(getattr(self.cfg, "obs_warn_threshold", 4.5))
        clip_val = float(getattr(self.cfg, "obs_clip", 5.0))
        self._norm_calls += 1
        max_abs = float(np.max(np.abs(z))) if z.size else 0.0
        warn_every = int(max(1, getattr(self.cfg, "obs_warn_every", 1000)))
        if max_abs > warn_thr and (self._norm_calls % warn_every == 1):
            idx = int(np.argmax(np.abs(z)))
            print(f"[warn] large normalized obs: max|z|={max_abs:.2f} "
                  f"at dim={idx}, raw={float(s[idx]):.3f}, scale={float(self._obs_scale[idx]):.3f}, "
                  f"raw_dim={original_dim}, cfg_dim={self.cfg.state_dim}")

        if clip_val > 0:
            z = np.clip(z, -clip_val, clip_val)
        return z.astype(np.float32, copy=False)

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.as_tensor(self.normalize(state).reshape(1, -1), device=self.device)
        with torch.no_grad():
            a, _, a_det = self.actor.sample(s)
        out = (a_det if deterministic else a)[0].cpu().numpy()
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(out, -1.0, 1.0).astype(np.float64)

    # -- learning ----------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        cfg = self.cfg
        batch = self.replay.sample(cfg.batch_size, self.device)
        s, a, r, s2, d = batch["s"], batch["a"], batch["r"], batch["s2"], batch["d"]

        with torch.no_grad():
            a2, logp2, _ = self.actor.sample(s2)
            q1t, q2t = self.critic_t(s2, a2)
            q_t = torch.min(q1t, q2t) - self.alpha * logp2
            target = r + cfg.gamma * (1.0 - d) * q_t
            target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        if not torch.isfinite(critic_loss):
            return {"critic_loss": float("nan"), "actor_loss": float("nan"),
                    "alpha": float(self.alpha.item())}
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_opt.step()

        a_pi, logp_pi, _ = self.actor.sample(s)
        q1p, q2p = self.critic(s, a_pi)
        actor_loss = (self.alpha.detach() * logp_pi - torch.min(q1p, q2p)).mean()
        if not torch.isfinite(actor_loss):
            return {"critic_loss": float(critic_loss.item()), "actor_loss": float("nan"),
                    "alpha": float(self.alpha.item())}
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # Project log_alpha to the configured interval. If projection clips the
        # parameter, remove only the stale Adam first-moment component that would
        # immediately push it farther outside the feasible interval on the next
        # step. The second moment is preserved, so the optimizer does not lose
        # its adaptive learning-rate scale.
        with torch.no_grad():
            lo = math.log(self.alpha_min)
            hi = math.log(self.alpha_max)
            before = self.log_alpha.detach().clone()
            self.log_alpha.clamp_(lo, hi)
            clipped_low = bool((before < lo).item())
            clipped_high = bool((before > hi).item())

        if clipped_low or clipped_high:
            opt_state = self.alpha_opt.state.get(self.log_alpha, {})
            exp_avg = opt_state.get("exp_avg")
            if exp_avg is not None:
                with torch.no_grad():
                    # Adam subtracts its first moment from the parameter.  At
                    # the lower wall positive momentum is outward; at the upper
                    # wall negative momentum is outward.  Remove only that
                    # outward component and preserve inward momentum.
                    if clipped_low:
                        exp_avg.clamp_(max=0.0)
                    if clipped_high:
                        exp_avg.clamp_(min=0.0)

        soft_update(self.critic, self.critic_t, cfg.tau)
        self.total_updates += 1
        return {"critic_loss": float(critic_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "alpha": float(self.alpha.item())}

    # -- persistence -------------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
                    "cfg": asdict(self.cfg),
                    "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "critic_t": self.critic_t.state_dict(),
                    "log_alpha": self.log_alpha.detach().cpu(),
                    "total_updates": self.total_updates,
                    # Optimizer states make --resume_checkpoint a true optimizer
                    # continuation when loading checkpoints produced by this
                    # version. Older checkpoints remain supported.
                    "actor_opt": self.actor_opt.state_dict(),
                    "critic_opt": self.critic_opt.state_dict(),
                    "alpha_opt": self.alpha_opt.state_dict(),
                    }, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SACAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(SACConfig(**ckpt["cfg"]), device=device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_t.load_state_dict(ckpt.get("critic_t", ckpt["critic"]))
        with torch.no_grad():
            agent.log_alpha.copy_(ckpt["log_alpha"].to(agent.device))
            agent.log_alpha.clamp_(math.log(agent.alpha_min), math.log(agent.alpha_max))
        agent.total_updates = int(ckpt.get("total_updates", 0))

        # Best-effort optimizer restoration. Checkpoints from older versions do
        # not contain these keys; they still resume safely with fresh optimizers.
        for key, opt in (
            ("actor_opt", agent.actor_opt),
            ("critic_opt", agent.critic_opt),
            ("alpha_opt", agent.alpha_opt),
        ):
            try:
                if key in ckpt:
                    opt.load_state_dict(ckpt[key])
            except Exception as exc:
                print(f"[warn] could not restore {key}: {exc}")
        return agent


# =============================================================================
# Action scaling and the black-box policy wrapper for E-COCSF
# =============================================================================

class ActionScaler:
    """Map policy output in [-1,1]^A to machine units and back."""

    def __init__(self, machine: ec.MachineCard):
        self.low = machine.low.astype(np.float64)
        self.high = machine.high.astype(np.float64)

    def to_env(self, a_norm: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(a_norm, dtype=np.float64), -1.0, 1.0)
        return self.low + 0.5 * (a + 1.0) * (self.high - self.low)

    def to_norm(self, a_env: np.ndarray) -> np.ndarray:
        a = np.asarray(a_env, dtype=np.float64)
        return np.clip(2.0 * (a - self.low) / (self.high - self.low) - 1.0, -1.0, 1.0)


def route_controller_steer_from_state(
    state: np.ndarray,
    kp_lat: float = 0.28,
    kp_head: float = 0.85,
    kp_route_head: float = 0.35,
    steer_limit: float = 0.60,
) -> float:
    """Route/lane stabilizing steering prior.

    The SAC policy is still a black-box policy, but raw SAC steering is not
    trusted to learn lane keeping from scratch in early CARLA training.  This
    controller uses the state variables that are already available:
      state[0] = lane lateral offset
      state[1] = lane heading error
      state[8] = route heading error, if the 12-D route-aware observation is used

    The output is in the same environment action unit as the steering component,
    i.e. approximately [-0.60, 0.60].
    """
    x = np.asarray(state, dtype=np.float64).reshape(-1)
    lat_err = float(x[0]) if x.size > 0 else 0.0
    lane_head_err = float(x[1]) if x.size > 1 else 0.0
    route_head_err = float(x[8]) if x.size > 8 else lane_head_err

    # Lane heading keeps the vehicle centered locally; route heading prevents
    # drift at long curves/junction successors selected by the waypoint route.
    # Progressive centering: gain grows with |lat| so riding the lane stripe
    # produces a firm pull back to center (mirror clearance vs parked cars).
    kp_lat_eff = kp_lat * (1.0 + 1.6 * min(abs(lat_err), 1.5))
    steer = -kp_lat_eff * lat_err - kp_head * lane_head_err - kp_route_head * route_head_err
    return float(np.clip(steer, -steer_limit, steer_limit))


def apply_acc_longitudinal_guard(
    state: np.ndarray,
    action: np.ndarray,
    target_speed: float = 10.0,
    acc_guard: bool = True,
    acc_time_headway: float = 1.8,
    acc_min_gap: float = 6.0,
    acc_kp: float = 0.75,
    acc_comfort_brake: float = 2.5,
    acc_emergency_brake: float = 4.0,
    acc_ttc_soft: float = 3.0,
    acc_ttc_hard: float = 1.4,
    acc_front_gap_active: float = 75.0,
    acc_soft_extra_gap: float = 1.5,
) -> np.ndarray:
    """Adaptive-cruise longitudinal guard for the nominal action.

    The SAC policy remains the black-box nominal policy, but the nominal action
    is passed through a lightweight ACC prior before E-COCSF/filtering. This is
    important during early training: positive warmup acceleration otherwise
    makes the ego catch the lead vehicle before SAC has learned braking.

    State convention:
      state[2] = ego forward speed (m/s)
      state[3] = lead/obstacle gap (m); 80 means "no relevant same-lane lead"
      state[4] = lead_speed - ego_speed; negative means ego is closing
      state[11] = TTC, if 12-D observation is used
    """
    a = np.asarray(action, dtype=np.float64).reshape(2).copy()
    if not acc_guard:
        return a

    x = np.asarray(state, dtype=np.float64).reshape(-1)
    if x.size < 5:
        return a

    speed = max(0.0, float(x[2]))
    gap = max(0.0, float(x[3]))
    rel_vel = float(x[4])                 # lead - ego
    closing = max(0.0, -rel_vel)

    # In this environment, _read_state() sets gap=80 when no same-lane lead is
    # relevant. Only guard when the lead is actually within an active range.
    if gap >= float(acc_front_gap_active):
        # Also cap free-road speed gently; otherwise the progress reward can
        # push the car to 18-20 m/s even when target_speed=10.
        free_speed_acc = float(acc_kp) * (float(target_speed) - speed)
        a[1] = min(float(a[1]), free_speed_acc)
        return a

    front_speed = max(0.0, speed + rel_vel)
    safe_gap = float(acc_min_gap) + float(acc_time_headway) * speed
    ttc = 20.0
    if x.size > 11:
        ttc = float(x[11])
    elif closing > 1e-3:
        ttc = gap / closing

    # Stopped/slow lead vehicles (typically a queue at a red light) need a
    # stopping-distance controller, not the old aggressive closing-speed rule
    #     desired_speed = speed - 0.9 * closing
    # which could collapse a 10 m/s command to ~1 m/s while the queue was still
    # about 20--25 m away.  Follow a continuous braking curve instead.
    slow_front = bool(front_speed <= 0.8)
    clearance = max(0.0, gap - float(acc_min_gap))
    required_stop = (closing * closing) / (2.0 * max(clearance, 0.30))

    if slow_front:
        desired_speed = min(
            float(target_speed),
            front_speed + math.sqrt(
                max(0.0, 2.0 * float(acc_comfort_brake) * clearance)
            ),
        )
        acc_cmd = float(acc_kp) * (desired_speed - speed)

        if gap <= float(acc_min_gap) or ttc < float(acc_ttc_hard):
            acc_cmd = -float(acc_emergency_brake)
        elif required_stop >= float(acc_comfort_brake):
            acc_cmd = min(
                acc_cmd,
                -float(np.clip(required_stop + 0.10, 0.20, float(acc_emergency_brake))),
            )
        elif speed > desired_speed + 0.20:
            acc_cmd = min(
                acc_cmd,
                -float(np.clip(max(0.20, required_stop), 0.20, float(acc_comfort_brake))),
            )

        a[1] = min(
            float(a[1]),
            float(np.clip(acc_cmd, -float(acc_emergency_brake), 2.5)),
        )
        return a

    # Moving front vehicle: keep time-headway following, but use a smooth
    # front-speed-relative target instead of instantly subtracting almost the
    # full closing speed from the ego target.
    gap_speed = max(
        0.0,
        (gap - float(acc_min_gap)) / max(float(acc_time_headway), 1e-6),
    )
    desired_speed = min(float(target_speed), gap_speed)
    if gap < safe_gap + float(acc_soft_extra_gap):
        smooth_follow_speed = front_speed + max(
            0.0,
            (gap - safe_gap) / max(float(acc_time_headway), 0.5),
        )
        desired_speed = min(desired_speed, smooth_follow_speed)

    acc_cmd = float(acc_kp) * (desired_speed - speed)

    # Hard TTC/gap override for genuinely urgent moving-vehicle conflicts.
    if ttc < float(acc_ttc_hard) or gap < max(1.5, 0.35 * safe_gap):
        acc_cmd = -float(acc_emergency_brake)
    elif required_stop >= float(acc_comfort_brake) or gap < safe_gap:
        acc_cmd = min(acc_cmd, -float(acc_comfort_brake))

    # Only reduce acceleration / increase braking. Do not force acceleration.
    a[1] = min(float(a[1]), float(np.clip(acc_cmd, -float(acc_emergency_brake), 2.5)))
    return a


def combine_controller_residual_action(
    state: np.ndarray,
    sac_env_action: np.ndarray,
    steer_mode: str = "residual",
    steer_residual_gain: float = 0.25,
    steer_blend_policy: float = 0.30,
    route_kp_lat: float = 0.28,
    route_kp_head: float = 0.85,
    route_kp_route_head: float = 0.35,
    steer_limit: float = 0.60,
    target_speed: float = 10.0,
    acc_guard: bool = True,
    acc_time_headway: float = 1.8,
    acc_min_gap: float = 6.0,
    acc_kp: float = 0.75,
    acc_comfort_brake: float = 2.5,
    acc_emergency_brake: float = 4.0,
    acc_ttc_soft: float = 3.0,
    acc_ttc_hard: float = 1.4,
    acc_front_gap_active: float = 75.0,
    acc_soft_extra_gap: float = 1.5,
) -> np.ndarray:
    """Convert SAC action into the final nominal action executed in CARLA.

    Modes:
      residual   : steer = route_controller + gain * SAC_residual
      blend      : steer = (1-policy_frac)*controller + policy_frac*SAC_steer
      controller : steer = controller only; SAC learns acceleration
      rl         : steer = SAC only; useful as a policy-only diagnostic

    Acceleration/brake remains learned by SAC, but is guarded by an ACC-style
    same-lane car-following prior so warmup and early exploration cannot simply
    drive into the lead vehicle.  E-COCSF still sees and filters the final 2-D
    nominal action [steer, acceleration].
    """
    a = np.asarray(sac_env_action, dtype=np.float64).reshape(2).copy()
    ctrl = route_controller_steer_from_state(
        state,
        kp_lat=route_kp_lat,
        kp_head=route_kp_head,
        kp_route_head=route_kp_route_head,
        steer_limit=steer_limit,
    )

    mode = str(steer_mode).lower()
    if mode == "residual":
        a[0] = np.clip(ctrl + float(steer_residual_gain) * a[0],
                       -steer_limit, steer_limit)
    elif mode == "blend":
        frac = float(np.clip(steer_blend_policy, 0.0, 1.0))
        a[0] = np.clip((1.0 - frac) * ctrl + frac * a[0],
                       -steer_limit, steer_limit)
    elif mode == "controller":
        a[0] = ctrl
    elif mode == "rl":
        a[0] = np.clip(a[0], -steer_limit, steer_limit)
    else:
        raise ValueError(f"Unknown steer_mode={steer_mode!r}. Use residual|blend|controller|rl.")

    a = apply_acc_longitudinal_guard(
        state, a,
        target_speed=target_speed,
        acc_guard=acc_guard,
        acc_time_headway=acc_time_headway,
        acc_min_gap=acc_min_gap,
        acc_kp=acc_kp,
        acc_comfort_brake=acc_comfort_brake,
        acc_emergency_brake=acc_emergency_brake,
        acc_ttc_soft=acc_ttc_soft,
        acc_ttc_hard=acc_ttc_hard,
        acc_front_gap_active=acc_front_gap_active,
        acc_soft_extra_gap=acc_soft_extra_gap,
    )
    return a



class TrainedDrivingPolicy:
    """Black-box policy interface expected by ECLCSAgent: state -> env action.

    The nominal policy is hybrid by default: a route-following steering prior
    stabilizes the car, while SAC supplies acceleration and a small residual
    steering correction.  This keeps the E-COCSF action interface 2-D and
    policy-agnostic, but prevents raw random/early SAC steering from causing
    repeated lane departures.
    """

    def __init__(self, agent: SACAgent, scaler: ActionScaler, deterministic: bool = True,
                 steer_mode: str = "residual", steer_residual_gain: float = 0.25,
                 steer_blend_policy: float = 0.30, route_kp_lat: float = 0.28,
                 route_kp_head: float = 0.85, route_kp_route_head: float = 0.35,
                 target_speed: float = 10.0, acc_guard: bool = True,
                 acc_time_headway: float = 1.8, acc_min_gap: float = 6.0,
                 acc_kp: float = 0.75, acc_comfort_brake: float = 2.5,
                 acc_emergency_brake: float = 4.0,
                 acc_ttc_soft: float = 3.0, acc_ttc_hard: float = 1.4,
                 acc_front_gap_active: float = 75.0, acc_soft_extra_gap: float = 1.5):
        self.agent = agent
        self.scaler = scaler
        self.deterministic = deterministic
        self.steer_mode = steer_mode
        self.steer_residual_gain = float(steer_residual_gain)
        self.steer_blend_policy = float(steer_blend_policy)
        self.route_kp_lat = float(route_kp_lat)
        self.route_kp_head = float(route_kp_head)
        self.route_kp_route_head = float(route_kp_route_head)
        self.target_speed = float(target_speed)
        self.acc_guard = bool(acc_guard)
        self.acc_time_headway = float(acc_time_headway)
        self.acc_min_gap = float(acc_min_gap)
        self.acc_kp = float(acc_kp)
        self.acc_comfort_brake = float(acc_comfort_brake)
        self.acc_emergency_brake = float(acc_emergency_brake)
        self.acc_ttc_soft = float(acc_ttc_soft)
        self.acc_ttc_hard = float(acc_ttc_hard)
        self.acc_front_gap_active = float(acc_front_gap_active)
        self.acc_soft_extra_gap = float(acc_soft_extra_gap)

    def __call__(self, state) -> np.ndarray:
        a_norm = self.agent.act(np.asarray(state, dtype=np.float64).reshape(-1),
                                deterministic=self.deterministic)
        a_env_raw = self.scaler.to_env(a_norm)
        return combine_controller_residual_action(
            state,
            a_env_raw,
            steer_mode=self.steer_mode,
            steer_residual_gain=self.steer_residual_gain,
            steer_blend_policy=self.steer_blend_policy,
            route_kp_lat=self.route_kp_lat,
            route_kp_head=self.route_kp_head,
            route_kp_route_head=self.route_kp_route_head,
            target_speed=self.target_speed,
            acc_guard=self.acc_guard,
            acc_time_headway=self.acc_time_headway,
            acc_min_gap=self.acc_min_gap,
            acc_kp=self.acc_kp,
            acc_comfort_brake=self.acc_comfort_brake,
            acc_emergency_brake=self.acc_emergency_brake,
            acc_ttc_soft=self.acc_ttc_soft,
            acc_ttc_hard=self.acc_ttc_hard,
            acc_front_gap_active=self.acc_front_gap_active,
            acc_soft_extra_gap=self.acc_soft_extra_gap,
        )


# =============================================================================
# Environment factory
# =============================================================================

def make_env(args, town: str, drift_scale: float, seed: int,
             load_town_override: Optional[bool] = None):
    load_town = (
        not bool(args.no_load_town)
        if load_town_override is None else bool(load_town_override)
    )
    return ec.CarlaDrivingEnv(host=args.carla_host, port=args.carla_port, town=town,
                              dt=args.dt, max_steps=args.max_steps, seed=seed,
                              drift_scale=drift_scale, target_speed=args.target_speed,
                              lead_gap0=args.lead_gap, spawn_lead_vehicle=not args.no_manual_lead,
                              throttle_floor=args.throttle_floor,
                              route_distance=args.route_distance,
                              success_bonus=args.success_bonus,
                              action_smoothing_beta=args.action_smoothing_beta,
                              terminate_on_headway_violation=args.terminate_on_headway_violation,
                              headway_hard_fail_gap=args.headway_hard_fail_gap,
                              route_step_m=args.route_step_m,
                              route_planner_mode=args.route_planner_mode,
                              allow_heuristic_route_fallback=args.allow_heuristic_route_fallback,
                              route_destination_candidates=args.route_destination_candidates,
                              route_turn_lookahead_m=args.route_turn_lookahead_m,
                              route_turn_speed=args.route_turn_speed,
                              route_turn_steer_gain=args.route_turn_steer_gain,
                              turn_recovery_speed=args.turn_recovery_speed,
                              turn_recovery_accel=args.turn_recovery_accel,
                              turn_recovery_patience_ticks=args.turn_recovery_patience_ticks,
                              failure_persistence_ticks=args.failure_persistence_ticks,
                              headway_failure_persistence_ticks=args.headway_failure_persistence_ticks,
                              use_augmented_state=not args.compact_obs,
                              num_traffic_vehicles=args.num_traffic_vehicles,
                              num_walkers=args.num_walkers,
                              tm_port=args.tm_port,
                              traffic_speed_difference=args.traffic_speed_difference,
                              traffic_min_distance=args.traffic_min_distance,
                              traffic_rear_safety_distance=args.traffic_rear_safety_distance,
                              traffic_auto_lane_change=args.traffic_auto_lane_change,
                              traffic_radius=args.traffic_radius,
                              walker_speed_min=args.walker_speed_min,
                              walker_speed_max=args.walker_speed_max,
                              traffic_warmup_ticks=args.traffic_warmup_ticks,
                              tm_hybrid_physics=args.tm_hybrid_physics,
                              protect_cross_town_walkers=not args.allow_cross_town_walkers,
                              episode_reset_settle_ticks=args.episode_reset_settle_ticks,
                              destroy_all_stale_actors=args.destroy_all_stale_actors,
                              weather_mode=args.weather_mode,
                              traffic_light_guard=not args.no_traffic_light_guard,
                              red_light_stop_distance=args.red_light_stop_distance,
                              traffic_light_hysteresis_ticks=args.traffic_light_hysteresis_ticks,
                              red_light_reaction_time_s=args.red_light_reaction_time_s,
                              red_light_activation_margin=args.red_light_activation_margin,
                              red_light_blend_distance=args.red_light_blend_distance,
                              yellow_light_stop=not args.no_yellow_light_stop,
                              traffic_light_use_affected_lanes=not args.no_traffic_light_affected_lanes,
                              traffic_light_heading_tolerance_deg=args.traffic_light_heading_tolerance_deg,
                              yellow_reaction_time_s=args.yellow_reaction_time_s,
                              yellow_comfort_decel=args.yellow_comfort_decel,
                              yellow_stop_margin=args.yellow_stop_margin,
                              junction_commit_clear_distance=args.junction_commit_clear_distance,
                              junction_commit_clear_ticks=args.junction_commit_clear_ticks,
                              stop_line_cross_tolerance=args.stop_line_cross_tolerance,
                              red_light_stop_buffer=args.red_light_stop_buffer,
                              red_light_virtual_offset=args.red_light_virtual_offset,
                              red_light_creep_speed=args.red_light_creep_speed,
                              red_light_creep_distance=args.red_light_creep_distance,
                              red_light_comfort_decel=args.red_light_comfort_decel,
                              red_light_keep_lead_gap=args.red_light_keep_lead_gap,
                              queue_stop_gap=args.queue_stop_gap,
                              queue_detect_distance=args.queue_detect_distance,
                              queue_creep_speed=args.queue_creep_speed,
                              vehicle_collision_guard=not args.no_vehicle_collision_guard,
                              vehicle_stop_gap=args.vehicle_stop_gap,
                              vehicle_detect_distance=args.vehicle_detect_distance,
                              vehicle_ttc_soft=args.vehicle_ttc_soft,
                              vehicle_ttc_hard=args.vehicle_ttc_hard,
                              vehicle_moving_time_headway=args.vehicle_moving_time_headway,
                              vehicle_soft_extra_gap=args.vehicle_soft_extra_gap,
                              vehicle_queue_speed_threshold=args.vehicle_queue_speed_threshold,
                              predictive_collision_guard=not args.no_predictive_collision_guard,
                              predictive_horizon_s=args.predictive_horizon_s,
                              predictive_step_s=args.predictive_step_s,
                              predictive_vehicle_radius_m=args.predictive_vehicle_radius_m,
                              predictive_lateral_margin_m=args.predictive_lateral_margin_m,
                              predictive_longitudinal_margin_m=args.predictive_longitudinal_margin_m,
                              predictive_ttc_soft=args.predictive_ttc_soft,
                              predictive_ttc_hard=args.predictive_ttc_hard,
                              predictive_clear_ticks=args.predictive_clear_ticks,
                              predictive_junction_lookahead_m=args.predictive_junction_lookahead_m,
                              predictive_junction_preview_speed=args.predictive_junction_preview_speed,
                              ego_overtake_enabled=not args.no_ego_overtake,
                              external_blockage_recovery=args.external_blockage_recovery,
                              external_blockage_patience_s=args.external_blockage_patience_s,
                              external_blockage_max_recoveries=args.external_blockage_max_recoveries,
                              road_edge_guard=not args.no_road_edge_guard,
                              lane_edge_soft_margin=args.lane_edge_soft_margin,
                              lane_edge_hard_margin=args.lane_edge_hard_margin,
                              lane_edge_target_speed=args.lane_edge_target_speed,
                              lane_edge_brake=args.lane_edge_brake,
                              lane_edge_steer_gain=args.lane_edge_steer_gain,
                              lane_edge_heading_gain=args.lane_edge_heading_gain,
                              traffic_light_route_scan=not args.no_traffic_light_route_scan,
                              traffic_light_route_scan_step=args.traffic_light_route_scan_step,
                              traffic_light_landmark_fallback=not args.no_traffic_light_landmark_fallback,
                              vehicle_route_corridor_factor=args.vehicle_route_corridor_factor,
                              vehicle_route_corridor_max=args.vehicle_route_corridor_max,
                              vary_weather=(not args.no_weather),
                              render_follow=not args.no_render,
                              load_town=load_town)


def default_machine(args) -> ec.MachineCard:
    cfg = ec.ECLCSConfig(dt=args.dt)
    return ec.MachineCard(action_low=cfg.action_low, action_high=cfg.action_high,
                          rate_limit=cfg.rate_limit, jerk_limit=cfg.jerk_limit,
                          neutral_action=cfg.neutral_action)


# =============================================================================
# Training
# =============================================================================

def _env_reset_with_retry(env, tries: int = 3, wait_s: float = 3.0):
    """Reset the CARLA env, absorbing transient episode-creation failures.

    env.reset() already retries internally (spawn, route, warmup).  If it still
    raises, one bad draw must not abort a multi-day training run: wait briefly
    (letting the server settle and Traffic Manager release actors) and try
    again with a fresh RNG draw.  Only after ``tries`` consecutive full
    failures is the error propagated.
    """
    last_exc: Optional[Exception] = None
    for i in range(1, int(tries) + 1):
        try:
            return env.reset()
        except Exception as exc:
            last_exc = exc
            print(f"[train][WARN] env.reset failed (outer attempt {i}/{tries}): "
                  f"{type(exc).__name__}: {exc}", flush=True)
            if i < tries:
                time.sleep(float(wait_s))
    raise RuntimeError(f"env.reset failed {tries} consecutive times") from last_exc


def _reconnect_env_with_retry(env, tries: int = 3, wait_s: float = 10.0) -> bool:
    """Attempt a full CARLA client rebuild after a server stall.

    A stalled server sometimes recovers (GPU contention, long GC pause), and a
    restarted server comes back on the same port.  Waiting between attempts
    gives both cases a chance before the run is declared unrecoverable.
    """
    reconnect = getattr(env, "reconnect", None)
    if reconnect is None:
        return False
    for i in range(1, int(tries) + 1):
        print(f"[train][WARN] reconnect attempt {i}/{tries} ...", flush=True)
        try:
            if bool(reconnect()):
                return True
        except Exception as exc:
            print(f"[train][WARN] reconnect raised: {type(exc).__name__}: {exc}",
                  flush=True)
        if i < tries:
            time.sleep(float(wait_s))
    return False


def train(args) -> Dict[str, Any]:
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    machine = default_machine(args)
    scaler = ActionScaler(machine)
    # Keep the CARLA state normalized for SAC while retaining raw physical state
    # for the E-COCSF safety filter.  The 5th state entry is relative velocity.
    obs_scale = list(SACConfig.obs_scale)
    if len(obs_scale) > 4:
        obs_scale[4] = float(args.rel_vel_obs_scale)
    sac_cfg = SACConfig(state_dim=args.state_dim, action_dim=machine.action_dim, hidden=args.hidden,
                        use_spectral_norm=not args.no_spectral_norm,
                        init_alpha=args.init_alpha, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
                        batch_size=args.batch_size, replay_size=args.replay_size,
                        warmup_steps=args.warmup_steps,
                        updates_per_step=args.updates_per_step, seed=args.seed,
                        obs_scale=tuple(obs_scale), obs_clip=args.obs_clip,
                        obs_warn_threshold=args.obs_warn_threshold,
                        obs_warn_every=args.obs_warn_every,
                        strict_state_dim=args.strict_state_dim)
    resumed = bool(str(getattr(args, "resume_checkpoint", "")).strip())
    if resumed:
        agent = SACAgent.load(args.resume_checkpoint, device=args.device)
        if int(agent.cfg.state_dim) != int(sac_cfg.state_dim):
            raise ValueError(
                f"Resume checkpoint state_dim={agent.cfg.state_dim} but current "
                f"configuration expects state_dim={sac_cfg.state_dim}."
            )
        if int(agent.cfg.action_dim) != int(sac_cfg.action_dim):
            raise ValueError(
                f"Resume checkpoint action_dim={agent.cfg.action_dim} but current "
                f"configuration expects action_dim={sac_cfg.action_dim}."
            )
        # Runtime-only normalization/debug options may be changed safely without
        # changing network shapes.
        agent.cfg.strict_state_dim = bool(args.strict_state_dim)
        agent.cfg.obs_clip = float(args.obs_clip)
        agent.cfg.obs_warn_threshold = float(args.obs_warn_threshold)
        agent.cfg.obs_warn_every = int(args.obs_warn_every)
        print(f"[train] RESUME checkpoint={args.resume_checkpoint} "
              f"updates={agent.total_updates} fresh_replay=1")
    else:
        agent = SACAgent(sac_cfg, device=args.device)

    print(f"[train] device={agent.device} town={args.train_town} "
          f"steps={args.total_steps} state_dim={sac_cfg.state_dim} resumed={int(resumed)}")

    env = make_env(args, args.train_town, args.train_drift, seed=args.seed)
    history: List[Dict[str, float]] = []
    ep_ret, ep_len, ep_viol, episode = 0.0, 0, 0, 0
    losses: Dict[str, float] = {}
    state = _env_reset_with_retry(env)
    t0 = time.time()
    ep_speed_sum = 0.0
    ep_red_violations = 0
    ep_red_stop_successes = 0
    ep_tl_dropouts = 0
    ep_progress_max = 0.0
    ep_reached_goal = False
    ep_termination = "running"
    ep_comp_viol: Dict[str, int] = {}
    try:
        for step in range(1, int(args.total_steps) + 1):
            if resumed:
                # Do not destroy a learned checkpoint with a second round of
                # random movement warmup. Use the stochastic learned policy from
                # the first resumed step while the fresh replay buffer fills.
                a_norm = agent.act(state, deterministic=False)
            elif step <= int(args.move_warmup_steps):
                # Movement-first warmup: fill replay with useful moving states.
                # a_norm[0] = steering, a_norm[1] = acceleration, both in [-1, 1].
                steer_span = 0.05 if args.steer_mode in ("controller", "residual") else 0.15
                a_norm = np.asarray([
                    np.random.uniform(-steer_span, steer_span),
                    np.random.uniform(args.warmup_accel_min, args.warmup_accel_max),
                ], dtype=np.float64)
            elif step <= sac_cfg.warmup_steps:
                # After movement warmup, keep broad exploration, but bias away
                # from excessive braking so the car does not relearn a static policy.
                steer_span = 0.20 if args.steer_mode in ("controller", "residual") else 0.50
                a_norm = np.asarray([
                    np.random.uniform(-steer_span, steer_span),
                    np.random.uniform(-0.2, 1.0),
                ], dtype=np.float64)
            else:
                a_norm = agent.act(state, deterministic=False)
            a_env_raw = scaler.to_env(a_norm)
            a_env = combine_controller_residual_action(
                state,
                a_env_raw,
                steer_mode=args.steer_mode,
                steer_residual_gain=args.steer_residual_gain,
                steer_blend_policy=args.steer_blend_policy,
                route_kp_lat=args.route_kp_lat,
                route_kp_head=args.route_kp_head,
                route_kp_route_head=args.route_kp_route_head,
                target_speed=args.target_speed,
                acc_guard=not args.no_acc_guard,
                acc_time_headway=args.acc_time_headway,
                acc_min_gap=args.acc_min_gap,
                acc_kp=args.acc_kp,
                acc_comfort_brake=args.acc_comfort_brake,
                acc_emergency_brake=args.acc_emergency_brake,
                acc_ttc_soft=args.acc_ttc_soft,
                acc_ttc_hard=args.acc_ttc_hard,
                acc_front_gap_active=args.acc_front_gap_active,
                acc_soft_extra_gap=args.acc_soft_extra_gap,
            )
            try:
                next_state, reward, done, info = env.step(a_env)
            except RuntimeError as exc:
                # CARLA server stall/crash mid-step.  Save the training state
                # FIRST (nothing about the agent is lost either way), then try
                # a full client reconnect.  On success the partial episode is
                # discarded and training continues; on failure the run exits
                # with a resumable checkpoint instead of a core dump.
                print(f"[train][WARN] env.step failed: {type(exc).__name__}: {exc}",
                      flush=True)
                emergency = out_dir / "policy_emergency.pt"
                try:
                    agent.save(str(emergency))
                    print(f"[train] emergency checkpoint saved: {emergency}",
                          flush=True)
                except Exception as save_exc:
                    print(f"[train][WARN] emergency save failed: {save_exc}",
                          flush=True)
                if not _reconnect_env_with_retry(env):
                    (out_dir / "train_history.json").write_text(
                        json.dumps(history, indent=2), encoding="utf-8")
                    raise RuntimeError(
                        "CARLA server is unreachable after repeated reconnect "
                        "attempts. Restart the CARLA server, then resume with "
                        f"--resume_checkpoint {emergency}"
                    ) from exc
                # Partial episode is invalid: no transition is stored, episode
                # accumulators are cleared, and a fresh episode begins.
                ep_ret, ep_len, ep_viol, ep_speed_sum = 0.0, 0, 0, 0.0
                ep_red_violations = 0
                ep_red_stop_successes = 0
                ep_tl_dropouts = 0
                ep_progress_max = 0.0
                ep_reached_goal = False
                ep_termination = "running"
                ep_comp_viol = {}
                state = _env_reset_with_retry(env)
                continue

            # SAC replay semantics: the critic action must use the SAME action
            # coordinates that the actor outputs and that actor_loss later queries.
            # Route steering, ACC, traffic-light, queue, turn, lane-edge and other
            # low-level guards are treated as state-dependent environment/wrapper
            # dynamics. Therefore replay stores the actor proposal a_norm, not the
            # post-guard executed physical action. The latter remains available in
            # info["executed_action_env"] for diagnostics and E-COCSF residuals.
            #
            # Time-limit bootstrapping: max-step truncation ends the rollout but is
            # not a true MDP terminal state. Only genuine terminations zero the SAC
            # bootstrap target. Older env versions remain backward compatible.
            true_terminal = bool(info.get(
                "terminated",
                bool(done) and not bool(info.get("timeout", False)),
            ))
            agent.replay.add(
                agent.normalize(state),
                np.asarray(a_norm, dtype=np.float32),
                float(reward),
                agent.normalize(next_state),
                float(true_terminal),
            )
            ep_ret += reward; ep_len += 1
            ep_viol += int(float(info.get("h", 1.0)) < 0.0)
            ep_speed_sum += float(info.get("speed", 0.0))
            ep_red_violations += int(bool(info.get("red_light_crossed_on_red", False)))
            ep_red_stop_successes += int(bool(info.get("red_light_stop_success", False)))
            ep_tl_dropouts += int(bool(info.get("traffic_light_detection_dropout", False)))
            ep_progress_max = max(ep_progress_max, float(info.get("progress_m", ep_progress_max)))
            ep_reached_goal = ep_reached_goal or bool(info.get("reached_goal", False))
            ep_termination = str(info.get("termination", ep_termination))
            for name, cval in info.get("h_components", {}).items():
                if cval < 0.0:
                    ep_comp_viol[name] = ep_comp_viol.get(name, 0) + 1
            state = next_state

            update_start_step = (
                int(args.resume_replay_warmup_steps) if resumed else int(sac_cfg.warmup_steps)
            )
            if step > update_start_step and len(agent.replay) >= int(agent.cfg.batch_size):
                for _ in range(int(args.updates_per_step)):
                    losses = agent.update()

            if done:
                episode += 1
                mean_speed = ep_speed_sum / max(ep_len, 1)
                # Per-component violation rates, sorted worst-first.
                comp_rates = {k: v / max(ep_len, 1) for k, v in ep_comp_viol.items()}
                worst = sorted(comp_rates.items(), key=lambda kv: -kv[1])[:3]
                comp_str = " ".join(f"{k}={r:.2f}" for k, r in worst) or "none"
                rec = {"episode": episode, "step": step, "return": ep_ret,
                       "length": ep_len, "violations": ep_viol,
                       "mean_speed": mean_speed, "progress_m": ep_progress_max,
                       "reached_goal": ep_reached_goal,
                       "termination": ep_termination,
                       "red_light_violations": int(ep_red_violations),
                       "red_light_stop_successes": int(ep_red_stop_successes),
                       "traffic_light_dropouts": int(ep_tl_dropouts),
                       "comp_viol": comp_rates, **losses}
                history.append(rec)
                if episode % max(1, args.log_every) == 0:
                    el = time.time() - t0
                    last_tmoving = int(info.get("traffic_moving_count", 0)) if isinstance(info, dict) else 0
                    last_ttotal = int(info.get("num_traffic_vehicles", 0)) if isinstance(info, dict) else 0
                    last_tmean = float(info.get("traffic_mean_speed", 0.0)) if isinstance(info, dict) else 0.0
                    last_tl = str(info.get("traffic_light_state", "none")) if isinstance(info, dict) else "none"
                    last_kind = str(info.get("lead_actor_kind", "none")) if isinstance(info, dict) else "none"
                    last_tl_dist = float(info.get("traffic_light_distance", 999.0)) if isinstance(info, dict) else 999.0
                    last_tl_stoperr = float(info.get("traffic_light_stop_error", 999.0)) if isinstance(info, dict) else 999.0
                    last_rlctrl = int(bool(info.get("red_light_control_active", False))) if isinstance(info, dict) else 0
                    last_rlbrake = int(bool(info.get("red_light_braking_active", False))) if isinstance(info, dict) else 0
                    last_rlreason = str(info.get("red_light_control_reason", "none")) if isinstance(info, dict) else "none"
                    last_qactive = int(bool(info.get("queue_active", False))) if isinstance(info, dict) else 0
                    last_qgap = float(info.get("queue_front_gap", 999.0)) if isinstance(info, dict) else 999.0
                    last_qerr = float(info.get("queue_gap_error", 999.0)) if isinstance(info, dict) else 999.0
                    last_fguard = int(bool(info.get("front_vehicle_guard_active", False))) if isinstance(info, dict) else 0
                    last_fgap = float(info.get("front_vehicle_gap", 999.0)) if isinstance(info, dict) else 999.0
                    last_fttc = float(info.get("front_vehicle_ttc", 20.0)) if isinstance(info, dict) else 20.0
                    last_lguard = int(bool(info.get("lane_edge_guard_active", False))) if isinstance(info, dict) else 0
                    last_lmargin = float(info.get("lane_edge_margin", 999.0)) if isinstance(info, dict) else 999.0
                    last_turn = int(bool(info.get("route_turn_guard_active", False))) if isinstance(info, dict) else 0
                    last_recover = int(bool(info.get("turn_recovery_active", False))) if isinstance(info, dict) else 0
                    last_turnerr = float(info.get("route_lookahead_yaw_error", 0.0)) if isinstance(info, dict) else 0.0
                    last_geom = str(info.get("state_geometry_source", "map")) if isinstance(info, dict) else "map"
                    last_maplat = float(info.get("map_lane_lat_error", 0.0)) if isinstance(info, dict) else 0.0
                    last_routelat = float(info.get("route_lat_error", 0.0)) if isinstance(info, dict) else 0.0
                    last_colzone = str(info.get("collision_zone", "none")) if isinstance(info, dict) else "none"
                    print(f"[train] ep={episode:4d} step={step:7d} "
                          f"ret={ep_ret:8.2f} len={ep_len:4d} viol={ep_viol:3d} "
                          f"speed={mean_speed:5.2f} progress={ep_progress_max:7.1f}m "
                          f"goal={int(ep_reached_goal)} term={ep_termination} "
                          f"alpha={losses.get('alpha', float(agent.alpha.item())):.3f} "
                          f"traffic={last_tmoving}/{last_ttotal}@{last_tmean:.1f}mps "
                          f"tl={last_tl} tl_dist={last_tl_dist:.1f}m stoperr={last_tl_stoperr:.1f}m "
                          f"rlctrl={last_rlctrl} rlbrake={last_rlbrake} rlreason={last_rlreason} "
                          f"lead={last_kind} q={last_qactive}@{last_qgap:.1f}m qe={last_qerr:.1f}m "
                          f"fg={last_fguard}@{last_fgap:.1f}m ttc={last_fttc:.1f}s "
                          f"lg={last_lguard}@{last_lmargin:.2f}m "
                          f"turn={last_turn} look={last_turnerr:.2f}rad recover={last_recover} "
                          f"geom={last_geom} maplat={last_maplat:.2f}m routelat={last_routelat:.2f}m "
                          f"colzone={last_colzone} | worst: {comp_str} ({el:.0f}s)")
                ep_ret, ep_len, ep_viol, ep_speed_sum = 0.0, 0, 0, 0.0
                ep_red_violations = 0
                ep_red_stop_successes = 0
                ep_tl_dropouts = 0
                ep_progress_max = 0.0
                ep_reached_goal = False
                ep_termination = "running"
                ep_comp_viol = {}
                state = _env_reset_with_retry(env)

            if step % int(args.ckpt_every) == 0:
                agent.save(str(out_dir / f"policy_step{step}.pt"))
    finally:
        try:
            env.close()
        except Exception:
            pass

    final = out_dir / "policy_final.pt"
    agent.save(str(final))
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2),
                                                encoding="utf-8")
    summary = {"checkpoint": str(final), "episodes": episode,
               "total_steps": int(args.total_steps),
               "successes": int(sum(1 for h in history if h.get("reached_goal"))),
               "last_returns": [h["return"] for h in history[-10:]],
               "last_progress_m": [h.get("progress_m", 0.0) for h in history[-10:]],
               "steer_mode": str(args.steer_mode),
               "steer_residual_gain": float(args.steer_residual_gain),
               "resumed_from": str(args.resume_checkpoint) if resumed else "",
               "out_dir": str(out_dir)}
    print(f"[train] saved {final}")
    return summary


# =============================================================================
# E-COCSF evaluation across towns
# =============================================================================

def _eval_ecocsf(policy: Callable, env, episodes: int,
                 max_steps: int, args) -> Dict[str, Any]:
    """Evaluate E-COCSF with independent episode resets.

    Critical correctness points:
      * reset q, actuator history, residual window, and audit state every episode;
      * preserve the cumulative filter log so aggregate metrics still cover all
        episodes;
      * print every episode, including fallback/infeasibility diagnostics, so a
        frozen policy can be identified immediately instead of after hours.
    """
    cfg = ec.ECLCSConfig(dt=args.dt, epsilon=args.epsilon, eta=args.eta,
                         q_init=args.q_init, q_max=args.q_max, zeta_max=args.zeta_max,
                         probe_probability=args.probe_probability, seed=args.seed,
                         certified_tube_delta=args.certified_tube_delta,
                         use_gain_schedule=args.gain_schedule,
                         feasibility_restoration=not args.no_feasibility_restoration,
                         restoration_grid_points=args.restoration_grid_points,
                         rebase_history_on_guard_override=
                         not args.no_guard_history_rebase)
    # Share the exact mutable barrier model with the CARLA environment so
    # lane width and context-dependent gap parameters are consistent.
    sf = ec.build_filter(cfg, "car", barrier=env.barrier)
    agent = ec.ECLCSAgent(policy, sf)

    rets, successes, progresses, collisions, steps_all, viols = [], 0, [], 0, 0, 0
    terminations: Dict[str, int] = {}
    episode_records: List[Dict[str, Any]] = []
    town_name = str(getattr(env, "town", "unknown"))
    episode_limit = max(int(max_steps), int(getattr(env, "max_steps", max_steps)))

    for ep in range(1, int(episodes) + 1):
        state = None
        goal_distance = float("nan")
        for route_attempt in range(1, int(args.eval_route_retries) + 1):
            state = _env_reset_with_retry(env)
            goal_distance = float(getattr(
                env, "_goal_distance_m", getattr(env, "route_distance", 1.0)
            ))
            requested_distance = float(getattr(env, "route_distance", args.route_distance))
            if (requested_distance <= 0.0 or
                    goal_distance + 1e-6 >=
                    float(args.eval_route_min_fraction) * requested_distance):
                break
            print(
                f"[eval][WARN] rejecting relaxed route town={town_name} "
                f"ep={ep} attempt={route_attempt}/{args.eval_route_retries} "
                f"goal={goal_distance:.1f}m requested={requested_distance:.1f}m",
                flush=True,
            )
        else:
            raise RuntimeError(
                f"Could not create a publication-valid route in {town_name}: "
                f"goal={goal_distance:.1f}m is below "
                f"{float(args.eval_route_min_fraction):.2f} of the requested "
                f"{float(args.route_distance):.1f}m after "
                f"{int(args.eval_route_retries)} attempts."
            )
        assert state is not None

        # Each CARLA episode is an independent evaluation trial.  Do not carry
        # previous actions, q adaptation, audit history, or residual history from
        # the previous episode.  Keep sf.log so final aggregate metrics remain
        # available across all episodes.
        sf.reset(q=args.q_init, clear_log=False)
        ep_log_start = len(sf.log)
        q_start = float(sf.q)

        ep_return = 0.0
        ep_progress = 0.0
        ep_collision = False
        ep_viols = 0
        ep_steps = 0
        ep_speed_sum = 0.0
        ep_red_violations = 0
        ep_red_stop_successes = 0
        ep_tl_dropouts = 0
        last_info: Dict[str, Any] = {}
        startup_low_speed = 0
        startup_infeasible = 0

        for _ in range(episode_limit):
            decision = agent.act(state)
            try:
                next_state, reward, done, info = env.step(decision.action)
            except RuntimeError as exc:
                # Server stall during evaluation: reconnect and abort this
                # episode; the outer loop continues with the next one.
                print(f"[eval][WARN] env.step failed: {exc}", flush=True)
                if not _reconnect_env_with_retry(env):
                    raise
                break
            actual_action = np.asarray(info.get("executed_action_env", decision.action), dtype=np.float64)
            agent.filter.update_after_transition(state, actual_action, next_state, decision)

            ep_return += float(reward)
            ep_steps += 1
            steps_all += 1
            bad = int(float(info.get("h", 1.0)) < 0.0)
            ep_viols += bad
            viols += bad
            ep_collision = ep_collision or bool(info.get("collided", False))
            ep_progress = max(ep_progress, float(info.get("progress_m", ep_progress)))
            ep_speed_sum += float(info.get("speed", 0.0))
            ep_red_violations += int(bool(info.get("red_light_crossed_on_red", False)))
            ep_red_stop_successes += int(bool(info.get("red_light_stop_success", False)))
            ep_tl_dropouts += int(bool(info.get("traffic_light_detection_dropout", False)))

            # Lightweight startup-deadlock diagnostic.  This should stay quiet
            # after the speed-min/restoration fixes; if it fires, the episode
            # line will also show a high strict-infeasibility rate.
            if ep_steps <= 100:
                startup_low_speed += int(float(info.get("speed", 0.0)) < 0.10)
                startup_infeasible += int(not bool(decision.feasible))

            state = next_state
            last_info = info
            if done:
                break

        success = bool(last_info.get("reached_goal", False))
        successes += int(success)
        collisions += int(ep_collision)
        term = str(last_info.get(
            "termination", "timeout" if ep_steps >= episode_limit else "unknown"
        ))
        terminations[term] = terminations.get(term, 0) + 1
        completion = float(min(1.0, ep_progress / max(goal_distance, 1e-6)))
        mean_speed = ep_speed_sum / max(ep_steps, 1)
        weather = str(last_info.get("weather_mode", getattr(env, "_current_weather_mode", "unknown")))
        ep_viol_rate = ep_viols / max(ep_steps, 1)

        ep_logs = sf.log[ep_log_start:]
        strict_infeasible_count = sum(
            int(not bool(r.get("feasible", True))) for r in ep_logs
        )
        executed_infeasible_count = sum(
            int(bool(r.get("infeasible", False))) for r in ep_logs
        )
        fallback_count = sum(int(str(r.get("mode", "")) == "fallback") for r in ep_logs)
        restoration_count = sum(int(bool(r.get("restoration_used", False))) for r in ep_logs)
        history_rebase_count = sum(int(bool(r.get("history_rebased", False))) for r in ep_logs)
        restoration_slacks = [
            float(r.get("restoration_slack", 0.0)) for r in ep_logs
            if np.isfinite(float(r.get("restoration_slack", 0.0)))
        ]
        calibration_logs = [
            r for r in ep_logs if bool(r.get("calibration_valid", False))
        ]
        exceed_count = sum(
            int(bool(r.get("hard_exceedance", False))) for r in calibration_logs
        )
        intervention_vals = [float(r.get("intervention_norm", 0.0)) for r in ep_logs]
        fallback_rate = fallback_count / max(len(ep_logs), 1)
        strict_infeasible_rate = strict_infeasible_count / max(len(ep_logs), 1)
        executed_infeasible_rate = executed_infeasible_count / max(len(ep_logs), 1)
        restoration_rate = restoration_count / max(len(ep_logs), 1)
        history_rebase_rate = history_rebase_count / max(len(ep_logs), 1)
        restoration_slack_mean = (
            float(np.mean(restoration_slacks)) if restoration_slacks else 0.0
        )
        restoration_slack_max = (
            float(np.max(restoration_slacks)) if restoration_slacks else 0.0
        )
        ep_exceed_rate = (
            exceed_count / len(calibration_logs)
            if calibration_logs else float("nan")
        )
        calibration_valid_rate = len(calibration_logs) / max(len(ep_logs), 1)
        intervention_mean = float(np.mean(intervention_vals)) if intervention_vals else 0.0
        q_final = float(sf.q)

        if ep_steps >= 100 and startup_low_speed >= 95 and startup_infeasible >= 50:
            print(
                f"[eval-warn] town={town_name} method=ecocsf ep={ep}/{int(episodes)} "
                f"possible_startup_deadlock low_speed={startup_low_speed}/100 "
                f"strict_infeasible={startup_infeasible}/100 q_start={q_start:.3f}",
                flush=True,
            )

        rec = {
            "episode": ep, "return": float(ep_return), "steps": int(ep_steps),
            "progress_m": float(ep_progress), "route_completion": completion,
            "route_goal_m": float(goal_distance),
            "requested_route_m": float(getattr(env, "route_distance", args.route_distance)),
            "success": int(success), "collision": int(ep_collision),
            "termination": term, "violation_rate": float(ep_viol_rate),
            "mean_speed": float(mean_speed), "weather_mode": weather,
            "red_light_violations": int(ep_red_violations),
            "red_light_stop_successes": int(ep_red_stop_successes),
            "traffic_light_dropouts": int(ep_tl_dropouts),
            "collision_zone": str(last_info.get("collision_zone", "none")),
            "collision_actor_type": str(last_info.get("collision_actor_type", "none")),
            "collision_impulse": float(last_info.get("collision_impulse", 0.0)),
            "predictive_conflict_kind": str(last_info.get("predictive_conflict_kind", "none")),
            "predictive_conflict_ttc": float(last_info.get("predictive_conflict_ttc", 999.0)),
            "predictive_conflict_actor_id": int(last_info.get("predictive_conflict_actor_id", -1)),
            "predictive_conflict_guard_active": int(bool(last_info.get("predictive_conflict_guard_active", False))),
            "external_blockage_recovery_count": int(last_info.get("external_blockage_recovery_count", 0)),
            "external_blockage_last_reason": str(last_info.get("external_blockage_last_reason", "none")),
            "pre_collision_trace": list(last_info.get("pre_collision_trace", [])),
            "route_turn_guard_active": int(bool(last_info.get("route_turn_guard_active", False))),
            "turn_recovery_active": int(bool(last_info.get("turn_recovery_active", False))),
            "q_start": q_start, "q_final": q_final,
            "fallback_rate": float(fallback_rate),
            "infeasible_rate": float(executed_infeasible_rate),
            "executed_infeasible_rate": float(executed_infeasible_rate),
            "strict_projection_infeasible_rate": float(strict_infeasible_rate),
            "calibration_valid_rate": float(calibration_valid_rate),
            "restoration_rate": float(restoration_rate),
            "restoration_slack_mean": float(restoration_slack_mean),
            "restoration_slack_max": float(restoration_slack_max),
            "history_rebase_rate": float(history_rebase_rate),
            "exceedance_rate": float(ep_exceed_rate),
            "intervention_mean": float(intervention_mean),
            "active_barrier_component": str(
                ep_logs[-1].get("active_barrier_component", "unknown")
                if ep_logs else "unknown"
            ),
            "active_barrier_value": float(
                ep_logs[-1].get("active_barrier_value", float("nan"))
                if ep_logs else float("nan")
            ),
            "last_filter_mode": str(
                ep_logs[-1].get("mode", "unknown") if ep_logs else "unknown"
            ),
            "last_restoration_kind": str(
                ep_logs[-1].get("restoration_kind", "none") if ep_logs else "none"
            ),
            "h_components": dict(last_info.get("h_components", {})),
            # A bounded tail is enough to diagnose a timeout without making
            # every successful episode result enormous.
            "filter_trace_tail": list(ep_logs[-200:]) if term == "timeout" else [],
        }
        episode_records.append(rec)
        progresses.append(ep_progress)
        rets.append(ep_return)

        print(
            f"[eval-ep] town={town_name} method=ecocsf "
            f"ep={ep:3d}/{int(episodes)} ret={ep_return:9.2f} steps={ep_steps:4d} "
            f"progress={ep_progress:7.1f}m completion={completion:5.3f} "
            f"success={int(success)} collision={int(ep_collision)} term={term} "
            f"viol={ep_viol_rate:.4f} speed={mean_speed:.2f}mps weather={weather} "
            f"turn={int(bool(last_info.get('route_turn_guard_active', False)))} "
            f"recover={int(bool(last_info.get('turn_recovery_active', False)))} "
            f"colzone={str(last_info.get('collision_zone', 'none'))} "
            f"pred={str(last_info.get('predictive_conflict_kind', 'none'))}@"
            f"{float(last_info.get('predictive_conflict_ttc', 999.0)):.1f}s "
            f"extclear={int(last_info.get('external_blockage_recovery_count', 0))} "
            f"q={q_start:.3f}->{q_final:.3f} "
            f"strict_infeas={strict_infeasible_rate:.3f} "
            f"exec_infeas={executed_infeasible_rate:.3f} "
            f"fallback={fallback_rate:.3f} "
            f"restore={restoration_rate:.3f}/{restoration_slack_mean:.3f} "
            f"rebase={history_rebase_rate:.3f} "
            f"calvalid={calibration_valid_rate:.3f} "
            f"exceed={ep_exceed_rate:.3f} interv={intervention_mean:.3f}",
            flush=True,
        )

    m = sf.metrics()
    n = max(steps_all, 1)
    m.update({"return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
              "violation_rate": viols / n,
              "collision_rate": collisions / max(episodes, 1),
              "success_rate": successes / max(episodes, 1),
              "progress_mean": float(np.mean(progresses)) if progresses else float("nan"),
              "progress_max": float(np.max(progresses)) if progresses else float("nan"),
              "route_completion_mean": float(np.mean([
                  float(r["route_completion"]) for r in episode_records
              ])) if episode_records else float("nan"),
              "steps": float(steps_all),
              "red_light_violations": int(sum(r.get("red_light_violations", 0) for r in episode_records)),
              "red_light_stop_successes": int(sum(r.get("red_light_stop_successes", 0) for r in episode_records)),
              "traffic_light_dropouts": int(sum(r.get("traffic_light_dropouts", 0) for r in episode_records)),
              "external_blockage_recoveries": int(sum(r.get("external_blockage_recovery_count", 0) for r in episode_records)),
              "terminations": terminations,
              "episode_records": episode_records})
    audit_prefix = f"ecocsf_{town_name.lower()}"
    m["audit_paths"] = sf.save_audit_log(
        Path(args.out_dir) / "audit", prefix=audit_prefix
    )
    return m


EVAL_COLUMNS = ["town", "method", "return_mean", "return_std", "success_rate",
                "route_completion_mean", "progress_mean", "progress_max",
                "violation_rate", "collision_rate",
                "exceedance_rate", "coverage_error_signed", "coverage_error_abs",
                "calibration_valid_rate", "certified_fraction",
                "certified_fraction_valid", "infeasible_rate",
                "executed_infeasible_rate", "strict_projection_infeasible_rate",
                "fallback_rate", "P_out", "P_cross", "intervention_mean",
                "restoration_rate", "restoration_slack_mean",
                "history_rebase_rate",
                "red_light_violations", "red_light_stop_successes",
                "traffic_light_dropouts", "external_blockage_recoveries", "steps"]

EVAL_EPISODE_COLUMNS = [
    "town", "method", "episode", "return", "steps", "progress_m",
    "route_completion", "route_goal_m", "requested_route_m",
    "success", "collision", "termination",
    "violation_rate", "mean_speed", "weather_mode", "collision_zone",
    "collision_actor_type", "collision_impulse",
    "predictive_conflict_kind", "predictive_conflict_ttc",
    "predictive_conflict_actor_id", "predictive_conflict_guard_active",
    "external_blockage_recovery_count", "external_blockage_last_reason",
    "route_turn_guard_active", "turn_recovery_active", "q_start", "q_final",
    "infeasible_rate", "executed_infeasible_rate",
    "strict_projection_infeasible_rate", "calibration_valid_rate",
    "fallback_rate", "restoration_rate", "restoration_slack_mean",
    "restoration_slack_max", "history_rebase_rate",
    "active_barrier_component", "active_barrier_value",
    "last_filter_mode", "last_restoration_kind",
    "exceedance_rate", "intervention_mean",
    "red_light_violations", "red_light_stop_successes",
    "traffic_light_dropouts",
]


def evaluate(args) -> Dict[str, Any]:
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    agent = SACAgent.load(args.checkpoint, device=args.device)
    machine = default_machine(args)
    policy = TrainedDrivingPolicy(
        agent,
        ActionScaler(machine),
        deterministic=True,
        steer_mode=args.steer_mode,
        steer_residual_gain=args.steer_residual_gain,
        steer_blend_policy=args.steer_blend_policy,
        route_kp_lat=args.route_kp_lat,
        route_kp_head=args.route_kp_head,
        route_kp_route_head=args.route_kp_route_head,
        target_speed=args.target_speed,
        acc_guard=not args.no_acc_guard,
        acc_time_headway=args.acc_time_headway,
        acc_min_gap=args.acc_min_gap,
        acc_kp=args.acc_kp,
        acc_comfort_brake=args.acc_comfort_brake,
        acc_emergency_brake=args.acc_emergency_brake,
        acc_ttc_soft=args.acc_ttc_soft,
        acc_ttc_hard=args.acc_ttc_hard,
        acc_front_gap_active=args.acc_front_gap_active,
        acc_soft_extra_gap=args.acc_soft_extra_gap,
    )
    towns = [t.strip() for t in args.eval_towns.split(",") if t.strip()]
    if not towns:
        raise ValueError("--eval_towns must contain at least one CARLA town.")
    print(f"[eval] checkpoint={args.checkpoint} towns={towns} method=ecocsf")

    rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []
    audit_paths_by_town: Dict[str, Dict[str, str]] = {}

    for town in towns:
        eval_seed = int(args.seed) + 1
        print(
            f"[eval] start town={town} method=ecocsf episodes={args.episodes} "
            f"seed={eval_seed}",
            flush=True,
        )
        isolated_switch = bool(
            not args.no_load_town and not args.no_isolated_map_switch
        )
        if isolated_switch:
            _prepare_carla_town_isolated(args, town)
        env = make_env(
            args, town, args.eval_drift, seed=eval_seed,
            # The isolated helper has already loaded and verified the town.
            # Prevent native load_world from running in this process.
            load_town_override=False if isolated_switch else None,
        )
        try:
            m = _eval_ecocsf(policy, env, args.episodes, args.max_steps, args)
        finally:
            try:
                env.close()
            except Exception:
                pass
            # Give CARLA a brief recovery window after actor teardown and
            # synchronous-world restoration before loading the next town.
            try:
                time.sleep(max(0.0, float(args.eval_cooldown_s)))
            except Exception:
                pass

        audit_paths_by_town[town] = dict(m.get("audit_paths", {}))

        row = {"town": town, "method": "ecocsf",
               **{k: m.get(k, float("nan")) for k in EVAL_COLUMNS[2:]}}
        rows.append(row)

        for ep_rec in m.get("episode_records", []):
            episode_rows.append({"town": town, "method": "ecocsf", **ep_rec})

        print(f"[eval] {town:8s} ecocsf       "
              f"ret={row['return_mean']:8.2f} success={row['success_rate']:.3f} "
              f"completion={row['route_completion_mean']:.3f} "
              f"collision={row['collision_rate']:.3f} "
              f"viol={row['violation_rate']:.4f} exceed={row['exceedance_rate']:.4f} "
              f"cert={row['certified_fraction']:.3f} "
              f"calvalid={row['calibration_valid_rate']:.3f} "
              f"exec_infeas={row['executed_infeasible_rate']:.3f} "
              f"strict_infeas={row['strict_projection_infeasible_rate']:.3f} "
              f"restore={row['restoration_rate']:.3f} "
              f"extclear={int(row['external_blockage_recoveries'])}", flush=True)
    csv_path = out_dir / "ecocsf_eval_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in EVAL_COLUMNS})

    episode_csv_path = out_dir / "ecocsf_eval_episode_results.csv"
    with episode_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_EPISODE_COLUMNS)
        w.writeheader()
        for r in episode_rows:
            w.writerow({k: r.get(k, "") for k in EVAL_EPISODE_COLUMNS})

    collision_trace_path = out_dir / "ecocsf_eval_collision_traces.jsonl"
    with collision_trace_path.open("w", encoding="utf-8") as f:
        for r in episode_rows:
            trace = r.get("pre_collision_trace", [])
            if trace:
                f.write(json.dumps({
                    "town": r.get("town"),
                    "method": r.get("method"),
                    "episode": r.get("episode"),
                    "collision_actor_type": r.get("collision_actor_type", "none"),
                    "collision_zone": r.get("collision_zone", "none"),
                    "collision_impulse": r.get("collision_impulse", 0.0),
                    "trace": trace,
                }) + "\n")

    deadlock_trace_path = out_dir / "ecocsf_eval_deadlock_traces.jsonl"
    with deadlock_trace_path.open("w", encoding="utf-8") as f:
        for r in episode_rows:
            trace = r.get("filter_trace_tail", [])
            if trace:
                f.write(json.dumps({
                    "town": r.get("town"),
                    "method": r.get("method"),
                    "episode": r.get("episode"),
                    "termination": r.get("termination"),
                    "progress_m": r.get("progress_m"),
                    "active_barrier_component": r.get(
                        "active_barrier_component", "unknown"
                    ),
                    "active_barrier_value": r.get(
                        "active_barrier_value", float("nan")
                    ),
                    "last_filter_mode": r.get("last_filter_mode", "unknown"),
                    "last_restoration_kind": r.get(
                        "last_restoration_kind", "none"
                    ),
                    "h_components": r.get("h_components", {}),
                    "trace": trace,
                }) + "\n")

    (out_dir / "ecocsf_eval_results.json").write_text(
        json.dumps({"rows": rows, "episodes": episode_rows,
                    "checkpoint": args.checkpoint, "epsilon": args.epsilon,
                    "audit_paths_by_town": audit_paths_by_town}, indent=2),
        encoding="utf-8")
    print(f"[eval] wrote {csv_path}")
    print(f"[eval] wrote {episode_csv_path}")
    print(f"[eval] wrote {collision_trace_path}")
    print(f"[eval] wrote {deadlock_trace_path}")
    return {"rows": rows, "episodes": episode_rows,
            "csv": str(csv_path), "episode_csv": str(episode_csv_path),
            "collision_traces": str(collision_trace_path),
            "deadlock_traces": str(deadlock_trace_path),
            "out_dir": str(out_dir)}


# =============================================================================
# =============================================================================
# CLI
# =============================================================================

def probe(args) -> None:
    """Spawn one car, floor the throttle, print speed. Answers 'can it move?'."""
    import carla
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(20.0)
    world = client.load_world(args.train_town) if not args.no_load_town else client.get_world()
    s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = args.dt
    world.apply_settings(s)
    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    car = None
    for sp in world.get_map().get_spawn_points():
        car = world.try_spawn_actor(bp, sp)
        if car is not None:
            break
    if car is None:
        print("[probe] could not spawn a vehicle"); return
    try:
        car.set_simulate_physics(True)
        for _ in range(20):
            world.tick()
        print("[probe] flooring throttle for 40 ticks ...")
        max_v = 0.0
        for i in range(40):
            car.apply_control(carla.VehicleControl(
                throttle=1.0, brake=0.0, hand_brake=False, manual_gear_shift=False))
            world.tick()
            v = car.get_velocity()
            spd = (v.x**2 + v.y**2 + v.z**2) ** 0.5
            max_v = max(max_v, spd)
            if i % 5 == 0:
                print(f"[probe] step {i:2d}: speed = {spd:.2f} m/s")
        print(f"[probe] MAX speed = {max_v:.2f} m/s")
        if max_v < 0.5:
            print("[probe] VERDICT: car did NOT move -> physics/gear issue "
                  "(check GPU physics, town, or CARLA build).")
        else:
            print("[probe] VERDICT: car moves fine -> any 'freeze' is the "
                  "spectator view, not the simulation.")
    finally:
        try:
            car.destroy()
        except Exception:
            pass
        s.synchronous_mode = False; world.apply_settings(s)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/evaluate a black-box CARLA driving "
                                            "policy for the E-COCSF paper experiments.")
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--probe", action="store_true",
                   help="Spawn one car, floor throttle, print speed (movement test).")

    # environment
    p.add_argument("--carla_host", type=str, default="localhost")
    p.add_argument("--carla_port", type=int, default=2000)
    p.add_argument("--train_town", type=str, default="Town04")
    p.add_argument("--eval_towns", type=str, default="Town03,Town04,Town05")
    p.add_argument("--no_weather", action="store_true",
                   help="Disable weather changes. Do NOT use this with --weather_mode night_rain.")
    p.add_argument("--weather_mode", type=str, default="clear",
                   choices=["clear", "rain", "night", "night_rain", "fog", "morning_rain", "random", "dynamic"],
                   help="CARLA weather/lighting preset. Use random for per-episode random clear/rain/fog/night.")
    p.add_argument("--no_render", action="store_true")
    p.add_argument("--no_load_town", action="store_true")
    p.add_argument("--no_isolated_map_switch", action="store_true",
                   help="Disable crash-contained evaluation map switching and call load_world in the main process (not recommended on packaged CARLA 0.9.15).")
    p.add_argument("--map_switch_timeout_s", type=float, default=90.0,
                   help="Timeout for isolated evaluation map switching and readiness verification.")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=4000,
                   help="Episode length in steps. Increase for long Town10HD / red-light episodes.")
    p.add_argument("--target_speed", type=float, default=18.0)
    p.add_argument("--throttle_floor", type=float, default=0.35,
                   help="Minimum ego throttle to overcome standstill (raise if the "
                        "car won't move, lower if it creeps when it should idle).")
    p.add_argument("--lead_gap", type=float, default=40.0,
                   help="Initial lead-vehicle gap (m). Must exceed the headway "
                        "requirement d0 + T*target_speed, or episodes start "
                        "already violating (bring-up finding).")
    p.add_argument("--no_manual_lead", action="store_true",
                   help="Disable the extra hand-coded lead vehicle. Use this for traffic-light/Town10HD experiments so the ego is not stopped far behind a non-TM lead car.")
    p.add_argument("--route_distance", type=float, default=1000.0,
                   help="Destination/route distance in meters; episode succeeds when ego reaches this progress.")
    p.add_argument("--success_bonus", type=float, default=100.0)
    p.add_argument("--action_smoothing_beta", type=float, default=0.15,
                   help="Low-pass smoothing for CARLA throttle/steer/brake commands.")
    p.add_argument("--steer_mode", type=str, default="residual",
                   choices=["residual", "blend", "controller", "rl"],
                   help="Steering interface: controller, residual, blend, or raw RL steering.")
    p.add_argument("--steer_residual_gain", type=float, default=0.20,
                   help="Residual steering strength for --steer_mode residual. Start with 0.15-0.25.")
    p.add_argument("--steer_blend_policy", type=float, default=0.30,
                   help="Policy steering fraction for --steer_mode blend.")
    p.add_argument("--route_kp_lat", type=float, default=0.30,
                   help="Route controller lateral-error gain.")
    p.add_argument("--route_kp_head", type=float, default=0.90,
                   help="Route controller lane-heading-error gain.")
    p.add_argument("--route_kp_route_head", type=float, default=0.40,
                   help="Route controller route-heading-error gain for 12-D observations.")
    p.add_argument("--route_turn_lookahead_m", type=float, default=12.0,
                   help="Look-ahead distance in meters for turn anticipation.")
    p.add_argument("--route_turn_speed", type=float, default=4.5,
                   help="Approximate target speed used by the common route-turn guard.")
    p.add_argument("--route_turn_steer_gain", type=float, default=0.85,
                   help="Look-ahead route-heading steering gain at curves/junctions.")
    p.add_argument("--turn_recovery_speed", type=float, default=0.8,
                   help="Below this speed, a safe unblocked turn may trigger recovery.")
    p.add_argument("--turn_recovery_accel", type=float, default=0.55,
                   help="Small positive acceleration used only for safe turn-stall recovery.")
    p.add_argument("--turn_recovery_patience_ticks", type=int, default=25,
                   help="Low-speed turn ticks before recovery is allowed.")
    p.add_argument("--no_acc_guard", action="store_true",
                   help="Disable the ACC-style longitudinal guard. Keep this OFF for training with a lead vehicle.")
    p.add_argument("--acc_time_headway", type=float, default=1.2,
                   help="ACC guard time headway T in seconds. Increase to 2.0-2.2 for safer following.")
    p.add_argument("--acc_min_gap", type=float, default=4.0,
                   help="ACC guard standstill/minimum following gap in meters.")
    p.add_argument("--acc_kp", type=float, default=0.75,
                   help="ACC guard speed error gain for acceleration limiting.")
    p.add_argument("--acc_comfort_brake", type=float, default=2.5,
                   help="ACC guard comfortable braking deceleration in m/s^2.")
    p.add_argument("--acc_emergency_brake", type=float, default=4.0,
                   help="ACC guard emergency braking deceleration in m/s^2.")
    p.add_argument("--acc_ttc_soft", type=float, default=3.0,
                   help="Start strong braking below this time-to-collision.")
    p.add_argument("--acc_ttc_hard", type=float, default=1.4,
                   help="Emergency braking below this time-to-collision.")
    p.add_argument("--acc_front_gap_active", type=float, default=75.0,
                   help="Only activate ACC guard if same-lane lead gap is below this value.")
    p.add_argument("--acc_soft_extra_gap", type=float, default=1.5,
                   help="Extra meters beyond dynamic safe gap where ACC begins gentle slowdown. Use 1--2 m to avoid stopping too early.")
    p.add_argument("--no_traffic_light_guard", action="store_true",
                   help="Disable red/yellow traffic-light stopping guard.")
    p.add_argument("--no_traffic_light_affected_lanes", action="store_true",
                   help="Disable TrafficLight.get_affected_lane_waypoints() filtering and use route/heading geometry only.")
    p.add_argument("--traffic_light_heading_tolerance_deg", type=float, default=35.0,
                   help="Maximum heading mismatch in degrees for route/ego-frame traffic-light relevance.")
    p.add_argument("--yellow_reaction_time_s", type=float, default=0.5,
                   help="Reaction time used by the human-like yellow-light dilemma-zone stopping test.")
    p.add_argument("--yellow_comfort_decel", type=float, default=3.0,
                   help="Comfortable deceleration in m/s^2 used by the yellow-light stop/go decision.")
    p.add_argument("--yellow_stop_margin", type=float, default=1.0,
                   help="Extra stopping-distance margin in meters for yellow-light decisions.")
    p.add_argument("--junction_commit_clear_distance", type=float, default=8.0,
                   help="Minimum route progress beyond a crossed stop line before normal junction commitment release.")
    p.add_argument("--junction_commit_clear_ticks", type=int, default=5,
                   help="Consecutive clear ticks required before releasing a crossed-signal junction commitment.")
    p.add_argument("--stop_line_cross_tolerance", type=float, default=0.20,
                   help="Front-bumper stop-line crossing tolerance in meters.")
    p.add_argument("--red_light_stop_distance", type=float, default=70.0,
                   help="Detection/look-ahead horizon in meters for route-relevant red/yellow stop lines; detection alone does not force braking or creep.")
    p.add_argument("--traffic_light_hysteresis_ticks", type=int, default=5,
                   help="Grace ticks for one-frame traffic-light detector dropouts.")
    p.add_argument("--red_light_reaction_time_s", type=float, default=0.35,
                   help="Reaction-time term used by speed-dependent red-light activation distance.")
    p.add_argument("--red_light_activation_margin", type=float, default=2.0,
                   help="Additional meters in the dynamic red-light braking activation distance.")
    p.add_argument("--red_light_blend_distance", type=float, default=2.0,
                   help="Meters used to smoothly blend braking-curve speed into creep speed.")
    p.add_argument("--red_light_stop_buffer", type=float, default=1.2,
                   help="Desired front-bumper distance before the physical CARLA stop waypoint/zebra line.")
    p.add_argument("--red_light_virtual_offset", type=float, default=0.0,
                   help="Deprecated compatibility flag. The current controller uses the physical CARLA stop-line distance directly.")
    p.add_argument("--red_light_creep_speed", type=float, default=0.8,
                   help="Maximum precision-creep speed inside the final red-light creep zone only.")
    p.add_argument("--red_light_creep_distance", type=float, default=3.0,
                   help="Final close-stop zone in meters where small positive crawl acceleration may be requested; farther away, red-light detection cannot force creep.")
    p.add_argument("--red_light_comfort_decel", type=float, default=3.0,
                   help="Comfort deceleration in m/s^2 used by the speed-dependent red-light braking curve.")
    p.add_argument("--red_light_keep_lead_gap", type=float, default=8.0,
                   help="Deprecated compatibility flag; queue control uses --queue_stop_gap/--queue_detect_distance.")
    p.add_argument("--queue_stop_gap", type=float, default=3.0,
                   help="Desired bumper-to-bumper stopping distance behind an already waiting vehicle at a red light.")
    p.add_argument("--queue_detect_distance", type=float, default=25.0,
                   help="If a front vehicle is within this distance during red/yellow light, treat it as a queue target instead of creeping to the stop line.")
    p.add_argument("--queue_creep_speed", type=float, default=1.5,
                   help="Slow crawl speed when forming a queue behind a stopped/waiting vehicle.")

    # universal collision / road-edge safety guards
    p.add_argument("--no_vehicle_collision_guard", action="store_true",
                   help="Disable the low-level same-route front-vehicle emergency brake.")
    p.add_argument("--vehicle_stop_gap", type=float, default=4.0,
                   help="Desired bumper clearance behind spawned/NPC vehicles when stopped.")
    p.add_argument("--vehicle_detect_distance", type=float, default=45.0,
                   help="Look-ahead distance for the universal front-vehicle collision guard.")
    p.add_argument("--vehicle_ttc_soft", type=float, default=3.5,
                   help="Soft TTC threshold for front-vehicle braking.")
    p.add_argument("--vehicle_ttc_hard", type=float, default=1.4,
                   help="Hard TTC threshold for emergency braking.")
    p.add_argument("--vehicle_moving_time_headway", type=float, default=1.2,
                   help="Extra dynamic gap for moving vehicles: desired gap = vehicle_stop_gap + this * ego_speed.")
    p.add_argument("--vehicle_soft_extra_gap", type=float, default=1.5,
                   help="Extra meters before desired gap where the front-vehicle guard begins gentle braking.")
    p.add_argument("--vehicle_queue_speed_threshold", type=float, default=0.7,
                   help="Front actor below this speed is treated as stopped/queued and uses queue_stop_gap.")
    p.add_argument("--no_predictive_collision_guard", action="store_true",
                   help="Disable velocity-swept OBB protection for cut-ins and junction crossing traffic.")
    p.add_argument("--predictive_horizon_s", type=float, default=3.0,
                   help="Prediction horizon for dynamic vehicle conflicts.")
    p.add_argument("--predictive_step_s", type=float, default=0.20,
                   help="Time spacing between predicted oriented-box samples.")
    p.add_argument("--predictive_vehicle_radius_m", type=float, default=45.0,
                   help="Maximum radius for predictive dynamic-vehicle scanning.")
    p.add_argument("--predictive_lateral_margin_m", type=float, default=0.35,
                   help="Per-vehicle lateral OBB inflation; keep below half the inter-lane free space.")
    p.add_argument("--predictive_longitudinal_margin_m", type=float, default=1.0,
                   help="Per-vehicle longitudinal OBB inflation for braking margin.")
    p.add_argument("--predictive_ttc_soft", type=float, default=3.0,
                   help="Begin smooth braking below this predicted conflict time.")
    p.add_argument("--predictive_ttc_hard", type=float, default=1.2,
                   help="Emergency braking threshold for predicted conflicts.")
    p.add_argument("--predictive_clear_ticks", type=int, default=5,
                   help="Consecutive conflict-free ticks before releasing a predictive yield.")
    p.add_argument("--predictive_junction_lookahead_m", type=float, default=30.0,
                   help="Route distance ahead where junction occupancy prediction becomes active.")
    p.add_argument("--predictive_junction_preview_speed", type=float, default=3.0,
                   help="Prospective speed used to test whether entering a junction would conflict.")
    p.add_argument("--no_ego_overtake", action="store_true",
                   help="Disable ego overtaking for a conservative diagnostic ablation.")
    p.add_argument("--external_blockage_recovery", action="store_true",
                   help="After a grace period, remove only an experiment-owned stationary NPC that persistently blocks the ego; every recovery is logged.")
    p.add_argument("--external_blockage_patience_s", type=float, default=15.0,
                   help="Seconds the same stationary owned NPC must block a stopped ego before optional recovery.")
    p.add_argument("--external_blockage_max_recoveries", type=int, default=2,
                   help="Maximum logged NPC blockage recoveries allowed per episode.")
    p.add_argument("--no_road_edge_guard", action="store_true",
                   help="Disable lane/road-edge steering and speed guard.")
    p.add_argument("--lane_edge_soft_margin", type=float, default=0.75,
                   help="Start road-edge correction when lane margin is below this many meters.")
    p.add_argument("--lane_edge_hard_margin", type=float, default=0.30,
                   help="Hard road-edge correction/braking threshold in meters.")
    p.add_argument("--lane_edge_target_speed", type=float, default=3.0,
                   help="Target speed near road/lane edge while recovering to lane center.")
    p.add_argument("--lane_edge_brake", type=float, default=2.5,
                   help="Braking deceleration cap used near road/lane edge.")
    p.add_argument("--lane_edge_steer_gain", type=float, default=0.75,
                   help="Lateral-error gain for low-level road-edge steering correction.")
    p.add_argument("--lane_edge_heading_gain", type=float, default=0.60,
                   help="Heading-error gain for low-level road-edge steering correction.")
    p.add_argument("--no_traffic_light_route_scan", action="store_true",
                   help="Disable route-sampled traffic-light search. Keep enabled for Town10HD junctions.")
    p.add_argument("--traffic_light_route_scan_step", type=float, default=4.0,
                   help="Meters between route samples used to find upcoming traffic lights.")
    p.add_argument("--no_traffic_light_landmark_fallback", action="store_true",
                   help="Disable OpenDRIVE landmark fallback for traffic-light detection.")
    p.add_argument("--vehicle_route_corridor_factor", type=float, default=0.55,
                   help="Fallback vehicle-lead route corridor factor. Lower rejects adjacent-lane cars.")
    p.add_argument("--vehicle_route_corridor_max", type=float, default=1.0,
                   help="Maximum fallback vehicle-lead corridor width in meters.")
    p.add_argument("--no_yellow_light_stop", action="store_true",
                   help="If set, the ego only stops for red lights, not yellow lights.")
    p.add_argument("--terminate_on_headway_violation", action="store_true",
                   help="If set, terminate on moderate headway violation. Default keeps episode alive so ego can slow/follow.")
    p.add_argument("--headway_hard_fail_gap", type=float, default=0.75,
                   help="Only terminate headway when bumper gap is below this hard-failure threshold.")
    p.add_argument("--route_step_m", type=float, default=2.0,
                   help="Waypoint spacing / GlobalRoutePlanner sampling resolution.")
    p.add_argument("--route_planner_mode", type=str, default="global",
                   choices=["global", "heuristic"],
                   help="Explicit destination-based global routing is the default and publication path.")
    p.add_argument("--allow_heuristic_route_fallback", action="store_true",
                   help="Explicitly permit legacy local Waypoint.next branch routing if global planning fails.")
    p.add_argument("--route_destination_candidates", type=int, default=32,
                   help="Number of deterministic map spawn destinations evaluated by GlobalRoutePlanner.")
    p.add_argument("--failure_persistence_ticks", type=int, default=10,
                   help="Consecutive severe lane/heading ticks before terminating.")
    p.add_argument("--headway_failure_persistence_ticks", type=int, default=8,
                   help="Consecutive dangerous headway ticks before terminating; prevents single-tick false hard_headway_failure.")
    p.add_argument("--compact_obs", action="store_true",
                   help="Use original 8-D state only; default uses 12-D route-aware state.")
    p.add_argument("--train_drift", type=float, default=0.3)
    p.add_argument("--eval_drift", type=float, default=0.5)

    # background traffic / pedestrians
    p.add_argument("--num_traffic_vehicles", type=int, default=0,
                   help="Number of extra Traffic-Manager vehicles to spawn near the ego route.")
    p.add_argument("--num_walkers", type=int, default=0,
                   help="Number of AI-controlled pedestrians/walkers to spawn near the route.")
    p.add_argument("--tm_port", type=int, default=8000,
                   help="Traffic Manager port for background autopilot vehicles.")
    p.add_argument("--traffic_speed_difference", type=float, default=35.0,
                   help="TM percentage speed difference. Positive is slower than speed limit; negative is faster.")
    p.add_argument("--traffic_min_distance", type=float, default=6.0,
                   help="Minimum distance for Traffic Manager vehicles.")
    p.add_argument("--traffic_rear_safety_distance", type=float, default=7.0,
                   help="TM following distance used to reduce NPC rear-end collisions into a slowing/stopped ego.")
    p.add_argument("--traffic_auto_lane_change", action="store_true",
                   help="Allow Traffic Manager vehicles to auto lane-change. Keep off for early training.")
    p.add_argument("--traffic_radius", type=float, default=0.0,
                   help="Optional max radius from ego start for background vehicle spawn. 0 means route-based only.")
    p.add_argument("--walker_speed_min", type=float, default=0.8)
    p.add_argument("--walker_speed_max", type=float, default=1.6)
    p.add_argument("--traffic_warmup_ticks", type=int, default=40,
                   help="Ticks after spawning traffic so Traffic Manager builds paths and NPCs start moving.")
    p.add_argument("--tm_hybrid_physics", action="store_true",
                   help="Enable Traffic Manager hybrid physics. Default is full physics to avoid frozen Town10HD NPCs.")
    p.add_argument("--allow_cross_town_walkers", action="store_true",
                   help="Override the safety guard that suppresses AI walkers after a map switch on CARLA 0.9.15-or-unknown servers. Do not use this override on vulnerable CARLA versions.")
    p.add_argument("--episode_reset_settle_ticks", type=int, default=1,
                   help="Controlled synchronous ticks after single-batch episode teardown. Default 1 avoids partially destroyed-world ticks.")
    p.add_argument("--destroy_all_stale_actors", action="store_true",
                   help="On a dedicated CARLA server only, remove every stale vehicle/walker/sensor/controller at connect time. Default cleanup is ownership-safe.")
    p.add_argument("--eval_cooldown_s", type=float, default=1.0,
                   help="Host-side cooldown after closing one town environment before loading the next town.")

    # training
    p.add_argument("--total_steps", type=int, default=150_000,
                   help="Number of NEW environment steps for this run.")
    p.add_argument("--resume_checkpoint", type=str, default="",
                   help="Continue training from a SAC checkpoint. Network/temperature and optimizer states are restored when available.")
    p.add_argument("--resume_replay_warmup_steps", type=int, default=2_000,
                   help="On resumed training, fill the fresh replay buffer for this many learned-policy steps before updates.")
    p.add_argument("--warmup_steps", type=int, default=2_000)
    p.add_argument("--move_warmup_steps", type=int, default=1_200,
                   help="First training steps use small steering and positive acceleration so replay contains moving transitions.")
    p.add_argument("--warmup_accel_min", type=float, default=0.35,
                   help="Minimum normalized acceleration during move warmup in [-1,1].")
    p.add_argument("--warmup_accel_max", type=float, default=0.95,
                   help="Maximum normalized acceleration during move warmup in [-1,1].")
    p.add_argument("--updates_per_step", type=int, default=1)
    p.add_argument("--state_dim", type=int, default=12,
                   help="SAC observation dimension. Use 12 with route-aware observations, 8 with --compact_obs.")
    p.add_argument("--obs_clip", type=float, default=5.0,
                   help="Clip normalized SAC observations to [-obs_clip, obs_clip]. Use 0 to disable clipping.")
    p.add_argument("--obs_warn_threshold", type=float, default=4.5,
                   help="Warn when any normalized observation dimension exceeds this absolute value.")
    p.add_argument("--obs_warn_every", type=int, default=1000,
                   help="Print at most one large-observation warning every N normalization calls.")
    p.add_argument("--strict_state_dim", action="store_true",
                   help="Fail if env state dimension does not exactly match --state_dim; useful for final debugging.")
    p.add_argument("--rel_vel_obs_scale", type=float, default=15.0,
                   help="Normalization scale for state[4] relative velocity. Dense traffic can exceed 10 m/s, so 15 is safer.")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--replay_size", type=int, default=300_000)
    p.add_argument("--no_spectral_norm", action="store_true",
                   help="Disable spectral norm on the ACTOR (critics never use it).")
    p.add_argument("--init_alpha", type=float, default=0.1,
                   help="Initial SAC temperature (bring-up finding: 1.0 stalls "
                        "learning at this reward scale).")
    p.add_argument("--alpha_min", type=float, default=0.01,
                   help="Minimum SAC entropy temperature after clamping.")
    p.add_argument("--alpha_max", type=float, default=1.0,
                   help="Maximum SAC entropy temperature; prevents alpha explosion in long CARLA runs.")
    p.add_argument("--ckpt_every", type=int, default=25_000)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--device", type=str, default=None)

    # evaluation
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--eval_route_retries", type=int, default=5,
                   help="Maximum resets used to obtain a publication-valid route per episode.")
    p.add_argument("--eval_route_min_fraction", type=float, default=0.98,
                   help="Reject evaluation routes shorter than this fraction of --route_distance.")

    # filter parameters (match ECLCS defaults / paper settings)
    p.add_argument("--epsilon", type=float, default=0.10)
    p.add_argument("--eta", type=float, default=0.03)
    p.add_argument("--q_init", type=float, default=0.10)
    p.add_argument("--q_max", type=float, default=5.0)
    p.add_argument("--zeta_max", type=float, default=0.02)
    p.add_argument("--probe_probability", type=float, default=0.10)
    p.add_argument("--certified_tube_delta", type=float, default=0.10,
                   help="Half-width delta of the theorem certified tube around estimated q_star.")
    p.add_argument("--gain_schedule", action="store_true")
    p.add_argument("--no_feasibility_restoration", action="store_true",
                   help="Disable the always-returning q=0/max-h recovery path (diagnostic ablation only).")
    p.add_argument("--restoration_grid_points", type=int, default=5,
                   help="Per-action-axis samples for bounded max-h feasibility restoration.")
    p.add_argument("--no_guard_history_rebase", action="store_true",
                   help="Disable rate/jerk history rebasing after an external CARLA guard overrides the filter action.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./runs")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.compact_obs and args.state_dim == 12:
        # Compact environment returns only the original barrier state.
        args.state_dim = 8
    if args.no_weather and str(args.weather_mode).lower() not in {"clear", "none", "off"}:
        print("[warn] --no_weather disables the requested weather_mode; remove --no_weather for rain/fog/night/random.")
    if args.alpha_max < args.alpha_min:
        raise SystemExit("--alpha_max must be >= --alpha_min")
    if int(args.eval_route_retries) < 1:
        raise SystemExit("--eval_route_retries must be >= 1")
    if not (0.0 < float(args.eval_route_min_fraction) <= 1.0):
        raise SystemExit("--eval_route_min_fraction must lie in (0, 1]")
    if float(args.map_switch_timeout_s) < 30.0:
        raise SystemExit("--map_switch_timeout_s must be >= 30")
    if float(args.external_blockage_patience_s) < 5.0:
        raise SystemExit("--external_blockage_patience_s must be >= 5")
    if int(args.external_blockage_max_recoveries) < 0:
        raise SystemExit("--external_blockage_max_recoveries must be >= 0")
    if args.probe:
        probe(args)
    elif args.train:
        print(json.dumps(train(args), indent=2))
    elif args.eval:
        if not args.checkpoint:
            raise SystemExit("--eval requires --checkpoint")
        evaluate(args)
    else:
        print("Use --train or --eval. See --help.")


if __name__ == "__main__":
    main()
