#!/usr/bin/env python3
"""Publication-oriented PPO + E-COCSF experiments on SafetyPointGoal2-v0.

This program deliberately separates:

* nominal-policy training, which uses only the standard environment observation;
* safety-filter evaluation, which augments that observation with simulator
  geometry through :mod:`safety_gym_env`;
* E-COCSF, imported from ``ECLCS.py`` so the CARLA and Safety-Gymnasium studies
  execute the same proposed algorithm.

Example
-------
Train an unfiltered nominal PPO policy::

    python -u safety_gym_train_eval.py train --device cuda \
      --total_steps 500000 --out_dir runs/pointgoal2_ppo_seed42

Evaluate all margin methods with paired seeds under action-noise shift::

    python -u safety_gym_train_eval.py eval \
      --checkpoint runs/pointgoal2_ppo_seed42/policy_final.pt \
      --device cuda --episodes 50 --noise_levels 0,0.05,0.10,0.20 \
      --methods unfiltered,fixed,uncertainty,naive_aci,ecocsf \
      --out_dir runs/pointgoal2_benchmark_seed42
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
except ImportError as exc:  # pragma: no cover - depends on user environment
    raise SystemExit(
        "PyTorch is required. Install the CUDA build recommended at "
        "https://pytorch.org/get-started/locally/"
    ) from exc

from safety_gym_env import (
    PointGoalBarrierModel,
    SafetyPointGoalAdapter,
    load_point_dynamics,
)


SCHEMA_VERSION = 1  # PPO checkpoint schema; dynamics artifacts have their own version.
EVALUATION_SCHEMA_VERSION = 2  # Adds explicit valid-soft-loss accounting.


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy and Torch without claiming full simulator determinism."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).strip().lower()
    if requested in {"", "auto", "none"}:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    device = torch.device(requested)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        required_arch = f"sm_{major}{minor}"
        compiled_arches = set(torch.cuda.get_arch_list())
        if compiled_arches and required_arch not in compiled_arches:
            name = torch.cuda.get_device_name(index)
            raise RuntimeError(
                f"Installed PyTorch {torch.__version__} does not contain kernels for "
                f"{name} ({required_arch}); compiled architectures are "
                f"{sorted(compiled_arches)}. Install a CUDA wheel supporting "
                f"{required_arch} before training. RTX 50-series GPUs require a "
                "current CUDA 13.x PyTorch build."
            )
    return device


def finite_vector(value: Any, *, expected: Optional[int] = None, name: str = "value") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if expected is not None and arr.size != int(expected):
        raise ValueError(f"{name} must contain {expected} values, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{name} contains NaN/Inf")
    return arr


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_manifest(args: argparse.Namespace, *, checkpoint: Optional[Path] = None) -> Dict[str, Any]:
    source_hashes: Dict[str, str] = {}
    for filename in ("safety_gym_train_eval.py", "safety_gym_env.py", "ECLCS.py"):
        path = Path(__file__).resolve().parent / filename
        if path.exists():
            source_hashes[filename] = sha256_file(path)
    gpu = None
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            gpu = "available"
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "arguments": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "safety-gymnasium": package_version("safety-gymnasium"),
            "gymnasium": package_version("gymnasium"),
            "mujoco": package_version("mujoco"),
        },
        "gpu": gpu,
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint and checkpoint.exists() else None,
        "source_sha256": source_hashes,
    }


class PPOActorCritic(nn.Module):
    """Tanh-squashed Gaussian PPO policy with a separate value network.

    The policy emits actions directly in ``[-1, 1]^action_dim`` so it matches
    Safety-Gymnasium's continuous Point action space.  PPO ratios are computed
    from the exact tanh-corrected log probability of the stored action.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0

    def __init__(self, observation_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.apply(self._initialize)
        # Small actor output gain is standard for PPO and prevents initially
        # saturated commands that would be difficult for the safety filter.
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        features = self.actor_body(observation)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)

    @staticmethod
    def _squashed_log_prob(
        distribution: torch.distributions.Normal,
        raw_action: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        base = distribution.log_prob(raw_action).sum(dim=-1)
        correction = torch.log(1.0 - action.square() + 1e-6).sum(dim=-1)
        return base - correction

    def sample(
        self, observation: torch.Tensor, *, deterministic: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        raw_action = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        log_probability = self._squashed_log_prob(distribution, raw_action, action)
        value = self.value(observation)
        return action, log_probability, value

    def evaluate_actions(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        clipped_action = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_action = torch.atanh(clipped_action)
        log_probability = self._squashed_log_prob(
            distribution, raw_action, clipped_action
        )
        # The entropy of a tanh-transformed normal has no simple closed form.
        # The base-normal entropy is a stable, standard exploration surrogate.
        entropy = distribution.entropy().sum(dim=-1)
        value = self.value(observation)
        return log_probability, entropy, value


class RolloutBuffer:
    """On-policy rollout storage with correct time-limit bootstrapping for GAE."""

    def __init__(self, capacity: int, observation_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.observations = np.empty(
            (self.capacity, observation_dim), dtype=np.float32
        )
        self.actions = np.empty((self.capacity, action_dim), dtype=np.float32)
        self.log_probabilities = np.empty(self.capacity, dtype=np.float32)
        self.values = np.empty(self.capacity, dtype=np.float32)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.next_values = np.empty(self.capacity, dtype=np.float32)
        self.episode_done = np.empty(self.capacity, dtype=np.float32)
        self.advantages = np.empty(self.capacity, dtype=np.float32)
        self.returns = np.empty(self.capacity, dtype=np.float32)
        self.size = 0

    def reset(self) -> None:
        self.size = 0

    def add(
        self,
        observation: Sequence[float],
        action: Sequence[float],
        log_probability: float,
        value: float,
        reward: float,
        next_value: float,
        episode_done: bool,
    ) -> None:
        if self.size >= self.capacity:
            raise RuntimeError("PPO rollout buffer overflow")
        i = self.size
        self.observations[i] = finite_vector(observation, name="observation")
        self.actions[i] = np.clip(
            finite_vector(action, name="action"), -1.0, 1.0
        )
        self.log_probabilities[i] = float(log_probability)
        self.values[i] = float(value)
        self.rewards[i] = float(np.clip(float(reward), -1e4, 1e4))
        self.next_values[i] = float(next_value)
        self.episode_done[i] = float(bool(episode_done))
        self.size += 1

    def compute_advantages(self, gamma: float, gae_lambda: float) -> None:
        if self.size < 1:
            raise RuntimeError("Cannot compute PPO advantages from an empty rollout")
        gae = 0.0
        for i in range(self.size - 1, -1, -1):
            delta = (
                float(self.rewards[i])
                + float(gamma) * float(self.next_values[i])
                - float(self.values[i])
            )
            # Stop GAE recursion at any episode boundary to avoid leaking
            # advantages across independently reset episodes. For a Gymnasium
            # time-limit truncation, next_value was already bootstrapped from
            # the final observation; for a true terminal state it is zero.
            continuation = 1.0 - float(self.episode_done[i])
            gae = delta + float(gamma) * float(gae_lambda) * continuation * gae
            self.advantages[i] = gae
        self.returns[: self.size] = (
            self.advantages[: self.size] + self.values[: self.size]
        )


@dataclass
class PPOConfig:
    observation_dim: int
    action_dim: int = 2
    hidden: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    rollout_steps: int = 2048
    update_epochs: int = 10
    batch_size: int = 64
    clip_range: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.50
    target_kl: float = 0.03
    observation_clip: float = 10.0
    seed: int = 42


class PPOAgent:
    def __init__(self, config: PPOConfig, device: str = "auto"):
        self.config = config
        self.device = resolve_device(device)
        set_seed(config.seed)
        self.model = PPOActorCritic(
            config.observation_dim, config.action_dim, config.hidden
        ).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        self.rollout = RolloutBuffer(
            config.rollout_steps, config.observation_dim, config.action_dim
        )
        self.updates = 0

    def preprocess(self, observation: Sequence[float]) -> np.ndarray:
        obs = finite_vector(
            observation,
            expected=self.config.observation_dim,
            name="policy observation",
        )
        clip = float(self.config.observation_clip)
        if clip > 0.0:
            obs = np.clip(obs, -clip, clip)
        return obs.astype(np.float32, copy=False)

    def _observation_tensor(self, observation: Sequence[float]) -> torch.Tensor:
        return torch.as_tensor(
            self.preprocess(observation)[None, :],
            dtype=torch.float32,
            device=self.device,
        )

    def act(self, observation: Sequence[float], deterministic: bool) -> np.ndarray:
        obs = self._observation_tensor(observation)
        with torch.no_grad():
            action, _, _ = self.model.sample(obs, deterministic=deterministic)
        return np.clip(action[0].cpu().numpy(), -1.0, 1.0).astype(np.float64)

    def act_with_info(
        self, observation: Sequence[float]
    ) -> Tuple[np.ndarray, float, float]:
        obs = self._observation_tensor(observation)
        with torch.no_grad():
            action, log_probability, value = self.model.sample(
                obs, deterministic=False
            )
        return (
            np.clip(action[0].cpu().numpy(), -1.0, 1.0).astype(np.float64),
            float(log_probability.item()),
            float(value.item()),
        )

    def value(self, observation: Sequence[float]) -> float:
        obs = self._observation_tensor(observation)
        with torch.no_grad():
            return float(self.model.value(obs).item())

    def store(
        self,
        observation: Sequence[float],
        action: Sequence[float],
        log_probability: float,
        value: float,
        reward: float,
        next_value: float,
        episode_done: bool,
    ) -> None:
        self.rollout.add(
            self.preprocess(observation),
            action,
            log_probability,
            value,
            reward,
            next_value,
            episode_done,
        )

    def update(self) -> Dict[str, float]:
        cfg = self.config
        buffer = self.rollout
        if buffer.size < 1:
            return {}
        buffer.compute_advantages(cfg.gamma, cfg.gae_lambda)

        n = buffer.size
        advantages_np = buffer.advantages[:n].astype(np.float32, copy=True)
        advantages_np = (
            advantages_np - float(advantages_np.mean())
        ) / (float(advantages_np.std()) + 1e-8)

        observations = torch.as_tensor(
            buffer.observations[:n], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            buffer.actions[:n], dtype=torch.float32, device=self.device
        )
        old_log_probabilities = torch.as_tensor(
            buffer.log_probabilities[:n], dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(
            buffer.returns[:n], dtype=torch.float32, device=self.device
        )
        advantages = torch.as_tensor(
            advantages_np, dtype=torch.float32, device=self.device
        )

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        approx_kls: List[float] = []
        clip_fractions: List[float] = []
        stop_early = False

        for _epoch in range(int(cfg.update_epochs)):
            permutation = torch.randperm(n, device=self.device)
            for start in range(0, n, int(cfg.batch_size)):
                indices = permutation[start : start + int(cfg.batch_size)]
                new_log_probability, entropy, value = self.model.evaluate_actions(
                    observations[indices], actions[indices]
                )
                log_ratio = new_log_probability - old_log_probabilities[indices]
                ratio = log_ratio.exp()
                batch_advantage = advantages[indices]

                unclipped = ratio * batch_advantage
                clipped = torch.clamp(
                    ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range
                ) * batch_advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(value, returns[indices])
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_mean
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float((old_log_probabilities[indices] - new_log_probability).mean().item())
                    clip_fraction = float(
                        ((ratio - 1.0).abs() > cfg.clip_range).float().mean().item()
                    )
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_mean.item()))
                approx_kls.append(approx_kl)
                clip_fractions.append(clip_fraction)

            if cfg.target_kl > 0.0 and approx_kls:
                epoch_kl = float(np.mean(approx_kls[-max(1, math.ceil(n / cfg.batch_size)):]))
                if epoch_kl > 1.5 * cfg.target_kl:
                    stop_early = True
                    break

        self.updates += 1
        explained_variance = float("nan")
        returns_np = buffer.returns[:n]
        variance = float(np.var(returns_np))
        if variance > 1e-12:
            explained_variance = 1.0 - float(
                np.var(returns_np - buffer.values[:n]) / variance
            )

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else float("nan"),
            "value_loss": float(np.mean(value_losses)) if value_losses else float("nan"),
            "entropy": float(np.mean(entropies)) if entropies else float("nan"),
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else float("nan"),
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else float("nan"),
            "explained_variance": explained_variance,
            "early_stop_kl": float(stop_early),
            "rollout_size": float(n),
        }

    def save(self, path: Path, *, env_id: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": "PPO",
            "env_id": str(env_id),
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "updates": int(self.updates),
            "optimizer": self.optimizer.state_dict(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        device: str = "auto",
        load_optimizers: bool = False,
    ) -> "PPOAgent":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("schema_version", -1)) != SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported checkpoint schema {checkpoint.get('schema_version')}"
            )
        if checkpoint.get("algorithm") != "PPO":
            raise RuntimeError(
                "Checkpoint is not a Safety-Gymnasium PPO checkpoint. "
                "Train a new PPO checkpoint with this script; SAC checkpoints "
                "cannot be converted into PPO checkpoints."
            )
        agent = cls(PPOConfig(**checkpoint["config"]), device=device)
        agent.model.load_state_dict(checkpoint["model"])
        agent.updates = int(checkpoint.get("updates", 0))
        if load_optimizers and "optimizer" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
        return agent


def load_policy(checkpoint: Path, *, device: str = "auto"):
    """Load a nominal-policy checkpoint, dispatching on its algorithm tag.

    PPO checkpoints (written by this script's ``train`` command) and SAC
    checkpoints (written by ``safety_gym_sac_train.py``) are both supported.
    Both agents expose the same evaluation interface:
    ``act(observation, deterministic)`` plus
    ``config.observation_dim``/``config.action_dim``.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    algorithm = str(payload.get("algorithm", "")).upper()
    if algorithm == "SAC":
        from safety_gym_sac_train import SACAgent

        return SACAgent.load(checkpoint, device=device)
    if algorithm == "PPO":
        return PPOAgent.load(checkpoint, device=device)
    raise RuntimeError(
        f"Unsupported checkpoint algorithm {algorithm!r}: expected 'PPO' "
        "(from this script's train command) or 'SAC' (from "
        "safety_gym_sac_train.py)."
    )


def make_adapter(args: argparse.Namespace, *, action_noise: float, render: bool = False):
    return SafetyPointGoalAdapter(
        env_id=args.env_id,
        render_mode="human" if render else None,
        action_noise=float(action_noise),
        frameskip_probability=float(args.frameskip_probability),
        max_objects=int(args.max_objects),
        hazard_buffer=float(args.hazard_buffer),
        vase_buffer=float(args.vase_buffer),
        agent_radius=float(args.agent_radius),
    )


def train(args: argparse.Namespace) -> Dict[str, Any]:
    """Train an unfiltered nominal PPO policy on the standard observation only."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, args.deterministic_torch)

    with make_adapter(args, action_noise=args.train_action_noise, render=args.render) as env:
        config = PPOConfig(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden=args.hidden,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            learning_rate=args.learning_rate,
            rollout_steps=args.rollout_steps,
            update_epochs=args.update_epochs,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            observation_clip=args.observation_clip,
            seed=args.seed,
        )
        agent = PPOAgent(config, device=args.device)
        observation, _ = env.reset(seed=args.seed)
        episode_return = 0.0
        episode_cost = 0.0
        episode_goals = 0
        episode_length = 0
        episode_index = 0
        episode_rows: List[Dict[str, Any]] = []
        last_losses: Dict[str, float] = {}
        start_time = time.time()
        global_step = 0
        next_log_step = int(args.log_every)
        next_checkpoint_step = int(args.checkpoint_every)

        while global_step < int(args.total_steps):
            agent.rollout.reset()
            rollout_target = min(
                int(args.rollout_steps), int(args.total_steps) - global_step
            )

            for _ in range(rollout_target):
                action, log_probability, value = agent.act_with_info(observation)
                next_observation, reward, cost, terminated, truncated, info = env.step(action)
                learning_reward = float(reward) - float(args.train_cost_penalty) * float(cost)

                # A true terminal state has zero bootstrap value. A time-limit
                # truncation is not terminal in the underlying MDP, so bootstrap
                # from its final observation but stop GAE recursion across reset.
                next_value = 0.0 if terminated else agent.value(next_observation)
                episode_done = bool(terminated or truncated)
                agent.store(
                    observation,
                    action,
                    log_probability,
                    value,
                    learning_reward,
                    next_value,
                    episode_done,
                )

                observation = next_observation
                global_step += 1
                episode_return += float(reward)
                episode_cost += float(cost)
                episode_goals += int(bool(info.get("goal_met", False)))
                episode_length += 1

                if episode_done:
                    episode_rows.append(
                        {
                            "episode": episode_index,
                            "global_step": global_step,
                            "return": episode_return,
                            "cost": episode_cost,
                            "cost_rate": episode_cost / max(episode_length, 1),
                            "goals": episode_goals,
                            "length": episode_length,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                    episode_index += 1
                    observation, _ = env.reset(seed=args.seed + episode_index)
                    episode_return = 0.0
                    episode_cost = 0.0
                    episode_goals = 0
                    episode_length = 0

                if global_step >= next_log_step:
                    elapsed = max(time.time() - start_time, 1e-9)
                    recent = episode_rows[-10:]
                    mean_return = (
                        float(np.mean([row["return"] for row in recent]))
                        if recent else float("nan")
                    )
                    mean_cost = (
                        float(np.mean([row["cost"] for row in recent]))
                        if recent else float("nan")
                    )
                    print(
                        f"[train:ppo] step={global_step}/{args.total_steps} "
                        f"episodes={episode_index} return10={mean_return:.3f} "
                        f"cost10={mean_cost:.3f} "
                        f"policy_loss={last_losses.get('policy_loss', float('nan')):.4f} "
                        f"value_loss={last_losses.get('value_loss', float('nan')):.4f} "
                        f"kl={last_losses.get('approx_kl', float('nan')):.5f} "
                        f"steps_per_s={global_step / elapsed:.1f}",
                        flush=True,
                    )
                    while next_log_step <= global_step:
                        next_log_step += int(args.log_every)

            # PPO must update only after collecting the on-policy rollout.
            last_losses = agent.update()

            if global_step >= next_checkpoint_step:
                agent.save(
                    out_dir / f"policy_step_{global_step}.pt",
                    env_id=args.env_id,
                )
                while next_checkpoint_step <= global_step:
                    next_checkpoint_step += int(args.checkpoint_every)

        final_checkpoint = out_dir / "policy_final.pt"
        agent.save(final_checkpoint, env_id=args.env_id)

    write_csv(out_dir / "train_episodes.csv", episode_rows)
    summary = {
        "algorithm": "PPO",
        "checkpoint": str(final_checkpoint),
        "total_steps": int(args.total_steps),
        "episodes": len(episode_rows),
        "updates": int(agent.updates),
        "mean_last_20_return": float(
            np.mean([row["return"] for row in episode_rows[-20:]])
        ) if episode_rows else None,
        "mean_last_20_cost": float(
            np.mean([row["cost"] for row in episode_rows[-20:]])
        ) if episode_rows else None,
        "last_update": last_losses,
    }
    atomic_json(out_dir / "train_summary.json", summary)
    atomic_json(out_dir / "manifest.json", build_manifest(args, checkpoint=final_checkpoint))
    print(json.dumps(json_safe(summary), indent=2), flush=True)
    return summary


def import_ecocsf():
    try:
        import ECLCS as ec
    except ImportError as exc:
        raise RuntimeError(
            "ECLCS.py must be in the same directory as safety_gym_train_eval.py"
        ) from exc
    required = (
        "ECLCSConfig",
        "MachineCard",
        "EndogenousClosedLoopConformalSafetyFilter",
        "ECLCSAgent",
    )
    missing = [name for name in required if not hasattr(ec, name)]
    if missing:
        raise RuntimeError(f"ECLCS.py is missing required symbols: {missing}")
    return ec


def build_ecocsf(
    args: argparse.Namespace,
    env: SafetyPointGoalAdapter,
    *,
    seed: int,
    method: str = "ecocsf",
):
    ec = import_ecocsf()
    dynamics = load_point_dynamics(args.dynamics_model)
    model_dt = float(dynamics.get("dt", args.model_dt)) if dynamics else float(args.model_dt)
    barrier = PointGoalBarrierModel(
        codec=env.codec,
        dt=model_dt,
        time_headway=float(args.barrier_time_headway),
        lookahead_steps=int(args.barrier_lookahead_steps),
        braking_deceleration=float(args.barrier_braking_deceleration),
        braking_distance_scale=float(args.barrier_braking_distance_scale),
        acceleration_gain=float(args.model_acceleration_gain),
        linear_drag=float(args.model_linear_drag),
        yaw_tracking=float(args.model_yaw_tracking),
        max_speed=float(args.model_max_speed),
        workspace_half_extent=float(args.workspace_half_extent),
        workspace_buffer=float(args.workspace_buffer),
        dynamics=dynamics,
    )
    machine = ec.MachineCard(
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
        rate_limit=(float(args.action_rate_limit),) * 2,
        jerk_limit=(float(args.action_jerk_limit),) * 2,
        neutral_action=(0.0, 0.0),
        action_names=("forward_force", "yaw_velocity"),
    )
    config = ec.ECLCSConfig(
        action_dim=2,
        dt=model_dt,
        epsilon=float(args.epsilon),
        eta=float(args.eta),
        q_init=float(args.q_init),
        q_max=float(args.q_max),
        ramp_tau=float(args.ramp_tau),
        method=str(method),
        fixed_margin=float(args.fixed_margin),
        uncertainty_quantile=float(args.uncertainty_quantile),
        uncertainty_scale=float(args.uncertainty_scale),
        uncertainty_min_samples=int(args.uncertainty_min_samples),
        zeta_max=float(args.zeta_max),
        probe_probability=float(args.probe_probability),
        alpha=float(args.barrier_alpha),
        projection_scp_iterations=int(args.projection_scp_iterations),
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
        rate_limit=(float(args.action_rate_limit),) * 2,
        jerk_limit=(float(args.action_jerk_limit),) * 2,
        neutral_action=(0.0, 0.0),
        action_weight=(1.0, float(args.yaw_action_weight)),
        audit_window=int(args.audit_window),
        audit_min_samples=int(args.audit_min_samples),
        audit_min_q_range=float(args.audit_min_q_range),
        audit_conf_z=float(args.audit_conf_z),
        audit_min_mu=float(args.audit_min_mu),
        audit_crossing_slack=float(args.audit_crossing_slack),
        certified_tube_delta=float(args.certified_tube_delta),
        use_gain_schedule=bool(args.gain_schedule),
        feasibility_restoration=not bool(args.no_feasibility_restoration),
        restoration_grid_points=int(args.restoration_grid_points),
        # The headroom-capped executed margin is part of the E-COCSF method
        # (endogenous projection of the executed margin onto the enforceable
        # set); the fixed/uncertainty/naive_aci baselines request their
        # margins without headroom awareness, as defined.
        headroom_margin_cap=(str(method) == "ecocsf"
                             and not bool(args.no_headroom_cap)),
        headroom_cap_delta=float(args.headroom_cap_delta),
        anti_windup_gamma=float(args.anti_windup_gamma),
        capped_positive_integration=float(
            args.capped_positive_integration
        ),
        freeze_backcalc_on_positive_cap=not bool(
            args.allow_positive_cap_backcalc
        ),
        seed=int(seed),
        save_jsonl=False,
        run_audit=(str(method) == "ecocsf"),
    )
    safety_filter = ec.EndogenousClosedLoopConformalSafetyFilter(
        config, machine=machine, barrier=barrier
    )
    return ec, barrier, safety_filter


def bootstrap_mean_ci(values: Sequence[float], *, seed: int, samples: int = 10_000):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1)) if array.size > 1 else 0.0
    if array.size == 1:
        return mean, standard_deviation, mean, mean
    rng = np.random.default_rng(int(seed))
    chunk = 1000
    boot_means: List[np.ndarray] = []
    remaining = int(samples)
    while remaining > 0:
        count = min(chunk, remaining)
        indices = rng.integers(0, array.size, size=(count, array.size))
        boot_means.append(array[indices].mean(axis=1))
        remaining -= count
    draws = np.concatenate(boot_means)
    low, high = np.quantile(draws, (0.025, 0.975))
    return mean, standard_deviation, float(low), float(high)


def parse_noise_levels(text: str) -> List[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values or any((not np.isfinite(value) or value < 0.0) for value in values):
        raise ValueError("noise_levels must be a comma-separated list of non-negative values")
    if len(set(values)) != len(values):
        raise ValueError("noise_levels contains duplicate values")
    return values


def parse_methods(text: str) -> List[str]:
    allowed = ("unfiltered", "fixed", "uncertainty", "naive_aci", "ecocsf")
    methods = [item.strip().lower() for item in str(text).split(",") if item.strip()]
    unknown = sorted(set(methods) - set(allowed))
    if not methods or unknown:
        raise ValueError(
            f"methods must be a comma-separated subset of {allowed}; unknown={unknown}"
        )
    if len(set(methods)) != len(methods):
        raise ValueError("methods contains duplicate values")
    return methods


def _wrapped_angle_difference(after: float, before: float) -> float:
    return float(np.arctan2(np.sin(after - before), np.cos(after - before)))


def _ridge_fit(features: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("Invalid identification design/target matrices")
    regularizer = np.eye(x.shape[1], dtype=np.float64) * float(ridge)
    regularizer[-1, -1] = 0.0  # do not shrink the intercept
    coefficients = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
    return coefficients.T


def _fit_point_dynamics(
    samples: Sequence[Mapping[str, Any]], *, seed: int, ridge: float,
    validation_fraction: float, trim_quantile: float, dt: float, env_id: str,
) -> Dict[str, Any]:
    if len(samples) < 100:
        raise RuntimeError("At least 100 collision-free transitions are required")
    planar = np.asarray([row["planar"] for row in samples], dtype=np.float64)
    angular = np.asarray([row["angular"] for row in samples], dtype=np.float64)
    position_target = np.asarray(
        [row["position_delta"] for row in samples], dtype=np.float64
    )
    velocity_target = np.asarray(
        [row["velocity_next"] for row in samples], dtype=np.float64
    )
    yaw_delta_target = np.asarray(
        [[row["yaw_delta"]] for row in samples], dtype=np.float64
    )
    yaw_rate_target = np.asarray(
        [[row["yaw_rate_next"]] for row in samples], dtype=np.float64
    )

    # Split by episode, not transition.  A random transition split leaks nearly
    # identical adjacent simulator states into validation and overstates model
    # quality precisely where the barrier relies on it.
    episode_ids = np.asarray(
        [int(row.get("episode", -1)) for row in samples], dtype=np.int64
    )
    unique_episodes = np.unique(episode_ids)
    if unique_episodes.size < 2:
        raise RuntimeError(
            "Identification requires transitions from at least two episodes"
        )
    rng = np.random.default_rng(int(seed))
    shuffled_episodes = rng.permutation(unique_episodes)
    validation_episode_count = int(np.clip(
        round(unique_episodes.size * validation_fraction),
        1,
        unique_episodes.size - 1,
    ))
    validation_episodes = shuffled_episodes[:validation_episode_count]
    validation_mask = np.isin(episode_ids, validation_episodes)
    validation_indices = np.flatnonzero(validation_mask)
    training_indices = np.flatnonzero(~validation_mask)
    if training_indices.size < 50:
        raise RuntimeError("Identification training split is too small")

    def fit(indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            _ridge_fit(planar[indices], position_target[indices], ridge),
            _ridge_fit(planar[indices], velocity_target[indices], ridge),
            _ridge_fit(angular[indices], yaw_delta_target[indices], ridge).reshape(-1),
            _ridge_fit(angular[indices], yaw_rate_target[indices], ridge).reshape(-1),
        )

    position_coef, velocity_coef, yaw_delta_coef, yaw_rate_coef = fit(training_indices)

    # One robust refit removes only extreme contact/numerical outliers.  The
    # threshold and retained count are saved in the artifact for reproducibility.
    train_planar_error = np.linalg.norm(
        planar[training_indices] @ velocity_coef.T
        - velocity_target[training_indices],
        axis=1,
    )
    threshold = float(np.quantile(train_planar_error, trim_quantile))
    retained = training_indices[train_planar_error <= threshold]
    if retained.size >= 50:
        position_coef, velocity_coef, yaw_delta_coef, yaw_rate_coef = fit(retained)

    def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(prediction - target))))

    validation = {
        "position_rmse_m": rmse(
            planar[validation_indices] @ position_coef.T,
            position_target[validation_indices],
        ),
        "velocity_rmse_mps": rmse(
            planar[validation_indices] @ velocity_coef.T,
            velocity_target[validation_indices],
        ),
        "yaw_delta_rmse_rad": rmse(
            angular[validation_indices] @ yaw_delta_coef[:, None],
            yaw_delta_target[validation_indices],
        ),
        "yaw_rate_rmse_radps": rmse(
            angular[validation_indices] @ yaw_rate_coef[:, None],
            yaw_rate_target[validation_indices],
        ),
    }
    return {
        "schema_version": 2,
        "model_type": "frozen_affine_point_dynamics",
        "env_id": str(env_id),
        "dt": float(dt),
        "feature_names": [
            "velocity_x", "velocity_y", "yaw_rate", "forward_x",
            "forward_y", "yaw_lateral_x", "yaw_lateral_y",
            "speed_yaw_lateral_x", "speed_yaw_lateral_y", "intercept",
        ],
        "angular_feature_names": [
            "yaw_rate", "yaw_action", "forward_action", "intercept",
        ],
        "position_coef": position_coef,
        "velocity_coef": velocity_coef,
        "yaw_delta_coef": yaw_delta_coef,
        "yaw_rate_coef": yaw_rate_coef,
        "samples_total": len(samples),
        "samples_train": int(training_indices.size),
        "samples_retained_after_trim": int(retained.size),
        "samples_validation": int(validation_indices.size),
        "episodes_total": int(unique_episodes.size),
        "episodes_validation": int(validation_episode_count),
        "ridge": float(ridge),
        "trim_quantile": float(trim_quantile),
        "validation": validation,
    }


def identify(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect collision-free transitions and fit a frozen local Point model."""
    set_seed(args.seed)
    output_path = Path(args.out_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples: List[Dict[str, Any]] = []
    attempts = 0
    episode = 0
    last_reported = 0
    rng = np.random.default_rng(int(args.seed))

    with make_adapter(args, action_noise=0.0, render=args.render) as env:
        observation, _ = env.reset(seed=int(args.seed))
        actual_dt = env.control_dt
        clearance_model = PointGoalBarrierModel(
            env.codec, dt=actual_dt, time_headway=0.0,
            lookahead_steps=0, braking_distance_scale=0.0,
        )
        action = np.zeros(env.action_dim, dtype=np.float64)
        while len(samples) < int(args.transitions):
            if attempts >= int(args.max_attempt_factor * args.transitions):
                raise RuntimeError(
                    f"Collected only {len(samples)} valid transitions after "
                    f"{attempts} attempts; lower --min_clearance or increase "
                    "--max_attempt_factor"
                )
            if attempts % int(args.action_hold_steps) == 0:
                action = rng.uniform(-1.0, 1.0, size=env.action_dim)

            state = env.filter_state(observation)
            _, yaw, position, yaw_rate, velocity, _ = env.codec.decode(state)
            planar, angular = PointGoalBarrierModel.dynamics_features(
                yaw, velocity, yaw_rate, action, schema_version=2
            )
            next_observation, _, cost, terminated, truncated, _ = env.step(action)
            next_state = env.filter_state(next_observation)
            _, next_yaw, next_position, next_yaw_rate, next_velocity, _ = (
                env.codec.decode(next_state)
            )
            attempts += 1

            clearance = min(
                clearance_model.geometric_h(state),
                clearance_model.geometric_h(next_state),
            )
            if float(cost) == 0.0 and clearance >= float(args.min_clearance):
                samples.append(
                    {
                        "planar": planar,
                        "angular": angular,
                        "position_delta": next_position - position,
                        "velocity_next": next_velocity,
                        "yaw_delta": _wrapped_angle_difference(next_yaw, yaw),
                        "yaw_rate_next": next_yaw_rate,
                        "episode": int(episode),
                        "clearance": float(clearance),
                    }
                )

            observation = next_observation
            if terminated or truncated:
                episode += 1
                observation, _ = env.reset(seed=int(args.seed + episode))

            if (
                len(samples) >= last_reported + int(args.log_every)
                or len(samples) == int(args.transitions)
            ):
                last_reported = len(samples)
                print(
                    f"[identify] accepted={len(samples)}/{args.transitions} "
                    f"attempts={attempts} episodes={episode + 1}",
                    flush=True,
                )

        model = _fit_point_dynamics(
            samples,
            seed=int(args.seed),
            ridge=float(args.ridge),
            validation_fraction=float(args.validation_fraction),
            trim_quantile=float(args.trim_quantile),
            dt=actual_dt,
            env_id=args.env_id,
        )

    model["collection_seed"] = int(args.seed)
    model["attempted_transitions"] = int(attempts)
    model["min_clearance"] = float(args.min_clearance)
    atomic_json(output_path, model)
    print(json.dumps(json_safe(model), indent=2), flush=True)
    return model


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """Run paired, same-policy comparisons of all safety-margin methods."""
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, args.deterministic_torch)
    policy = load_policy(checkpoint, device=args.device)
    noise_levels = parse_noise_levels(args.noise_levels)
    methods = parse_methods(args.methods)
    episode_rows: List[Dict[str, Any]] = []
    step_log_path = out_dir / "benchmark_steps.jsonl.gz"
    step_handle = None if args.no_step_log else gzip.open(
        step_log_path, "wt", encoding="utf-8"
    )

    try:
        for condition_index, noise in enumerate(noise_levels):
            render_condition = bool(args.render and condition_index == 0)
            with make_adapter(args, action_noise=noise, render=render_condition) as env:
                frozen_dynamics = load_point_dynamics(args.dynamics_model)
                effective_model_dt = float(
                    frozen_dynamics.get("dt", args.model_dt)
                ) if frozen_dynamics else float(args.model_dt)
                if not np.isclose(
                    effective_model_dt, env.control_dt, rtol=0.0, atol=1e-6
                ):
                    raise RuntimeError(
                        f"Dynamics dt={effective_model_dt} does not match environment "
                        f"control interval {env.control_dt:.9g}."
                    )
                if policy.config.observation_dim != env.observation_dim:
                    raise RuntimeError(
                        "Checkpoint/environment observation mismatch: "
                        f"{policy.config.observation_dim} != {env.observation_dim}"
                    )
                if policy.config.action_dim != env.action_dim:
                    raise RuntimeError("Checkpoint/environment action mismatch")

                for method_index, method in enumerate(methods):
                    for episode in range(int(args.episodes)):
                        episode_seed = int(args.seed + episode)
                        filter_seed = int(args.seed + 1_000_000 + episode)
                        config_method = method if method != "unfiltered" else "fixed"
                        ec, barrier, safety_filter = build_ecocsf(
                            args,
                            env,
                            seed=filter_seed,
                            method=config_method,
                        )
                        ec.set_seed(filter_seed)
                        observation, _ = env.reset(seed=episode_seed)
                        state = env.filter_state(observation)
                        safety_filter.reset(clear_log=True)

                        def nominal_policy(filter_state: Sequence[float]) -> np.ndarray:
                            policy_observation = env.codec.observation(filter_state)
                            return policy.act(policy_observation, deterministic=True)

                        agent = ec.ECLCSAgent(nominal_policy, safety_filter)
                        totals = {
                            "reward": 0.0, "cost": 0.0, "goals": 0,
                            "interventions": 0, "intervention": 0.0,
                            "barrier_violations": 0, "geometric_violations": 0,
                            "requested_exceedances": 0,
                            # Enforcement-valid calibration accounting.
                            "valid_steps": 0,
                            "valid_soft_loss_sum": 0.0,
                            "hard_exceedances": 0,
                            "strict_valid_steps": 0,
                            "strict_valid_soft_loss_sum": 0.0,
                            "strict_valid_hard_exceedances": 0,
                            "capped_valid_steps": 0,
                            "capped_valid_soft_loss_sum": 0.0,
                            "capped_valid_hard_exceedances": 0,
                            "operational_support_steps": 0,
                            "certified_steps": 0,  # backward-compatible alias
                            "cap_attempted_steps": 0,
                            "margin_capped_steps": 0,
                            "infeasible_steps": 0,
                            "cost_events": 0,
                            "contact_events": 0, "disagreements": 0,
                            "false_positives": 0, "false_negatives": 0,
                            "actuator_error": 0.0,
                        }
                        component_cost_totals: Dict[str, float] = {}
                        length = 0
                        terminated = truncated = False

                        for step in range(int(args.max_steps)):
                            if method == "unfiltered":
                                nominal = nominal_policy(state)
                                requested_action = np.asarray(nominal, dtype=np.float64)
                                decision = update = None
                            else:
                                decision = agent.act(state, deterministic_probe=False)
                                nominal = decision.nominal_action
                                requested_action = decision.action

                            next_observation, reward, cost, terminated, truncated, info = (
                                env.step(requested_action)
                            )
                            next_state = env.filter_state(next_observation)
                            residual_action = (
                                requested_action if args.residual_action == "commanded"
                                else env.last_executed_action
                            )
                            if decision is not None:
                                update = agent.observe(
                                    state,
                                    decision,
                                    next_state,
                                    executed_action=residual_action,
                                )

                            predictive_unsafe = bool(
                                update.violation if update is not None
                                else barrier.h(next_state) < 0.0
                            )
                            geometric_unsafe = bool(barrier.geometric_h(next_state) < 0.0)
                            intervention = float(
                                decision.intervention_norm if decision is not None else 0.0
                            )
                            components = {
                                str(key): float(value)
                                for key, value in dict(
                                    info.get("cost_components", {})
                                ).items()
                            }
                            for key, value in components.items():
                                component_cost_totals[key] = (
                                    component_cost_totals.get(key, 0.0) + value
                                )
                            cost_event = bool(float(cost) > 0.0)
                            direct_contact = bool(
                                components.get("cost_hazards", 0.0) > 0.0
                                or components.get("cost_vases_contact", 0.0) > 0.0
                            )
                            contact_event = direct_contact if components else cost_event

                            totals["reward"] += float(reward)
                            totals["cost"] += float(cost)
                            totals["goals"] += int(bool(info.get("goal_met", False)))
                            totals["intervention"] += intervention
                            totals["interventions"] += int(
                                intervention > args.intervention_threshold
                            )
                            totals["barrier_violations"] += int(predictive_unsafe)
                            totals["geometric_violations"] += int(geometric_unsafe)
                            totals["cost_events"] += int(cost_event)
                            totals["contact_events"] += int(contact_event)
                            totals["disagreements"] += int(
                                geometric_unsafe != contact_event
                            )
                            totals["false_positives"] += int(
                                geometric_unsafe and not contact_event
                            )
                            totals["false_negatives"] += int(
                                contact_event and not geometric_unsafe
                            )
                            totals["actuator_error"] += float(np.linalg.norm(
                                np.asarray(env.last_executed_action, dtype=np.float64)
                                - np.asarray(requested_action, dtype=np.float64)
                            ))
                            if update is not None:
                                calibration_valid = bool(update.calibration_valid)
                                verified_capped = bool(
                                    calibration_valid
                                    and getattr(update, "margin_capped", False)
                                )
                                strict_valid = bool(
                                    calibration_valid
                                    and getattr(
                                        update,
                                        "direct_enforcement_valid",
                                        not verified_capped,
                                    )
                                )
                                audit_supported = bool(
                                    getattr(
                                        update,
                                        "audit_supported",
                                        getattr(update, "certified", False),
                                    )
                                )
                                cap_attempted = bool(
                                    getattr(
                                        update,
                                        "cap_attempted",
                                        getattr(decision, "margin_capped", False),
                                    )
                                )

                                totals["requested_exceedances"] += int(
                                    update.hard_exceedance_requested
                                )
                                totals["operational_support_steps"] += int(
                                    audit_supported
                                )
                                totals["certified_steps"] += int(audit_supported)
                                totals["cap_attempted_steps"] += int(cap_attempted)
                                totals["margin_capped_steps"] += int(
                                    verified_capped
                                )
                                totals["infeasible_steps"] += int(update.infeasible)

                                if calibration_valid:
                                    soft_loss = float(update.soft_exceedance)
                                    if not np.isfinite(soft_loss):
                                        raise RuntimeError(
                                            "A calibration-valid transition returned "
                                            "a non-finite soft_exceedance"
                                        )
                                    hard_exceedance = int(update.hard_exceedance)
                                    totals["valid_steps"] += 1
                                    totals["valid_soft_loss_sum"] += soft_loss
                                    totals["hard_exceedances"] += hard_exceedance

                                    if verified_capped:
                                        totals["capped_valid_steps"] += 1
                                        totals[
                                            "capped_valid_soft_loss_sum"
                                        ] += soft_loss
                                        totals[
                                            "capped_valid_hard_exceedances"
                                        ] += hard_exceedance
                                    elif strict_valid:
                                        totals["strict_valid_steps"] += 1
                                        totals[
                                            "strict_valid_soft_loss_sum"
                                        ] += soft_loss
                                        totals[
                                            "strict_valid_hard_exceedances"
                                        ] += hard_exceedance
                                    else:
                                        raise RuntimeError(
                                            "A calibration-valid transition is neither "
                                            "verified strict nor verified capped"
                                        )

                            if step_handle is not None:
                                record = {
                                    "schema_version": EVALUATION_SCHEMA_VERSION,
                                    "method": method,
                                    "action_noise": noise,
                                    "episode": episode,
                                    "seed": episode_seed,
                                    "step": step,
                                    "reward": reward,
                                    "cost": cost,
                                    "cost_components": components,
                                    "goal_met": bool(info.get("goal_met", False)),
                                    "q": getattr(decision, "q", None),
                                    "q_tilde": getattr(decision, "q_tilde", None),
                                    "q_after": getattr(update, "q_after", None),
                                    # Final issued-margin semantics.  q_enforced is
                                    # undefined (serialized as null) on invalid steps.
                                    "q_enforced": getattr(
                                        update, "q_enforced", None
                                    ),
                                    "calibration_margin": getattr(
                                        update, "calibration_margin", None
                                    ),
                                    "residual": getattr(update, "residual", None),
                                    "soft_exceedance": getattr(
                                        update, "soft_exceedance", None
                                    ),
                                    "hard_exceedance": getattr(
                                        update, "hard_exceedance", None
                                    ),
                                    "hard_exceedance_requested": getattr(
                                        update, "hard_exceedance_requested", None
                                    ),
                                    "V_t": (
                                        int(bool(update.calibration_valid))
                                        if update is not None else None
                                    ),
                                    "B_t": (
                                        int(bool(
                                            update.calibration_valid
                                            and getattr(
                                                update, "margin_capped", False
                                            )
                                        ))
                                        if update is not None else None
                                    ),
                                    "calibration_valid": getattr(
                                        update, "calibration_valid", None
                                    ),
                                    "predictive_barrier_violation": predictive_unsafe,
                                    "geometric_barrier_violation": geometric_unsafe,
                                    "direct_enforcement_valid": getattr(
                                        update, "direct_enforcement_valid", None
                                    ),
                                    "execution_mode": getattr(
                                        update, "execution_mode", None
                                    ),
                                    "cap_attempted": getattr(
                                        update, "cap_attempted", False
                                    ),
                                    # Verified capping, not merely a pre-execution
                                    # cap attempt by the decision object.
                                    "margin_capped": getattr(
                                        update, "margin_capped", False
                                    ),
                                    "audit_supported": getattr(
                                        update, "audit_supported", False
                                    ),
                                    "certified": getattr(update, "certified", False),
                                    "certified_after_update": getattr(
                                        update, "certified_after_update", False
                                    ),
                                    "anti_windup_backcalc": getattr(
                                        update, "anti_windup_backcalc", None
                                    ),
                                    "infeasible": getattr(update, "infeasible", False),
                                    "restoration_used": getattr(
                                        decision, "restoration_used", False
                                    ),
                                    # Retained for backward compatibility with old
                                    # post-processing scripts; use q_enforced for new
                                    # calibration accounting.
                                    "effective_q": getattr(
                                        decision, "effective_q", None
                                    ),
                                    "headroom_estimate": getattr(
                                        decision, "headroom_estimate", None
                                    ),
                                    "intervention_norm": intervention,
                                    "nominal_action": nominal,
                                    "requested_action": requested_action,
                                    "executed_action": env.last_executed_action,
                                    "residual_action_mode": args.residual_action,
                                    "action_perturbation": info.get(
                                        "action_perturbation"
                                    ),
                                    "terminated": terminated,
                                    "truncated": truncated,
                                }
                                step_handle.write(json.dumps(
                                    json_safe(record), separators=(",", ":"),
                                    allow_nan=False,
                                ) + "\n")

                            state = next_state
                            length += 1
                            if terminated or truncated:
                                break

                        filter_metrics = (
                            safety_filter.metrics() if method != "unfiltered" else {}
                        )
                        denominator = max(length, 1)
                        valid_steps = int(totals["valid_steps"])
                        strict_valid_steps = int(totals["strict_valid_steps"])
                        capped_valid_steps = int(totals["capped_valid_steps"])
                        if strict_valid_steps + capped_valid_steps != valid_steps:
                            raise RuntimeError(
                                "Valid-step accounting mismatch: strict + capped "
                                "does not equal all enforcement-valid transitions"
                            )
                        valid_soft_loss = (
                            float(totals["valid_soft_loss_sum"]) / valid_steps
                            if valid_steps else float("nan")
                        )
                        valid_hard_exceedance_rate = (
                            float(totals["hard_exceedances"]) / valid_steps
                            if valid_steps else float("nan")
                        )
                        strict_valid_soft_loss = (
                            float(totals["strict_valid_soft_loss_sum"])
                            / strict_valid_steps
                            if strict_valid_steps else float("nan")
                        )
                        strict_valid_hard_exceedance_rate = (
                            float(totals["strict_valid_hard_exceedances"])
                            / strict_valid_steps
                            if strict_valid_steps else float("nan")
                        )
                        capped_valid_soft_loss = (
                            float(totals["capped_valid_soft_loss_sum"])
                            / capped_valid_steps
                            if capped_valid_steps else float("nan")
                        )
                        capped_valid_hard_exceedance_rate = (
                            float(totals["capped_valid_hard_exceedances"])
                            / capped_valid_steps
                            if capped_valid_steps else float("nan")
                        )
                        row = {
                            "method": method,
                            "action_noise": float(noise),
                            "episode": episode,
                            "seed": episode_seed,
                            "return": float(totals["reward"]),
                            "cost": float(totals["cost"]),
                            "safe_utility": float(
                                totals["reward"]
                                - float(args.eval_cost_weight) * totals["cost"]
                            ),
                            "cost_rate": float(totals["cost"]) / denominator,
                            "zero_cost": float(totals["cost"] == 0.0),
                            "goals": int(totals["goals"]),
                            "goals_per_1000_steps": (
                                1000.0 * float(totals["goals"]) / denominator
                            ),
                            "any_goal": float(totals["goals"] > 0),
                            "length": length,
                            "terminated": terminated,
                            "truncated": truncated,
                            "barrier_violation_rate": (
                                float(totals["barrier_violations"]) / denominator
                            ),
                            "geometric_violation_rate": (
                                float(totals["geometric_violations"]) / denominator
                            ),
                            # Episode-level enforcement-valid calibration.
                            # exceedance_rate is retained as a backward-compatible
                            # alias for valid_hard_exceedance_rate.
                            "exceedance_rate": valid_hard_exceedance_rate,
                            "valid_hard_exceedance_rate": (
                                valid_hard_exceedance_rate
                            ),
                            "valid_soft_loss": valid_soft_loss,
                            "valid_soft_loss_minus_epsilon": (
                                valid_soft_loss - float(args.epsilon)
                                if np.isfinite(valid_soft_loss)
                                else float("nan")
                            ),
                            "valid_steps": valid_steps,
                            "invalid_steps": (
                                denominator - valid_steps
                                if method != "unfiltered" else 0
                            ),
                            "valid_soft_loss_sum": float(
                                totals["valid_soft_loss_sum"]
                            ),
                            "valid_hard_exceedance_count": int(
                                totals["hard_exceedances"]
                            ),
                            "strict_valid_steps": strict_valid_steps,
                            "strict_valid_soft_loss": strict_valid_soft_loss,
                            "strict_valid_soft_loss_sum": float(
                                totals["strict_valid_soft_loss_sum"]
                            ),
                            "strict_valid_hard_exceedance_rate": (
                                strict_valid_hard_exceedance_rate
                            ),
                            "strict_valid_hard_exceedance_count": int(
                                totals["strict_valid_hard_exceedances"]
                            ),
                            "capped_valid_steps": capped_valid_steps,
                            "capped_valid_soft_loss": capped_valid_soft_loss,
                            "capped_valid_soft_loss_sum": float(
                                totals["capped_valid_soft_loss_sum"]
                            ),
                            "capped_valid_hard_exceedance_rate": (
                                capped_valid_hard_exceedance_rate
                            ),
                            "capped_valid_hard_exceedance_count": int(
                                totals["capped_valid_hard_exceedances"]
                            ),
                            "requested_exceedance_rate_all_steps": (
                                float(totals["requested_exceedances"]) / denominator
                                if method != "unfiltered" else float("nan")
                            ),
                            "calibration_valid_rate": (
                                valid_steps / denominator
                                if method != "unfiltered" else float("nan")
                            ),
                            "invalid_rate": (
                                1.0 - valid_steps / denominator
                                if method != "unfiltered" else float("nan")
                            ),
                            "intervention_rate": (
                                float(totals["interventions"]) / denominator
                            ),
                            "intervention_mean": (
                                float(totals["intervention"]) / denominator
                            ),
                            "certified_fraction": (
                                float(totals["certified_steps"]) / denominator
                            ),
                            "operational_support_steps": int(
                                totals["operational_support_steps"]
                            ),
                            "operational_support_rate": (
                                float(totals["operational_support_steps"])
                                / denominator
                                if method == "ecocsf" else float("nan")
                            ),
                            "infeasible_rate": (
                                float(totals["infeasible_steps"]) / denominator
                            ),
                            "cap_attempted_steps": int(
                                totals["cap_attempted_steps"]
                            ),
                            "cap_attempt_rate": (
                                float(totals["cap_attempted_steps"]) / denominator
                                if method != "unfiltered" else float("nan")
                            ),
                            "margin_capped_rate": (
                                float(totals["margin_capped_steps"]) / denominator
                                if method != "unfiltered" else float("nan")
                            ),
                            "cost_event_rate": (
                                float(totals["cost_events"]) / denominator
                            ),
                            "direct_contact_event_rate": (
                                float(totals["contact_events"]) / denominator
                            ),
                            "barrier_contact_disagreement_rate": (
                                float(totals["disagreements"]) / denominator
                            ),
                            "barrier_false_positive_rate": (
                                float(totals["false_positives"]) / denominator
                            ),
                            "barrier_false_negative_rate": (
                                float(totals["false_negatives"]) / denominator
                            ),
                            "actuator_error_mean": (
                                float(totals["actuator_error"]) / denominator
                            ),
                            "q_final": (
                                float(safety_filter.q)
                                if method != "unfiltered" else float("nan")
                            ),
                            **{
                                f"total_{key}": value
                                for key, value in sorted(
                                    component_cost_totals.items()
                                )
                            },
                            **{
                                f"filter_{key}": value
                                for key, value in filter_metrics.items()
                            },
                        }
                        episode_rows.append(row)
                        print(
                            f"[eval:{method}] noise={noise:.3f} "
                            f"ep={episode + 1}/{args.episodes} "
                            f"return={row['return']:.3f} cost={row['cost']:.1f} "
                            f"goals={row['goals']} intervene={row['intervention_rate']:.4f}",
                            flush=True,
                        )
    finally:
        if step_handle is not None:
            step_handle.close()

    metrics = (
        "return", "cost", "safe_utility", "cost_rate", "zero_cost",
        "goals", "goals_per_1000_steps", "any_goal",
        "barrier_violation_rate", "geometric_violation_rate",
        "exceedance_rate", "valid_hard_exceedance_rate",
        "valid_soft_loss", "valid_soft_loss_minus_epsilon",
        "strict_valid_hard_exceedance_rate", "strict_valid_soft_loss",
        "capped_valid_hard_exceedance_rate", "capped_valid_soft_loss",
        "requested_exceedance_rate_all_steps", "calibration_valid_rate",
        "invalid_rate", "intervention_rate", "intervention_mean",
        "certified_fraction", "operational_support_rate", "infeasible_rate",
        "cap_attempt_rate", "margin_capped_rate", "cost_event_rate",
        "direct_contact_event_rate", "barrier_contact_disagreement_rate",
        "barrier_false_positive_rate", "barrier_false_negative_rate",
        "actuator_error_mean", "q_final",
    )
    metrics += tuple(sorted({
        str(key) for row in episode_rows for key in row
        if str(key).startswith("total_cost_")
    }))
    aggregate_rows: List[Dict[str, Any]] = []
    for condition_index, noise in enumerate(noise_levels):
        for method_index, method in enumerate(methods):
            group = [
                row for row in episode_rows
                if row["method"] == method
                and float(row["action_noise"]) == float(noise)
            ]
            total_steps = int(sum(int(row["length"]) for row in group))
            pooled_valid_steps = int(
                sum(int(row.get("valid_steps", 0)) for row in group)
            )
            pooled_strict_valid_steps = int(
                sum(int(row.get("strict_valid_steps", 0)) for row in group)
            )
            pooled_capped_valid_steps = int(
                sum(int(row.get("capped_valid_steps", 0)) for row in group)
            )
            pooled_valid_soft_sum = float(
                sum(float(row.get("valid_soft_loss_sum", 0.0)) for row in group)
            )
            pooled_valid_hard_count = int(
                sum(
                    int(row.get("valid_hard_exceedance_count", 0))
                    for row in group
                )
            )
            pooled_strict_soft_sum = float(
                sum(
                    float(row.get("strict_valid_soft_loss_sum", 0.0))
                    for row in group
                )
            )
            pooled_strict_hard_count = int(
                sum(
                    int(row.get("strict_valid_hard_exceedance_count", 0))
                    for row in group
                )
            )
            pooled_capped_soft_sum = float(
                sum(
                    float(row.get("capped_valid_soft_loss_sum", 0.0))
                    for row in group
                )
            )
            pooled_capped_hard_count = int(
                sum(
                    int(row.get("capped_valid_hard_exceedance_count", 0))
                    for row in group
                )
            )
            pooled_support_steps = int(
                sum(int(row.get("operational_support_steps", 0)) for row in group)
            )
            pooled_cap_attempted_steps = int(
                sum(int(row.get("cap_attempted_steps", 0)) for row in group)
            )

            filtered_method = method != "unfiltered"
            aggregate: Dict[str, Any] = {
                "method": method,
                "action_noise": float(noise),
                "episodes": len(group),
                "total_steps": total_steps,
                "target_epsilon": float(args.epsilon),
                "ramp_tau": float(args.ramp_tau),
                # Exact pooled quantities matching the paper definitions.
                "pooled_valid_steps": pooled_valid_steps,
                "pooled_valid_soft_loss": (
                    pooled_valid_soft_sum / pooled_valid_steps
                    if filtered_method and pooled_valid_steps
                    else float("nan")
                ),
                "pooled_valid_hard_exceedance_rate": (
                    pooled_valid_hard_count / pooled_valid_steps
                    if filtered_method and pooled_valid_steps
                    else float("nan")
                ),
                "pooled_calibration_valid_rate": (
                    pooled_valid_steps / total_steps
                    if filtered_method and total_steps else float("nan")
                ),
                "pooled_invalid_rate": (
                    1.0 - pooled_valid_steps / total_steps
                    if filtered_method and total_steps else float("nan")
                ),
                "pooled_strict_valid_steps": pooled_strict_valid_steps,
                "pooled_strict_valid_soft_loss": (
                    pooled_strict_soft_sum / pooled_strict_valid_steps
                    if pooled_strict_valid_steps else float("nan")
                ),
                "pooled_strict_valid_hard_exceedance_rate": (
                    pooled_strict_hard_count / pooled_strict_valid_steps
                    if pooled_strict_valid_steps else float("nan")
                ),
                "pooled_capped_valid_steps": pooled_capped_valid_steps,
                "pooled_capped_valid_soft_loss": (
                    pooled_capped_soft_sum / pooled_capped_valid_steps
                    if pooled_capped_valid_steps else float("nan")
                ),
                "pooled_capped_valid_hard_exceedance_rate": (
                    pooled_capped_hard_count / pooled_capped_valid_steps
                    if pooled_capped_valid_steps else float("nan")
                ),
                "pooled_verified_capped_rate": (
                    pooled_capped_valid_steps / total_steps
                    if filtered_method and total_steps else float("nan")
                ),
                "pooled_cap_attempt_rate": (
                    pooled_cap_attempted_steps / total_steps
                    if filtered_method and total_steps else float("nan")
                ),
                "pooled_operational_support_rate": (
                    pooled_support_steps / total_steps
                    if method == "ecocsf" and total_steps else float("nan")
                ),
            }
            for metric_index, metric in enumerate(metrics):
                mean, standard_deviation, low, high = bootstrap_mean_ci(
                    [float(row.get(metric, float("nan"))) for row in group],
                    seed=(args.seed + 100_000 * condition_index
                          + 10_000 * method_index + metric_index),
                    samples=args.bootstrap_samples,
                )
                aggregate[f"{metric}_mean"] = mean
                aggregate[f"{metric}_std"] = standard_deviation
                aggregate[f"{metric}_ci95_low"] = low
                aggregate[f"{metric}_ci95_high"] = high
            aggregate_rows.append(aggregate)

    comparison_metrics = (
        "return", "cost", "safe_utility", "cost_rate", "zero_cost",
        "goals", "any_goal", "barrier_violation_rate",
        "geometric_violation_rate", "valid_soft_loss",
        "valid_hard_exceedance_rate", "calibration_valid_rate",
        "invalid_rate", "intervention_rate",
    )
    higher_is_better = {
        "return", "safe_utility", "zero_cost", "goals", "any_goal",
        "calibration_valid_rate",
    }
    paired_rows: List[Dict[str, Any]] = []
    for condition_index, noise in enumerate(noise_levels):
        comparisons: List[Tuple[str, str]] = []
        if "unfiltered" in methods:
            comparisons.extend(
                ("unfiltered", method)
                for method in methods if method != "unfiltered"
            )
        if "ecocsf" in methods:
            comparisons.extend(
                (method, "ecocsf")
                for method in methods
                if method not in {"unfiltered", "ecocsf"}
            )
        for comparison_index, (reference_method, target_method) in enumerate(comparisons):
            reference = {
                int(row["seed"]): row for row in episode_rows
                if row["method"] == reference_method
                and float(row["action_noise"]) == float(noise)
            }
            target = {
                int(row["seed"]): row for row in episode_rows
                if row["method"] == target_method
                and float(row["action_noise"]) == float(noise)
            }
            shared_seeds = sorted(set(reference) & set(target))
            for metric_index, metric in enumerate(comparison_metrics):
                differences: List[float] = []
                for seed in shared_seeds:
                    target_value = float(target[seed].get(metric, float("nan")))
                    reference_value = float(
                        reference[seed].get(metric, float("nan"))
                    )
                    if np.isfinite(target_value) and np.isfinite(reference_value):
                        differences.append(target_value - reference_value)
                mean, standard_deviation, low, high = bootstrap_mean_ci(
                    differences,
                    seed=(args.seed + 1_000_000 + 100_000 * condition_index
                          + 10_000 * comparison_index + metric_index),
                    samples=args.bootstrap_samples,
                )
                paired_rows.append({
                    "action_noise": float(noise),
                    "reference_method": reference_method,
                    "target_method": target_method,
                    "metric": metric,
                    "higher_is_better": metric in higher_is_better,
                    "paired_episodes": len(differences),
                    "mean_difference_target_minus_reference": mean,
                    "difference_std": standard_deviation,
                    "difference_ci95_low": low,
                    "difference_ci95_high": high,
                })

    calibration_rows: List[Dict[str, Any]] = []
    for aggregate in aggregate_rows:
        if aggregate["method"] == "unfiltered":
            continue
        calibration_rows.append({
            "method": aggregate["method"],
            "action_noise": aggregate["action_noise"],
            "episodes": aggregate["episodes"],
            "total_steps": aggregate["total_steps"],
            "target_epsilon": aggregate["target_epsilon"],
            "ramp_tau": aggregate["ramp_tau"],
            "valid_soft_loss": aggregate["pooled_valid_soft_loss"],
            "valid_hard_exceedance_rate": aggregate[
                "pooled_valid_hard_exceedance_rate"
            ],
            "valid_steps": aggregate["pooled_valid_steps"],
            "invalid_rate": aggregate["pooled_invalid_rate"],
            "verified_capped_rate": aggregate[
                "pooled_verified_capped_rate"
            ],
            "operational_support_rate": aggregate[
                "pooled_operational_support_rate"
            ],
            "strict_valid_soft_loss": aggregate[
                "pooled_strict_valid_soft_loss"
            ],
            "strict_valid_hard_exceedance_rate": aggregate[
                "pooled_strict_valid_hard_exceedance_rate"
            ],
            "strict_valid_steps": aggregate["pooled_strict_valid_steps"],
            "capped_valid_soft_loss": aggregate[
                "pooled_capped_valid_soft_loss"
            ],
            "capped_valid_hard_exceedance_rate": aggregate[
                "pooled_capped_valid_hard_exceedance_rate"
            ],
            "capped_valid_steps": aggregate["pooled_capped_valid_steps"],
            "episode_valid_soft_loss_mean": aggregate[
                "valid_soft_loss_mean"
            ],
            "episode_valid_soft_loss_std": aggregate[
                "valid_soft_loss_std"
            ],
            "episode_valid_soft_loss_ci95_low": aggregate[
                "valid_soft_loss_ci95_low"
            ],
            "episode_valid_soft_loss_ci95_high": aggregate[
                "valid_soft_loss_ci95_high"
            ],
        })

    write_csv(out_dir / "benchmark_episode_results.csv", episode_rows)
    write_csv(out_dir / "benchmark_aggregate_results.csv", aggregate_rows)
    write_csv(out_dir / "benchmark_calibration_accounting.csv", calibration_rows)
    write_csv(out_dir / "benchmark_paired_method_deltas.csv", paired_rows)
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "methods": methods,
        "episode_results": episode_rows,
        "aggregate_results": aggregate_rows,
        "calibration_accounting": calibration_rows,
        "paired_method_deltas": paired_rows,
        "step_log": None if args.no_step_log else str(step_log_path),
    }
    atomic_json(out_dir / "benchmark_results.json", result)
    atomic_json(out_dir / "manifest.json", build_manifest(args, checkpoint=checkpoint))
    print(json.dumps(json_safe(aggregate_rows), indent=2), flush=True)
    return result


def smoke(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    with make_adapter(args, action_noise=0.0, render=args.render) as env:
        observation, _ = env.reset(seed=args.seed)
        barrier = PointGoalBarrierModel(env.codec)
        total_cost = 0.0
        for step in range(int(args.steps)):
            state = env.filter_state(observation)
            action = env.action_space.sample()
            prediction = barrier.h_hat(state, action)
            observation, reward, cost, terminated, truncated, _ = env.step(action)
            next_state = env.filter_state(observation)
            total_cost += cost
            if not np.isfinite(prediction) or not np.isfinite(barrier.h(next_state)):
                raise RuntimeError("Barrier smoke test produced NaN/Inf")
            if terminated or truncated:
                break
        print(
            f"[smoke] env={args.env_id} obs_dim={env.observation_dim} "
            f"filter_state_dim={env.codec.state_dim} steps={step + 1} cost={total_cost:.1f}"
        )


def add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env_id", default="SafetyPointGoal2-v0")
    parser.add_argument("--frameskip_probability", type=float, default=1.0)
    parser.add_argument("--max_objects", type=int, default=32)
    parser.add_argument("--hazard_buffer", type=float, default=0.02)
    parser.add_argument("--vase_buffer", type=float, default=0.02)
    parser.add_argument("--agent_radius", type=float, default=0.10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO and evaluate E-COCSF on Safety-Gymnasium PointGoal2. "
            "The eval command accepts both PPO checkpoints from this script "
            "and SAC checkpoints from safety_gym_sac_train.py."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train the nominal PPO policy")
    add_environment_arguments(train_parser)
    train_parser.add_argument("--out_dir", default="runs/pointgoal2_ppo_seed42")
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--total_steps", type=int, default=500_000)
    train_parser.add_argument("--hidden", type=int, default=256)
    train_parser.add_argument("--rollout_steps", type=int, default=2048)
    train_parser.add_argument("--update_epochs", type=int, default=10)
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--gae_lambda", type=float, default=0.95)
    train_parser.add_argument("--learning_rate", type=float, default=3e-4)
    train_parser.add_argument("--clip_range", type=float, default=0.20)
    train_parser.add_argument("--value_coef", type=float, default=0.50)
    train_parser.add_argument("--entropy_coef", type=float, default=0.0)
    train_parser.add_argument("--max_grad_norm", type=float, default=0.50)
    train_parser.add_argument(
        "--target_kl",
        type=float,
        default=0.03,
        help="Early-stop a PPO update epoch when approximate KL exceeds 1.5x this value; use 0 to disable",
    )
    train_parser.add_argument("--observation_clip", type=float, default=10.0)
    train_parser.add_argument("--train_action_noise", type=float, default=0.0)
    train_parser.add_argument("--train_cost_penalty", type=float, default=0.0)
    train_parser.add_argument("--log_every", type=int, default=5_000)
    train_parser.add_argument("--checkpoint_every", type=int, default=100_000)
    train_parser.add_argument("--deterministic_torch", action="store_true")

    identify_parser = subparsers.add_parser(
        "identify",
        help="fit a frozen Point dynamics model on disjoint collision-free data",
    )
    add_environment_arguments(identify_parser)
    identify_parser.add_argument(
        "--out_model", default="runs/pointgoal2_dynamics_seed10042.json"
    )
    identify_parser.add_argument("--transitions", type=int, default=20_000)
    identify_parser.add_argument("--action_hold_steps", type=int, default=5)
    identify_parser.add_argument("--min_clearance", type=float, default=0.02)
    identify_parser.add_argument("--max_attempt_factor", type=float, default=10.0)
    identify_parser.add_argument("--ridge", type=float, default=1e-6)
    identify_parser.add_argument("--validation_fraction", type=float, default=0.20)
    identify_parser.add_argument("--trim_quantile", type=float, default=0.995)
    identify_parser.add_argument("--log_every", type=int, default=1_000)

    eval_parser = subparsers.add_parser(
        "eval", help="run paired unfiltered/baseline/E-COCSF benchmarks"
    )
    add_environment_arguments(eval_parser)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--out_dir", default="runs/pointgoal2_ecocsf_seed42")
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--episodes", type=int, default=50)
    eval_parser.add_argument("--max_steps", type=int, default=1000)
    eval_parser.add_argument("--noise_levels", default="0,0.05,0.10,0.20")
    eval_parser.add_argument(
        "--methods",
        default="unfiltered,fixed,uncertainty,naive_aci,ecocsf",
    )
    eval_parser.add_argument(
        "--eval_cost_weight", type=float, default=1.0,
        help="Reporting-only weight in safe_utility = return - weight * cost",
    )
    eval_parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    eval_parser.add_argument("--no_step_log", action="store_true")
    eval_parser.add_argument("--deterministic_torch", action="store_true")
    eval_parser.add_argument("--intervention_threshold", type=float, default=1e-6)
    eval_parser.add_argument(
        "--residual_action",
        choices=("commanded", "executed"),
        default="commanded",
        help=(
            "Use commanded action so unknown actuator noise enters the conformal "
            "residual; executed is only for fully observed low-level overrides"
        ),
    )

    # E-COCSF calibration and audit.
    eval_parser.add_argument("--epsilon", type=float, default=0.10)
    # Validation-calibrated profile for SafetyPointGoal2.  The earlier
    # q_init=0.006, q_max=0.0125, eta=0.00025 profile was below the empirical
    # 0.10-loss root (about 0.014--0.016 on the validation residuals) and adapted
    # too slowly after every episode reset.  This profile removes that cold-start
    # bias while retaining a bounded online recursion.
    eval_parser.add_argument("--eta", type=float, default=0.001)
    eval_parser.add_argument("--q_init", type=float, default=0.016)
    eval_parser.add_argument("--q_max", type=float, default=0.05)
    eval_parser.add_argument("--ramp_tau", type=float, default=0.001)
    eval_parser.add_argument("--zeta_max", type=float, default=0.001)
    eval_parser.add_argument("--probe_probability", type=float, default=0.20)
    # alpha=0.70 lowers the (1-alpha)h decay RHS, enlarging the per-step
    # controllable headroom relative to the conformal margin.
    eval_parser.add_argument("--barrier_alpha", type=float, default=0.70)
    eval_parser.add_argument(
        "--headroom_cap_delta", type=float, default=0.0001,
        help=(
            "Safety buffer subtracted from estimated feasible headroom. "
            "The final command is still reverified, so this should be small "
            "relative to the conformal margin and ramp width."
        ),
    )
    eval_parser.add_argument(
        "--anti_windup_gamma", type=float, default=0.01,
        help=(
            "Back-calculation gain on verified capped transitions. "
            "A small positive value avoids the aggressive q collapse caused "
            "by the former hidden ECLCS default of 0.25."
        ),
    )
    eval_parser.add_argument(
        "--capped_positive_integration", type=float, default=1.0,
        help=(
            "Fraction in [0,1] of positive loss innovation retained on a "
            "verified capped transition. Use 1 for pooled-valid target tracking; "
            "use 0 to reproduce conditional-integration ablation."
        ),
    )
    eval_parser.add_argument(
        "--allow_positive_cap_backcalc", action="store_true",
        help=(
            "Also apply downward back-calculation on a capped transition whose "
            "loss exceeds epsilon. Disabled by default because it gives a "
            "sign-inconsistent high-loss update."
        ),
    )
    eval_parser.add_argument(
        "--no_headroom_cap", action="store_true",
        help="Disable the E-COCSF headroom-capped executed margin (ablation)",
    )
    eval_parser.add_argument("--projection_scp_iterations", type=int, default=3)
    eval_parser.add_argument("--fixed_margin", type=float, default=0.004)
    eval_parser.add_argument("--uncertainty_quantile", type=float, default=0.90)
    eval_parser.add_argument("--uncertainty_scale", type=float, default=1.0)
    eval_parser.add_argument("--uncertainty_min_samples", type=int, default=30)
    eval_parser.add_argument("--certified_tube_delta", type=float, default=0.005)
    eval_parser.add_argument("--audit_window", type=int, default=200)
    eval_parser.add_argument("--audit_min_samples", type=int, default=50)
    eval_parser.add_argument("--audit_min_q_range", type=float, default=5e-4)
    eval_parser.add_argument("--audit_conf_z", type=float, default=1.645)
    eval_parser.add_argument("--audit_min_mu", type=float, default=1e-3)
    eval_parser.add_argument("--audit_crossing_slack", type=float, default=0.02)
    eval_parser.add_argument("--gain_schedule", action="store_true")
    eval_parser.add_argument("--no_feasibility_restoration", action="store_true")
    eval_parser.add_argument("--restoration_grid_points", type=int, default=21)

    # Point dynamics model and barrier.
    eval_parser.add_argument("--model_dt", type=float, default=0.02)
    eval_parser.add_argument("--dynamics_model", default=None)
    eval_parser.add_argument("--barrier_time_headway", type=float, default=0.10)
    # 4 backup steps retain the steer-around effect of the rollout while
    # leaving one-step action authority over h_hat above the margin scale.
    eval_parser.add_argument("--barrier_lookahead_steps", type=int, default=4)
    eval_parser.add_argument(
        "--barrier_braking_deceleration", type=float, default=3.0
    )
    eval_parser.add_argument(
        "--barrier_braking_distance_scale", type=float, default=0.35
    )
    eval_parser.add_argument("--model_acceleration_gain", type=float, default=3.0)
    eval_parser.add_argument("--model_linear_drag", type=float, default=0.15)
    eval_parser.add_argument("--model_yaw_tracking", type=float, default=0.85)
    eval_parser.add_argument("--model_max_speed", type=float, default=3.0)
    eval_parser.add_argument("--workspace_half_extent", type=float, default=0.0)
    eval_parser.add_argument("--workspace_buffer", type=float, default=0.10)
    eval_parser.add_argument("--action_rate_limit", type=float, default=2.0)
    eval_parser.add_argument("--action_jerk_limit", type=float, default=4.0)
    eval_parser.add_argument("--yaw_action_weight", type=float, default=0.50)

    smoke_parser = subparsers.add_parser("smoke", help="verify environment and barrier integration")
    add_environment_arguments(smoke_parser)
    smoke_parser.add_argument("--steps", type=int, default=100)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not (0.0 < float(args.frameskip_probability) <= 1.0):
        raise SystemExit("--frameskip_probability must lie in (0,1]")
    if int(args.max_objects) < 20 and args.env_id == "SafetyPointGoal2-v0":
        raise SystemExit("PointGoal2 has 20 constrained objects; use --max_objects >= 20")
    if args.command == "train":
        if min(
            args.total_steps,
            args.rollout_steps,
            args.update_epochs,
            args.batch_size,
            args.log_every,
            args.checkpoint_every,
        ) < 1:
            raise SystemExit("PPO step/epoch/batch/log arguments must be positive")
        if args.batch_size > args.rollout_steps:
            raise SystemExit("--batch_size must be <= --rollout_steps")
        if not (0.0 < args.gamma <= 1.0):
            raise SystemExit("--gamma must lie in (0,1]")
        if not (0.0 <= args.gae_lambda <= 1.0):
            raise SystemExit("--gae_lambda must lie in [0,1]")
        if args.learning_rate <= 0.0 or args.clip_range <= 0.0:
            raise SystemExit("--learning_rate and --clip_range must be positive")
        if args.value_coef < 0.0 or args.entropy_coef < 0.0:
            raise SystemExit("--value_coef and --entropy_coef must be non-negative")
        if args.max_grad_norm <= 0.0 or args.target_kl < 0.0:
            raise SystemExit("Require --max_grad_norm > 0 and --target_kl >= 0")
    if args.command == "identify":
        if min(args.transitions, args.action_hold_steps, args.log_every) < 1:
            raise SystemExit("identification counts must be positive")
        if args.min_clearance < 0.0 or args.max_attempt_factor < 1.0:
            raise SystemExit(
                "Require --min_clearance >= 0 and --max_attempt_factor >= 1"
            )
        if args.ridge < 0.0:
            raise SystemExit("--ridge must be non-negative")
        if not (0.05 <= args.validation_fraction <= 0.50):
            raise SystemExit("--validation_fraction must lie in [0.05,0.50]")
        if not (0.90 <= args.trim_quantile <= 1.0):
            raise SystemExit("--trim_quantile must lie in [0.90,1]")
    if args.command == "eval":
        parse_methods(args.methods)
        if not (0.0 < args.epsilon < 1.0):
            raise SystemExit("--epsilon must lie in (0,1)")
        if min(args.episodes, args.max_steps, args.bootstrap_samples) < 1:
            raise SystemExit("episode/step/bootstrap counts must be positive")
        if not (0.0 <= args.probe_probability <= 1.0):
            raise SystemExit("--probe_probability must lie in [0,1]")
        if not (0.0 < args.barrier_alpha <= 1.0):
            raise SystemExit("--barrier_alpha must lie in (0,1]")
        if args.projection_scp_iterations < 1:
            raise SystemExit("--projection_scp_iterations must be >= 1")
        if args.eta <= 0.0 or args.ramp_tau <= 0.0:
            raise SystemExit("--eta and --ramp_tau must be positive")
        if args.zeta_max < 0.0:
            raise SystemExit("--zeta_max must be non-negative")
        if args.q_init < 0.0 or args.q_max <= 0.0 or args.q_init > args.q_max:
            raise SystemExit("Require 0 <= --q_init <= --q_max and --q_max > 0")
        if not (0.0 <= args.fixed_margin <= args.q_max):
            raise SystemExit("--fixed_margin must lie in [0,q_max]")
        if not (0.0 < args.uncertainty_quantile < 1.0):
            raise SystemExit("--uncertainty_quantile must lie in (0,1)")
        if args.uncertainty_scale < 0.0 or args.uncertainty_min_samples < 1:
            raise SystemExit(
                "Require --uncertainty_scale >= 0 and --uncertainty_min_samples >= 1"
            )
        if args.audit_min_samples > args.audit_window:
            raise SystemExit("--audit_min_samples cannot exceed --audit_window")
        if args.model_dt <= 0.0 or args.barrier_time_headway < 0.0:
            raise SystemExit(
                "Require --model_dt > 0 and --barrier_time_headway >= 0"
            )
        if (
            args.barrier_lookahead_steps < 0
            or args.barrier_braking_deceleration <= 0.0
            or args.barrier_braking_distance_scale < 0.0
        ):
            raise SystemExit(
                "Require lookahead >= 0, braking deceleration > 0, and "
                "braking-distance scale >= 0"
            )
        if args.eval_cost_weight < 0.0:
            raise SystemExit("--eval_cost_weight must be non-negative")
        if args.headroom_cap_delta <= 0.0:
            raise SystemExit("--headroom_cap_delta must be positive")
        if not (0.0 < args.anti_windup_gamma < 1.0):
            raise SystemExit("--anti_windup_gamma must lie in (0,1)")
        if not (0.0 <= args.capped_positive_integration <= 1.0):
            raise SystemExit(
                "--capped_positive_integration must lie in [0,1]"
            )
        if args.dynamics_model and not Path(args.dynamics_model).is_file():
            raise SystemExit(
                f"--dynamics_model does not exist: {args.dynamics_model}"
            )
        parse_noise_levels(args.noise_levels)


def main() -> None:
    args = build_parser().parse_args()
    validate_arguments(args)
    if args.command == "train":
        train(args)
    elif args.command == "identify":
        identify(args)
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "smoke":
        smoke(args)
    else:  # pragma: no cover
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
