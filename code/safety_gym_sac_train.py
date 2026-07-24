#!/usr/bin/env python3
"""Train a nominal Soft Actor-Critic policy on SafetyPointGoal2-v0.

The policy uses only the standard Safety-Gymnasium observation. Privileged
geometry remains exclusively inside ``safety_gym_env.py`` and the E-COCSF
runtime filter, so PPO and SAC checkpoints can be compared with the same safety
layer without changing the policy's information set.

Example
-------
python -u safety_gym_sac_train.py \
  --device auto --seed 42 --total_steps 1000000 \
  --out_dir runs/pointgoal2_sac_seed42
"""

from __future__ import annotations

import argparse
import csv
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "PyTorch is required. Install the CUDA/CPU build recommended at "
        "https://pytorch.org/get-started/locally/"
    ) from exc

from safety_gym_env import SafetyPointGoalAdapter


CHECKPOINT_SCHEMA_VERSION = 1


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
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
    value = str(requested).strip().lower()
    if value in {"", "auto", "none"}:
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(value)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        required_arch = f"sm_{major}{minor}"
        compiled = set(torch.cuda.get_arch_list())
        if compiled and required_arch not in compiled:
            raise RuntimeError(
                f"PyTorch {torch.__version__} lacks kernels for {required_arch}; "
                f"compiled architectures are {sorted(compiled)}"
            )
    return device


def finite_vector(
    value: Any, *, expected: Optional[int] = None, name: str = "value"
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if expected is not None and array.size != int(expected):
        raise ValueError(f"{name} expected {expected} values, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} contains NaN/Inf")
    return array


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
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


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class RunningMeanStd:
    """Numerically stable observation normalization, saved in the checkpoint."""

    def __init__(self, shape: int, epsilon: float = 1e-4):
        self.mean = np.zeros(int(shape), dtype=np.float64)
        self.variance = np.ones(int(shape), dtype=np.float64)
        self.count = float(epsilon)

    def update(self, observation: Sequence[float]) -> None:
        batch = np.asarray(observation, dtype=np.float64).reshape(1, -1)
        if batch.shape[1] != self.mean.size or not np.all(np.isfinite(batch)):
            raise RuntimeError("Invalid observation supplied to normalizer")
        batch_mean = batch.mean(axis=0)
        batch_variance = batch.var(axis=0)
        batch_count = float(batch.shape[0])
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        old_m2 = self.variance * self.count
        batch_m2 = batch_variance * batch_count
        new_m2 = old_m2 + batch_m2 + delta * delta * self.count * batch_count / total
        self.mean = new_mean
        self.variance = np.maximum(new_m2 / total, 1e-8)
        self.count = total

    def normalize(self, observation: np.ndarray, clip: float) -> np.ndarray:
        normalized = (observation - self.mean) / np.sqrt(self.variance + 1e-8)
        if float(clip) > 0.0:
            normalized = np.clip(normalized, -float(clip), float(clip))
        return normalized.astype(np.float32, copy=False)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "variance": self.variance.copy(),
            "count": float(self.count),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        mean = np.asarray(payload["mean"], dtype=np.float64).reshape(-1)
        variance = np.asarray(payload["variance"], dtype=np.float64).reshape(-1)
        if mean.shape != self.mean.shape or variance.shape != self.variance.shape:
            raise RuntimeError("Observation-normalizer shape mismatch")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise RuntimeError("Observation-normalizer state contains NaN/Inf")
        self.mean = mean.copy()
        self.variance = np.maximum(variance, 1e-8)
        self.count = float(payload["count"])


def initialize_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(module.bias)


class SquashedGaussianActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, observation_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.apply(initialize_linear)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.constant_(self.log_std.bias, -0.5)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.body(observation)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(observation)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw_action = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        log_probability = distribution.log_prob(raw_action).sum(dim=-1)
        log_probability -= torch.log(1.0 - action.square() + 1e-6).sum(dim=-1)
        return action, log_probability


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.apply(initialize_linear)
        nn.init.orthogonal_(self.network[-1].weight, gain=1.0)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1)).squeeze(-1)


class ReplayBuffer:
    """Fixed-size raw-observation replay with true-terminal bootstrapping flags."""

    def __init__(
        self, capacity: int, observation_dim: int, action_dim: int, seed: int
    ):
        self.capacity = int(capacity)
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(int(seed))

    def add(
        self,
        observation: Sequence[float],
        action: Sequence[float],
        reward: float,
        next_observation: Sequence[float],
        terminated: bool,
    ) -> None:
        index = self.position
        self.observations[index] = finite_vector(observation, name="observation")
        self.next_observations[index] = finite_vector(
            next_observation, name="next observation"
        )
        self.actions[index] = np.clip(finite_vector(action, name="action"), -1.0, 1.0)
        self.rewards[index] = float(np.clip(reward, -1e4, 1e4))
        self.terminated[index] = float(bool(terminated))
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.size < int(batch_size):
            raise RuntimeError("Replay buffer has fewer samples than batch_size")
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return {
            "observations": self.observations[indices],
            "next_observations": self.next_observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "terminated": self.terminated[indices],
        }

    def state_dict(self) -> Dict[str, Any]:
        count = self.size if self.size < self.capacity else self.capacity
        return {
            "capacity": self.capacity,
            "position": self.position,
            "size": self.size,
            "observations": self.observations[:count].copy(),
            "next_observations": self.next_observations[:count].copy(),
            "actions": self.actions[:count].copy(),
            "rewards": self.rewards[:count].copy(),
            "terminated": self.terminated[:count].copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if int(payload["capacity"]) != self.capacity:
            raise RuntimeError("Replay-buffer capacity mismatch")
        size = int(payload["size"])
        if not (0 <= size <= self.capacity):
            raise RuntimeError("Invalid replay-buffer size")
        count = size if size < self.capacity else self.capacity
        self.observations[:count] = np.asarray(payload["observations"], dtype=np.float32)
        self.next_observations[:count] = np.asarray(
            payload["next_observations"], dtype=np.float32
        )
        self.actions[:count] = np.asarray(payload["actions"], dtype=np.float32)
        self.rewards[:count] = np.asarray(payload["rewards"], dtype=np.float32)
        self.terminated[:count] = np.asarray(payload["terminated"], dtype=np.float32)
        self.position = int(payload["position"])
        self.size = size
        self.rng.bit_generator.state = payload["rng_state"]


@dataclass
class SACConfig:
    observation_dim: int
    action_dim: int = 2
    hidden: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 500_000
    learning_starts: int = 10_000
    gradient_steps: int = 1
    target_update_interval: int = 1
    reward_scale: float = 1.0
    automatic_entropy_tuning: bool = True
    initial_alpha: float = 0.20
    target_entropy: Optional[float] = None
    observation_clip: float = 10.0
    normalizer_freeze_steps: int = 100_000
    max_grad_norm: float = 10.0
    seed: int = 42


class SACAgent:
    """Twin-critic SAC with automatic temperature tuning and target networks."""

    def __init__(
        self,
        config: SACConfig,
        device: str = "auto",
        *,
        replay_capacity_override: Optional[int] = None,
    ):
        self.config = config
        self.device = resolve_device(device)
        set_seed(config.seed)
        self.actor = SquashedGaussianActor(
            config.observation_dim, config.action_dim, config.hidden
        ).to(self.device)
        self.q1 = QNetwork(config.observation_dim, config.action_dim, config.hidden).to(
            self.device
        )
        self.q2 = QNetwork(config.observation_dim, config.action_dim, config.hidden).to(
            self.device
        )
        self.target_q1 = QNetwork(
            config.observation_dim, config.action_dim, config.hidden
        ).to(self.device)
        self.target_q2 = QNetwork(
            config.observation_dim, config.action_dim, config.hidden
        ).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.learning_rate)
        self.q1_optimizer = Adam(self.q1.parameters(), lr=config.learning_rate)
        self.q2_optimizer = Adam(self.q2.parameters(), lr=config.learning_rate)
        self.log_alpha = torch.tensor(
            math.log(float(config.initial_alpha)),
            dtype=torch.float32,
            device=self.device,
            requires_grad=bool(config.automatic_entropy_tuning),
        )
        self.alpha_optimizer = (
            Adam([self.log_alpha], lr=config.alpha_learning_rate)
            if config.automatic_entropy_tuning else None
        )
        self.target_entropy = float(
            -config.action_dim if config.target_entropy is None
            else config.target_entropy
        )
        self.normalizer = RunningMeanStd(config.observation_dim)
        self.replay = ReplayBuffer(
            (
                config.replay_capacity
                if replay_capacity_override is None
                else int(replay_capacity_override)
            ),
            config.observation_dim,
            config.action_dim,
            seed=config.seed + 17,
        )
        self.updates = 0
        self.global_step = 0
        self.episodes = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    def update_normalizer(self, observation: Sequence[float]) -> None:
        if self.global_step < int(self.config.normalizer_freeze_steps):
            self.normalizer.update(observation)

    def preprocess(self, observation: Sequence[float]) -> np.ndarray:
        raw = finite_vector(
            observation,
            expected=self.config.observation_dim,
            name="policy observation",
        ).astype(np.float64)
        return self.normalizer.normalize(raw, self.config.observation_clip)

    def _normalize_batch(self, observations: np.ndarray) -> torch.Tensor:
        normalized = self.normalizer.normalize(
            np.asarray(observations, dtype=np.float64),
            self.config.observation_clip,
        )
        return torch.as_tensor(normalized, dtype=torch.float32, device=self.device)

    def act(self, observation: Sequence[float], deterministic: bool) -> np.ndarray:
        tensor = torch.as_tensor(
            self.preprocess(observation)[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            action, _ = self.actor.sample(tensor, deterministic=deterministic)
        return np.clip(action[0].cpu().numpy(), -1.0, 1.0).astype(np.float64)

    def store(
        self,
        observation: Sequence[float],
        action: Sequence[float],
        reward: float,
        next_observation: Sequence[float],
        terminated: bool,
    ) -> None:
        self.replay.add(observation, action, reward, next_observation, terminated)

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    @torch.no_grad()
    def _polyak_update(self) -> None:
        tau = float(self.config.tau)
        for online, target in (
            (self.q1, self.target_q1), (self.q2, self.target_q2)
        ):
            for online_parameter, target_parameter in zip(
                online.parameters(), target.parameters()
            ):
                target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)

    def update(self) -> Dict[str, float]:
        cfg = self.config
        batch = self.replay.sample(cfg.batch_size)
        observations = self._normalize_batch(batch["observations"])
        next_observations = self._normalize_batch(batch["next_observations"])
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )
        terminated = torch.as_tensor(
            batch["terminated"], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            next_actions, next_log_probability = self.actor.sample(next_observations)
            target_q = torch.minimum(
                self.target_q1(next_observations, next_actions),
                self.target_q2(next_observations, next_actions),
            ) - self.alpha * next_log_probability
            target = (
                float(cfg.reward_scale) * rewards
                + float(cfg.gamma) * (1.0 - terminated) * target_q
            )

        q1_prediction = self.q1(observations, actions)
        q2_prediction = self.q2(observations, actions)
        q1_loss = F.mse_loss(q1_prediction, target)
        q2_loss = F.mse_loss(q2_prediction, target)
        if not torch.isfinite(q1_loss + q2_loss):
            raise FloatingPointError("Non-finite SAC critic loss")

        self.q1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), cfg.max_grad_norm)
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), cfg.max_grad_norm)
        self.q2_optimizer.step()

        self._set_requires_grad(self.q1, False)
        self._set_requires_grad(self.q2, False)
        sampled_actions, log_probability = self.actor.sample(observations)
        actor_q = torch.minimum(
            self.q1(observations, sampled_actions),
            self.q2(observations, sampled_actions),
        )
        actor_loss = (self.alpha * log_probability - actor_q).mean()
        if not torch.isfinite(actor_loss):
            raise FloatingPointError("Non-finite SAC actor loss")
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
        self.actor_optimizer.step()
        self._set_requires_grad(self.q1, True)
        self._set_requires_grad(self.q2, True)

        alpha_loss_value = 0.0
        if self.alpha_optimizer is not None:
            alpha_loss = -(
                self.log_alpha * (log_probability.detach() + self.target_entropy)
            ).mean()
            if not torch.isfinite(alpha_loss):
                raise FloatingPointError("Non-finite SAC temperature loss")
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.item())

        self.updates += 1
        if self.updates % int(cfg.target_update_interval) == 0:
            self._polyak_update()

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha.item()),
            "target_q_mean": float(target.mean().item()),
            "q_mean": float(torch.minimum(q1_prediction, q2_prediction).mean().item()),
            "log_probability_mean": float(log_probability.mean().item()),
        }

    def checkpoint_payload(
        self, *, env_id: str, include_replay: bool
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "algorithm": "SAC",
            "env_id": str(env_id),
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": (
                self.alpha_optimizer.state_dict()
                if self.alpha_optimizer is not None else None
            ),
            "normalizer": self.normalizer.state_dict(),
            "updates": int(self.updates),
            "global_step": int(self.global_step),
            "episodes": int(self.episodes),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        if include_replay:
            payload["replay"] = self.replay.state_dict()
        return payload

    def save(self, path: Path, *, env_id: str, include_replay: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            self.checkpoint_payload(env_id=env_id, include_replay=include_replay),
            temporary,
        )
        os.replace(temporary, path)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        device: str = "auto",
        load_optimizers: bool = False,
        load_replay: bool = False,
    ) -> "SACAgent":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported SAC checkpoint schema {checkpoint.get('schema_version')}"
            )
        if checkpoint.get("algorithm") != "SAC":
            raise RuntimeError("Checkpoint is not a SAC checkpoint")
        config = SACConfig(**checkpoint["config"])
        # Deterministic evaluation needs only the actor and normalizer. Avoid
        # allocating the full off-policy replay buffer in that path.
        replay_capacity = (
            config.replay_capacity
            if (load_optimizers or load_replay) else config.batch_size
        )
        agent = cls(
            config,
            device=device,
            replay_capacity_override=replay_capacity,
        )
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.q1.load_state_dict(checkpoint["q1"])
        agent.q2.load_state_dict(checkpoint["q2"])
        agent.target_q1.load_state_dict(checkpoint["target_q1"])
        agent.target_q2.load_state_dict(checkpoint["target_q2"])
        agent.normalizer.load_state_dict(checkpoint["normalizer"])
        agent.log_alpha.data.copy_(
            torch.as_tensor(checkpoint["log_alpha"], device=agent.device)
        )
        agent.updates = int(checkpoint.get("updates", 0))
        agent.global_step = int(checkpoint.get("global_step", 0))
        agent.episodes = int(checkpoint.get("episodes", 0))
        if load_optimizers:
            agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            agent.q1_optimizer.load_state_dict(checkpoint["q1_optimizer"])
            agent.q2_optimizer.load_state_dict(checkpoint["q2_optimizer"])
            if agent.alpha_optimizer is not None and checkpoint.get("alpha_optimizer"):
                agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        if load_replay:
            if "replay" not in checkpoint:
                raise RuntimeError(
                    "Checkpoint has no replay state; train with --save_replay"
                )
            agent.replay.load_state_dict(checkpoint["replay"])
        if load_optimizers:
            if "python_random_state" in checkpoint:
                random.setstate(checkpoint["python_random_state"])
            if "numpy_random_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_random_state"])
            if "torch_random_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_random_state"])
            if torch.cuda.is_available() and checkpoint.get("torch_cuda_random_state"):
                torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state"])
        return agent


def make_adapter(
    args: argparse.Namespace, *, action_noise: float, render: bool = False
) -> SafetyPointGoalAdapter:
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


def evaluate_nominal(
    agent: SACAgent, args: argparse.Namespace
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    if int(args.eval_episodes) <= 0:
        return rows, {}
    with make_adapter(args, action_noise=0.0, render=False) as env:
        for episode in range(int(args.eval_episodes)):
            observation, _ = env.reset(seed=int(args.eval_seed + episode))
            total_return = total_cost = 0.0
            goals = length = 0
            terminated = truncated = False
            for _ in range(int(args.eval_max_steps)):
                action = agent.act(observation, deterministic=True)
                observation, reward, cost, terminated, truncated, info = env.step(action)
                total_return += float(reward)
                total_cost += float(cost)
                goals += int(bool(info.get("goal_met", False)))
                length += 1
                if terminated or truncated:
                    break
            rows.append({
                "episode": episode,
                "seed": int(args.eval_seed + episode),
                "return": total_return,
                "cost": total_cost,
                "cost_rate": total_cost / max(length, 1),
                "goals": goals,
                "length": length,
                "terminated": terminated,
                "truncated": truncated,
            })
    summary = {
        "episodes": float(len(rows)),
        "return_mean": float(np.mean([row["return"] for row in rows])),
        "cost_mean": float(np.mean([row["cost"] for row in rows])),
        "goals_mean": float(np.mean([row["goals"] for row in rows])),
        "zero_cost_fraction": float(np.mean([row["cost"] == 0.0 for row in rows])),
    }
    return rows, summary


def build_manifest(
    args: argparse.Namespace, *, checkpoint: Path
) -> Dict[str, Any]:
    source_path = Path(__file__).resolve()
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
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
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_sha256": sha256_file(source_path),
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, args.deterministic_torch)

    with make_adapter(
        args, action_noise=args.train_action_noise, render=args.render
    ) as env:
        if args.resume:
            agent = SACAgent.load(
                Path(args.resume),
                device=args.device,
                load_optimizers=True,
                load_replay=bool(args.resume_replay),
            )
            if agent.config.observation_dim != env.observation_dim:
                raise RuntimeError("Resume checkpoint observation dimension mismatch")
            if agent.config.action_dim != env.action_dim:
                raise RuntimeError("Resume checkpoint action dimension mismatch")
        else:
            agent = SACAgent(SACConfig(
                observation_dim=env.observation_dim,
                action_dim=env.action_dim,
                hidden=args.hidden,
                gamma=args.gamma,
                tau=args.tau,
                learning_rate=args.learning_rate,
                alpha_learning_rate=args.alpha_learning_rate,
                batch_size=args.batch_size,
                replay_capacity=args.replay_capacity,
                learning_starts=args.learning_starts,
                gradient_steps=args.gradient_steps,
                target_update_interval=args.target_update_interval,
                reward_scale=args.reward_scale,
                automatic_entropy_tuning=not args.fixed_alpha,
                initial_alpha=args.initial_alpha,
                target_entropy=args.target_entropy,
                observation_clip=args.observation_clip,
                normalizer_freeze_steps=args.normalizer_freeze_steps,
                max_grad_norm=args.max_grad_norm,
                seed=args.seed,
            ), device=args.device)

        observation, _ = env.reset(seed=int(args.seed + agent.episodes))
        agent.update_normalizer(observation)
        episode_return = episode_cost = 0.0
        episode_goals = episode_length = 0
        episode_rows: List[Dict[str, Any]] = []
        last_losses: Dict[str, float] = {}
        start_time = time.time()
        start_step = int(agent.global_step)
        next_log_step = (
            (agent.global_step // int(args.log_every)) + 1
        ) * int(args.log_every)
        next_checkpoint_step = (
            (agent.global_step // int(args.checkpoint_every)) + 1
        ) * int(args.checkpoint_every)

        while agent.global_step < int(args.total_steps):
            if agent.global_step < int(agent.config.learning_starts):
                action = np.asarray(env.action_space.sample(), dtype=np.float64)
            else:
                action = agent.act(observation, deterministic=False)

            next_observation, reward, cost, terminated, truncated, info = env.step(action)
            executed_action = np.asarray(
                info.get("executed_action", action), dtype=np.float64
            )
            learning_reward = (
                float(reward) - float(args.train_cost_penalty) * float(cost)
            )
            agent.store(
                observation,
                # The critic must be trained on the action that generated the
                # transition.  This differs from the requested action whenever
                # --train_action_noise is nonzero.
                executed_action,
                learning_reward,
                next_observation,
                terminated=bool(terminated),
            )
            agent.global_step += 1
            agent.update_normalizer(next_observation)
            observation = next_observation
            episode_return += float(reward)
            episode_cost += float(cost)
            episode_goals += int(bool(info.get("goal_met", False)))
            episode_length += 1

            if (
                agent.global_step >= int(agent.config.learning_starts)
                and agent.replay.size >= int(agent.config.batch_size)
            ):
                for _ in range(int(agent.config.gradient_steps)):
                    last_losses = agent.update()

            if terminated or truncated:
                episode_rows.append({
                    "episode": agent.episodes,
                    "global_step": agent.global_step,
                    "return": episode_return,
                    "cost": episode_cost,
                    "cost_rate": episode_cost / max(episode_length, 1),
                    "goals": episode_goals,
                    "length": episode_length,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                })
                agent.episodes += 1
                observation, _ = env.reset(seed=int(args.seed + agent.episodes))
                agent.update_normalizer(observation)
                episode_return = episode_cost = 0.0
                episode_goals = episode_length = 0

            if agent.global_step >= next_log_step:
                elapsed = max(time.time() - start_time, 1e-9)
                recent = episode_rows[-10:]
                return10 = (
                    float(np.mean([row["return"] for row in recent]))
                    if recent else float("nan")
                )
                cost10 = (
                    float(np.mean([row["cost"] for row in recent]))
                    if recent else float("nan")
                )
                print(
                    f"[train:sac] step={agent.global_step}/{args.total_steps} "
                    f"episodes={agent.episodes} replay={agent.replay.size} "
                    f"return10={return10:.3f} cost10={cost10:.3f} "
                    f"actor_loss={last_losses.get('actor_loss', float('nan')):.4f} "
                    f"q1_loss={last_losses.get('q1_loss', float('nan')):.4f} "
                    f"alpha={last_losses.get('alpha', float(agent.alpha.item())):.4f} "
                    f"steps_per_s={(agent.global_step - start_step) / elapsed:.1f}",
                    flush=True,
                )
                next_log_step += int(args.log_every)

            if agent.global_step >= next_checkpoint_step:
                checkpoint = out_dir / f"sac_step_{agent.global_step}.pt"
                agent.save(
                    checkpoint,
                    env_id=args.env_id,
                    include_replay=bool(args.save_replay),
                )
                write_csv(out_dir / "sac_train_episodes.csv", episode_rows)
                next_checkpoint_step += int(args.checkpoint_every)

        final_checkpoint = out_dir / "sac_policy_final.pt"
        agent.save(
            final_checkpoint,
            env_id=args.env_id,
            include_replay=bool(args.save_replay),
        )

    write_csv(out_dir / "sac_train_episodes.csv", episode_rows)
    evaluation_rows, evaluation_summary = evaluate_nominal(agent, args)
    write_csv(out_dir / "sac_eval_episodes.csv", evaluation_rows)
    summary = {
        "algorithm": "SAC",
        "checkpoint": str(final_checkpoint),
        "total_steps": int(agent.global_step),
        "episodes": int(agent.episodes),
        "updates": int(agent.updates),
        "replay_size": int(agent.replay.size),
        "alpha_final": float(agent.alpha.item()),
        "mean_last_20_return": (
            float(np.mean([row["return"] for row in episode_rows[-20:]]))
            if episode_rows else None
        ),
        "mean_last_20_cost": (
            float(np.mean([row["cost"] for row in episode_rows[-20:]]))
            if episode_rows else None
        ),
        "last_update": last_losses,
        "deterministic_evaluation": evaluation_summary,
    }
    atomic_json(out_dir / "sac_train_summary.json", summary)
    atomic_json(
        out_dir / "sac_manifest.json",
        build_manifest(args, checkpoint=final_checkpoint),
    )
    print(json.dumps(json_safe(summary), indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train twin-critic Soft Actor-Critic on SafetyPointGoal2-v0."
    )
    parser.add_argument("--env_id", default="SafetyPointGoal2-v0")
    parser.add_argument("--frameskip_probability", type=float, default=1.0)
    parser.add_argument("--max_objects", type=int, default=32)
    parser.add_argument("--hazard_buffer", type=float, default=0.02)
    parser.add_argument("--vase_buffer", type=float, default=0.02)
    parser.add_argument("--agent_radius", type=float, default=0.10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_dir", default="runs/pointgoal2_sac_seed42")
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--alpha_learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replay_capacity", type=int, default=500_000)
    parser.add_argument("--learning_starts", type=int, default=10_000)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--target_update_interval", type=int, default=1)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--initial_alpha", type=float, default=0.20)
    parser.add_argument("--target_entropy", type=float, default=None)
    parser.add_argument(
        "--fixed_alpha", action="store_true",
        help="Disable automatic entropy tuning and keep initial_alpha fixed",
    )
    parser.add_argument("--observation_clip", type=float, default=10.0)
    parser.add_argument("--normalizer_freeze_steps", type=int, default=100_000)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--train_action_noise", type=float, default=0.0)
    parser.add_argument("--train_cost_penalty", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=5_000)
    parser.add_argument("--checkpoint_every", type=int, default=100_000)
    parser.add_argument("--save_replay", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume_replay", action="store_true")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_max_steps", type=int, default=1000)
    parser.add_argument("--eval_seed", type=int, default=1_000_042)
    parser.add_argument("--deterministic_torch", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not (0.0 < args.frameskip_probability <= 1.0):
        raise SystemExit("--frameskip_probability must lie in (0,1]")
    if args.env_id == "SafetyPointGoal2-v0" and args.max_objects < 20:
        raise SystemExit("SafetyPointGoal2 requires --max_objects >= 20")
    positive_counts = (
        args.total_steps, args.hidden, args.batch_size, args.replay_capacity,
        args.gradient_steps, args.target_update_interval, args.log_every,
        args.checkpoint_every,
    )
    if min(positive_counts) < 1:
        raise SystemExit("SAC step/network/buffer/update counts must be positive")
    if args.replay_capacity < args.batch_size:
        raise SystemExit("--replay_capacity must be >= --batch_size")
    if args.learning_starts < args.batch_size:
        raise SystemExit("--learning_starts must be >= --batch_size")
    if not (0.0 < args.gamma <= 1.0):
        raise SystemExit("--gamma must lie in (0,1]")
    if not (0.0 < args.tau <= 1.0):
        raise SystemExit("--tau must lie in (0,1]")
    if min(args.learning_rate, args.alpha_learning_rate, args.initial_alpha) <= 0.0:
        raise SystemExit("learning rates and initial alpha must be positive")
    if args.reward_scale <= 0.0 or args.max_grad_norm <= 0.0:
        raise SystemExit("reward scale and max grad norm must be positive")
    if min(
        args.normalizer_freeze_steps, args.eval_episodes, args.eval_max_steps
    ) < 0:
        raise SystemExit("normalizer/evaluation counts must be non-negative")
    if min(args.train_action_noise, args.train_cost_penalty) < 0.0:
        raise SystemExit("training noise and cost penalty must be non-negative")
    if args.resume and not Path(args.resume).is_file():
        raise SystemExit(f"--resume checkpoint does not exist: {args.resume}")
    if args.resume_replay and not args.resume:
        raise SystemExit("--resume_replay requires --resume")


def main() -> None:
    args = build_parser().parse_args()
    validate_arguments(args)
    train(args)


if __name__ == "__main__":
    main()
