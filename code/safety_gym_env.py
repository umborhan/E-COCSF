#!/usr/bin/env python3
"""Safety-Gymnasium adapter and CBF model for E-COCSF experiments.

The nominal policy observes the environment's standard flattened observation.
The safety filter additionally receives a fixed-size physical-state suffix with
the Point robot pose/velocity and the positions/radii of constrained objects.
This keeps the learned policy independent of privileged simulator state while
allowing the safety layer to use ground-truth geometry, as is standard in CBF
evaluation.

Tested target API: safety-gymnasium==1.0.0, SafetyPointGoal2-v0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


def _flat_finite(value: Any, *, name: str) -> Array:
    """Return a finite, one-dimensional float64 array."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{name} contains NaN/Inf values")
    return arr


def _scalar(value: Any, default: float = 0.0) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr[0]):
        return float(default)
    return float(arr[0])


@dataclass(frozen=True)
class PointGoalStateCodec:
    """Layout of the augmented filter state.

    State layout::

        [standard observation,
         agent_x, agent_y, yaw, velocity_x, velocity_y, yaw_rate,
         object_count,
         object_0_kind, object_0_x, object_0_y,
         object_0_vx, object_0_vy, object_0_size,
         object_0_yaw, object_0_margin, ...]

    ``kind`` is 0 for a circular hazard and 1 for a movable vase.  ``size`` is
    the hazard radius or vase box half-extent; ``margin`` is an additional
    geometric margin (agent radius plus the configured buffer for vases).
    Padding objects use a negative kind and are ignored by the barrier.
    """

    observation_dim: int
    max_objects: int = 32

    PHYSICAL_DIM: ClassVar[int] = 7
    OBJECT_DIM: ClassVar[int] = 8

    def __post_init__(self) -> None:
        if int(self.observation_dim) < 1:
            raise ValueError("observation_dim must be positive")
        if int(self.max_objects) < 1:
            raise ValueError("max_objects must be positive")

    @property
    def physical_start(self) -> int:
        return int(self.observation_dim)

    @property
    def objects_start(self) -> int:
        return self.physical_start + self.PHYSICAL_DIM

    @property
    def state_dim(self) -> int:
        return self.objects_start + self.OBJECT_DIM * int(self.max_objects)

    def observation(self, state: Sequence[float]) -> Array:
        x = _flat_finite(state, name="filter state")
        if x.size != self.state_dim:
            raise ValueError(f"Expected filter state_dim={self.state_dim}, got {x.size}")
        return x[: self.observation_dim].copy()

    def decode(self, state: Sequence[float]) -> Tuple[Array, float, Array, float, Array, Array]:
        x = _flat_finite(state, name="filter state")
        if x.size != self.state_dim:
            raise ValueError(f"Expected filter state_dim={self.state_dim}, got {x.size}")

        p = self.physical_start
        position = x[p : p + 2].copy()
        yaw = float(x[p + 2])
        velocity = x[p + 3 : p + 5].copy()
        yaw_rate = float(x[p + 5])
        object_count = int(np.clip(round(float(x[p + 6])), 0, self.max_objects))

        packed = x[self.objects_start :].reshape(self.max_objects, self.OBJECT_DIM)
        packed = packed[:object_count]
        valid = (
            np.isfinite(packed).all(axis=1)
            & (packed[:, 0] >= 0.0)
            & (packed[:, 5] >= 0.0)
            & (packed[:, 7] >= 0.0)
        )
        return (
            x[: self.observation_dim].copy(),
            yaw,
            position,
            yaw_rate,
            velocity,
            packed[valid].copy(),
        )

    def encode(
        self,
        observation: Sequence[float],
        position: Sequence[float],
        yaw: float,
        velocity: Sequence[float],
        yaw_rate: float,
        objects: Iterable[Sequence[float]],
    ) -> Array:
        obs = _flat_finite(observation, name="observation")
        if obs.size != self.observation_dim:
            raise ValueError(
                f"Expected observation_dim={self.observation_dim}, got {obs.size}"
            )
        pos = _flat_finite(position, name="agent position")
        vel = _flat_finite(velocity, name="agent velocity")
        if pos.size != 2 or vel.size != 2:
            raise ValueError("position and velocity must each contain two values")

        object_rows = np.asarray(list(objects), dtype=np.float64).reshape(
            -1, self.OBJECT_DIM
        )
        if object_rows.shape[0] > self.max_objects:
            # Keep nearest objects; far objects cannot be the active barrier.
            distances = np.linalg.norm(object_rows[:, 1:3] - pos[None, :], axis=1)
            object_rows = object_rows[np.argsort(distances)[: self.max_objects]]
        if object_rows.size and (
            not np.all(np.isfinite(object_rows))
            or np.any(object_rows[:, 0] < 0.0)
            or np.any(object_rows[:, 5] < 0.0)
            or np.any(object_rows[:, 7] < 0.0)
        ):
            raise RuntimeError("Invalid constrained-object geometry")

        out = np.zeros(self.state_dim, dtype=np.float64)
        out[: self.observation_dim] = obs
        p = self.physical_start
        out[p : p + 2] = pos
        out[p + 2] = float(yaw)
        out[p + 3 : p + 5] = vel
        out[p + 5] = float(yaw_rate)
        out[p + 6] = float(object_rows.shape[0])

        packed = out[self.objects_start :].reshape(self.max_objects, self.OBJECT_DIM)
        packed[:, 0] = -1.0
        if object_rows.shape[0]:
            packed[: object_rows.shape[0]] = object_rows
        return out


def load_point_dynamics(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load and validate a frozen dynamics model produced by ``identify``.

    Final paper evaluations should identify this model on disjoint seeds and
    then keep it frozen.  This avoids adapting the plant model on test data;
    only the proposed conformal margin remains online.
    """
    if path is None or not str(path).strip():
        return None
    model_path = Path(path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in {1, 2}:
        raise RuntimeError(f"Unsupported dynamics model schema: {model_path}")
    planar_features = 7 if schema_version == 1 else 10
    expected = {
        "position_coef": (2, planar_features),
        "velocity_coef": (2, planar_features),
        "yaw_delta_coef": (4,),
        "yaw_rate_coef": (4,),
    }
    model: Dict[str, Any] = dict(payload)
    for name, shape in expected.items():
        value = np.asarray(payload.get(name), dtype=np.float64)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise RuntimeError(
                f"Invalid {name} in {model_path}: expected finite shape {shape}, "
                f"got {value.shape}"
            )
        model[name] = value
    return model


@dataclass
class PointGoalBarrierModel:
    """Cost-aligned, velocity-aware barrier for Safety-Gymnasium PointGoal.

    Hazards use the benchmark's exact center-distance threshold.  Vases use a
    signed distance to their oriented square footprint, expanded by the Point
    robot radius.  Movable-vase velocity is included in one-step prediction.

    A frozen affine model identified on disjoint data can be supplied through
    ``dynamics``.  Without one, a documented physics prior is used.  Exogenous
    actuator noise is intentionally *not* inserted into ``h_hat``; it belongs in
    the commanded-action conformal residual.
    """

    HAZARD_KIND: ClassVar[int] = 0
    VASE_KIND: ClassVar[int] = 1

    codec: PointGoalStateCodec
    dt: float = 0.02
    time_headway: float = 0.10
    lookahead_steps: int = 8
    braking_deceleration: float = 3.0
    braking_distance_scale: float = 0.35
    acceleration_gain: float = 3.0
    linear_drag: float = 0.15
    yaw_tracking: float = 0.85
    max_speed: float = 3.0
    workspace_half_extent: float = 0.0
    workspace_buffer: float = 0.10
    dynamics: Optional[Mapping[str, Any]] = None
    action_dim: int = 2

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.time_headway < 0.0:
            raise ValueError("time_headway must be non-negative")
        if int(self.lookahead_steps) < 0:
            raise ValueError("lookahead_steps must be non-negative")
        if self.braking_deceleration <= 0.0:
            raise ValueError("braking_deceleration must be positive")
        if self.braking_distance_scale < 0.0:
            raise ValueError("braking_distance_scale must be non-negative")
        if self.acceleration_gain <= 0.0 or self.max_speed <= 0.0:
            raise ValueError("acceleration_gain and max_speed must be positive")
        if self.dynamics is not None:
            model_dt = float(self.dynamics.get("dt", self.dt))
            if not np.isclose(model_dt, self.dt, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"Dynamics-model dt={model_dt} does not match barrier dt={self.dt}"
                )
        names = [f"object_{i:02d}" for i in range(self.codec.max_objects)]
        if self.workspace_half_extent > 0.0:
            names.extend(
                ("workspace_right", "workspace_left", "workspace_top", "workspace_bottom")
            )
        self.COMPONENT_NAMES = tuple(names)

    def _unpack(
        self, state: Sequence[float]
    ) -> Tuple[Array, float, Array, float, Array, Array]:
        return self.codec.decode(state)

    @staticmethod
    def _oriented_box_sdf(
        point: Array, center: Array, yaw: float, half_extent: float, margin: float
    ) -> float:
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        rotation_world_to_local = np.asarray([[c, s], [-s, c]], dtype=np.float64)
        local = rotation_world_to_local @ (point - center)
        q = np.abs(local) - float(half_extent)
        outside = float(np.linalg.norm(np.maximum(q, 0.0)))
        inside = float(min(max(float(q[0]), float(q[1])), 0.0))
        return outside + inside - float(margin)

    def _components(
        self,
        position: Array,
        velocity: Array,
        objects: Array,
        *,
        predictive_margin: bool = True,
    ) -> Array:
        values: List[float] = []
        for row in objects:
            kind = int(round(float(row[0])))
            center = row[1:3]
            object_velocity = row[3:5]
            size = float(row[5])
            object_yaw = float(row[6])
            margin = float(row[7])

            delta = center - position
            distance = float(np.linalg.norm(delta))
            direction = delta / max(distance, 1e-9)
            closing_speed = float(direction @ (velocity - object_velocity))
            if kind == self.HAZARD_KIND:
                clearance = distance - size - margin
            elif kind == self.VASE_KIND:
                clearance = self._oriented_box_sdf(
                    position, center, object_yaw, size, margin
                )
            else:
                continue
            positive_closing = max(closing_speed, 0.0)
            kinematic_margin = 0.0
            if predictive_margin:
                kinematic_margin = (
                    self.time_headway * positive_closing
                    + self.braking_distance_scale
                    * positive_closing * positive_closing
                    / (2.0 * self.braking_deceleration)
                )
            values.append(float(clearance - kinematic_margin))

        if self.workspace_half_extent > 0.0:
            bound = float(self.workspace_half_extent - self.workspace_buffer)
            values.extend(
                (
                    bound - float(position[0]),
                    bound + float(position[0]),
                    bound - float(position[1]),
                    bound + float(position[1]),
                )
            )
        if not values:
            return np.asarray([1e3], dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    def geometric_barrier_values(self, state: Sequence[float]) -> Array:
        """Exact instantaneous cost/contact geometry, without prediction margin."""
        _, _, position, _, velocity, objects = self._unpack(state)
        return self._components(
            position, velocity, objects, predictive_margin=False
        )

    def geometric_h(self, state: Sequence[float]) -> float:
        return float(np.min(self.geometric_barrier_values(state)))

    def barrier_values(self, state: Sequence[float]) -> Array:
        """Predictive high-order state barrier under a bounded braking backup.

        Taking the component-wise minimum over the short backup rollout makes
        the next-state barrier depend on both Point actions.  In particular,
        yaw can steer the predicted path around an object instead of appearing
        action-insensitive in a one-step distance CBF.

        The analytic kinematic margin (time headway plus the v^2/2a braking
        distance) is applied only at the root state.  The rollout states are
        generated under the braking backup itself, so re-adding a braking
        margin at every rollout step would count the same stopping distance
        twice, shrink the controllable headroom of the action box to the order
        of the conformal margin, and force the filter into restoration.  Along
        the rollout the exact clearance geometry is used instead.
        """
        _, yaw, position, yaw_rate, velocity, objects = self._unpack(state)
        rollout_position = position.copy()
        rollout_velocity = velocity.copy()
        rollout_yaw = float(yaw)
        rollout_yaw_rate = float(yaw_rate)
        rollout_objects = objects.copy()
        values_over_time: List[Array] = []
        for step in range(int(self.lookahead_steps) + 1):
            values_over_time.append(
                self._components(
                    rollout_position,
                    rollout_velocity,
                    rollout_objects,
                    predictive_margin=(step == 0),
                )
            )
            if step == int(self.lookahead_steps):
                break
            backup_turn = float(np.clip(rollout_yaw_rate, -1.0, 1.0))
            backup_action = np.asarray([-1.0, backup_turn], dtype=np.float64)
            (
                rollout_position,
                rollout_yaw,
                rollout_velocity,
                rollout_yaw_rate,
            ) = self._predict_robot(
                rollout_position,
                rollout_yaw,
                rollout_velocity,
                rollout_yaw_rate,
                backup_action,
            )
            if rollout_objects.size:
                rollout_objects[:, 1:3] += (
                    self.dt * rollout_objects[:, 3:5]
                )
        return np.min(np.vstack(values_over_time), axis=0)

    def h_components(self, state: Sequence[float]) -> Array:
        return self.barrier_values(state)

    def component_names(self, state: Sequence[float]) -> Tuple[str, ...]:
        """Names aligned with the non-padded components in this state."""
        _, _, _, _, _, objects = self._unpack(state)
        names = [f"object_{index:02d}" for index in range(objects.shape[0])]
        if self.workspace_half_extent > 0.0:
            names.extend(
                ("workspace_right", "workspace_left", "workspace_top", "workspace_bottom")
            )
        return tuple(names) if names else ("free_space",)

    def h(self, state: Sequence[float]) -> float:
        return float(np.min(self.barrier_values(state)))

    @staticmethod
    def dynamics_features(
        yaw: float,
        velocity: Sequence[float],
        yaw_rate: float,
        action: Sequence[float],
        *,
        schema_version: int = 2,
    ) -> Tuple[Array, Array]:
        vel = np.asarray(velocity, dtype=np.float64).reshape(2)
        u = np.asarray(action, dtype=np.float64).reshape(2)
        heading = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
        if int(schema_version) == 1:
            planar = np.asarray(
                [vel[0], vel[1], yaw_rate, u[0] * heading[0],
                 u[0] * heading[1], u[1], 1.0],
                dtype=np.float64,
            )
        elif int(schema_version) == 2:
            lateral = np.asarray([-heading[1], heading[0]], dtype=np.float64)
            speed = float(np.linalg.norm(vel))
            planar = np.asarray(
                [
                    vel[0], vel[1], yaw_rate,
                    u[0] * heading[0], u[0] * heading[1],
                    u[1] * lateral[0], u[1] * lateral[1],
                    u[1] * speed * lateral[0],
                    u[1] * speed * lateral[1],
                    1.0,
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"Unsupported dynamics feature schema {schema_version}")
        angular = np.asarray([yaw_rate, u[1], u[0], 1.0], dtype=np.float64)
        return planar, angular

    def _predict_robot(
        self,
        position: Array,
        yaw: float,
        velocity: Array,
        yaw_rate: float,
        action: Array,
    ) -> Tuple[Array, float, Array, float]:
        if self.dynamics is not None:
            schema_version = int(self.dynamics.get("schema_version", 1))
            planar, angular = self.dynamics_features(
                yaw, velocity, yaw_rate, action,
                schema_version=schema_version,
            )
            position_next = position + np.asarray(
                self.dynamics["position_coef"], dtype=np.float64
            ) @ planar
            velocity_next = np.asarray(
                self.dynamics["velocity_coef"], dtype=np.float64
            ) @ planar
            yaw_delta = float(
                np.asarray(self.dynamics["yaw_delta_coef"], dtype=np.float64)
                @ angular
            )
            yaw_rate_next = float(
                np.asarray(self.dynamics["yaw_rate_coef"], dtype=np.float64)
                @ angular
            )
            yaw_next = float(
                np.arctan2(np.sin(yaw + yaw_delta), np.cos(yaw + yaw_delta))
            )
        else:
            yaw_rate_next = (
                (1.0 - self.yaw_tracking) * yaw_rate
                + self.yaw_tracking * float(action[1])
            )
            midpoint_yaw = yaw + 0.5 * self.dt * yaw_rate_next
            heading = np.asarray(
                [np.cos(midpoint_yaw), np.sin(midpoint_yaw)], dtype=np.float64
            )
            acceleration = (
                self.acceleration_gain * float(action[0]) * heading
                - self.linear_drag * velocity
            )
            velocity_next = velocity + self.dt * acceleration
            yaw_next = float(
                np.arctan2(
                    np.sin(yaw + self.dt * yaw_rate_next),
                    np.cos(yaw + self.dt * yaw_rate_next),
                )
            )
            position_next = position + 0.5 * self.dt * (velocity + velocity_next)

        speed = float(np.linalg.norm(velocity_next))
        if speed > self.max_speed:
            velocity_next = velocity_next * self.max_speed / max(speed, 1e-12)
        return position_next, yaw_next, velocity_next, yaw_rate_next

    def predict_next_state(self, state: Sequence[float], action: Sequence[float]) -> Array:
        obs, yaw, position, yaw_rate, velocity, objects = self._unpack(state)
        u = np.clip(_flat_finite(action, name="action"), -1.0, 1.0)
        if u.size != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {u.size}")

        position_next, yaw_next, velocity_next, yaw_rate_next = self._predict_robot(
            position, yaw, velocity, yaw_rate, u
        )
        predicted_objects = objects.copy()
        if predicted_objects.size:
            predicted_objects[:, 1:3] += self.dt * predicted_objects[:, 3:5]
        rows = [tuple(row.tolist()) for row in predicted_objects]
        return self.codec.encode(
            obs,
            position_next,
            yaw_next,
            velocity_next,
            yaw_rate_next,
            rows,
        )

    def h_hat(self, state: Sequence[float], action: Sequence[float]) -> float:
        return self.h(self.predict_next_state(state, action))

    def h_hat_components(
        self, state: Sequence[float], action: Sequence[float]
    ) -> Array:
        return self.h_components(self.predict_next_state(state, action))

    def shortfall(
        self,
        state: Sequence[float],
        action: Sequence[float],
        next_state: Sequence[float],
    ) -> float:
        return float(max(0.0, self.h_hat(state, action) - self.h(next_state)))


class SafetyPointGoalAdapter:
    """Thin, version-checked adapter around a Safety-Gymnasium environment."""

    def __init__(
        self,
        env_id: str = "SafetyPointGoal2-v0",
        *,
        render_mode: Optional[str] = None,
        action_noise: float = 0.0,
        frameskip_probability: float = 1.0,
        max_objects: int = 32,
        hazard_buffer: float = 0.02,
        vase_buffer: float = 0.02,
        agent_radius: float = 0.10,
    ) -> None:
        if action_noise < 0.0:
            raise ValueError("action_noise must be non-negative")
        if not (0.0 < frameskip_probability <= 1.0):
            raise ValueError("frameskip_probability must lie in (0, 1]")
        if min(hazard_buffer, vase_buffer, agent_radius) < 0.0:
            raise ValueError("buffers/radius must be non-negative")

        try:
            import safety_gymnasium as safety_gymnasium
        except ImportError as exc:  # pragma: no cover - depends on user environment
            raise RuntimeError(
                "Safety-Gymnasium is not installed. Run: "
                "python -m pip install safety-gymnasium==1.0.0"
            ) from exc

        # Action noise is applied in ``step`` rather than passed through the
        # task config.  Older Safety-Gymnasium releases reject ``action_noise``
        # as a task key, and explicit injection also lets us record the exact
        # actuator command used by the conformal update.
        make_kwargs: Dict[str, Any] = {"render_mode": render_mode}
        if not np.isclose(frameskip_probability, 1.0):
            make_kwargs["config"] = {
                "sim_conf.frameskip_binom_p": float(frameskip_probability),
            }
        self.env = safety_gymnasium.make(env_id, **make_kwargs)
        self.env_id = str(env_id)
        self.render_mode = render_mode
        self.action_noise = float(action_noise)
        self.hazard_buffer = float(hazard_buffer)
        self.vase_buffer = float(vase_buffer)
        self.agent_radius = float(agent_radius)

        obs_space = self.env.observation_space
        action_space = self.env.action_space
        if len(getattr(obs_space, "shape", ())) != 1:
            self.close()
            raise RuntimeError(
                f"{env_id} must expose a flat vector observation; got {obs_space}"
            )
        if tuple(getattr(action_space, "shape", ())) != (2,):
            self.close()
            raise RuntimeError(f"PointGoal requires a two-dimensional action; got {action_space}")
        if not (
            np.allclose(np.asarray(action_space.low), -1.0)
            and np.allclose(np.asarray(action_space.high), 1.0)
        ):
            self.close()
            raise RuntimeError(f"Expected Point action bounds [-1,1], got {action_space}")

        self.observation_dim = int(obs_space.shape[0])
        self.action_dim = 2
        self.codec = PointGoalStateCodec(self.observation_dim, max_objects=max_objects)
        self.last_observation: Optional[Array] = None
        self.last_requested_action: Optional[Array] = None
        self.last_executed_action: Optional[Array] = None
        self._action_rng: Optional[np.random.Generator] = None

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def task(self):
        builder = self.env.unwrapped
        if not hasattr(builder, "task"):
            raise RuntimeError("Unsupported Safety-Gymnasium version: env.unwrapped.task is missing")
        return builder.task

    @property
    def control_dt(self) -> float:
        """Return the MuJoCo control interval used by one environment step."""
        task = self.task
        # Safety-Gymnasium's ``task.dt`` is the MuJoCo physics timestep.  One
        # policy/environment step advances a binomial number of physics steps;
        # with the publication setting p=1 this is exactly frameskip_binom_n.
        # Prefer this control interval over task.dt (0.002 s in PointGoal2).
        try:
            model = getattr(task, "model", None)
            if model is None:
                model = getattr(getattr(task, "engine", None), "model", None)
            timestep = float(model.opt.timestep)
            simulation = getattr(task, "sim_conf", None)
            if simulation is None:
                simulation = getattr(task, "simulation_conf", None)
            trials = int(getattr(simulation, "frameskip_binom_n", 1))
            probability = float(getattr(simulation, "frameskip_binom_p", 1.0))
            value = timestep * trials * probability
            if np.isfinite(value) and value > 0.0:
                return value
        except Exception:
            pass
        for owner in (task, getattr(task, "engine", None)):
            if owner is not None and hasattr(owner, "dt"):
                value = float(getattr(owner, "dt"))
                if np.isfinite(value) and value > 0.0:
                    return value
        return 0.02

    def _agent_state(self) -> Tuple[Array, float, Array, float]:
        task = self.task
        try:
            body = task.data.body("agent")
            position = np.asarray(body.xpos[:2], dtype=np.float64).copy()
            matrix = np.asarray(body.xmat, dtype=np.float64).reshape(3, 3)
            yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
            qvel = np.asarray(task.data.qvel, dtype=np.float64).reshape(-1)
            if qvel.size < 3:
                raise RuntimeError("Point model qvel has fewer than three entries")
            velocity = qvel[:2].copy()
            yaw_rate = float(qvel[2])
        except Exception as exc:
            raise RuntimeError("Could not read Point robot ground-truth state") from exc
        if not np.all(np.isfinite(np.r_[position, yaw, velocity, yaw_rate])):
            raise RuntimeError("Point robot physical state contains NaN/Inf")
        return position, yaw, velocity, yaw_rate

    @staticmethod
    def _body_planar_state(owner: Any, name: str) -> Tuple[Array, Array, float]:
        """Read body position, translational velocity and yaw across v1 APIs."""
        data = getattr(owner, "data", None)
        model = getattr(owner, "model", None)
        if data is None or model is None:
            engine = getattr(owner, "engine", None)
            data = getattr(engine, "data", data)
            model = getattr(engine, "model", model)
        if data is None or model is None:
            raise RuntimeError(f"No MuJoCo model/data available for body {name}")
        body = data.body(name)
        position = np.asarray(body.xpos[:2], dtype=np.float64).copy()
        matrix = np.asarray(body.xmat, dtype=np.float64).reshape(3, 3)
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        try:
            from safety_gymnasium.utils.task_utils import get_body_xvelp

            velocity = np.asarray(
                get_body_xvelp(model, data, name), dtype=np.float64
            ).reshape(-1)[:2]
        except Exception:
            # MuJoCo cvel stores rotational followed by translational velocity.
            cvel = np.asarray(body.cvel, dtype=np.float64).reshape(-1)
            velocity = cvel[3:5] if cvel.size >= 5 else np.zeros(2, dtype=np.float64)
        if not np.all(np.isfinite(np.r_[position, velocity, yaw])):
            raise RuntimeError(f"Body {name} state contains NaN/Inf")
        return position, velocity.copy(), yaw

    def _constrained_objects(self) -> List[Tuple[float, ...]]:
        task = self.task
        objects: List[Tuple[float, ...]] = []

        hazards = getattr(task, "hazards", None)
        if hazards is not None and bool(getattr(hazards, "is_constrained", False)):
            radius = float(getattr(hazards, "size", 0.2))
            for pos in hazards.pos:
                p = np.asarray(pos, dtype=np.float64).reshape(-1)
                objects.append(
                    (
                        float(PointGoalBarrierModel.HAZARD_KIND),
                        float(p[0]),
                        float(p[1]),
                        0.0,
                        0.0,
                        radius,
                        0.0,
                        self.hazard_buffer,
                    )
                )

        vases = getattr(task, "vases", None)
        if vases is not None and bool(getattr(vases, "is_constrained", False)):
            half_extent = float(getattr(vases, "size", 0.1))
            count = int(getattr(vases, "num", len(getattr(vases, "pos", ()))))
            for index in range(count):
                position, velocity, yaw = self._body_planar_state(
                    vases, f"vase{index}"
                )
                objects.append(
                    (
                        float(PointGoalBarrierModel.VASE_KIND),
                        float(position[0]),
                        float(position[1]),
                        float(velocity[0]),
                        float(velocity[1]),
                        half_extent,
                        yaw,
                        self.agent_radius + self.vase_buffer,
                    )
                )
        return objects

    def filter_state(self, observation: Optional[Sequence[float]] = None) -> Array:
        if observation is None:
            if self.last_observation is None:
                raise RuntimeError("reset() must be called before filter_state()")
            observation = self.last_observation
        position, yaw, velocity, yaw_rate = self._agent_state()
        return self.codec.encode(
            observation,
            position,
            yaw,
            velocity,
            yaw_rate,
            self._constrained_objects(),
        )

    def reset(self, *, seed: int) -> Tuple[Array, Dict[str, Any]]:
        # Gymnasium does not guarantee that reset(seed=...) also seeds the
        # action space. Explicit seeding makes random warm-up actions exactly
        # reproducible across independent training runs.
        self.action_space.seed(int(seed))
        # A separate deterministic stream avoids coupling task layout draws to
        # actuator-noise draws. Reusing the episode seed across conditions gives
        # paired layouts and paired standard-normal perturbations.
        self._action_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 0xEC0C5F])
        )
        self.last_requested_action = None
        self.last_executed_action = None
        observation, info = self.env.reset(seed=int(seed))
        self.last_observation = _flat_finite(observation, name="reset observation")
        return self.last_observation.copy(), dict(info)

    def step(self, action: Sequence[float]):
        requested = np.clip(
            _flat_finite(action, name="environment action"), -1.0, 1.0
        )
        if requested.size != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got {requested.size}"
            )
        if self._action_rng is None:
            raise RuntimeError("reset() must be called before step()")
        perturbation = np.zeros(self.action_dim, dtype=np.float64)
        if self.action_noise > 0.0:
            perturbation = self._action_rng.normal(
                loc=0.0, scale=self.action_noise, size=self.action_dim
            )
        executed = np.clip(requested + perturbation, -1.0, 1.0)
        self.last_requested_action = requested.copy()
        self.last_executed_action = executed.copy()

        observation, reward, cost, terminated, truncated, info = self.env.step(executed)
        self.last_observation = _flat_finite(observation, name="step observation")
        step_info = dict(info)
        cost_components = {
            str(key): _scalar(value)
            for key, value in step_info.items()
            if str(key).startswith("cost_")
        }
        step_info.update(
            {
                "requested_action": requested.tolist(),
                "executed_action": executed.tolist(),
                "action_perturbation": (executed - requested).tolist(),
                "cost_components": cost_components,
            }
        )
        return (
            self.last_observation.copy(),
            _scalar(reward),
            _scalar(cost),
            bool(terminated),
            bool(truncated),
            step_info,
        )

    def close(self) -> None:
        if hasattr(self, "env"):
            self.env.close()

    def __enter__(self) -> "SafetyPointGoalAdapter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
