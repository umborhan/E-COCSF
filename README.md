# E-COCSF: Enforcement-Aware Conformal Safety Filtering for Reinforcement Learning

**Adaptive safety-margin calibration from valid closed-loop evidence**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-0b8f87)](https://carla.org/)
[![Safety Gymnasium](https://img.shields.io/badge/Safety%20Gymnasium-SafetyPointGoal2--v0-purple)](https://github.com/PKU-Alignment/safety-gymnasium)
[![Task](https://img.shields.io/badge/Task-Safe%20Reinforcement%20Learning-orange)](#overview)
[![License](https://img.shields.io/badge/License-Add%20Before%20Release-lightgrey)](#license)

> **Paper:** Enforcement-Aware Conformal Safety Filtering for Reinforcement Learning  
> **Method:** Endogenous Closed-Loop Conformal Safety Filtering (E-COCSF)  
> **Target venue:** AAAI Conference on Artificial Intelligence  
> **Task:** Policy-agnostic runtime safety filtering for fixed RL policies  
> **Main idea:** Update an adaptive safety margin only when the requested margin verifiably governed the executed transition.

---

## Overview

Adaptive safety margins are part of the closed loop. A requested margin changes the feasible action set, the safety filter changes the executed action, and the resulting next state produces the residual used for later calibration. In deployed systems, infeasibility handling, restoration logic, actuator effects, or downstream guards can prevent the requested margin from governing the transition. Updating from such a transition attributes its residual to the wrong control semantics.

**E-COCSF** addresses this enforcement mismatch. It:

- perturbs the requested margin with bounded probing for local identifiability;
- verifies whether the requested margin was actually enforced;
- freezes calibration on invalid, restored, or constraint-changing transitions;
- uses only past data to screen whether the visited margin--risk response appears locally self-correcting; and
- reports invalid and audit-unsupported operation separately instead of hiding it inside a conditional average.

The safety filter wraps a fixed black-box policy. The policy can be replaced without changing the margin-calibration interface, provided that the environment supplies the state, executed action, barrier quantities, and transition diagnostics required by the filter.

> **Scope of the claims:** The prospective audit is an operational diagnostic, not a universal finite-sample certificate. Valid-transition exceedance is conditional on enforcement-valid transitions and is not equivalent to collision probability or an unconditional all-step safety guarantee.

---

## System Model Figure

<p align="center">
  <img src="./figures/E_COCSF.png" width="92%" alt="E-COCSF enforcement-aware closed-loop conformal safety-filter pipeline"/>
</p>

The standard adaptive loop updates its margin from every observed residual, even when the requested margin did not govern the action. E-COCSF adds three explicit mechanisms:

1. **Executed-margin attribution:** determine whether the requested margin governed the executed action;
2. **Validity gating:** update only from enforcement-valid transitions; and
3. **Prospective auditing:** use past randomized margins to screen excitation, negative local response, and target crossing before the current outcome is observed.

---

## Why E-COCSF?

General online conformal controllers already allow calibrated parameters to influence decisions, and performative risk control studies parameter-induced distribution change. The deployment gap addressed here is narrower: a safety filter can request one constraint margin while the environment receives an action produced under different semantics.

E-COCSF is designed to:

- prevent residuals from infeasible or restored transitions from being assigned to the requested margin;
- expose downstream action modifications that invalidate calibration attribution;
- detect when actuator saturation or restoration can flatten or reverse local margin--risk feedback;
- separate supported tracking from unsupported operation in both analysis and reporting;
- work as a runtime layer around fixed SAC, PPO, or other black-box policies; and
- retain detailed logs for infeasibility, restoration, validity, audit support, intervention, and exceedance.

---

## Method Overview

### 1. Requested-Margin Safety Filtering

At time `t`, the fixed policy proposes an action `u_pi`. E-COCSF maintains a base margin `q_t` and requests a bounded, randomly probed margin:

```text
q_requested = clip(q_t + probe_t, 0, q_max)
```

The runtime filter finds the action nearest to the policy proposal while enforcing the requested barrier margin together with action, rate, and second-difference limits. If strict projection is infeasible, a domain-specific restoration or fallback action may be used, but that transition is not treated as if the original margin had been enforced.

The realized certificate shortfall is:

```text
R_(t+1) = max(predicted_barrier(x_t, u_t) - observed_barrier(x_(t+1)), 0)
```

---

### 2. Enforcement-Validity Gating

Let `V_t = 1` only when all requested semantics remain true for the command sent to the environment. In particular:

- the strict projection was feasible;
- restoration did not substitute different constraint semantics;
- downstream guards did not invalidate the requested barrier condition; and
- the final action respected the requested action, rate, and smoothness envelope.

The margin recursion is:

```text
q_(t+1) = clip(
    q_t + eta * V_t * (soft_exceedance(R_(t+1) - q_requested) - epsilon),
    0,
    q_max,
)
```

Therefore, `q_(t+1) = q_t` whenever `V_t = 0`. Invalid transitions remain safety-relevant and stay in the logs; they are excluded only from calibration attribution.

---

### 3. Prospective Self-Correction Screening

Only one outcome is observed at each executed margin. E-COCSF therefore injects bounded margin probes and fits a local response using recent valid records. Before the current outcome is observed, the audit screens whether historical data provide:

- sufficient margin excitation;
- a negative local margin--risk slope; and
- a crossing of the target exceedance level `epsilon`.

If the screen fails, the step is marked audit-unsupported. This screen does not prove that the true response is self-correcting; the theoretical result is conditional on the true local response satisfying the stated dissipativity and drift assumptions.

---

### 4. Failure-Explicit Reporting

The implementation distinguishes quantities that should not be collapsed into a single coverage number:

| Quantity | Meaning |
|---|---|
| Valid-transition exceedance | Hard exceedance rate conditioned on `V_t = 1`. |
| Enforcement validity | Fraction of transitions for which the requested semantics were preserved. |
| Audit support | Fraction of transitions supported by the prospective local-response screen. |
| Strict infeasibility | Fraction for which the original strict projection was infeasible. |
| Restoration | Fraction using an alternative feasibility or fallback mechanism. |
| Barrier violation | Observed environmental barrier violation; distinct from conformal exceedance. |
| Collision / task success | End-to-end system outcomes, including all guards and environment dynamics. |

---

## Repository Layout

Recommended release structure:

```text
E-COCSF/
├── README.md
├── LICENSE
├── requirements.txt
│
├── code/
│   ├── ECLCS.py                       <- E-COCSF filter, audit, metrics, and wrappers
│   └── carla_train_eval.py            <- black-box SAC training and CARLA evaluation
│
├── figures/
│   └── E_COCSF.png                    <- method overview used in this README
│
├── graphs/
│   ├── fig1_carla_validity_audit.png
│   ├── fig2_seed_robustness.png
│   ├── fig3a_safetygym_runtime_diagnostics.png
│   ├── fig3b_cube_root_scaling.png
│   └── fig3c_gain_tradeoff.png
│
├── checkpoints/
│   ├── carla_sac_town10hd.pt
│   ├── safetygym_sac_seed42.zip
│   └── safetygym_ppo_seed42.zip
│
├── results/
│   ├── controlled/
│   ├── safetygym/
│   └── carla/
│
└── paper/
    └── E_COCSF_AAAI27_submission.tex
```

`carla_train_eval.py` imports `ECLCS` directly. Keep both Python files in the same directory or install the filter module on `PYTHONPATH`.

---

## Graphs and Visual Results

### Runtime Validity and Audit Behavior

<p align="center">
  <img src="./graphs/fig1_carla_validity_audit.png" width="82%" alt="CARLA enforcement validity and prospective audit behavior"/>
</p>

CARLA is primarily used to evaluate enforcement accounting and full guarded-pipeline behavior. Its reported valid-transition exceedance is well below the target, so it is not treated as an empirical demonstration of the theorem's interior-root regime.

---

### Safety Gymnasium Seed Robustness

<p align="center">
  <img src="./graphs/fig2_seed_robustness.png" width="82%" alt="Safety Gymnasium evaluation-seed robustness"/>
</p>

For the fixed SAC checkpoint, valid-transition exceedance remains below the prescribed `10%` target across the four evaluation seeds. These are evaluation seeds, not independently trained policies.

---

### Controlled Tracking and Gain Scaling

<p align="center">
  <img src="./graphs/fig3b_cube_root_scaling.png" width="48%" alt="Cube-root drift scaling"/>
  <img src="./graphs/fig3c_gain_tradeoff.png" width="48%" alt="Gain trade-off"/>
</p>

The controlled process isolates the tracking law. The fitted log--log slope is `0.352`, compared with the predicted `1/3`, with `R^2 = 0.999`. The gain sweep is U-shaped, with theoretical `eta* = 0.0138` and an empirical grid minimum at `0.0207`.

---

## Evaluation Environments

| Environment | Policy interface | Purpose | Paper protocol |
|---|---|---|---|
| Controlled endogenous process | Synthetic fixed controller | Isolate drift tracking and gain scaling | 20,000 steps; three seeds per drift level. |
| `SafetyPointGoal2-v0` | Fixed SAC and PPO checkpoints | Cross-backbone interface and actuator-noise evaluation | One checkpoint per backbone, trained for 500,000 steps with seed 42. |
| CARLA 0.9.15 | Fixed black-box SAC policy | Cross-town deployment, traffic, guards, restoration, and validity accounting | SAC trained in Town10HD for 250,000 steps; 20 evaluation episodes per town at 20 Hz. |

Safety Gymnasium evaluation uses seeds `{7, 42, 51, 72}`, ten `1000`-step episodes per seed and noise level, and action-noise magnitudes `{0, 0.05, 0.10, 0.20}`. CARLA evaluation uses approximately `500 m` routes under random weather in held-out Town02, held-out Town05, and in-domain Town10HD.

---

## Installation

### 1. Create a Python Environment

```bash
conda create -n ecocsf python=3.10 -y
conda activate ecocsf
```

### 2. Install Core Packages

```bash
python -m pip install --upgrade pip setuptools wheel
pip install numpy scipy pandas matplotlib
pip install torch
pip install safety-gymnasium stable-baselines3
```

Use the PyTorch installation command appropriate for the local CUDA driver when GPU acceleration is required.

### 3. Install CARLA 0.9.15

Install the CARLA simulator and its matching Python API. Add the API package to the active environment or `PYTHONPATH`, then verify:

```bash
python -c "import carla; print('CARLA Python API is available')"
```

The CARLA server must be started separately. Example:

```bash
./CarlaUE4.sh -carla-rpc-port=2000
```

Headless or off-screen flags depend on the CARLA installation and host GPU configuration.

---

## Quick Start

### 1. Clone and Enter the Repository

```bash
git clone https://github.com/<your-user-or-lab>/E-COCSF.git
cd E-COCSF
```

### 2. Verify the CARLA Connection

With the server running on port `2000`:

```bash
python code/carla_train_eval.py \
  --probe \
  --carla_port 2000 \
  --train_town Town10HD
```

The probe spawns one vehicle, applies throttle for 40 ticks, and reports whether the simulator vehicle moves.

### 3. Train the Black-Box CARLA Policy

The policy is trained without E-COCSF so that filter evaluation remains separate from policy learning:

```bash
python code/carla_train_eval.py \
  --train \
  --carla_port 2000 \
  --train_town Town10HD \
  --total_steps 250000 \
  --route_distance 500 \
  --weather_mode random \
  --no_manual_lead \
  --out_dir runs/carla_train_town10hd
```

The final checkpoint is written to:

```text
runs/carla_train_town10hd/policy_final.pt
```

### 4. Evaluate E-COCSF in CARLA

Town02 uses the paper's lighter `5V/0W` traffic condition:

```bash
python code/carla_train_eval.py \
  --eval \
  --carla_port 2000 \
  --checkpoint runs/carla_train_town10hd/policy_final.pt \
  --eval_towns Town02 \
  --episodes 20 \
  --route_distance 500 \
  --weather_mode random \
  --num_traffic_vehicles 5 \
  --num_walkers 0 \
  --no_manual_lead \
  --gain_schedule \
  --external_blockage_recovery \
  --out_dir runs/carla_eval_town02
```

Town05 and Town10HD use `20V/10W`:

```bash
python code/carla_train_eval.py \
  --eval \
  --carla_port 2000 \
  --checkpoint runs/carla_train_town10hd/policy_final.pt \
  --eval_towns Town05,Town10HD \
  --episodes 20 \
  --route_distance 500 \
  --weather_mode random \
  --num_traffic_vehicles 20 \
  --num_walkers 10 \
  --no_manual_lead \
  --gain_schedule \
  --external_blockage_recovery \
  --out_dir runs/carla_eval_dense
```

The script uses strict route-length validation and isolated cross-town map switching by default. Run each command on a dedicated CARLA server when possible.

### 5. Inspect Evaluation Outputs

```bash
ls -R runs/carla_eval_town02
ls -R runs/carla_eval_dense
```

The two result directories can be aggregated after both traffic conditions finish.

> The public release should include the exact controlled-process and Safety Gymnasium launchers used for the paper. Do not infer those command-line interfaces from the CARLA runner.

---

## Code Tour

```text
code/ECLCS.py
├── ECLCSConfig
│   └── margin, probing, validity, restoration, and audit configuration
├── MachineCard
│   └── action bounds, rate limits, jerk limits, and neutral action
├── build_filter()
│   └── constructs the domain-specific safety filter
├── ECLCSAgent
│   └── wraps an arbitrary state-to-action policy
├── transition update
│   ├── records the final executed action
│   ├── checks calibration validity
│   ├── computes the realized residual
│   └── updates or freezes the margin
└── audit and metrics
    ├── prospective local-response screen
    ├── supported/unsupported accounting
    └── audit-log export

code/carla_train_eval.py
├── reproducibility and CARLA map-switch utilities
├── GaussianPolicy, TwinQ, ReplayBuffer, and SACAgent
├── normalized 12-D route-aware policy observations
├── CarLA environment and black-box policy wrapper
├── independent SAC training
├── E-COCSF cross-town evaluation
│   ├── per-episode filter reset
│   ├── post-guard executed-action update
│   ├── feasibility and restoration diagnostics
│   └── validity, exceedance, collision, and task metrics
├── collision and deadlock trace export
└── command-line entry points: --probe, --train, and --eval
```

The neural policy receives normalized observations, while E-COCSF and the barrier functions receive the raw physical state. This separation is required for meaningful barrier and residual calculations.

---

## Main Reported Results

### Safety Gymnasium: Cross-Backbone Noise Evaluation

Values are mean ± standard deviation across four evaluation-seed means, with ten episodes per seed and noise level.

| Backbone | Noise | Return ↑ | Total cost ↓ | Goals/1000 ↑ | Any-goal ↑ | Barrier violation ↓ | Valid exceedance ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAC | 0.00 | 15.73 ± 3.29 | 129.60 ± 23.01 | 6.10 ± 1.44 | 97.5% | 23.18 ± 7.24% | 8.34 ± 0.56% |
| SAC | 0.05 | 16.90 ± 3.76 | 138.53 ± 29.23 | 6.75 ± 1.54 | 97.5% | 20.98 ± 4.71% | 8.30 ± 0.84% |
| SAC | 0.10 | 16.25 ± 2.29 | 155.90 ± 19.73 | 6.10 ± 0.94 | 97.5% | 22.88 ± 5.07% | 7.61 ± 0.48% |
| SAC | 0.20 | 16.04 ± 2.94 | 153.03 ± 29.26 | 6.18 ± 1.24 | 95.0% | 22.34 ± 4.71% | 7.79 ± 0.61% |
| PPO | 0.00 | 14.62 ± 2.15 | 162.53 ± 16.95 | 5.63 ± 0.90 | 97.5% | 21.06 ± 3.94% | 8.30 ± 0.70% |
| PPO | 0.05 | 15.21 ± 1.97 | 174.78 ± 36.36 | 5.70 ± 0.99 | 100% | 19.61 ± 4.49% | 8.37 ± 0.37% |
| PPO | 0.10 | 13.17 ± 3.09 | 165.23 ± 30.13 | 4.78 ± 1.43 | 97.5% | 21.98 ± 5.84% | 8.34 ± 0.57% |
| PPO | 0.20 | 13.20 ± 2.07 | 176.23 ± 29.06 | 4.98 ± 1.17 | 97.5% | 22.62 ± 4.80% | 8.17 ± 0.83% |

These results support interface portability across two fixed policy backbones. They do not establish robustness across independently trained policies.

### CARLA: Cross-Town Guarded-Pipeline Evaluation

Each town contains 20 episodes. Town02 and Town05 are held out from policy training; Town10HD is in-domain.

| Town | Train/test | Traffic | Return ↑ | Success ↑ | Route completion ↑ | Collision ↓ | Barrier violation ↓ | Valid exceedance ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Town02 | Held out | 5V/0W | 3917.41 ± 639.18 | 100% | 100% | 0% | 0.35% | 1.18% |
| Town05 | Held out | 20V/10W | 3815.05 ± 787.60 | 100% | 100% | 0% | 0.58% | 1.08% |
| Town10HD | In-domain | 20V/10W | 3581.00 ± 848.47 | 100% | 100% | 0% | 1.37% | 1.58% |
| **Overall** | — | Mixed | **3771.15** | **100%** | **100%** | **0% observed** | **0.77%** | **1.28%** |

The collision result describes the complete guarded CARLA pipeline and is not attributed to the conformal margin alone. Zero observed collisions in 60 simulated episodes is not a proof of zero collision probability.

---

## Reproducibility Settings

### Core Calibration Defaults Exposed by the CARLA Runner

```text
epsilon             = 0.10
eta                 = 0.03
q_init              = 0.10
q_max               = 5.00
zeta_max            = 0.02
probe_probability   = 0.10
dt                   = 0.05 s
seed                 = 42
```

### CARLA Policy Defaults

```text
state_dim            = 12
hidden               = 256
batch_size           = 256
replay_size          = 300000
gamma                = 0.99
tau                  = 0.005
initial SAC alpha    = 0.10
training warmup      = 2000 steps
checkpoint interval  = 25000 steps
```

For paper reproduction, record the command line, git commit, CARLA server version, GPU/driver information, simulator map assets, checkpoint checksum, and every random seed. Runtime filter state is reset between evaluation episodes; aggregates therefore summarize restarted trials rather than one measured infinite-horizon run.

---

## Output Files

### CARLA Training Outputs

| File | Description |
|---|---|
| `policy_step<N>.pt` | Periodic SAC checkpoint. |
| `policy_final.pt` | Final trained black-box policy. |
| `train_history.json` | Episode returns, lengths, violations, task progress, termination reasons, and optimizer diagnostics. |

### CARLA Evaluation Outputs

| File | Description |
|---|---|
| `ecocsf_eval_results.csv` | Per-town aggregate task, safety, validity, restoration, audit, and exceedance metrics. |
| `ecocsf_eval_episode_results.csv` | Per-episode task and filter diagnostics. |
| `ecocsf_eval_results.json` | Structured aggregates, episode records, checkpoint, target, and audit paths. |
| `ecocsf_eval_collision_traces.jsonl` | Pre-collision traces for episodes where a trace is available. |
| `ecocsf_eval_deadlock_traces.jsonl` | Tail filter traces for timeout/deadlock diagnosis. |
| `audit/` | Filter-generated audit logs for each evaluated town. |

### Figure Files

| File | Use |
|---|---|
| `figures/E_COCSF.png` | Method overview for the paper and README. |
| `graphs/fig1_carla_validity_audit.png` | CARLA validity and audit behavior. |
| `graphs/fig2_seed_robustness.png` | Safety Gymnasium evaluation-seed robustness. |
| `graphs/fig3a_safetygym_runtime_diagnostics.png` | Safety Gymnasium runtime diagnostics. |
| `graphs/fig3b_cube_root_scaling.png` | Controlled drift-scaling validation. |
| `graphs/fig3c_gain_tradeoff.png` | Controlled gain trade-off. |

Use PNG files for GitHub rendering and retain PDF versions for the paper.

---

## Interpreting the Results

E-COCSF should be interpreted as an **enforcement-aware calibration and accounting mechanism**, not as a standalone proof that every executed action is safe.

- `valid exceedance < epsilon` describes only transitions whose requested margin was verifiably enforced;
- invalid and restored transitions must be reported separately;
- the prospective audit screens empirical local behavior but does not certify the true response with universal finite-sample validity;
- the failure-explicit theorem is conditional on true local self-correction, bounded root drift, and controlled unsupported excursions;
- CARLA outcomes include shared traffic, collision, road-edge, turn, and recovery guards; and
- barrier violations, conformal exceedances, environmental costs, and collisions are different quantities.

In the reported Safety Gymnasium runs, enforcement validity is approximately `61--66%` and audit support is near `12%` under actuator noise. The low support fraction makes the unconditional failure-explicit bound numerically loose in that domain; it should be presented as transparent diagnostic accounting rather than a tight deployment certificate.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'ECLCS'`

Keep the files together:

```text
code/ECLCS.py
code/carla_train_eval.py
```

Then launch the runner from the repository root:

```bash
python code/carla_train_eval.py --help
```

### `ModuleNotFoundError: No module named 'carla'`

Install the Python API matching CARLA 0.9.15 and confirm that its package or egg is on `PYTHONPATH`. A version mismatch between the server and Python API can produce connection or serialization failures.

### CARLA Vehicle Does Not Move

Run the built-in movement probe:

```bash
python code/carla_train_eval.py --probe --carla_port 2000 --train_town Town10HD
```

If the vehicle remains stationary, check synchronous-mode ownership, physics state, server health, gear behavior, and the configured throttle floor before starting a long training run.

### Cross-Town Evaluation Crashes During Map Loading

The runner uses a crash-contained subprocess for map switching by default. Do not add `--no_isolated_map_switch` on packaged CARLA 0.9.15 unless the local installation has been validated for in-process map changes.

### Route Rejected Before Evaluation

The evaluator rejects routes shorter than `--eval_route_min_fraction` of the requested distance. Keep the default `0.98`, increase `--eval_route_retries`, or choose a route distance supported by the map. Do not silently relax this check for paper results.

### High Strict-Infeasibility or Low Calibration Validity

Inspect:

- requested and executed actions;
- restoration counts and slack;
- action rate and second-difference limits;
- downstream guard overrides;
- initial barrier feasibility; and
- filter audit logs.

Do not reduce the reported infeasibility rate by reclassifying restored transitions as enforcement-valid.

### CUDA Memory Errors

For CARLA policy training, reduce `--hidden`, `--batch_size`, or `--replay_size`, or run on CPU with `--device cpu` for a short smoke test. Any changed setting must be disclosed when reporting non-paper-aligned results.

---

## Citation

If you use this code or build on the method, cite the paper:

```bibtex
@misc{ecocsf2026,
  title  = {Enforcement-Aware Conformal Safety Filtering for Reinforcement Learning},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Manuscript under review}
}
```

Replace the anonymous author field and update the venue, pages, DOI, and year after publication.

---

## License

Add a license before public release. Common research-code options include:

- MIT for permissive reuse;
- Apache-2.0 for permissive reuse with explicit patent terms; or
- GPL-3.0 when derivative code should remain open source.

CARLA, Safety Gymnasium, PyTorch, Stable-Baselines3, and any pretrained assets remain subject to their own licenses.

---

## Acknowledgment

This repository supports the paper **Enforcement-Aware Conformal Safety Filtering for Reinforcement Learning**. The implementation is organized around traceable margin enforcement, prospective closed-loop diagnostics, failure-explicit reporting, and policy-agnostic evaluation in controlled simulation, Safety Gymnasium, and CARLA.
