# Autonomous Ship Course Keeping Controller

Two controllers for ship course-keeping, compared on the same problem which is holding a target heading under
wave noise, on a ship whose dynamics are unknown and change every episode.

1. **Reinforcement learning (PPO)**: a `RecurrentPPO` agent with an LSTM policy, trained by reward
   alone.
2. **System identification + MPC**: a two-stage controller. A GRU network (`SysIDNet`) estimates
   the ship parameters online, and a model predictive controller (solved with `OSQP`) computes the
   rudder command from those estimates.

![Course-correction demo](visuals/example.gif)

## Environment

The ship follows the discrete-time first-order Nomoto model:

$$ T \dot{r} + r = K \delta $$

where $r$ is the yaw rate, $\delta$ the rudder angle, $K$ the steering gain, and $T$ the turning
inertia (time constant). See [docs/nomoto_model.md](./docs/nomoto_model.md) for the derivation and
the discretization.

Every episode resets to a different ship. `K` and `T` are re-sampled from uniform ranges
(`K_range`, `T_range` in [conf/env/nomoto.yaml](./conf/env/nomoto.yaml)) and are never exposed to
the controller. The initial heading error is sampled within ±180°, and zero-mean Gaussian noise is
added to the yaw rate at each step to represent wave action.

The observation is `[heading_error, yaw_rate, rudder_angle, integral_heading_error]`. The integral
term allows a controller to remove steady-state offset. The action is a rudder command in
`[-1, 1]`, scaled to the ±35° mechanical limit.

The reward penalizes heading error, rudder angle, and rudder movement:

$$ R = -\left( w_1\,\psi^2 + w_2\,\delta^2 + w_3\,\Delta\delta^2 \right) $$

An episode ends after `max_steps`, or earlier once the heading has stayed within `bound_trial_rad`
(~2°) for `step_trial` consecutive steps. The count resets on any step outside the tolerance, so the
ship has to hold the course rather than cross it once.

## Controllers

**PPO** never identifies the ship explicitly. A single observation cannot distinguish one hull from
another, so the LSTM's recurrent state has to carry whatever the policy infers about `K` and `T`
from the sequence of observations so far. There is no model and no separate identification step.

**SysID + MPC** splits the problem in two. The GRU is trained first, on data collected while the
rudder moves at random, to predict the true `K`/`T` from a rolling window of `[yaw_rate, rudder]`
pairs. At run time the network re-estimates `K`/`T` at every step, and the MPC re-solves a
short-horizon QP from those estimates. It optimizes over the change in rudder rather than the
absolute angle, which keeps the steering smooth and reduces the rudder limit to a box constraint.
The full derivation is in [docs/mpc_formulation.md](./docs/mpc_formulation.md). Because the
estimates are explicit, the visualizer can plot them against the true `K`/`T`.

## Setup

Dependencies are managed with [`uv`](https://github.com/astral-sh/uv), on Python 3.13.

```bash
git clone https://github.com/Dogerion/Autonomous-Ship-Course-Keeping-Controller.git
cd Autonomous-Ship-Course-Keeping-Controller
uv sync
source .venv/bin/activate
```

## Usage

All modes run through `main.py`, configured by Hydra. Two options apply to every command:

- **`rl=`** selects the controller, `ppo` or `sysid_mpc`.
- **`model_name=`** names the checkpoint. It is the base name used to save during training and to
  load for evaluation and visualization (`models/{project_name}/{model_name}`). Give each
  experiment its own name, otherwise it overwrites the previous one.

Any other config value can be overridden the same way, for example
`python main.py rl=ppo mode=train rl.total_timesteps=500000 seed=7`.

```bash
# PPO
python main.py rl=ppo mode=train    model_name=coastal_run
python main.py rl=ppo mode=eval     model_name=coastal_run
python main.py rl=ppo mode=optimize                          # Optuna search over lr and gamma

# SysID + MPC
python main.py rl=sysid_mpc mode=train model_name=coastal_run
python main.py rl=sysid_mpc mode=eval  model_name=coastal_run
```

`mode=optimize` applies to PPO only. The SysID network is a supervised fit, and the MPC has no
learned parameters.

### Visualization

`mode=visualize` runs one episode with a trained controller and animates a top-down view of the ship
steering back onto course, next to its heading-error, rudder, and yaw-rate traces. For `sysid_mpc`
it also plots the network's estimated `K`/`T` against the true values.

```bash
python main.py rl=ppo       mode=visualize model_name=coastal_run env.max_steps=120
python main.py rl=sysid_mpc mode=visualize model_name=coastal_run env.max_steps=120
```

The animation is written as a GIF to `visuals/{project_name}/{model_name}/` before the live window
opens. The top-down path is illustrative: the Nomoto model tracks heading, not position, so the 2D
track is reconstructed by advancing the ship at a constant forward speed along its heading. Nothing
feeds back cross-track error, so the ship straightens out parallel to the target course rather than
rejoining it.

### Monitoring

Both controllers log to TensorBoard under `tensorboard_runs/{project_name}/{model_name}/`. PPO
writes the standard Stable-Baselines3 scalars (episode reward and length, policy and value losses,
success rate). SysID writes its per-epoch training loss as `sysid/mse_loss`.

```bash
tensorboard --logdir ./tensorboard_runs/
```

Evaluation runs report the mean and standard deviation of the episode reward to the console.

## Repository layout

```text
├── main.py                 # Entry point (dispatches on mode + agent)
├── pyproject.toml          # Dependencies (uv)
├── conf/                   # Hydra configs
│   ├── config.yaml         # Project name, seed, mode, model name
│   ├── env/nomoto.yaml     # Ship ranges, disturbance, reward weights, success criteria
│   └── rl/
│       ├── ppo.yaml        # RecurrentPPO, evaluation, and Optuna settings
│       └── sysid_mpc.yaml  # GRU SysID and MPC settings
├── src/
│   ├── env.py              # Gymnasium Nomoto environment
│   ├── visualize.py        # Episode animation
│   ├── utils.py            # Agent router (reads the selected rl config group)
│   └── agents/
│       ├── base_manager.py # Paths, seeding, rollout, visualization
│       ├── ppo_manager.py  # RecurrentPPO
│       └── mpc_manager.py  # SysIDNet (GRU) + OSQP MPC
├── docs/
│   ├── nomoto_model.md     # The ship model and its discretization
│   └── mpc_formulation.md  # From the model and cost to the QP
├── models/                 # Checkpoints (generated)
├── visuals/                # Saved episode GIFs (generated)
├── tensorboard_runs/       # TensorBoard logs (generated)
└── outputs/                # Hydra run logs (generated)
```
