# The First-Order Nomoto Model

This is the ship model implemented by `NomotoEnv` in [src/env.py](../src/env.py). The MPC baseline
predicts with the same model, so read the discretization at the end of this page before
[mpc_formulation.md](./mpc_formulation.md).

## Background

Steering a ship is a hydrodynamic problem in three degrees of freedom, namely surge (forward
motion), sway (sideways drift), and yaw (rotation). Solving all three together is costly and
requires hull coefficients that are rarely available.

In 1957 K. Nomoto showed that for course-keeping at a roughly constant forward speed, the three
coupled equations reduce to one transfer function from rudder angle $\delta$ to yaw rate $r$. Surge
is assumed constant and sway is absorbed into the yaw response, which leaves one ordinary
differential equation with two parameters. The model is still a common starting point for autopilot
design.

The result is second order,

$$ T_1 T_2 \ddot{r} + (T_1 + T_2)\dot{r} + r = K\left(\delta + T_3 \dot{\delta}\right) $$

but the fast mode contributes little at course-keeping frequencies. Reducing it to one effective
time constant $T = T_1 + T_2 - T_3$ gives the first-order form used here.

## Model equation

$$ T \dot{r}(t) + r(t) = K \delta(t) $$

| Symbol | Meaning | Units |
| --- | --- | --- |
| $r$ | Yaw rate, the rotational speed of the vessel | rad/s |
| $\dot{r}$ | Yaw acceleration | rad/s² |
| $\delta$ | Commanded rudder angle | rad |
| $K$ | Turning gain | 1/s |
| $T$ | Time constant, the rotational inertia | s |

Heading is the integral of the yaw rate.

$$ \dot{\psi}(t) = r(t) $$

Here $\psi$ is stored as the heading error, since the target course is fixed at zero. Steering onto
course therefore means driving $\psi \rightarrow 0$.

## Physical meaning of $K$ and $T$

Both parameters follow from the ship's yaw equation of motion. For a yaw-only model,

$$ (I_z - N_{\dot r})\,\dot r = N_r\,r + N_\delta\,\delta $$

where $I_z$ is the hull's yaw moment of inertia, $N_{\dot r}$ the added inertia from the water the
hull drags with it, $N_r$ the yaw damping, and $N_\delta$ the yaw moment produced per unit rudder
angle. Dividing through by $-N_r$ gives the Nomoto form.

$$ \underbrace{\frac{I_z - N_{\dot r}}{-N_r}}_{T}\,\dot r + r = \underbrace{\frac{N_\delta}{-N_r}}_{K}\,\delta $$

A directionally stable ship has $N_r < 0$, so both $K$ and $T$ come out positive. This is why
`SysIDNet` applies a Softplus to its output head. A negative estimate has no meaning, and a
near-zero $\hat{T}$ makes the $1/\hat{T}$ terms in the MPC matrices diverge.

### Turning gain $K$

$K$ describes the turning ability of the vessel. If the rudder is held at a constant angle
$\delta_{ss}$, the yaw acceleration eventually reaches zero and leaves a steady turn rate.

$$ r_{ss} = K \delta_{ss} $$

A high $K$ means the ship answers the rudder sharply. A low $K$ means it resists turning even at
full deflection. In non-dimensional form $K$ scales with ship length $L$ and forward speed $U$ as
$K = K'\,U/L$, so the same hull answers the rudder faster at speed.

### Time constant $T$

$T$ is the ship's rotational inertia divided by its yaw damping, and it sets how long the yaw rate
takes to build up to $r_{ss}$. After a step change in rudder angle, the yaw rate reaches about 63%
of its steady value in $T$ seconds.

A high $T$ describes a sluggish ship. A loaded tanker takes a long time to start turning, and just
as long to stop once the rudder is centred. A low $T$ describes a ship that reaches its steady turn
rate almost at once, such as a patrol boat. The non-dimensional form is $T = T'\,L/U$, so longer
hulls and lower speeds both make the response slower.

### Parameter ranges

$K$ and $T$ are re-sampled uniformly at every episode reset and are never shown to the controller,
so a controller has to work across the range rather than fit one ship. The ranges are set in
[conf/env/nomoto.yaml](../conf/env/nomoto.yaml).

| Parameter | Range | Meaning |
| --- | --- | --- |
| $K$ | 0.1 to 0.5 1/s | weak to strong rudder authority |
| $T$ | 5 to 30 s | quick to sluggish response |

## Discrete-time integration

The simulation advances in fixed steps of $dt$, 1 second by default, so the differential equations
are integrated numerically. Each call to `NomotoEnv.step` applies the rudder, then updates the
states in this order.

$$ \dot{r}_t = \frac{K \delta_t - r_t}{T} $$

$$ r_{t+1} = r_t + \dot{r}_t \, dt $$

$$ \psi_{t+1} = \psi_t + r_{t+1} \, dt $$

$$ I_{t+1} = I_t + \psi_{t+1} \, dt $$

The ordering matters. Heading is updated from the new yaw rate rather than the previous one, and the
integral term $I$ from the new heading, which makes the scheme semi-implicit rather than plain
forward Euler. The MPC's $A$ and $B$ matrices follow the same order, which is where their $dt^2$ and
$dt^3$ entries come from.

Wave action is applied last, as zero-mean Gaussian noise on the yaw rate.

$$ r_{t+1} \leftarrow r_{t+1} + w_t, \qquad w_t \sim \mathcal{N}(0, \sigma^2) $$

Since the noise is applied after the heading update, a disturbance at step $t$ reaches $\psi$ only
at step $t+1$.

## Assumptions and limitations

The model builds in the following assumptions.

- **The model is linear.** Rudder force saturates at large deflections on a real ship, and yaw
  damping is not linear in $r$. The model fits small course corrections and degrades on hard turns.
- **Forward speed is constant.** A real ship loses speed in a turn, which changes both $K$ and $T$
  while it is turning. Here they are fixed for the length of an episode.
- **Position is not modelled.** The state tracks heading only, so the model has no notion of
  cross-track error. The top-down view in the visualizer builds a path by advancing the ship at a
  constant speed along its heading. That path is drawn, not simulated.
- **Steering gear dynamics are omitted.** The commanded rudder angle takes effect at once, with no
  slew rate limit. The reward's $\Delta\delta$ penalty and the MPC's rate bound represent the cost
  of moving a real rudder quickly.
