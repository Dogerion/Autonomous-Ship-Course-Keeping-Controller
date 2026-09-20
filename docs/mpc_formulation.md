# QP Formulation for the Nomoto MPC

This page derives the quadratic program that OSQP solves in
[src/agents/mpc_manager.py](../src/agents/mpc_manager.py) from the Nomoto model and the control
cost. It assumes the discretization given at the end of [nomoto_model.md](./nomoto_model.md).

## Control loop

The controller runs one cycle per environment step.

1. `SysIDNet` estimates $\hat{K}$ and $\hat{T}$ from the recent history of yaw rate and rudder.
2. The MPC rebuilds its prediction matrices from those estimates.
3. The QP is solved over a horizon of $N$ steps.
4. Only the first rudder move of the solution is applied, and the rest is discarded.

The cycle repeats at the next step with updated estimates and a new measured state.

## State and control

With $\delta$ as the control, the rudder-rate penalty cannot be written without coupling consecutive
controls in the cost. That coupling puts off-diagonal blocks in the Hessian, and it leaves the
rudder limit as a constraint on a quantity the solver tracks only implicitly.

The delta-input form avoids both problems. The rudder angle is carried inside the state, and the
control becomes the change in rudder.

$$ x_k = \begin{bmatrix} \psi_k \\ r_k \\ I_{\psi, k} \\ \delta_{k-1} \end{bmatrix}, \qquad u_k = \begin{bmatrix} v_k \end{bmatrix}, \qquad \delta_k = \delta_{k-1} + v_k $$

Penalizing rudder movement then reduces to a penalty on $v_k$, the mechanical rudder limit becomes a
bound on the fourth state, and the Hessian stays block-diagonal. The formulation carries one more
state in exchange.

## State-space matrices

Substitute the estimates $\hat{K}$ and $\hat{T}$ into the environment's update order and keep
$\delta_k = \delta_{k-1} + v_k$ symbolic.

$$ r_{k+1} = \left(1 - \frac{dt}{\hat{T}}\right) r_k + \frac{\hat{K}\,dt}{\hat{T}}\,\delta_k $$

$$ \psi_{k+1} = \psi_k + dt\,r_{k+1} = \psi_k + \left(dt - \frac{dt^2}{\hat{T}}\right) r_k + \frac{\hat{K}\,dt^2}{\hat{T}}\,\delta_k $$

$$ I_{\psi,k+1} = I_{\psi,k} + dt\,\psi_{k+1} = dt\,\psi_k + \left(dt^2 - \frac{dt^3}{\hat{T}}\right) r_k + I_{\psi,k} + \frac{\hat{K}\,dt^3}{\hat{T}}\,\delta_k $$

Each equation has one more factor of $dt$ than the one above it, because each state is integrated
from the already-updated state below it. Separating $\delta_k$ into $\delta_{k-1}$, which is a state
and therefore belongs in $A$, and $v_k$, which is the control and belongs in $B$, gives
$x_{k+1} = A x_k + B u_k$.

$$
A = \begin{bmatrix}
1 & dt - \frac{dt^2}{\hat{T}} & 0 & \frac{\hat{K} \cdot dt^2}{\hat{T}} \\
0 & 1 - \frac{dt}{\hat{T}} & 0 & \frac{\hat{K} \cdot dt}{\hat{T}} \\
dt & dt^2 - \frac{dt^3}{\hat{T}} & 1 & \frac{\hat{K} \cdot dt^3}{\hat{T}} \\
0 & 0 & 0 & 1
\end{bmatrix}
, \qquad
B = \begin{bmatrix}
\frac{\hat{K} \cdot dt^2}{\hat{T}} \\
\frac{\hat{K} \cdot dt}{\hat{T}} \\
\frac{\hat{K} \cdot dt^3}{\hat{T}} \\
1
\end{bmatrix}
$$

The last row implements $\delta_k = \delta_{k-1} + v_k$, so the rudder angle carries forward and the
control adds to it.

Since $A$ and $B$ depend on $\hat{K}$ and $\hat{T}$, they change at every step. Any error in the
estimates is a modelling error that the controller cannot observe.

## Cost function

The per-step penalty is quadratic in state and control.

$$ \text{Cost}_k = x_k^{\top} Q x_k + u_k^{\top} R u_k = w_1 \psi_k^2 + w_2 \delta_{k-1}^2 + w_3 v_k^2 $$

$$
Q = \begin{bmatrix} w_1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & w_2 \end{bmatrix}, \qquad R = \begin{bmatrix} w_3 \end{bmatrix}
$$

Only heading error and rudder angle are penalized in $Q$. Yaw rate is left unpenalized by design,
because a ship far off course has to develop a yaw rate in order to correct, and a penalty on $r$
would oppose the turn. The integral term is carried in the state so that the model stays consistent
with the environment's observation, but it is not penalized either.

The weights $w_1$, $w_2$, and $w_3$ are read from the same `conf/env/nomoto.yaml` block that defines
the environment's reward. The MPC therefore minimizes the negative of the reward that PPO is trained
to maximize. The two controllers differ in method, not in objective.

Summed over the horizon, the objective is

$$ J = \sum_{k=0}^{N-1} \left( x_k^{\top} Q x_k + u_k^{\top} R u_k \right) + x_N^{\top} Q_N x_N $$

### Terminal cost

$Q_N = Q$, with no terminal multiplier and no infinite-horizon approximation such as a DARE
solution. Under wave noise, a heavy weight on the final predicted state makes the controller commit
to a prediction whose accuracy degrades over the horizon, which produces chattering. Using the same
terminal and per-step cost keeps the behaviour smooth.

## OSQP standard form

OSQP minimizes one quadratic over one decision vector.

$$ \min_z \quad \tfrac{1}{2} z^{\top} P z + q^{\top} z \qquad \text{subject to} \qquad l \leq A_{\text{osqp}} z \leq u $$

The states and controls of the whole horizon are stacked into $z$, states first.

$$ z = \begin{bmatrix} x_0 & \cdots & x_N & u_0 & \cdots & u_{N-1} \end{bmatrix}^{\top} $$

With $n_x = 4$, $n_u = 1$, and the default $N = 20$, this gives
$(N{+}1)\,n_x + N\,n_u = 104$ unknowns.

**The Hessian $P$** is block-diagonal, holding $N$ copies of $Q$, then $Q_N$, then $N$ copies of
$R$. There are no off-diagonal blocks, because moving the rudder angle into the state removed the
only coupling between consecutive steps.

$$
P = \operatorname{blkdiag}\left( \underbrace{Q, \ldots, Q}_{N}, \; Q_N, \; \underbrace{R, \ldots, R}_{N} \right)
$$

**The linear term $q$ is zero.** Every target in this problem is the origin, meaning zero heading
error, zero rudder angle, and zero rudder movement. A non-zero reference, such as a rudder offset
held against a steady current, would appear in $q$.

OSQP's objective carries a factor of $\frac{1}{2}$ that $J$ does not. The code inserts $Q$ and $R$
into $P$ without rescaling, so the solver minimizes $J/2$. Since $q = 0$, this scales every term
equally and leaves the minimizer unchanged.

## Constraints

All constraints are stacked into the single system $l \leq A_{\text{osqp}} z \leq u$. Equalities are
written by setting the lower and upper bound to the same value.

**Initial state (equality).** The horizon is fixed to the ship's current measured state, with the
rudder angle currently commanded placed in the fourth state.

$$ x_0 = \begin{bmatrix} \psi_0 & r_0 & I_{\psi,0} & \delta_{\text{prev}} \end{bmatrix}^{\top} $$

Because $x_0$ is fixed, its cost terms are constants. The free variables are $\psi_1 \ldots \psi_N$
and $\delta_0 \ldots \delta_{N-1}$, which are the headings and rudder angles the ship would pass
through over the horizon.

**Dynamics (equality).** Every later state has to obey the model.

$$ x_{k+1} - A x_k - B u_k = 0, \qquad k = 0 \ldots N-1 $$

**Rudder angle (inequality).** The mechanical limit applies to the fourth state at every step.

$$ -\delta_{\max} \leq x_{4,k} \leq \delta_{\max}, \qquad k = 0 \ldots N $$

**Rudder rate (inequality).** The control is bounded in the same way.

$$ -v_{\max} \leq v_k \leq v_{\max}, \qquad k = 0 \ldots N-1 $$

The code sets $v_{\max} = \delta_{\max}$, so one step may move the rudder by at most a full
deflection. This is a loose bound, and it is the point at which a steering-gear slew rate would be
imposed. Tightening it models a rudder that cannot move faster than a few degrees per second.

For $N = 20$ this produces 84 equality rows and 41 inequality rows, 125 in total.

## Receding-horizon execution

The solver returns an optimal sequence $v_0^{*} \ldots v_{N-1}^{*}$, of which only the first element
is used.

$$ \delta_{\text{cmd}} = \delta_{\text{prev}} + v_0^{*} $$

The manager scales this back into the environment's normalized `[-1, 1]` action range and clips it.
The remaining $N-1$ moves are discarded, and the next step re-solves from the new measured state and
new parameter estimates. Re-planning from the measured state at every step allows the controller to
absorb both wave disturbances and its own model error.

If OSQP returns any status other than `solved`, the manager commands zero rudder for that step and
continues. Centring the rudder is the safe fallback, since the ship coasts and the next solve starts
from a new state.

## Implementation notes

- $A$ and $B$ change at every step with the parameter estimates, so the problem is rebuilt and OSQP
  is set up from scratch on each solve rather than warm-started. This is fast enough at $N = 20$,
  and warm-starting would be the first thing to add if the horizon is increased.
- The formulas above keep $dt$ symbolic. The default config uses $dt = 1$ s, which makes the $dt^2$
  and $dt^3$ entries equal to $dt$. They are not interchangeable in general, and the code keeps them
  separate.
- The horizon $N$ is `mpc.prediction_horizon` in [conf/rl/sysid_mpc.yaml](../conf/rl/sysid_mpc.yaml).
  A longer horizon looks further ahead at a cost that grows with the size of the QP, and beyond a
  certain length the extra steps rest on parameter estimates that are not accurate enough to be
  useful.
