---
layout: page
title: "Learning Lyapunov Functions for Stable Nonlinear Control"
---

# Learning Lyapunov Functions for Stable Nonlinear Control

**Authors:** Aneri Muni · Ria Arora · Kaustubh Mani \
**Repo:** [github.com/amuni3/RobustifAI](https://github.com/amuni3/RobustifAI/tree/final-final-sub) \
**Based on:** Neural Lyapunov Control — Chang, Roohi & Gao (NeurIPS 2019) \
**Stack:** PyTorch · dReal (SMT) · Soft Actor-Critic · LQR · SOS/SDP

---

> Chang et al. (2019) jointly learn a neural-network control policy and a neural Lyapunov function, checked against an SMT solver (dReal) that returns a formal counterexample whenever the Lyapunov conditions are violated on some state. We reproduce their Inverted Pendulum and Wheeled Vehicle results, then push on three things the paper leaves open: it never measures task performance (only stability), the falsifier returns one counterexample per call, and the method is never tested past a 2-dimensional state space.
>
> **Headline findings.** Replacing the LQR-initialized linear controller with a *frozen* SAC policy roughly halves task cost (return of −340 vs. −725 for the paper's method) while still yielding a valid, more circular region of attraction — but letting the Lyapunov loss update that same policy's weights collapses task performance (return drops to −1767), exposing a stability/performance tradeoff the original Lyapunov risk objective has no term for. Splitting the falsifier's domain into a 5×5 grid nearly halves training time (32.4s → 26.5s) without materially growing the region of attraction. And the method breaks down past 2 state dimensions: a 2-link (4D) pendulum never converges to a valid Lyapunov function, and a 3-link (6D) pendulum is computationally intractable for the SMT verifier.

---

## The Gap

A model-based controller is only as trustworthy as its guarantees. For a nonlinear dynamical system ẋ = f(x, t), a **Lyapunov function** V(x) — positive everywhere except at the equilibrium, and strictly decreasing along trajectories — certifies that the system converges to that equilibrium from anywhere inside a **Region of Attraction (ROA)**, without ever having to integrate the dynamics forward. The catch is that finding a valid V(x) for a general nonlinear system is, in the authors' words, "more of an art than a science": convex-optimization approaches (SOS/SDP) require polynomial dynamics, and hand-designed candidates rarely scale.

Chang et al. (2019) replace the art with a learner: a neural network proposes both the control policy and the Lyapunov candidate, and an SMT solver (dReal) formally checks the candidate against the Lyapunov conditions over the *entire* continuous domain — not just a finite sample of states. When the check fails, dReal returns an actual counterexample state, which is folded back into training. This loop is what makes the guarantee different from a normal training curve going down: the certificate is either verified over the whole domain or it isn't.

<div align="center">
  <figure>
    <img src="/projects/assets/nlc/fig1_loop.png" alt="Learner and verifier loop" width="800">
    <figcaption>Learner (Synthesizer) and Verifier loop used to design formal safety certificates. A failed verification returns a counterexample that is added back into training.</figcaption>
  </figure>
</div>

We reproduce the paper's core results, then focus the rest of the project on three places the original method is incomplete or untested.

---

## Background

### Lyapunov Stability

For ẋ = f(x, t) with equilibrium at the origin, a valid Lyapunov function V(x) must satisfy:

```
V(0) = 0,      V(x) > 0,      L_fu V(x) ≤ 0
```

where L_fu V(x) is the Lie derivative of V along the closed-loop dynamics — informally, "does V decrease as the system evolves under this controller." If such a V exists on a domain, the origin is a stable equilibrium there; if the last inequality is strict, it's asymptotically stable.

### Verification via SMT

Lyapunov conditions have to hold over the *entire* continuous domain, which is a global, non-convex feasibility problem — NP-hard in general. The paper uses **dReal**, a δ-complete SMT (Satisfiability Modulo Theories) solver: given a formula, it returns either UNSAT (no violation exists) or a δ-SAT counterexample (a specific state that violates the conditions, up to a user-specified numerical tolerance δ). If the falsifier can't find a violation, the Lyapunov conditions provably hold over the whole domain — not just wherever training happened to sample.

### The Lyapunov Risk

The learner and Lyapunov candidate share parameters (θ for V, u for the controller), trained by minimizing the degree to which the Lyapunov conditions are violated:

```
Lyapunov Risk = max(0, −V_θ(x)) + max(0, L_fu V_θ(x)) + V_θ(0)²
```

Each term penalizes one Lyapunov condition being broken. Counterexamples returned by dReal are added to the training set for x, closing the learner–verifier loop in Figure 1.

---

## Three Gaps We Targeted

1. **No notion of task performance.** The Lyapunov Risk only rewards stability — a controller that satisfies it perfectly could still be a poor controller. The paper sidesteps this by initializing with an LQR controller, which happens to already encode task information, and by only ever updating a *linear* network on top of it. We test what happens when the underlying policy is genuinely nonlinear and trained by RL instead.
2. **One counterexample per verifier call.** Because dReal stops as soon as it finds a single violation, each expensive SMT call yields exactly one new training point. We test whether splitting the domain into a grid and calling the verifier on each cell in parallel yields more counterexamples per training loop.
3. **Untested beyond 2D.** The paper's own experiments never go past a 2-dimensional state space. We evaluate an n-link planar pendulum, which lets us dial up dimensionality directly, to see where the verifier stops scaling.

---

## Reproduced Results

We first reproduce the paper's baseline comparison — the proposed method ("NN"), against LQR and SOS baselines — on the Inverted Pendulum and Wheeled Vehicle path-following tasks.

<div align="center">
  <figure>
    <img src="/projects/assets/nlc/fig3_reproduced.png" alt="Reproduced Lyapunov functions and regions of attraction" width="1000">
    <figcaption>Reproduced results. Lyapunov function and Region of Attraction for Inverted Pendulum (first two plots) and Wheeled path following (last two plots). The dashed red circle is the valid domain; the region of attraction is the largest level curve of V contained within it.</figcaption>
  </figure>
</div>

The learned Lyapunov function (NN) gives a larger ROA than both baselines on both tasks, consistent with the paper — LQR's region is a fixed ellipse set by the linearization at the origin, while the learned candidate can shape itself to the domain.

---

## Replacing LQR with an RL Policy

The controller in the paper is a single linear layer with no nonlinear activation, initialized from an LQR solution — so despite being called a "neural network controller," it never actually behaves nonlinearly. We test what happens when the controller is genuinely nonlinear: a Soft Actor-Critic (SAC) policy trained by RL on the Inverted Pendulum, with reward

```
r = −(θ² + 0.1·θ̇² + 0.001·u²)
```

penalizing angle, angular velocity, and torque away from the upright fixed point.

We then run two variants: (1) update *both* the Lyapunov candidate and the SAC policy's weights to minimize the Lyapunov Risk, and (2) freeze the SAC policy and only fit a Lyapunov candidate around it.

<div align="center">
  <figure>
    <img src="/projects/assets/nlc/fig4_rl_controller.png" alt="Lyapunov function and ROA for RL controller, updated vs. frozen" width="1000">
    <figcaption>RL controller. First two plots: both the SAC policy and Lyapunov candidate are updated — a valid Lyapunov function is found. Last two plots: the SAC policy is frozen — no valid Lyapunov function is found within 2,000 iterations, though the Lyapunov risk gets very low.</figcaption>
  </figure>
</div>

| Algorithm | Expected Return |
|---|---|
| NN (paper's method) | −725.4 |
| RL policy (updated) | −1767 |
| **RL policy (frozen)** | **−340** |

**What this shows:** the frozen RL policy gives the best task performance by a wide margin — more than double the paper's method — confirming the Inverted Pendulum swing-up is genuinely nonlinear and a linear LQR-style controller leaves performance on the table. But it's also the case where we *fail* to find a valid Lyapunov function after 2,000 iterations: the Lyapunov risk keeps dropping without ever satisfying the conditions everywhere. When we let the Lyapunov loss update the SAC weights instead, we do get a valid certificate — but task performance collapses to −1767, worse than the paper's linear controller. Updating a nonlinear policy's weights to satisfy a stability loss with no performance term pulls it away from the reward-maximizing behavior it was trained for.

The ROA shapes tell a consistent story: LQR's and NN's regions are elliptical, a direct consequence of the quadratic form a linear controller induces, and visibly more stable along one axis than the other. The RL policy's region — in both the updated and frozen cases — is closer to circular, which is the more desirable shape since it doesn't privilege one state dimension's disturbances over another's.

**The tradeoff this exposes:** for a genuinely nonlinear task, there is no free lunch between stability certification and task performance under this framework — the Lyapunov Risk objective has no term that protects reward once the controller it's shaping is expressive enough to actually trade one off against the other.

---

## Gridding the State Space to Accelerate the Falsifier

Because dReal halts at the first violation it finds, every SMT call yields exactly one counterexample — regardless of how large or high-dimensional the domain is. We test splitting the Inverted Pendulum's 2D state space (θ, θ̇) into an N×N grid and running the falsifier on each cell, so a single training loop can surface up to N² counterexamples instead of one.

<div align="center">
  <figure>
    <img src="/projects/assets/nlc/fig5_grid.png" alt="Learned Lyapunov function under increasing grid splits" width="1000">
    <figcaption>Learned Lyapunov function for the inverted pendulum with the domain split into (a) 1, (b) 9, (c) 25, and (d) 49 sub-domains. D is the number of splits per state dimension.</figcaption>
  </figure>
</div>

| #Splits per dim | Tuning param. α | #Iterations | Train Time (s) | Verifier Time (s) |
|---|---|---|---|---|
| 0 (nominal) | 2.2 | 650 | 32.42 | 2.19 |
| 3 | 2.22 | 460 | 33.06 | 0.87 |
| **5** | 3.2 | **320** | **26.51** | **0.82** |
| 7 | 2.18 | 370 | 51.29 | 1.09 |

Gridding does cut training time — the 5-split configuration nearly halves it (32.4s → 26.5s) with fewer iterations to convergence — but the improvement isn't monotonic: going from 5 to 7 splits *increases* total time again, since coordinating and calling the verifier across more sub-domains adds its own overhead. The number of splits behaves like a hyperparameter to tune per system rather than a dial to max out.

It's also worth being clear about what this buys: comparing the learned surfaces in Figure 5 across splits, the resulting ROA isn't meaningfully larger than the nominal (ungridded) case. Gridding is a speed optimization on the *verifier*, not a way to certify a bigger region of attraction.

---

## Does It Scale Past 2D?

The paper's experiments — Inverted Pendulum, Wheeled Vehicle — never exceed a 2-dimensional state space. We test an n-link planar pendulum, which lets dimensionality scale directly with n: n control inputs, 2n state variables [θ₁...θₙ, θ̇₁...θ̇ₙ], dynamics

```
M(θ)θ̈ + C(θ, θ̇)θ̇ + τ(θ) = Bu
```

The paper doesn't release code for this environment, so the dynamics and falsifier were implemented from scratch, using the paper's LQR initialization weights.

<div align="center">
  <figure>
    <img src="/projects/assets/nlc/fig6_2link.png" alt="2-link pendulum Lyapunov function and system diagram" width="1000">
    <figcaption>2-link pendulum (4D state space). The Lyapunov function is shown as a function of one (θ, θ̇) pair with the remaining two state dimensions held constant. A valid Lyapunov function is never found, though the Lyapunov risk continues to decrease.</figcaption>
  </figure>
</div>

| | Pendulum (2D) | 2-link pendulum (4D) | 3-link pendulum (6D) |
|---|---|---|---|
| Verifier time / iteration (s) | 0.063 | 0.174 | 0.51 |

**The method does not scale past 2 dimensions in our experiments.** For the 2-link pendulum (4D), the Lyapunov risk keeps decreasing but never converges to a valid certificate. For the 3-link pendulum (6D), the SMT verifier's per-iteration cost alone makes the experiment computationally intractable to complete — and that cost is understated, since it only reflects easy early-training counterexamples; verification gets harder, not easier, as the candidate approaches convergence and violations become rarer and harder for the solver to locate. The roughly 3x jump in per-iteration verifier time from 2D → 4D → 6D is consistent with the known poor scaling of SMT solvers with dimensionality — this is a property of the *verification backend*, not something a better learner could route around.

---

## Conclusion

Three limitations, three outcomes:

- **Task performance.** The Lyapunov Risk alone is silent on task performance, and once the controller is expressive enough (RL/SAC) for that to matter, the gap is stark — a frozen RL policy roughly halves task cost versus the paper's method, but training that same policy to also satisfy the Lyapunov Risk more than doubles it relative to the paper. Stability and performance genuinely trade off here; the objective needs a performance term to navigate that tradeoff rather than ignore it.
- **Falsifier throughput.** Gridding the domain is a real but bounded win — up to ~18% faster training at the right grid resolution — and doesn't come with a bigger certified region of attraction. It buys speed, not more safety.
- **Dimensionality.** The method as specified does not scale past 2D in our tests. This is a hard limitation of using an SMT solver as the verification backend, not a training or tuning issue.

**Future work:** a joint objective that explicitly balances task reward against Lyapunov Risk during RL training (rather than post-hoc freezing); combining reinforcement learning with sampling-based Lyapunov methods for probabilistic — rather than SMT-verified — guarantees, which would sidestep the verifier's dimensionality bottleneck entirely; and testing whether any of these guarantees survive contact with a real robot rather than a simulated dynamics model.

---

## References

Chang, Y., Roohi, N., & Gao, S. (2019). *Neural Lyapunov Control*. Advances in Neural Information Processing Systems (NeurIPS 32).

Zhou, R., Quartz, T., De Sterck, H., & Liu, J. (2022). *Neural Lyapunov Control of Unknown Nonlinear Systems with Stability Guarantees*. NeurIPS. [openreview.net/forum?id=QvlcRh8hd8X](https://openreview.net/forum?id=QvlcRh8hd8X)

Richards, S. M., Berkenkamp, F., & Krause, A. (2018). *The Lyapunov Neural Network: Adaptive Stability Certification for Safe Learning of Dynamical Systems*. Conference on Robot Learning.

Barrett, C. & Tinelli, C. (2018). *Satisfiability Modulo Theories*, pp. 305–343. Springer International Publishing. [doi.org/10.1007/978-3-319-10575-8_11](https://doi.org/10.1007/978-3-319-10575-8_11)

Gao, S., Kong, S., & Clarke, E. M. (2013). *dReal: An SMT Solver for Nonlinear Theories over the Reals*. CADE-24.

Gao, S., Avigad, J., & Clarke, E. M. (2012). *δ-Complete Decision Procedures for Satisfiability over the Reals*. IJCAR.

Katz, G. et al. (2019). *The Marabou Framework for Verification and Analysis of Deep Neural Networks*, pp. 443–452.

Parrilo, P. A. (2004). *Structured Semidefinite Programs and Semialgebraic Geometry Methods in Robustness and Optimization*.

Johansen, T. A. (2000). *Computation of Lyapunov Functions for Smooth Nonlinear Systems Using Convex Optimization*.

Tedrake, R. (2009). *LQR-Trees: Feedback Motion Planning on Sparse Randomized Trees*.

Khansari-Zadeh, S. M. & Billard, A. (2014). *Learning Control Lyapunov Function to Ensure Stability of Dynamical System-Based Robot Reaching Motions*. Robotics and Autonomous Systems, 62, 752–765.

Giesl, P. & Hafstein, S. (2016). *Review on Computational Methods for Lyapunov Functions*. Discrete and Continuous Dynamical Systems, Series B, 20(8).
