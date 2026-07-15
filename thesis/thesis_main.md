---
layout: post
title: "Differentiable Bayesian Filtering with Harmonic Exponential Distributions"
---

# Differentiable Bayesian Filtering with Harmonic Exponential Distributions

**M.Sc. Thesis, Université de Montréal and MILA· 2025** \
**Supervisor:** Liam Paull and Guy Wolf \
**Status:** Accepted \
[Thesis](https://umontreal.scholaris.ca/items/70d3f215-6d63-4307-8311-bc33ed8c3522)

---

<div align="center">
  <figure>
    <img src="/thesis/assets/perseverance_landing.png" alt="Localisation without GPS" width="380">
    <figcaption><em>Mars Perseverance landing ellipse shrunk from 200 km to 7 km axis. Even within 7 km, hazardous terrain remains.</em></figcaption>
  </figure>
</div>

## Abstract

State estimation under non-linear, heteroscedastic noise remains a fundamental challenge in autonomous robotics. Real-world robots operate under two compounding sources of uncertainty: sensor noise and actuator degradation. Exteroceptive sensors are corrupted by the environment — cameras lose signal under poor lighting or partial occlusion, LiDAR returns are unreliable on smooth or specular surfaces, and sonar degrades in turbulent or acoustically complex settings. Proprioceptive actuators introduce a separate class of error — worn wheel encoders accumulate systematic bias, and mechanical wear produces drift that grows unbounded over time. Without explicitly accounting for these noise characteristics, a robot's pose estimate becomes inaccurate and overconfident, propagating errors into downstream tasks such as motion planning and feedback control.

Classical filters — EKF, UKF, and particle filters — are all interpretations of the recursive Bayes filter and share a common requirement: a specified observation likelihood `p(z|x)`. In practice this distribution is assumed Gaussian, which either overestimates certainty in the presence of heavy-tailed or multimodal sensor noise, or underestimates it in well-behaved regimes. Neither failure mode is acceptable for safety-critical systems. Learning-based methods have progressively addressed this by fitting aleatoric uncertainty directly from data, allowing the noise model to reflect what sensors actually produce rather than what is analytically convenient.

This work introduces **Diff-HEF (Differential Harmonic Exponential Filter)**, built over HEF <sup>[[2]](#r2)</sup> — a Bayesian filter that represents probability distributions as Harmonic Exponential Distributions (HED) <sup>[[1]](#r1)</sup> on compact Lie groups. Diff-HEF extends HEF with a differentiable, learned observation likelihood whose parameters are optimized end-to-end from trajectory data via backpropagation through time. Rather than prescribing a noise shape, Diff-HEF learns one that minimizes downstream state estimation error. Experiments on circular state spaces and SE(2) range-bearing simulations show that Diff-HEF produces better-calibrated posteriors than EKF and particle filter baselines, with consistent gains in non-linear heteroscedastic environments where classical assumptions most severely break down.

**TL;DR — this work contributes:**
1. **HED as an NLL-trained density estimator and aleatoric uncertainty model**, evaluated across synthetic (Gaussian, Beta, multimodal) and real (disc tracking, NYU depth) datasets against Gaussian, non-parametric, and mixture baselines.
2. **A local grid strategy** that keeps gradients dense as state-space dimensionality grows — addressing the compute blowup any non-parametric filter (grid- or particle-based) faces as it scales.
3. **A differentiable spline-interpolation implementation in PyTorch**, needed to move values between the Cartesian grid the filter operates on and the polar-coordinate grid SE(2)'s harmonic basis naturally requires.
4. **Diff-HEF**, the full differentiable filter on SE(2), benchmarked and analyzed against classical and differentiable baselines (EKF, PF, HistF, LSTM).

---

## Introduction

GPS is rarely as reliable as we assume. In open sky it carries a **3–5 meter** error radius; in cities, multipath reflections off buildings push that to **10–50 meters**. Indoors it fails entirely. For a pedestrian this means the occasional wrong street. For an autonomous system operating without a fallback, it is a mission failure.

The problem is more acute when there is no GPS at all. NASA's Perseverance rover navigates Jezero Crater on Mars with no satellite infrastructure — at landing, the best achievable targeting precision was a **~7 km ellipse** of terrain containing boulders, ridgelines, and slopes hazardous enough to end the mission (Figure 1). On the surface, Perseverance localizes by dead reckoning: integrating wheel encoders and inertial sensors forward from a known point. Errors accumulate with every meter traveled.

Robots navigating GPS-denied environments — indoor warehouses, underground mines, urban canyons — must maintain a probabilistic belief over their own state using only onboard sensors. Dead reckoning integrates velocity and heading estimates forward in time, but small biases compound: a 1% odometry error over a 100-meter traverse produces meter-scale positional uncertainty. Correcting drift requires fusing external sensor observations (LIDAR returns, depth images, WiFi fingerprints) into the state estimate.

The difficulty is that these sensors do not produce Gaussian-distributed noise. A depth camera reporting distance to a partially-occluded surface yields a distribution with a sharp mode near the true range and a heavy tail of outlier returns. A LIDAR beam clipping the edge of a door frame produces a bimodal return. Real sensor noise is **heteroscedastic** — its shape and variance depend on the scene geometry, lighting, and dynamic objects present — properties no fixed Gaussian can capture.

Classical and differentiable filters alike — EKF, UKF, particle filters, and their learned variants — either assume Gaussian observation noise or require the noise model to be specified by hand. When the true likelihood is multimodal or heavy-tailed, a misspecified noise model produces posteriors that are overconfident or underconfident regardless of which filter carries them. This work proposes an end-to-end differentiable framework that learns calibrated measurement distributions directly from data, producing more accurate localization estimates in the non-linear, heteroscedastic settings where fixed noise assumptions break down.

---

## Background

### Classical Bayesian Filters

A Bayesian filter maintains a belief `bel(x_t) = p(x_t | z_{1:t}, u_{1:t})` and updates it recursively:

```
Prediction:  bel̄(x_t) = ∫ p(x_t | x_{t-1}, u_t) · bel(x_{t-1}) dx_{t-1}
Update:      bel(x_t)  ∝ p(z_t | x_t) · bel̄(x_t)
```

The observation likelihood `p(z_t | x_t)` is the term this work targets.

| Filter | Likelihood model | State representation | Cost |
|---|---|---|---|
| EKF | Gaussian (linearized) | Gaussian | O(n²) |
| UKF | Gaussian (sigma points) | Gaussian | O(n²) |
| Particle Filter | Arbitrary | Weighted samples | O(N·n) |
| Histogram Filter | Arbitrary | Discrete grid | O(G) |
| **Diff-HEF (ours)** | **HED (learned)** | **HED on Lie group** | **O(K·n)** |

### Harmonic Exponential Distribution

<div align="center">
  <figure>
    <img src="/thesis/assets/hed_hetero_s1.png" alt="HED density estimation on S¹, heteroscedastic case" width="500">
    <figcaption><em>HED-predicted vs. true density on S¹ in a heteroscedastic setting — the predicted distribution (blue) tracks the true density (green) as its shape and spread change with the underlying state.</em></figcaption>
  </figure>
</div>

The Harmonic Exponential Distribution generalizes the exponential family to compact manifolds using a Fourier/harmonic basis. On the circle S¹, the density is:

```
p(g | η) = (1 / Z_η) · exp(η · T(g))
```

where `T(g) = [cos(g), sin(g), cos(2g), sin(2g), …, cos(Kg), sin(Kg)]` is the vector of harmonic features up to order K, `η ∈ ℝ^{2K}` is the natural parameter, and `Z_η = ∫ exp(η · T(g)) dg` is the normalizing constant (computable in closed form on S¹ via Bessel functions).

Key properties:
- **Closed under multiplication**: the product of two HEDs is an HED — enabling exact Bayesian updates in the harmonic domain.
- **Expressiveness**: by increasing K, HED approximates any distribution on the group arbitrarily well.
- **Group-structured**: defined natively on SO(2), S¹, and by extension SE(2), so the density is consistent with the manifold geometry of the state space.

### Differentiable Filters

A differentiable filter is a filter whose full update equations — prediction, likelihood evaluation, Bayesian update — are implemented as differentiable operations so that a loss computed on the output state estimate can be backpropagated through the entire filtering trajectory. This enables end-to-end training: all learnable parameters (neural networks encoding the likelihood, noise covariances, process models) are optimized jointly to minimize estimation error on real trajectories.

Prior differentiable filter work — Differentiable Particle Filters (DPF) <sup>[[4]](#r4)</sup>, particle filter networks <sup>[[5]](#r5)</sup>, and Backprop KF <sup>[[6]](#r6)</sup> — either relies on costly particle representations that scale poorly, or inherits the Gaussian likelihood assumption from the EKF. Diff-HEF targets the gap: expressive non-Gaussian likelihoods with the computational structure of a parametric filter.

The particle representation itself is the deeper problem, not just its cost. A particle filter's resampling step discards low-weight particles and duplicates high-weight ones to fight weight degeneracy — but this concentrates particles in already-likely regions, thinning out coverage everywhere else. This is **particle deprivation**: the posterior collapses toward a shrinking set of duplicated points, and if the true state later moves somewhere the particles have vacated, there is no probability mass left there to recover with. Resampling is also a discrete, non-differentiable operation — selecting particle *i* is a hard categorical choice — so a plain particle filter cannot be trained end-to-end. DPF works around this with **soft resampling**: instead of sampling from the raw importance weights, it samples from a mixture of the weights and a uniform distribution, which keeps each resampled particle's weight a differentiable function of its ancestor's weight, at the cost of a biased, higher-variance gradient estimate.

That tradeoff shows up directly in the results below. On S¹, Diff-PF's ATE (0.01612) is within 0.4% of Diff-HEF's (0.01606) — resampling toward the mode is an efficient way to track a point estimate — but its NLP (-0.50) trails Diff-HEF's (-0.70) by a wide margin, because a thinned-out particle set understates the true spread of the posterior. On SE(2), where the state is three-dimensional instead of one, the same mechanism compounds: Diff-PF's ATE gap to Diff-HEF widens to 38% and its NLP gap widens by more than an order of magnitude. This is the particle filter's classic curse of dimensionality — the number of particles needed to cover a volume scales exponentially with its dimension, so a particle budget that gives adequate coverage on S¹ is comparatively sparse on SE(2), leaving soft resampling's bias toward high-weight regions more empty space to concentrate away from.

---

## Methodology

### Learning the Observation Likelihood

<figure>
  <img src="/thesis/assets/hed_density_estimation.png" alt="Non-parametric likelihood via Fourier basis" width="600">
  <figcaption><em>Density estimation and uncertainty modelling with HED, learned using a grid-based representation. The input may be the true pose x, a noisy observation x̃ = x + ε, or a full prior p(x̃). Measurement z is incorporated into the negative log-likelihood of the HED-parameterised conditional to update the model.</em></figcaption>
</figure>

The observation likelihood `p(z_t | x_t)` is parameterized as an HED whose natural parameters are the output of a neural network:

```
η = f_θ(z_t, x_t)
p(z_t | x_t) = HED(η)
```

The network `f_θ` maps a raw sensor reading and a queried state to the harmonic parameters of the likelihood at that state. This is evaluated at each point of a discretized state grid during the Bayesian update step.

### Training Objective

The filter is trained end-to-end over full trajectories. The primary loss is Negative Log-Likelihood of the ground-truth state under the predicted posterior:

```
L_NLL = -1/T · Σ_t log bel(x_t*)
```

where `x_t*` is the ground-truth state at time t.

To prevent the likelihood network from overfitting to sharp, degenerate distributions, a **smoothness regularizer** penalizes high-frequency harmonic components:

```
L_smooth = Σ_k k² · ‖η_k‖²

L_total = L_NLL + λ_s · L_smooth
```

The λ_s coefficient controls the bias-variance tradeoff: higher λ_s encourages smoother, more conservative posteriors; lower λ_s allows sharper but potentially overfitted distributions.

### Local Grid Strategy

<div align="center">
  <figure>
    <img src="/thesis/assets/local_vs_whole_grid.png" alt="Local grid vs. whole grid on S¹" width="700">
    <figcaption><em>Predicted vs. true distribution on S¹ using a local grid centered on the belief mode (left) vs. a grid spanning the whole state space (right). The whole-grid predicted distribution spreads spurious mass toward the far side of the circle, away from the pose and measurement; the local grid stays concentrated near the true mode.</em></figcaption>
  </figure>
</div>

State spaces such as SE(2) (planar position + heading) are high-dimensional relative to the precision needed at any given filter step. A global grid over the full state space would require exponentially many cells and produce near-zero gradients for cells far from the true state — the sparse gradient problem.

The local grid strategy addresses this by maintaining a small grid **centered on the current belief mode** and updating the grid position each step. Gradients remain dense and informative; the filter trades global coverage for local precision around its current estimate.

### End-to-End Training via BPTT

<div align="center">
  <figure>
    <img src="/thesis/assets/e2e_hef.png" alt="End-to-end Diff-HEF training loop" width="800">
    <figcaption><em>The end-to-end loop: sensor readings z and the current belief p(x̃) feed the likelihood network f_θ, which outputs a discretized likelihood grid p_θ(z|x). This combines with the prior via the Bayesian update to produce the posterior p(x_k | x̃_0, v_{1:k}, z_{0:k-1}), and the smoothness-regularized loss L_total backpropagates through the whole loop to update θ.</em></figcaption>
  </figure>
</div>

The full filtering loop — grid initialization, likelihood evaluation, Bayesian update, belief normalization — is implemented in PyTorch with autograd-compatible operations. Backpropagation Through Time (BPTT) unrolls the filter across T timesteps and computes gradients of `L_total` with respect to `θ`, the parameters of the likelihood network. Standard Adam optimization is used.

---

## Experiments and Results

### Density Estimation on S¹

HED density estimation is first evaluated independently of filtering, on target distributions on S¹ of increasing complexity: a Gaussian-equivalent unimodal case (von Mises, the circular analogue of a Gaussian — more on that distinction in the Ablation section), an asymmetric unimodal case (Beta), and a multimodal case, all detailed below.

The comparison is less clear-cut when the target itself is Gaussian — the setting where the analytic Gaussian model should have every advantage.

**Model** | **KL Divergence** | **WD** | **NLL**
---|---|---|---
*measurement noise = 0.3* | | |
HED-Gaussian | 0.0145 ± 0.005 | **0.3839 ± 0.0623** | 0.2223 ± 0.0002
Gaussian | **0.0108 ± 0.0015** | 0.3868 ± 0.0263 | **0.2221 ± 0.0001**
*measurement noise = 0.5* | | |
HED-Gaussian | **0.0896 ± 0.0119** | **1.0954 ± 0.0796** | **0.7218 ± 0.0001**
Gaussian | 0.1020 ± 0.0057 | 1.1756 ± 0.0170 | **0.7218 ± 0.0001**

*Comparison of HED–Gaussian and Gaussian models across KL divergence, Wasserstein distance (WD), and NLL, evaluated under two measurement-noise settings over 5 seeds.*

At low noise (0.3), the analytic Gaussian model has a slight edge on KL divergence and NLL — expected, since the target distribution matches its assumptions exactly and it pays no cost for the extra harmonic parameters HED carries. At higher noise (0.5), where the target distribution has higher variance, HED-Gaussian overtakes it on both KL divergence and Wasserstein distance, and training is more stable in higher-dimensional settings. The takeaway is not that HED beats Gaussian everywhere — on its home turf, a correctly-specified Gaussian model is a fair fight — but that HED's advantage grows with distributional complexity and does not meaningfully cost anything when the simpler model would have sufficed.

The Beta distribution on S¹ tests the middle ground directly: a family of targets that sweeps from Gaussian-like to progressively more skewed and asymmetrical.

<div align="center">
  <figure>
    <img src="/thesis/assets/beta_gp_s1.png" alt="Gaussian density modelling of Beta distribution on S¹, five configurations" width="900">
    <img src="/thesis/assets/beta_s1_hed.png" alt="HED density modelling of Beta distribution on S¹, five configurations" width="900">
    <figcaption><em>Density estimation on S¹ for a Beta distribution, five configurations sweeping from Gaussian-like (left) to progressively more skewed and asymmetrical (right). Top row: Gaussian model. Bottom row: HED. HED tracks the true distribution closely across all five configurations; the Gaussian model's fit visibly degrades as the target skews further from symmetric.</em></figcaption>
  </figure>
</div>

HED outperforms the Gaussian model across every configuration tested, and the gap widens as skew increases — precisely where a fixed unimodal-symmetric assumption breaks down. This is the flexibility argument for HED in one picture: it does not need to be told the shape of the target distribution in advance to fit it well.

The multimodal case removes the Gaussian comparison entirely — a unimodal Gaussian has no way to represent two separated modes, so the natural baseline instead is a Mixture Density Network (MDN) <sup>[[8]](#r8)</sup>.

| Model | KL Divergence | WD | NLL |
|---|---|---|---|
| **HED** | **0.7771 ± 0.1** | **1.331 ± 0.04** | **1.3161 ± 0.007** |
| MDN | 0.785 ± 0.0208 | 2.0467 ± 0.0342 | 1.3301 ± 0.002 |

*Comparison of HED and MDN on a target density with two modes separated by π and measurement noise of 0.5, over 5 seeds.*

HED wins on all three metrics, but the gap is narrowest on KL divergence and NLL and widest on Wasserstein distance — MDN's mixture components land close to the right density values overall but are less precisely placed. The more important asymmetry is architectural rather than numerical: MDN must be told the number of modes in advance (fixed at 10 components here, well above the true 2, to give it a fair chance of capturing the mode locations), while HED's harmonic basis makes no such assumption and recovers the multimodal structure without knowing the mode count ahead of time.

### Aleatoric Uncertainty Calibration

**Disc tracking**: a simulated robot tracks a target moving on a circular track with synthetic sensor noise of varying type (Gaussian, bimodal, heavy-tailed). Diff-HEF is evaluated against EKF and UKF on Expected Calibration Error (ECE) — how closely the predicted posterior coverage matches empirical hit rates.

<div align="center">
  <figure>
    <img src="/thesis/assets/disc_tracking_input.png" alt="Sample input frame from the disc tracking dataset" width="400">
    <figcaption><em>Sample input frame: the target red disc among randomly colored, randomly sized distractor discs.</em></figcaption>
  </figure>
</div>

The simulated dataset is adapted from Bhatt et al. <sup>[[3]](#r3)</sup>, where disc positions are uniformly sampled over the image space and randomly colored distractors with uniformly sampled radii are added to occlude the target red disc. For this experiment, only the pose information is considered, and velocity is ignored.

| Method | L1 | ECE | WD | KL Divergence |
|---|---|---|---|---|
| NLL | 18.34 | 0.22 | 16.04 | 0.98 |
| beta-NLL | **14.21** | 0.15 | 14.52 | 0.87 |
| f-Cal | 14.24 | 0.14 | 14.22 | 0.85 |
| **HED loss** | 14.22 | **0.04** | **5.22** | **0.31** |

*The NLL baseline is the Gaussian negative log-likelihood; beta-NLL <sup>[[9]](#r9)</sup> and f-Cal are prior calibration-focused loss functions. The HED loss is the smoothness-regularized NLL introduced above (L_total).*

The HED loss matches beta-NLL on point estimation (L1) but is not competing on that axis — it wins decisively on every distributional metric: ECE roughly 3-5x lower, and both Wasserstein distance and KL divergence less than half of the next-best baseline (f-Cal). Point accuracy alone does not distinguish these methods; calibration and distributional fidelity do, and that is where the learned harmonic likelihood separates from prior loss functions.

This result is also a concrete instance of the optimization asymmetry detailed in the Ablation section below: every baseline here — NLL, beta-NLL, f-Cal — parameterizes its predictive distribution as Gaussian, and so needs a careful mean warm-up to avoid the variance-inflation shortcut described there. HED, trained directly over its frequency coefficients rather than an explicit mean/variance pair, does not need that warm-up. The ECE and KL divergence gaps above are the empirical consequence.

**NYU depth dataset**: real-world depth images from the NYU Depth V2 dataset <sup>[[7]](#r7)</sup> are used to evaluate whether the learned likelihood correctly represents the aleatoric (irreducible) uncertainty present in depth sensor returns. Diff-HEF shows lower ECE and better Negative Log Posterior (NLP) scores than Gaussian baselines, confirming that the learned HED captures sensor-specific noise characteristics.

| Method | RMSE ↓ | d1 ↑ | d2 ↑ | d3 ↑ | NLL ↓ |
|---|---|---|---|---|---|
| Base (Deterministic) | **0.419** | **0.865** | 0.975 | 0.993 | — |
| NLL Loss | 0.694 | 0.637 | 0.870 | 0.944 | 1.701 |
| beta-NLL | 0.439 | 0.843 | 0.974 | **0.995** | 1.432 |
| f-Cal | 0.441 | 0.842 | 0.973 | **0.995** | 1.14 |
| **HED Loss** | 0.421 | 0.863 | **0.976** | 0.994 | **0.266** |

*All baselines share the same backbone and differ only in loss function; NLL Loss refers specifically to the Gaussian NLL formulation. d1/d2/d3 are standard depth-estimation accuracy thresholds.*

The pattern here mirrors the disc tracking result: pointwise depth accuracy (RMSE, d1–d3) is close across every method with a real likelihood term — HED Loss essentially matches the deterministic baseline it's built on. The separation is in NLL, where HED Loss scores 0.266 against a next-best of 1.14 for f-Cal, more than 4x tighter. Depth noise is plausibly non-Gaussian and possibly multi-modal — occlusion boundaries and reflective surfaces don't fail gracefully into a bell curve — so a Gaussian-parameterized loss (NLL, beta-NLL, f-Cal) can only ever misrepresent that structure, regardless of how well it's calibrated within the Gaussian family.

One caveat, unlike the synthetic disc tracking case: the true depth distribution here is unknown, so there is no ground truth to check the predicted uncertainty against directly. NLL is the only metric available that evaluates the probabilistic prediction itself rather than just the point estimate, and HED wins on it decisively while staying competitive everywhere else — the strongest claim the evidence supports is that HED's uncertainty estimates are both competitive in pointwise accuracy and superior in probabilistic consistency, not that they are demonstrably closer to the true noise distribution.

### Filter Performance on S¹

| Model | ATE ↓ | NLP ↓ | KL Divergence (Posterior) ↓ |
|---|---|---|---|
| **Diff-HEF** | **0.01606** | **-0.7013** | **0.0037** |
| Diff PF | 0.01612 | -0.5013 | 0.0049 |
| Diff HistF | 0.02150 | -0.5113 | 0.0127 |
| Diff EKF | 0.26805 | 0.1837 | 0.1200 |
| LSTM | 0.43680 | 0.4011 | 0.7693 |
| HEF | 0.11840 | -0.4204 | 0.0316 |
| PF | 0.11870 | -0.3504 | 0.0415 |
| HistF | 0.12060 | -0.3284 | 0.0449 |
| EKF | 0.42190 | 0.3176 | 0.3520 |

*Measurement noise is a multimodal Gaussian mixture (component means 0 and 0.1, std 0.3). The analytical filters assume a fixed-std mixture measurement model (std 0.2; EKF linearizes around mean 0.05, std 0.2) and are overconfident as a result, while the differentiable filters learn the measurement model end-to-end.*

The main signal is the gap between differentiable and analytical variants of the same filter, not between filter types: every differentiable filter beats its analytical counterpart by roughly an order of magnitude on ATE, since the analytical filters carry a fixed, misspecified measurement model while the differentiable ones learn it from data. Within the differentiable group, Diff-HEF leads on all three metrics, and its edge on NLP (-0.70 vs. -0.50 for Diff PF) despite near-identical ATE shows it is fitting the shape of the posterior — not just its mode — more faithfully than a particle representation. LSTM, with no explicit probabilistic structure, is worst across the board.

A planar robot pose (x, y, θ) could, in principle, be treated as three independent axes — heading on S¹, position on R² — and filtered with the S¹ machinery just demonstrated, applied separately to each factor. Whether that's valid comes down to a simple question: if a robot turns and moves, does the order matter? For a robot, it does — "move forward" means forward relative to wherever it's currently facing, so turning first and then moving lands somewhere different than moving first and then turning, even though both end up facing the same way.

<div align="center">
  <figure>
    <img src="/thesis/assets/intrinsic_vs_extrinsic.svg" alt="Extrinsic (independent) vs. intrinsic (coupled) framing of rotation and translation" width="800">
    <figcaption><em>Left: if position and heading are just two independent readouts (extrinsic framing), there's no order to speak of — this is S¹ × R². Right: robot controls are issued in the body frame, so rotating changes what "forward" means for the next move (intrinsic framing) — the same two moves in a different order land at a different position. This is SE(2).</em></figcaption>
  </figure>
</div>

That order-dependence is exactly why SE(2) does not decompose into S¹ × R². It is a semidirect product, not a direct product: rotation and translation are coupled, since heading determines the direction a motion update pushes the robot in, and a change of heading acts on the translation component rather than leaving it fixed. S¹ × R² would model these as independent — a rotation marginal and a position marginal that never interact. That independence assumption discards exactly the structure that makes SE(2) SE(2). The harmonic basis and the local grid strategy both have to be built on SE(2) directly, respecting how rotation and translation interact, rather than composed from two separate S¹ and R² solutions.

### SE(2) Range-Bearing Simulator

The full filter is evaluated on a simulated planar robot navigating an environment with range-bearing sensors. The SE(2) state space (x, y, θ) uses the local grid strategy. Evaluation metrics include ATE and NLP over 30 simulated trajectories.

<div align="center">
  <figure>
    <img src="/thesis/assets/se2_main002.png" alt="Belief update pipeline and mean estimate comparison on SE(2)" width="800">
    <figcaption><em>The belief update pipeline at a single timestep: predicted belief, measurement likelihood (a ring, consistent with range-only observation of a beacon), and posterior belief after fusing the two. Right: mean position estimate at that step for HEF, EKF, PF, and HistF against ground truth and the beacon layout — EKF's linearization pulls its estimate furthest from GT, while HEF, PF, and HistF cluster closer together.</em></figcaption>
  </figure>
</div>

The ring shape in the figure above is not an artifact — it is the actual measurement geometry. A single range-only beacon constrains position to a circle, not a point, and over the course of a trajectory the true posterior shifts between this circular ambiguity, a genuinely bimodal one (once a second beacon narrows the ring to its two intersection points), and a unimodal one once enough beacons disambiguate it. This shifting multimodality is exactly where the soft-resampling bias described earlier compounds: in the bimodal phase, two roughly equal-mass components sit on either side of the beacon baseline, and soft resampling only damps — rather than removes — the tendency of whichever mode holds a slight majority to overtake the other. Once one mode has thinned enough, the mixture's weighted component is itself already built from a particle set that under-represents it, leaving no path back. Diff-HEF does not face this: its posterior is a density over the grid, not resampled particles, so both modes keep exactly the mass the likelihood assigns them.

<div align="center">
  <figure>
    <img src="/thesis/assets/se2_filter_comparison.png" alt="Posterior shape comparison across filters on SE(2)" width="700">
    <figcaption><em>Posterior belief shape at the same timestep, by filter. HEF and HistF represent the posterior as a density over the grid and recover a mode close to ground truth; EKF's Gaussian ellipse is elongated along the range direction and offset from GT; PF's particles cluster tightly but slightly short of GT. Only HEF and HistF give a genuine density rather than a point estimate with a parametric or sample-based spread.</em></figcaption>
  </figure>
</div>

| Filter | ATE ↓ | NLP ↓ |
|---|---|---|
| **Diff-HEF** | **0.0097** | **-7.3197** |
| Diff PF | 0.0134 | -0.3681 |
| Diff HistF | 0.0759 | -0.2714 |
| Diff EKF | 0.1124 | 213.70 |
| LSTM | 0.1786 | 597.81 |
| HEF | 0.1278 | -4.4087 |
| PF | 0.1280 | 3.04 |
| HistF | 0.1385 | 17.77 |
| EKF | 0.1635 | 19517.59 |

*Comparison of differentiable and analytical filters on SE(2) range data, averaged over 30 trajectories. Analytical filters estimate the measurement noise at 0.0005, against a true noise of 0.0001.*

The overconfidence problem is visible directly in NLP here: EKF's fixed, overestimated measurement covariance produces an NLP of 19,517 — the posterior assigns near-zero probability to the true state on average, a far more severe failure than its ATE alone would suggest. Every differentiable filter avoids this by learning the measurement noise from data rather than carrying a fixed misestimate, and Diff-HEF is the only method with a comfortably negative NLP (-7.32), reflecting a posterior that is both accurate and appropriately confident rather than merely accurate. The ATE gap between Diff-HEF and the next best differentiable filter (Diff PF, 0.0134) is smaller than the NLP gap (-0.37 vs. -7.32) — consistent with the S¹ result, and consistent with the bimodal-collapse mechanism above: Diff-PF's mean position is often close enough to ground truth to keep ATE competitive, since collapsing onto either symmetric mode still lands near one plausible answer, but the posterior it reports has already discarded whichever mode it didn't collapse onto. NLP penalizes exactly this — a posterior that assigns near-zero density to the true state whenever it happens to sit under the mode that was thinned out — while ATE, being a point comparison, cannot see it at all.

## Ablation

### Bandwidth

<div align="center">
  <figure style="display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin: 0;">
    <span style="flex: 1 1 300px; max-width: 48%;">
      <img src="/thesis/assets/s1_DE_b_30.png" alt="HED density estimation on S¹, bandwidth 30" style="width: 100%;">
    </span>
    <span style="flex: 1 1 300px; max-width: 48%;">
      <img src="/thesis/assets/s1_DE_b_100.png" alt="HED density estimation on S¹, bandwidth 100" style="width: 100%;">
    </span>
    <figcaption style="width: 100%;"><em>HED density estimation on S¹ at bandlimit 30 (left) vs. 100 (right). The higher bandlimit visibly overfits — the predicted distribution tracks high-frequency noise rather than the smooth underlying density.</em></figcaption>
  </figure>
</div>

The bandlimit sets the number of harmonics available to the HED, and its effect on fit quality depends entirely on the underlying distribution being estimated — there is no single bandlimit that is best across tasks.

| Bandlimit | KL Divergence | NLL |
|---|---|---|
| **10** | **0.0317 ± 0.006** | **0.7322 ± 0.0027** |
| 30 | 0.1319 ± 0.0164 | 0.7343 ± 0.0036 |
| 100 | 0.5432 ± 0.0684 | 0.7421 ± 0.0035 |

*Bandlimit ablation on a unimodal, homoscedastic (near-Gaussian) target density on S¹, 5 seeds, validation set. Lower bandlimits perform better here, likely because the true distribution is simple enough that added harmonic capacity only adds variance rather than fitting real structure.*

| Bandlimit | KL Divergence | NLL |
|---|---|---|
| 10 | 0.2301 ± 0.0052 | 1.1232 ± 0.0043 |
| **30** | **0.1155 ± 0.0241** | **1.1227 ± 0.0062** |
| 100 | 0.2372 ± 0.0733 | 1.1227 ± 0.0050 |

*Bandlimit ablation on a Beta(α=5, β=10) target density on S¹, 5 seeds, validation set. Here a bandlimit of 30 beats both extremes — 10 underfits the asymmetric target, while 100 overfits and produces a noisier estimate.*

**Bandlimit is a task-dependent hyperparameter, not a fixed setting.** For a near-Gaussian target, the lowest bandlimit tested wins outright — extra harmonics buy nothing and only add variance. For an asymmetric target (Beta), the lowest bandlimit underfits and the highest overfits; a mid-range value is optimal. As distribution complexity increases, more harmonics are needed to represent it faithfully — but past that point, additional bandlimit trades bias for variance. In practice, the bandlimit needs to be tuned per-task rather than fixed globally.

### Optimization Landscape

A terminology note first: on S¹, "Gaussian" is not actually a well-defined distribution — the circle is compact and periodic, while the Gaussian density is defined on the real line. The natural circular analogue used throughout the density estimation results above is the **von Mises distribution**, parameterized by a mean direction and concentration in place of a Gaussian's mean and variance. Wherever earlier sections say "Gaussian" on S¹, von Mises is the distribution actually being fit — the two are treated as interchangeable shorthand here because von Mises converges to a Gaussian in the high-concentration limit, but they are not the same object.

Von Mises/Gaussian NLL carries a well-documented failure mode during training: because the loss trades off a quadratic mean-fit term against a log-variance (or log-concentration) term, the optimizer has a cheap way to lower NLL without actually fitting the mean — inflate the variance instead. This is not a hypothetical risk; it is the exact pathology that motivated beta-NLL <sup>[[9]](#r9)</sup> as a corrective reweighting of the standard NLL loss. Without a stable mean initialization (a "warm-up"), training can converge to an over-dispersed, semantically uninformative solution that looks fine on the loss curve while representing nothing about the true data. HED is not immune to poor initialization either — being non-parametric, it relies on the initial density shape to anchor the optimization, and without a mean estimate to provide that anchor, early gradients carry little directional signal, leading to slow convergence or collapse to an uninformative distribution.

The difference between the two shows up once the target grows more complex. In higher-dimensional or multimodal settings, von Mises/Gaussian mixtures give the variance-inflation shortcut to each component independently, and the components' interactions compound it — correcting one mode's mean can perturb another's variance estimate, so training a mixture is markedly less stable than training a single unimodal component, even with careful initialization. HED's harmonic coefficients jointly shape the whole density rather than decomposing it into separate per-mode means and variances, so there is no isolated variance term for the optimizer to inflate while leaving the rest of the fit untouched. Its failure mode under poor initialization is therefore different in kind, not just degree: instead of blowing up into a confidently-wrong, over-dispersed solution, it starves for gradient signal and converges toward a flat, honestly-uncertain density. Given a reasonable initialization, HED's loss landscape does not reward the optimizer for giving up on the mean the way von Mises/Gaussian NLL does — and that gap is widest exactly where von Mises/Gaussian training is most fragile: multimodal, higher-dimensional targets.

---

## Limitations

- **Local grid coverage**: the local grid strategy centers on the belief mode, which means a catastrophic divergence (mode far from ground truth) may not self-correct without a global recovery step.
- **Computational cost**: evaluating the likelihood network at every grid cell every timestep is more expensive than a closed-form EKF update, though cheaper than a particle filter at equivalent distributional quality.

- **Interpolation**: SE(2)'s group composition couples rotation and translation, so its natural harmonic basis uses polar coordinates for position (paired with angular frequency for heading) rather than Cartesian (x, y) — a standard result in the harmonic analysis of the Euclidean motion group, whose irreducible representations are indexed by radial frequency with Bessel-function matrix elements <sup>[[10]](#r10)</sup>. But the filter's grid is queried and updated in Cartesian coordinates (odometry, beacon positions), so every update requires resampling between a Cartesian grid and a polar one, and grid points from one rarely land on the other exactly. Linear interpolation, the simplest option, gave poor results; order-2 spline interpolation with 3x oversampling improved accuracy considerably. Fourier-domain interpolation would be the natural fit given the harmonic representation, but standard tools for it (non-uniform FFTs) aren't differentiable by default — differentiable variants exist in adjacent domains like MRI reconstruction, suggesting this is feasible future work rather than a fundamental blocker.
- **Manifold generality**: the current implementation is validated on S¹ and SE(2). Extension to higher-dimensional Lie groups (SO(3), SE(3)) requires adapting the harmonic basis and grid construction, which is left to future work.
- **Training data dependence**: the likelihood network learns the noise characteristics of the training distribution. Domain shifts — a new sensor, a new environment — require retraining or fine-tuning.
- **Baseline distribution families are still mostly Gaussian**: HED is shown to out-calibrate Gaussian-parameterized methods (NLL, beta-NLL <sup>[[9]](#r9)</sup>, f-Cal <sup>[[3]](#r3)</sup>) under higher variance, out-fit a Gaussian model on a non-Gaussian target (Beta), out-fit a fixed-mode-count MDN <sup>[[8]](#r8)</sup> on a multimodal target, and out-perform non-parametric filters (PF, HistF) on both S¹ and SE(2). Missing is the more flexible end of density estimation — normalizing flows and diffusion models — both capable in principle of representing arbitrary distributions despite a Gaussian base or noise process. Neither is a drop-in baseline: standard flows and diffusion are built for Euclidean data, and manifold-native flows exist for the circle and torus <sup>[[11]](#r11)</sup> but not yet for SE(2)'s specific rotation-translation coupling, while exact diffusion likelihoods require solving a probability-flow ODE per query — a nontrivial cost inside a filtering loop against HED's closed-form normalizer. Whether HED's edge would hold against these families, and which is best suited to SE(2) specifically, remains open.



---

## Conclusion

This work introduced Diff-HEF, a differentiable Bayesian filter that replaces the fixed Gaussian observation likelihood with a learned Harmonic Exponential Distribution. Training end-to-end via BPTT allows the filter to adapt its uncertainty representation directly to downstream estimation loss. Experiments demonstrate measurable improvements in posterior calibration and trajectory error over classical Gaussian filters in settings where the true sensor noise is non-Gaussian — precisely the settings most common in real-world robot deployment.

The central insight is that the filter structure need not change — only the likelihood model does. By making that model expressive and learnable while preserving the recursive Bayesian update, Diff-HEF bridges the gap between principled probabilistic estimation and data-driven adaptability.

---

## References

<a id="r1">[1]</a> Cohen, T. S., & Welling, M. (2015). *Harmonic Exponential Families on Manifolds*. ICML. [PDF](https://tacocohen.wordpress.com/wp-content/uploads/2015/05/hef.pdf)

<a id="r2">[2]</a> *Harmonic Exponential Filter (HEF)*. Montreal Robotics. [Project page](https://montrealrobotics.ca/hef/)

<a id="r3">[3]</a> Bhatt, D., Mani, K., Bansal, D., Murthy, K., Lee, H., & Paull, L. (2021). *f-Cal: Calibrated Aleatoric Uncertainty Estimation from Neural Networks for Robot Perception*. [arXiv:2109.13913](https://arxiv.org/abs/2109.13913)

<a id="r4">[4]</a> Jonschkowski, R., Rastogi, D., & Brock, O. (2018). *Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors*. RSS.

<a id="r5">[5]</a> Karkus, P., Hsu, D., & Lee, W. S. (2018). *Particle Filter Networks with Application to Visual Localization*. CoRL.

<a id="r6">[6]</a> Haarnoja, T., Ajay, A., Levine, S., & Abbeel, P. (2016). *Backprop KF: Learning Discriminative Deterministic State Estimators*. NeurIPS.

<a id="r7">[7]</a> Silberman, N., Hoiem, D., Kohli, P., & Fergus, R. (2012). *Indoor Segmentation and Support Inference from RGBD Images*. ECCV.

<a id="r8">[8]</a> Bishop, C. M. (1994). *Mixture Density Networks*. Neural Computing Research Group Report NCRG/94/004, Aston University.

<a id="r9">[9]</a> Seitzer, M., Tavakoli, A., Antic, D., & Martius, G. (2022). *On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks*. ICLR. [arXiv:2203.09168](https://arxiv.org/abs/2203.09168)

<a id="r10">[10]</a> Chirikjian, G. S. (2009). *Stochastic Models, Information Theory, and Lie Groups*. Birkhäuser.

<a id="r11">[11]</a> Rezende, D. J., Papamakarios, G., Racanière, S., & Albergo, M. S. (2020). *Normalizing Flows on Tori and Spheres*. ICML.

- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
- Mardia, K. V., & Jupp, P. E. (2000). *Directional Statistics*. Wiley.
