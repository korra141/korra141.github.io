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

This work introduces **Diff-HEF (Differential Harmonic Exponential Filter)**, built over HEF <sup>[[2]](#r2)</sup> — a Bayesian filter that represents probability distributions as Harmonic Exponential Distributions (HED) <sup>[[1]](#r1)</sup> on compact Lie groups. Diff-HEF extends HEF with a differentiable, learned observation likelihood whose parameters are optimized end-to-end from trajectory data via backpropagation through time. Rather than prescribing a noise shape, Diff-HEF learns one that minimizes downstream state estimation error. Experiments on circular state spaces and SE(2) range-bearing simulations show that Diff-HEF produces better-calibrated posteriors than EKF, UKF, and particle filter baselines, with consistent gains in non-linear heteroscedastic environments where classical assumptions most severely break down.

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

<!--- DIAGRAM PLACEHOLDER: HED density plots on S¹ — showing unimodal, bimodal, and uniform configurations by varying η coefficients; contrasted with a Gaussian approximation on the circle --->

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

Prior differentiable filter work (Differentiable Particle Filter, DPF; Differentiable Extended Kalman Filter, DEKF) either rely on costly particle representations that scale poorly, or inherit the Gaussian likelihood assumption from the EKF. Diff-HEF targets the gap: expressive non-Gaussian likelihoods with the computational structure of a parametric filter.

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

<!--- DIAGRAM PLACEHOLDER: Sparse gradient problem — global grid in SE(2) state space; true state is in one corner; likelihood gradients at distant grid cells are near-zero, providing no training signal. Contrast with local grid centered near current belief mode --->

State spaces such as SE(2) (planar position + heading) are high-dimensional relative to the precision needed at any given filter step. A global grid over the full state space would require exponentially many cells and produce near-zero gradients for cells far from the true state — the sparse gradient problem.

The local grid strategy addresses this by maintaining a small grid **centered on the current belief mode** and updating the grid position each step. Gradients remain dense and informative; the filter trades global coverage for local precision around its current estimate.

### End-to-End Training via BPTT

The full filtering loop — grid initialization, likelihood evaluation, Bayesian update, belief normalization — is implemented in PyTorch with autograd-compatible operations. Backpropagation Through Time (BPTT) unrolls the filter across T timesteps and computes gradients of `L_total` with respect to `θ`, the parameters of the likelihood network. Standard Adam optimization is used.

---

## Experiments and Results

### Density Estimation on S¹ and R²

<!--- DIAGRAM PLACEHOLDER: HED fitting results on synthetic datasets — (a) unimodal circular distribution, (b) bimodal circular distribution, (c) wrapped Cauchy (heavy tail). Each panel shows ground truth density, HED fit at K=4, and Gaussian baseline. NLL and KL divergence reported per method --->

HED density estimation is first evaluated independently of filtering, on synthetic circular and planar distributions designed to be non-Gaussian: unimodal von Mises, bimodal mixtures, and heavy-tailed wrapped Cauchy. 

HED consistently achieves lower NLL and KL divergence than Gaussian baselines. Bimodal distributions — where the Gaussian collapses to a broad unimodal approximation — show the largest gains.

### Aleatoric Uncertainty Calibration

<!--- DIAGRAM PLACEHOLDER: Disc tracking experiment — robot tracks a disc moving on a circular track; sensor noise is injected at known levels. Expected Calibration Error (ECE) plotted against noise level for DHEF vs EKF vs UKF. DHEF calibration curve stays near zero across noise levels; EKF ECE rises sharply at high noise --->

**Disc tracking**: a simulated robot tracks a target moving on a circular track with synthetic sensor noise of varying type (Gaussian, bimodal, heavy-tailed). Diff-HEF is evaluated against EKF and UKF on Expected Calibration Error (ECE) — how closely the predicted posterior coverage matches empirical hit rates.

**NYU depth dataset**: real-world depth images from the NYU Depth V2 dataset are used to evaluate whether the learned likelihood correctly represents the aleatoric (irreducible) uncertainty present in depth sensor returns. Diff-HEF shows lower ECE and better Negative Log Posterior (NLP) scores than Gaussian baselines, confirming that the learned HED captures sensor-specific noise characteristics.

### Filter Performance on S¹

<!--- DIAGRAM PLACEHOLDER: Time-series plots — ground truth circular trajectory, EKF estimate with shaded ±1σ confidence, Diff-HEF estimate with HED posterior shown as color-coded density on circle at each step --->

### SE(2) Range-Bearing Simulator

<!--- DIAGRAM PLACEHOLDER: SE(2) simulator — top-down 2D map with robot trajectory (blue), range-bearing sensor field of view (grey arcs), ground truth path (green dashed), DHEF estimate (red), EKF estimate (orange). Inset: posterior density on SE(2) at a selected timestep shown as a heat map over the 2D position plane --->

The full filter is evaluated on a simulated planar robot navigating an environment with range-bearing sensors. The SE(2) state space (x, y, θ) uses the local grid strategy. Evaluation metrics include ATE and NLP over 50 simulated trajectories.

## Ablation 

<!-- Bandwidth -->

## Optimisation smoother over Gaussian NLL.
---

## Limitations

- **Local grid coverage**: the local grid strategy centers on the belief mode, which means a catastrophic divergence (mode far from ground truth) may not self-correct without a global recovery step. 
<!-- Better method can be the upsampling, downsampling -->
- **Computational cost**: evaluating the likelihood network at every grid cell every timestep is more expensive than a closed-form EKF update, though cheaper than a particle filter at equivalent distributional quality.

- **Interpolation**
- **Manifold generality**: the current implementation is validated on S¹ and SE(2). Extension to higher-dimensional Lie groups (SO(3), SE(3)) requires adapting the harmonic basis and grid construction, which is left to future work.
- **Training data dependence**: the likelihood network learns the noise characteristics of the training distribution. Domain shifts — a new sensor, a new environment — require retraining or fine-tuning.



---

## Conclusion

This work introduced DHEF, a differentiable Bayesian filter that replaces the fixed Gaussian observation likelihood with a learned Harmonic Exponential Distribution. Training end-to-end via BPTT allows the filter to adapt its uncertainty representation directly to downstream estimation loss. Experiments demonstrate measurable improvements in posterior calibration and trajectory error over classical Gaussian filters in settings where the true sensor noise is non-Gaussian — precisely the settings most common in real-world robot deployment.

The central insight is that the filter structure need not change — only the likelihood model does. By making that model expressive and learnable while preserving the recursive Bayesian update, DHEF bridges the gap between principled probabilistic estimation and data-driven adaptability.

---

## References

<a id="r1">[1]</a> Cohen, T. S., & Welling, M. (2015). *Harmonic Exponential Families on Manifolds*. ICML. [PDF](https://tacocohen.wordpress.com/wp-content/uploads/2015/05/hef.pdf)

<a id="r2">[2]</a> *Harmonic Exponential Filter (HEF)*. Montreal Robotics. [Project page](https://montrealrobotics.ca/hef/)

- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
- Haarnoja, T., Ajay, A., Levine, S., & Abbeel, P. (2016). Backprop KF: Learning discriminative deterministic state estimators. *NeurIPS*.
- Jonschkowski, R., Rastogi, D., & Brock, O. (2018). Differentiable particle filters: End-to-end learning with algorithmic priors. *RSS*.
- Karkus, P., Hsu, D., & Lee, W. S. (2018). Particle filter networks with application to visual localization. *CoRL*.
- Mardia, K. V., & Jupp, P. E. (2000). *Directional Statistics*. Wiley.
- Chirikjian, G. S. (2009). *Stochastic Models, Information Theory, and Lie Groups*. Birkhäuser.
- Silberman, N., Hoiem, D., Kohli, P., & Fergus, R. (2012). Indoor segmentation and support inference from RGBD images. *ECCV*.
