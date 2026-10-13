---
layout: post
title: "Entropy, Log-Likelihood, and KL Divergence: When Does Lower Loss Mean Calibrated?"
date: 2026-10-13
---

# Entropy, Log-Likelihood, and KL Divergence: When Does Lower Loss Mean Calibrated?

## The identity

Three quantities, one equation. Let $$p$$ be the true data distribution and $$q$$ the model's predicted distribution.

$$
H(p) = H(p,q) - KL(p \| q)
$$

- **Entropy** $$H(p) = -\mathbb{E}_{x \sim p}[\log p(x)]$$ — the irreducible uncertainty in the data itself. A property of $$p$$ alone; no model touches it.
- **Cross-entropy** $$H(p,q) = -\mathbb{E}_{x \sim p}[\log q(x)]$$ — the average surprise of using $$q$$ to describe samples actually drawn from $$p$$.
- **KL divergence** $$KL(p \| q) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]$$ — the gap between them. Always $$\geq 0$$, zero iff $$q = p$$ almost everywhere.

Rearranged the usual way, $$H(p,q) = H(p) + KL(p \| q)$$: cross-entropy is entropy plus a non-negative penalty for $$q$$ being wrong. That penalty is the only thing a model can move.

## Minimizing NLL is minimizing KL, not entropy

Training by maximum likelihood — minimizing negative log-likelihood (NLL) over a dataset drawn from $$p$$ — is a Monte Carlo estimate of minimizing $$H(p,q)$$. So it's worth being precise about what "reducing log-likelihood reduces entropy" actually means, because taken literally it's wrong: **$$H(p)$$ is fixed.** You cannot lower the entropy of the true data distribution by training harder. What you *can* lower is the gap above it:

$$
\arg\min_\theta \; H(p, q_\theta) \;=\; \arg\min_\theta \; KL(p \,\|\, q_\theta)
$$

since $$H(p)$$ doesn't depend on $$\theta$$. NLL minimization is maximum-likelihood estimation is cross-entropy minimization is **forward-KL minimization** — four names for the same optimization. "Reducing log-likelihood is a proxy for reducing entropy" is really: reducing log-likelihood closes the KL gap, dragging cross-entropy down toward the entropy floor it can never cross. At $$KL(p \| q)=0$$, $$q = p$$ a.e. — exact distributional match.

## Two notions of calibration

That last sentence — $$q=p$$ at every input, not just on average — is worth separating from a weaker, more commonly measured idea, because "calibration" gets used for both:

- **Confidence (reliability) calibration.** Among all inputs where the model reports confidence $$c$$, the empirical accuracy is $$c$$. This is an *aggregate* statement — it checks a marginal relationship between confidence and correctness, not any specific input's distribution.
- **Distributional calibration.** $$q(y \mid x) = p(y \mid x)$$ for (almost) every $$x$$. A *pointwise* statement, strictly stronger: get this right and confidence calibration follows for free, along with everything else about the predictive distribution's shape.

The identity makes the strong version's story clean: if $$KL(p \| q) \to 0$$ in the population limit, you get full distributional calibration — not because anyone optimized for it, but as a side effect of optimizing likelihood to convergence. In that idealized limit, **NLL minimization and calibration are the same thing.**

## Where it breaks in practice

That limit is never reached, and the reason is structural, not just "finite data is noisy."

$$KL(p \| q)$$ as written is an *average* over $$x$$ of the pointwise divergence $$KL\big(p(y \mid x)\,\|\,q(y \mid x)\big)$$. A small average doesn't force every term to be small. A model can be confidently wrong on one slice of the input space and confidently right on another, and the errors wash out in the mean — low average KL, locally miscalibrated everywhere that matters.

Finite-sample training makes this worse by construction: the loss is computed against the *empirical* distribution $$\hat{p}$$ — point masses on the observed training examples — not the true $$p$$. Pushing $$q$$ to match $$\hat p$$ pushes it toward something sharper and more degenerate than $$p(y \mid x)$$ actually is, especially past the point where the model has enough capacity to start memorizing. The predictable consequence is **overconfidence**: modern neural networks, trained by minimizing exactly this loss, are frequently *more* accurate than their predecessors and *worse* calibrated, because their reported confidence systematically overstates how often they're right (Guo et al., 2017; [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)).

## The bound you can't beat, and the axis NLL can't see

The identity also gives a hard floor: $$H(p,q) \geq H(p)$$, always. Average NLL can never legitimately drop below the true entropy of the data — that floor is the irreducible, aleatoric part of the problem, and no amount of training removes it.

But there's a second, different quantity worth not confusing with that floor: the model's *own* predictive entropy, $$H(q)$$ — how peaked or spread out its predictions are, on their own, independent of whether they're right. Nothing stops a model from driving $$H(q)$$ arbitrarily low just by committing harder to an answer. Doing that *below* what the true conditional entropy $$H(p(y \mid x))$$ actually warrants isn't progress — it's the overconfidence from the section above, wearing a different name.

This is exactly the two-axis structure Gneiting, Balabdaoui & Raftery make precise: a good predictive distribution should maximize **sharpness** (how concentrated it is) *subject to* **calibration** (being no more concentrated than the truth supports) (Gneiting et al., 2007, *JRSS-B* 69(2):243–268). A single scalar loss like NLL blends both axes into one number — which is precisely why two models can land on the same NLL, or the same KL divergence, while one is sharp-and-correct and the other is either timidly hedging or confidently wrong.

## Measuring the two axes: ECE and sharpness

If NLL can't separate them, you need metrics that do.

**Expected Calibration Error (ECE)** operationalizes confidence calibration directly: bin predictions by reported confidence, and in each bin compare the average confidence to the empirical accuracy.

$$
\text{ECE} = \sum_{b} \frac{n_b}{N} \left\vert \text{acc}_b - \text{conf}_b \right\vert
$$

— a weighted average of the gap, bin by bin. Its blind spot follows directly from the two definitions above: ECE only checks the *aggregate* confidence-accuracy relationship, the weaker of the two calibration notions, and it's sensitive to how the bins are drawn. A model can score well on ECE while still being locally miscalibrated in exactly the pointwise-vs-average way described earlier.

**Sharpness** measures the other axis, and deliberately ignores correctness: how concentrated the predictions are, full stop. For classification, the average predictive entropy $$\mathbb{E}_x[H(q(\cdot \mid x))]$$; for regression, average predicted variance. Lower is sharper. A model can be arbitrarily sharp and arbitrarily wrong — sharpness alone certifies nothing — which is exactly why it only means something read *alongside* ECE: a confidently-wrong model posts excellent sharpness and terrible ECE, a flat maximum-entropy model posts excellent ECE and terrible sharpness, and only sharp-*and*-calibrated is the actual target.

The visual version of ECE is the **reliability diagram** — bin confidence on the x-axis, empirical accuracy on the y-axis, and compare the resulting curve to the $$y=x$$ identity line; the vertical gap at each bin is exactly the term being averaged in the ECE sum above.

Put together: NLL (equivalently, average KL) tells you the *combined* distance between $$q$$ and $$p$$, but not which axis it's spent on. ECE and sharpness are how you find out.

## Where this lands

This is the same split that shows up concretely in the calibration results on [the Diff-HEF work](/thesis/thesis_main) — Gaussian-NLL, beta-NLL, and f-Cal baselines get compared against a learned harmonic likelihood on exactly ECE and KL divergence side by side, and the interesting results are the ones where two methods land close on one axis and far apart on the other. The derivation here is the reason that comparison is the right one to run.
