---
layout: post
title: "How Much Numerical Reasoning Do LLMs Actually Do?"
date: 2026-05-18
---

# How Much Numerical Reasoning Do LLMs Actually Do?

## The target

**Numerical reasoning** = recovering what a quantity represents (from context or units) and acting on it. The key property: **required fidelity is task-conditional**.

| Quantity | Discrimination needed | Regime |
|---|---|---|
| Market cap $5B | order-of-magnitude | approximate suffices |
| Temp 21 vs 23°C | ~9% | tolerant |
| P/E 15.9 vs 16.1 | ~1.2% | precision-critical |

Ratios are the sharp end: quotients of noisy estimates (error compounds) where meaningful differences are small. Two skills get conflated — **metric selection** (which ratio? LLMs are good, it's linguistic association) vs **metric calibration** (what does P/E 16 *mean* here? LLMs are weak, and this is reasoning proper).

## The representation: two layers

**Layer 1 — coarse logarithmic magnitude.** 10→20 resembles 1→2, not 1→11. Precision declines as values grow (AlQuabeh 2025; arXiv:2502.16147, tying this to the human logarithmic mental number line). Log-magnitude explains ~55–63% of representational variance.

**Layer 2 — per-digit circular (base-10 Fourier) features.** Levy & Geva (2025, NAACL): direct value probes fail; **orthogonal circular representations per digit** succeed (Llama-3-8B, Mistral-7B), causally verified. Diagnostic signature: **errors distribute across the answer's digits rather than clustering around its value** — digit-wise, not magnitude-wise, computation. Kantamneni & Tegmark (2025) find a **helix** (linear + periods 2, 5, 10, 100) and the **"Clock" algorithm** — addition as rotation. Zhou et al. (2024, NeurIPS): MLPs approximate magnitude (low frequencies), attention does modular arithmetic (high frequencies).

**Why circles:** modular arithmetic *is* the arithmetic of angles — adding 3 to 7 rotates to 0, and the wraparound *is* the carry. A linear code would need unattainable precision to separate 1,000,047 from 1,000,048; circles give error-correcting redundancy.

> **Model it as an odometer, not a ruler**: stacked dials, dial-count ≈ scale, rotation = addition, wrapping = carry. The dials are well-formed; the *reading procedure* is unreliable.

## Frequency ≠ context

- **Context** (co-occurrence) determines **where** a number sits — the geometry. Similar magnitudes are **substitutable** in similar frames ("temperature was 71/73 degrees"; "7,100 degrees" is not). The number line is a shadow of substitutability.
- **Frequency** determines **how confidently** it sits there — sample size, not content.

At equal frequency, **"50" beats "1999"**: 50's contexts are saturated with units and comparatives; 1999's are calendar-flavored. What matters is diversity of **magnitude-bearing frames**, not raw count. Razeghi et al. (2022) established frequency→accuracy (>70% absolute gap, top vs bottom decile) — but that governs *fidelity*, not the code.

> *Context is what a number means. Frequency is how well we know it.*

## Benford's Curse: a correct prior in the wrong place

Corpora inherit P(d) = log₁₀(1 + 1/d) — leading **1 ≈ 30.1%** vs **9 ≈ 4.6%** (verified on OLMo2's Olmo-Mix-1124; arXiv:2506.01734). Models absorb this as a prior and handle small-leading-digit numbers more accurately.

**The prior isn't the pathology** — it's correct under genuine uncertainty. The failure is **leakage into deterministic computation**: computing 6,847 + 2,913 has a fixed answer, yet digit bias still tugs the output. Hence *systematic* hallucination, not random error.

## Tokenization: three distinct failures

**(a) Round numbers < 1000** get their own token — frequent, well-estimated embeddings. The regime where strong value-recovery holds.

**(b) Large integers fail *compositionally*.** `60000001` isn't a rare token — it's **no token**. Place value must be reassembled across chunks, and GPT's greedy **left-to-right** chunking shifts boundaries under carries ("1000" → ["100","0"]). Singh & Strouse (2024, arXiv:2402.14903): forcing **R2L** grouping via commas measurably improves GPT-3.5/4 arithmetic. **L2R is the problem; R2L is the fix.**

**(c) Decimals fail *worse*.** The `.` carries a strong **sentence-boundary prior** to override, and the fractional part **inverts place-value semantics** (rightward = smaller). **P/E ratios live exactly here** — the hardest example sits in the regime with the most structural interference. Testable, not coincidental.

## What digit-level tokenization buys

It doesn't *supply* magnitude — it **removes any whole-number embedding** (no vector means "47"). Magnitude becomes a property of *computation over a sequence*, not of the embedding.

- **Gain:** order of magnitude nearly free (token count ≈ log₁₀ — a natural explanation for *why* the manifold is logarithmic); immunity to rare-token blur (only ten tokens, all well-estimated).
- **Cost:** place-value composition becomes inference. Errors turn **procedural** (carries, mid-digit corruption) rather than **statistical** (ignorance).

Hence positional fixes dominate: **Abacus embeddings** (McLeish et al., 2024, NeurIPS) give equal-place digits identical positional embeddings; with reversed output and input injection, models trained on ≤20 digits generalize to **120-digit** addition (looped variant: 92.9% → 99.1% OOD). **Position Coupling** (Cho et al., 2024) is the independent twin.

> Same lesson as vanilla positional encoding: significance must be *injected*, not inferred.

## Is it reasoning?

**Pure pattern-matching is contested, not consensus.** Apple's **GSM-Symbolic** (ICLR 2025) reported drops up to 65% (GPT-4o 94.9% → 63.0% on GSM-NoOp). But Ivanova's re-analysis showed much of that variance sits inside confidence intervals, and a **2026 replication** (Sturgeon) found that once distractors are audited to genuinely irrelevant ones, **"every audited drop is statistically indistinguishable from zero."** Meanwhile Clock/Fourier circuits are *actual computation*, not lookup.

**The defensible middle:** Nikankin et al. (2024), *Arithmetic Without Algorithms* — neither robust algorithm nor memorization, but a sparse **bag of heuristics** (operand-range and modulo detectors) combining unordered. Anthropic's circuit tracing of **Claude 3.5 Haiku** (2025) corroborates: parallel **approximation** ("near 92") and **modular** ("ends in 5") paths recombine — and the model **cannot introspect it**, reporting textbook carrying instead.

> *imprecise ≠ absent. heuristic ≠ pattern-matching.*

## The unifying hypothesis

The P/E-vs-temperature intuition **is Weber's Law**: just-noticeable difference scales with magnitude. A log-compressed code has **relative, not absolute, resolution**. This makes the account predictive:

> **Numerical reasoning fails where task-required precision drops below the model's magnitude-dependent discrimination threshold.**

| Case | Separation | Task demand | Prediction |
|---|---|---|---|
| $5B market cap | order-of-magnitude | order-of-magnitude | **safe** |
| 21 vs 23°C | ~9% | coarse | **safe** |
| P/E 15.9 vs 16.1 | ~1.2% | fine | **breaks** |

Ratios, decimals, and large-integer arithmetic fall out of one mechanism.

---

## Corrections from the original draft

1. **Magnitude was under-claimed.** Kadlčík et al. (2025, arXiv:2506.08966): with a **Fourier-basis probe**, values are recoverable from single-token number embeddings near-perfectly — and they argue arithmetic failure should *not* be blamed on representation. Earlier "imprecise representation" verdicts were largely an artifact of **linear probes on a periodic geometry**. → **The representation may be sounder than the computation using it.** Blame the circuit, not the embedding. *(Scope: "single-token" = one token in vocabulary. The digit-tokenization case above is mechanistic reasoning, not a verified citation.)*

2. **Regression-initialization link removed — wrong.** Initialization bias is an *optimization artifact* (untrained weights near zero); Benford's Curse is a *successfully-learned data statistic*. Both point "small," but shared direction ≠ shared cause. Keeping the link predicts re-initialization helps; it won't.

3. **Two tokenization failures separated** (compositional vs radix-prior + inverted place value) — the distinction predicts the P/E difficulty. And **L2R → R2L**.

4. **Frequency and context disentangled** — context builds the geometry, frequency sets its resolution.

5. **Weber's Law promoted** from aside to organizing principle — it's what makes the account predictive rather than descriptive.
