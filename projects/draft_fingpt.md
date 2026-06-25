---
layout: post
title: "Probing FinGPT's Forecasting Module"
subtitle: "An empirical and ablative evaluation of unpublished claims"
date: 2026-06-02
author: "[FILL IN: your name]"
categories: [research, nlp, finance]
tags: [fingpt, llm, financial-forecasting, ablation, llama]
---

> FinGPT released a forecasting module. Only its sentiment results have been published. We ran the forecaster against classical and neural baselines on Dow 30 over a 2-week horizon, and probed what information it actually uses with an ablation that masks numerical values versus financial vocabulary.
>
> **Headline finding.** [FILL IN: one honest sentence — e.g., "FinGPT's forecasting performance is comparable to base-model and sentiment-only baselines, and our ablations are consistent with the forecasts depending more on textual context than on the numerical price history in the prompt — though confounds prevent a clean conclusion."]

---

## TL;DR

- **Gap.** FinGPT publishes sentiment results; its forecasting module has not been benchmarked in the original work.
- **Questions.** (1) Does it forecast, or is it sentiment analysis in forecasting clothing? (2) When it forecasts, how much does it rely on numerical values vs. textual context?
- **Setup.** Dow 30, 2-week horizon. Baselines: ARIMA, FinBERT, base Llama-2, base Llama-3, ChatGPT. Metrics: direction accuracy, MSE, ROUGE, and BERTScore (added — see why below).
- **What we found.** [FILL IN: 2–3 bullets with the actual findings, phrased honestly.]
- **What we couldn't conclude.** [FILL IN: the most important open question your evidence cannot settle.]

---

## Why this work exists

[FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) is an open-source LLM framework for financial NLP. The project includes a forecasting module that takes recent price history and news as input and produces a predicted direction (up/down), magnitude (% change), and a natural-language rationale for a 2-week horizon.

Published work around FinGPT focuses on **sentiment analysis** — the forecasting module's predictive performance has not been empirically benchmarked in the original release. That is the gap this project addresses: not a new model, but an honest evaluation of a claim that has been widely repeated and never tested.

The two questions that organize the rest of this writeup:

1. **Does the forecasting module actually forecast?** Specifically, does it produce predictions that beat sensible baselines — including a classical time-series model that uses no text at all?
2. **What is the model using when it produces a forecast?** Is the prediction driven by the numerical price history in the prompt, by the surrounding qualitative text, or by some combination — and in what proportion?

These questions matter beyond FinGPT. The first is a methodological check on a class of claims that has become common in LLM-for-finance work. The second probes a known weakness of LLMs (numerical reasoning) in a domain where numbers are not decorative.

---

## Why this is hard

Two separate difficulties stack on top of each other.

**Financial time series are stochastic.** Daily and weekly returns are noisy, non-stationary, and regime-dependent. Even well-tuned classical models (ARIMA, GARCH) struggle to predict directional movement at multi-week horizons much above chance. Any LLM-based forecaster competing in this space inherits the same difficulty plus its own.

**LLMs are weak on numbers.** This is well-documented across simple tasks (arithmetic, counting characters, comparison) and has structural roots in how tokenizers fragment numerical values. A number like `3,247.89` is not seen by the model as a quantity — it is seen as a sequence of digit-fragments and punctuation tokens that the attention mechanism must reassemble. Whether that reassembly happens reliably, especially in long prompts dense with other numbers, is an empirical question — and one our ablation is designed to probe.

---

## How FinGPT's forecaster works

The forecaster is built on Llama-2 7B and Llama-3 8B (instruction-tuned variants), fine-tuned with LoRA adapters on financial instruction data. The forecasting prompt is structured: it includes recent price history (numerical), recent news headlines or summaries (textual), and company-level context.

The output is a structured natural-language response: a directional prediction (up/down), a percentage magnitude, and a rationale.

[FILL IN: a small diagram or code block showing the prompt template, if you want to make the structure concrete for readers.]

---

## The right baseline: ARIMA

A common failure mode in LLM-for-X papers is to evaluate only against other LLMs. That comparison can show that one LLM is better than another while saying nothing about whether *any* of them beat the obvious non-LLM approach.

For univariate time-series prediction on price returns, the obvious non-LLM approach is ARIMA — an Auto-Regressive Integrated Moving Average model. It uses only the numerical price history. No text, no semantics. If the LLM-based forecaster cannot beat ARIMA at the same horizon, then the LLM machinery is not earning its keep on the forecasting task — whatever else it might be doing.

We include ARIMA as the floor of the comparison, not because we expect it to win, but because the comparison is dishonest without it.

---

## Data

- **Universe.** Dow Jones 30 constituents.
- **Time window.** [FILL IN: start date — end date].
- **Train/test split.** [FILL IN: describe the split policy — purely temporal, rolling window, etc.].
- **Horizon.** 2 weeks ahead.

A representative prompt (excerpted):

```
[FILL IN: paste a real prompt — price history block + news block + instruction.]
```

The prompt format matters more than it sounds. Subtle differences in how the price history is formatted (raw values, percentage returns, log returns) and how news is summarized can shift performance meaningfully. We held the format fixed across all model conditions to make comparisons fair; we did not systematically study prompt-format sensitivity, which is a limitation we return to below.

---

## Metrics — and why we added one

The original FinGPT work reports several metrics, of which the most relevant for our purposes are:

- **Binary direction accuracy.** Fraction of correctly predicted up/down moves. Clean, interpretable.
- **MSE on percentage return.** Continuous error on the magnitude prediction.
- **ROUGE.** N-gram overlap on the rationale text.

ROUGE has a known weakness for this kind of evaluation: it rewards lexical match, not semantic correctness. A forecast that says *"the stock will rise"* gets penalized against a reference that says *"the price will go up"* — even though the two are equivalent claims. For sentiment-rich, paraphrase-rich domains like financial commentary, this matters.

We added **BERTScore** as a complement. BERTScore uses contextual embeddings to measure semantic similarity between the predicted and reference rationale. It does not replace ROUGE — it sits alongside it and tells you when ROUGE is penalizing a model for stylistic differences rather than substantive ones.

The numerical metrics (direction accuracy, MSE) are unaffected by this. The change is in how we evaluate the rationale.

---

## Question 1: Does it forecast?

The headline results table:

| Model              | Dir. Acc. | MSE       | ROUGE     | BERTScore |
|--------------------|-----------|-----------|-----------|-----------|
| ARIMA              | [FILL IN] | [FILL IN] | —         | —         |
| FinBERT            | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Llama-2 (base)     | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Llama-3 (base)     | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| ChatGPT            | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| FinGPT (Llama-2)   | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| FinGPT (Llama-3)   | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |

[FILL IN: 2–3 paragraphs walking the reader through what the table shows. Lead with the most important comparison (FinGPT vs ARIMA, or FinGPT vs base Llama, depending on what's most striking in your results). Be honest about the size of any gaps.]

### What we can say

- [FILL IN: specific claim 1 — phrased as "consistent with" rather than "shows that"]
- [FILL IN: specific claim 2]
- [FILL IN: specific claim 3]

### What we cannot say

- [FILL IN: e.g., that any observed lead is robust across market regimes or time windows]
- [FILL IN: e.g., that performance gaps reflect forecasting skill rather than sentiment-prediction skill — the two are confounded on this task]

### Why these results look the way they do

Two structural factors shape the picture.

**Encoder vs. decoder.** FinBERT is encoder-only with a classification head — bidirectional attention, no generative rationale. The Llama family and ChatGPT are decoder-only — autoregressive, generative, producing both forecast and rationale in one pass. The architectural mismatch means FinBERT's strong showing on certain metrics is not directly comparable to the LLMs' showings on others; the models are doing partially different things.

**Tokenizer differences.** Llama-3 expanded its vocabulary substantially relative to Llama-2, which changes how numbers and financial vocabulary are fragmented. Some of the Llama-2 vs Llama-3 performance delta may reflect tokenizer effects more than reasoning-capability effects. This is the bridge into Question 2.

---

## Question 2: What is the model using?

If a forecast is being produced, what is it being produced *from*? The prompt contains both numerical content (price history) and textual content (news, context). We can probe which the model relies on by masking each in turn and measuring the change in performance.

### How LLMs see numbers

Before showing the ablation, it helps to make the tokenization concern concrete. Consider the value `3,247.89`:

- **Llama-2 tokenizer:** [FILL IN: token split — e.g., `['3', ',', '247', '.', '89']` — 5 tokens]
- **Llama-3 tokenizer:** [FILL IN: token split — N tokens]

The model never sees this as a single number. It sees a sequence of fragments that attention must compose. Two consequences:

1. **Numerical comprehension is emergent, not built-in.** The model has to learn — across training — that these fragments together represent a quantity.
2. **Long, number-dense prompts dilute attention.** The price-history portion of a forecasting prompt can run to many tokens, with numerical positions spread far apart. Cross-number reasoning (e.g., comparing values, computing returns) competes for the same attention budget that the model uses to read news context.

This motivates the testable conjecture behind the ablation: **if the model is mostly using textual context to forecast, then masking numbers should cost relatively little — but masking financial terms should cost a lot.**

### Ablation design

Three conditions, evaluated on the same test set:

1. **Baseline.** Full prompt — numbers and financial vocabulary intact.
2. **Numbers masked.** Numerical values replaced with [FILL IN: describe mask token and strategy — e.g., `<NUM>`].
3. **Financial terms masked.** Domain-specific terminology replaced with [FILL IN: describe mask token, strategy, and how terms were identified].

Comparing each masked condition to baseline isolates the contribution of that information channel.

[FILL IN: a short inline example showing the same prompt in all three conditions makes this concrete for readers.]

### Ablation results

| Condition              | Dir. Acc. | MSE       | BERTScore |
|------------------------|-----------|-----------|-----------|
| Full prompt            | [FILL IN] | [FILL IN] | [FILL IN] |
| Numbers masked         | [FILL IN] | [FILL IN] | [FILL IN] |
| Financial terms masked | [FILL IN] | [FILL IN] | [FILL IN] |

[FILL IN: 1–2 paragraphs interpreting the table. Lead with the most striking comparison. State the magnitude of each drop and what it implies, with appropriate hedging.]

### What the ablation does — and doesn't — show

The result is suggestive but not decisive. Several confounds remain.

- **Distribution shift from masking.** A prompt full of mask tokens is out-of-distribution for the model. Performance degradation may reflect distribution shift rather than information loss.
- **Asymmetric token counts.** Masking financial terms removes more tokens (and more information density per token) than masking numbers. The two conditions are not strictly apples-to-apples.
- **Signal carried by the mask token itself.** A `<NUM>` token communicates that something numerical was there. The mask is not the same as deletion.

Each of these is addressable with follow-up experiments. None of them are addressed in the present work.

### A separate probe: chain-of-thought

As a behavioral cross-check, we prompted the model to produce step-by-step reasoning before its forecast, and inspected the reasoning text for references to numerical values versus textual cues.

[FILL IN: 1–2 representative example outputs.]

The pattern: [FILL IN: e.g., "reasoning chains rarely reference specific price values, even when those values are provided in the prompt — they tend to summarize the news context and produce a directional verdict directly."]

An important caveat: chain-of-thought output is not a faithful trace of internal computation. It is what the model is willing to *say* about its reasoning, which can come apart from what the model is actually doing. We treat the CoT result as one piece of behavioral evidence, not as a mechanistic claim.

---

## Coming back to the two questions

**On Question 1 — does it forecast?** [FILL IN: honest one-paragraph answer. The phrase "the evidence is informative but does not settle the question" is acceptable if true — and is preferable to manufactured confidence in either direction.]

**On Question 2 — what is it using?** [FILL IN: honest one-paragraph answer. Pull together the ablation result and the CoT probe. Be explicit about the leading interpretation and the alternative that the evidence cannot rule out.]

---

## Limitations

The boundaries of these findings, stated up front:

- **Time window.** A single 2-week horizon on one date range. Regime sensitivity is not characterized.
- **Universe.** Dow 30 only — large-cap US equities. Results may not generalize to small caps, other markets, or other asset classes.
- **Model versions.** Fixed Llama-2 and Llama-3 checkpoints. Newer base models are not evaluated.
- **Prompt format.** Held constant across conditions. Sensitivity to phrasing not systematically studied.
- **Masking strategy.** A single design. Alternative designs (partial masking, granular masking, replacement with plausible-but-wrong values) may yield different conclusions.
- **Statistical power.** [FILL IN: sample size, variance, multiple-comparison considerations].

---

## What would settle these questions

A short list of experiments that would convert "suggestive" into "decisive":

- **Multiple time windows across regimes** (volatile / quiet / trending). The most important next step. Single-window results on stochastic time series are intrinsically fragile.
- **Counterfactual numeric perturbation.** Replace numbers with wrong-but-plausible values and measure how much the forecast changes. If the forecast is invariant, the model is not using the numbers; if it changes meaningfully, the model is.
- **Cross-tokenizer experiments** holding the base model constant, to separate tokenization effects from capability effects.
- **Finer-grained ablation.** Per-sentence and per-entity masking; partial masking of numbers (e.g., decimals only); masking news headlines vs. analyst commentary separately.
- **Mechanistic analysis.** Attention pattern inspection and activation patching to complement the behavioral probes used here.

---

## Code and data

[FILL IN: link to the repository, datasets, fine-tuning scripts, evaluation harness, anything that would allow another researcher to replicate or extend the work.]

---

## Acknowledgments

[FILL IN: people who helped, compute providers, advisors, etc.]

## References

1. [FILL IN: FinGPT paper citation]
2. [FILL IN: relevant LLM-numerical-reasoning literature]
3. [FILL IN: BERTScore, ROUGE references]
4. [FILL IN: ARIMA / classical forecasting reference]