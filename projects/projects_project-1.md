---
layout: post
title: "Probing FinGPT's Forecasting Module"
---

# Probing FinGPT's Forecasting Module

**An Empirical and Ablative Evaluation of Unpublished Claims**  

**Repo:** [github.com/korra141/FinGPT](https://github.com/korra141/FinGPT) \
**Based on:** [FinGPT — Yang et al., 2023](https://arxiv.org/abs/2306.06031) \
**Dataset:** [DOW30 · HuggingFace](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405) · [Sequential DOW30 (42-week)](https://huggingface.co/datasets/korra141/fingpt-forecaster-dow30-sequential_2/settings) \
**Models:** [fingpt-forecaster-llama3-lora](https://huggingface.co/korra141/fingpt-forecaster-llama3-lora) · [fingpt-forecaster-llama2-lora](https://huggingface.co/korra141/fingpt-forecaster-llama2-lora) \
**CoT Dataset:** [korra141/fingpt-dow30-cot-reasoning](https://huggingface.co/datasets/korra141/fingpt-dow30-cot-reasoning) \
**Stack:** PyTorch · HuggingFace · LLaMA 2/3 · LoRA · Chain-of-Thought · ARIMA · XGBoost

---

> FinGPT released a forecasting module. Only its sentiment results have been published. We ran the forecaster against classical and neural baselines on Dow 30 over a 2-week horizon, and probed what information it actually uses with ablations that mask numerical values versus financial vocabulary, before attempting to improve it with Chain-of-Thought fine-tuning.
>
> **Headline finding.** Fine-tuning extracts genuine directional signal — FinGPT (LLaMA-3) leads all baselines on accuracy — but magnitude prediction sits at the level of a driftless random walk. After fine-tuning, the model builds joint representations of numbers and financial vocabulary, though this does not imply emergent numerical reasoning: whether it has learnt that specific ratios are more important, or that decimals carry magnitude, cannot be deduced from these results. CoT fine-tuning produced a marginal directional improvement at the cost of magnitude accuracy.

---

## The Gap

FinGPT is an open-source LLM framework for financial NLP. It ships a forecasting module that takes recent price history, news headlines, and company fundamentals as input and outputs a directional prediction (up/down), a magnitude estimate (% change), and a natural-language rationale. The original paper published results almost exclusively for sentiment analysis. **The forecasting module's actual predictive performance was never benchmarked.**

This project fills that gap — and uses the benchmarking exercise to probe two deeper questions about what the model is actually doing, before attempting to improve it with Chain-of-Thought fine-tuning.  

---

## Two Questions

**Q1 — Does it forecast?**
Or is it effectively a sentiment classifier dressed in forecasting clothing?

**Q2 — What is it using?**
When it produces a forecast, how much does it rely on numerical content (prices, ratios, EPS) versus qualitative financial language ("beat", "missed", "surged")?

Both are evaluated empirically on the Dow 30 dataset over a 2-week forecasting horizon, with models uploaded to HuggingFace.

---

## Why This Is Hard

Two separate difficulties compound each other.

**The task is structurally hard.** Weekly price movement is noisy, non-stationary, and regime-dependent. Even well-tuned classical models struggle to predict direction at multi-week horizons much above chance. The two-price-per-week data format here gives time-series models almost no sequential structure to exploit — any forecaster is working from thin signal before it even opens the news.

**LLMs are structurally weak on numbers.** Tokenizers fragment numerical values into sub-word pieces. A price like `163.56` does not arrive as a quantity — it arrives as a sequence of digit-tokens whose meaning the attention mechanism must reassemble. Whether that reassembly happens reliably in long, number-dense prompts is exactly what the ablation below is designed to probe.

---

## The Dataset

The input to each prompt is a heterogeneous mix of three data types:

| Source | Content | Quality |
|---|---|---|
| **Price points** | 2 price endpoints per week (e.g. 163.70 → 163.56, −0.085%) | Weak — too sparse for any time series model |
| **News articles** | ~5 headlines per week | Mixed — 3/5 typically relevant, rest noise |
| **Fundamentals** | 30+ quarterly ratios (PE, ROE, D/E, FCF…) | Moderate — context, not a weekly price driver |

Two price endpoints per week is too sparse for classical time series models to work with — ARIMA needs a sequence, not two snapshots. To give it a fighting chance, 42 weeks of data were collated into sequential windows before training ARIMA.

Even with more data points, short-term prices are hard to predict from fundamentals or headlines alone. News and quarterly financials tend to move stock valuations over weeks or months, not the next trading week. There is a lag — by the time the market fully prices in a development, the window you are trying to predict has already closed. This makes the forecasting task genuinely hard for every model here, not just the LLMs.  

### Class Distribution and Distribution Shift

|  | Up | Down | Total | % Up | % Down |
|---|---|---|---|---|---|
| Train | 687 | 543 | 1,230 | 55.9% | 44.1% |
| Test | 137 | 163 | 300 | 45.7% | 54.3% |

The train set covers a **bullish market window**; the test set covers a **bearish one**. Any model that absorbed the upward bias during training is structurally penalised at test time. The meaningful bar to beat on the test set is **54.3%** (majority class) — not 50%.

---

## Results

A common failure mode in LLM benchmarks is comparing only against other LLMs — which can show one model beats another while saying nothing about whether any of them beat the obvious non-LLM approach. For price prediction, that approach is ARIMA: no text, no semantics, just the price sequence. ARIMA is included not because it is expected to win, but because the comparison is dishonest without it. Its performance sets the floor for what structured numerical signal alone can achieve.

| Model | Dir. Acc. ↑ | MSE ↓ | ROUGE ↑ |
|---|---|---|---|
| **FinGPT (LLaMA-3)** | **0.6122** | 7.2653 | 0.2467 |
| FinGPT (LLaMA-2) | 0.5102 | 9.7142 | 0.2425 |
| LLaMA-3 (base) | 0.4568 | 19.9748 | 0.2387 |
| LLaMA-2 (base) | 0.4201 | 28.447 | 0.2023 |
| GPT-4 | 0.3506 | 24.5682 | 0.1674 |
| FinBERT | 0.4107 | 17.9348 | — |
| ARIMA | 0.5111 | 8.2926 | — |
| XGBoost | 0.4782 | 8.8607 | — |
| Linear Regression | 0.4600 | 7.4170 | — |
| **Driftless Random Walk** | — | **7.1150** | — |
| Class Distribution | 0.5040 | — | — |

**What these results show:**
- FinGPT (LLaMA-3) leads all baselines across directional accuracy, MSE, and ROUGE. Fine-tuning clearly improves which direction the model predicts, but its MSE of 7.27 is barely above the driftless random walk (7.12) — a baseline that predicts zero price change every week. This gap between strong directional accuracy and weak magnitude accuracy is the first hint that the model is learning something closer to sentiment signal than true price forecasting.

- The ~20 percentage point accuracy gap between FinGPT (LLaMA-3) and FinGPT (LLaMA-2) is striking given that the fine-tuning recipe is nearly identical. The most concrete structural difference is the tokenizer: LLaMA-3's 128K vocabulary represents many numbers as single tokens, while LLaMA-2's 32K vocabulary breaks them into digit-level fragments that share no numerical relationship. Whether this actually explains the gap is tested in the masking ablation below.

- FinBERT scores below the class distribution baseline (0.41 vs 0.50). FinBERT is encoder-only with a classification head — bidirectional attention, trained to label sentiment in earnings reports and financial news. The LLaMA models are decoder-only, generating a directional prediction and rationale in a single autoregressive pass. These are architecturally different tasks using the same prompt format. FinBERT's poor performance reflects that mismatch, not a general weakness.

- GPT-4 and both base LLaMA models also fall below the class distribution baseline. General language understanding and scale are not sufficient here — without fine-tuning on domain-specific examples, these models have no basis for calibrating their predictions to this data format.

- ARIMA operates on a fundamentally different input than the LLMs: 42 weeks of sequential price history, constructed specifically to give it enough autocorrelation signal to work with. The LLMs see only 2 weeks of prices but also receive news headlines and financial ratios. Despite this input disadvantage, ARIMA still trails FinGPT — which suggests the structured sequential signal is less useful here than the qualitative context. XGBoost is the more like-for-like comparison: it ingests the same heterogeneous mix of prices and fundamentals as the LLMs, and its accuracy of 0.478 reflects what a well-regularised model can extract from those same inputs without any language understanding.

**What these results do not show:**
- Whether the forecasts are calibrated. Directional accuracy of 61% says the model is right more often than chance, but not whether it is confident when it is right and uncertain when it is wrong. A model that is 61% accurate and well-calibrated is useful; one that is confidently wrong half the time is not.
- How much of the error is irreducible. Short-term price movement is inherently noisy — some portion of the loss is not recoverable regardless of model quality. Without an estimate of the noise floor for this dataset and horizon, it is impossible to say whether FinGPT is close to the ceiling or still well below it.
- Whether the model is reasoning over numbers or responding to sentiment-coded language. Terms like "beat expectations" or "record earnings" directionally imply price movement without any numerical computation. 
- Whether FinGPT is computing price magnitude or simply predicting the mean. Weekly price moves in this dataset are small — typically 0 to 2%. A model that always predicts near-zero change will score a competitive MSE, which is exactly what the driftless random walk does at 7.12. FinGPT's MSE of 7.27 is marginally worse, not better — so there is no evidence it is recovering magnitude from the numerical inputs rather than outputting values close to the training distribution mean.

## How Numbers Are Tokenised

Before running ablations, it helps to understand why numerical sensitivity is structurally hard for these models. Tokenisers segment numbers into sub-word pieces — not numeric units. The model sees token IDs, not quantities.

| Number | LLaMA-2 (SentencePiece, 32K) | GPT-4 / LLaMA-3 (tiktoken, ~100K–128K) |
|---|---|---|
| `9.11` | `[9] [.] [1] [1]` | `[9] [.] [11]` |
| `9.2` | `[9] [.] [2]` | `[9] [.] [2]` |
| `1000` | `[1] [000]` | `[1000]` |
| `3.14159` | `[3] [.] [1] [4] [1] [5] [9]` | `[3] [.] [14] [159]` |

**What those token IDs can and cannot encode:** Through training on large text corpora, common single-digit integers develop weak ordinal structure — `6` tends to appear in contexts between `5` and `7`, so embeddings reflect something like relative ordering. But this breaks down quickly for multi-digit numbers, decimals, and financial ratios. `163.56` and `164.02` share no tokens and nothing in their embeddings encodes that they differ by 0.46. The model has no representation of numeric magnitude — only of which tokens tend to appear near which other tokens.

**Why this is especially problematic here:** The DOW30 dataset is dense with prices in decimals, PE ratios, EPS figures, and percentage changes — precisely the numbers that fragment most and whose embeddings reflect text co-occurrence rather than numeric relationships. As the table above shows, the same number produces different token sequences in LLaMA-2 and LLaMA-3, meaning the two models have learned structurally different — and likely arbitrary — representations for the same quantity.

**A concrete mitigation:** Replace absolute values with relative rank within a peer group. Instead of "market cap of $5 billion," say "3rd largest market cap in the sector this quarter." Instead of a raw price change, express it as a percentile rank among DOW30 stocks for that week. Percentile ranks are bounded between 0 and 100, stable across time and stocks, and do not fragment under tokenisation. Whether this would improve magnitude prediction is an open and testable question — this study did not evaluate it. 

---

## Ablation 1: Does Token Length Matter?

FinGPT's output format is verbose — it generates an analysis section, a pros section, a cons section, and a directional prediction. The token budget consumed by this structure affects what the model can attend to.

90% of samples fit within 2,048 tokens, with the input prompt consuming roughly 80% of that budget. Yet **4,096 tokens yields the best forecast accuracy**.

<div align="center">
  <figure>
    <img src="/projects/assets/fingpt/token_ablation.png" alt="Token length ablation" width="1500">
    <figcaption>MSE and directional accuracy vs. token length for FinGPT (LLaMA-3)</figcaption>
  </figure>
</div>

The additional tokens at 4,096 are consumed by the structured reasoning output (analysis → pros → cons → prediction) that *grounds* the final prediction in intermediate steps. Beyond 4,096, the structure becomes verbose — **ROUGE and BERTScore improve** (the text is richer and more fluent) but **MSE and directional accuracy degrade** (the prediction is buried in padding).

| Token Length | Bin Accuracy | MSE |
|---|---|---|
| 2,048 | 0.571 | 12.434 |
| **4,096** | **0.612** | **7.265** |
| 8,192 | 0.571 | 8.347 |

This is a useful signal in itself: the model's best performance comes not from seeing more input, but from being given enough room to produce structured intermediate output.

---

## Ablation 2: Financial Terms vs. Numericals

To probe Q2 directly, two masking experiments were run over LLaMA-3 (base) and FinGPT (LLaMA-3 fine-tuned):

- **Mask numbers**: replace all numeric tokens with `[MASK]`
- **Mask financial terms**: replace domain-specific financial vocabulary with `[MASK]`
- **Mask both**

<div align="center">
  <figure>
    <img src="/projects/assets/fingpt/ablation_fingpt.png" alt="Masking ablation — LLaMA-3 base" width="700">
    <figcaption>LLaMA-3 base: masking numbers has almost no effect, masking financial terms degrades accuracy. Words carry the signal.</figcaption>
  </figure>
  <figure>
    <img src="/projects/assets/fingpt/ablation_fingpt_1.png" alt="Masking ablation — FinGPT" width="700">
    <figcaption>FinGPT (LLaMA-3): both numbers and financial terms degrade performance symmetrically when masked. Fine-tuning built joint representations across both modalities.</figcaption>
  </figure>
</div>

The logic: **if the model relies primarily on financial language, masking numbers should cost little — but masking financial terms should hurt significantly.** The reverse pattern would point to numerical content as the primary signal. Symmetric degradation under both conditions would suggest the two modalities are jointly encoded and cannot be separated. Each outcome answers Q2 differently.

### Base LLaMA-3: Words ≫ Numbers

| Condition | Δ Dir. Acc | Δ MSE |
|---|---|---|
| Mask Numbers | +1.0% | −8.3% (improves) |
| Mask Financial | −4.2% | −8.7% (improves) |

The asymmetry is consistent with what a base model's pre-training actually optimises for. Next-token prediction on internet text builds strong associations between financial language and market direction — "beat expectations" precedes positive outcomes, "missed guidance" precedes negative ones. It does not build representations of numeric magnitude. Words carry the contextual signal; numbers are largely noise.

The +1% improvement when masking numbers introduces a confound: masking removes tokens, shortening the prompt and freeing attention for the remaining language. To isolate whether the actual numeric *values* matter — not just their presence — the experiment was repeated with random number substitution: same positions, same token count, different values. Directional accuracy dropped by only 0.63%. The model is not using the numeric values to determine direction; it is using the language around them.

### FinGPT (Fine-tuned): Words ≈ Numbers

| Condition | Δ Dir. Acc | Δ MSE |
|---|---|---|
| Mask Numbers | −25.0% | +64.7% (degrades) |
| Mask Financial | −25.2% | +38.7% (degrades) |
| Mask Both | −18.9% | +55.3% (degrades) |

FinGPT degrades symmetrically when either modality is removed. LoRA fine-tuning on the Dow30 dataset has built **joint representations** — phrases like "stock rose 4% this quarter" or "bullish run ending at $235" are encoded together. Neither the number nor the qualifier can be dropped without losing the signal.

**What the ablation shows:** Fine-tuning built joint representations across both modalities — reflected in the 63.6% MSE reduction over base LLaMA-3, and in a striking shift in sensitivity: masking financial terms causes a 25% accuracy drop in FinGPT versus only 4% in the base model. Numbers matter too, in a way they did not before fine-tuning.

**What the ablation cannot show:** FinGPT clearly attends to numeric tokens — particularly for price magnitude — but whether this constitutes *true numerical reasoning* or *learned pattern completion* over the fine-tuning distribution remains open. Distinguishing the two would require out-of-distribution inputs where numeric values are systematically perturbed while language is held fixed — which this study did not test.

---

## Ablation 3: Chain-of-Thought as a Probe

If the model is doing real reasoning, making that reasoning explicit should improve it. Chain-of-Thought fine-tuning trains the model to produce intermediate steps before its final prediction — a proxy for structured reasoning, where the explanation is intended to causally lead to the answer rather than follow from it.

LLaMA-2 (7B) was fine-tuned on CoT traces generated by GPT (`openai/gpt-oss-20b`) on the DOW30 dataset, with the resulting dataset published to HuggingFace.

| Model | Dir. Acc. | MSE | ROUGE |
|---|---|---|---|
| FinGPT (LLaMA-2) | 0.5102 | 9.7142 | 0.2425 |
| FinGPT + CoT   |  0.5152 | 10.2586 | 0.2495 |
| Δ | **+0.98%** | **+5.6%** | +2.88% |

The results are mixed: directional accuracy improved marginally (+0.98%) and text quality improved slightly (ROUGE +2.88%), but magnitude accuracy degraded (MSE +5.6%). CoT did not produce a meaningful analysis section. Three structural reasons explain why:

**Strong Q&A prior.** The instruction-tuned base model used for FinGPT was trained to answer directly and concisely. To override this with CoT behaviour, a higher LoRA rank was used to update more weights across the attention and projection layers — but overriding a deeply reinforced prior with a small gradient update is fundamentally difficult.

**Trace quality mismatch.** Traces from a larger teacher model may be too long or discursive for a 7B model to reproduce faithfully. The dataset was cleaned to remove incoherent outputs, but the reasoning quality was not independently verified — it is not guaranteed that the traces reflect genuine financial inference rather than well-formatted rationalisation after the fact.

**LoRA constraint.** LoRA updates a low-rank perturbation of frozen weights — sufficient for adapting style and domain vocabulary, but likely too constrained to instil a fundamentally different generation pattern. Reliable CoT behaviour in small models typically requires distillation from a significantly larger teacher, not just fine-tuning on teacher-generated traces. May require more training.

A sample of what the generated CoT traces looked like in practice:

```
"We need to produce structured output: Positive Developments, Potential Concerns, 
Prediction Use data given We must consider news headlines and financials Also mention 
prediction but we can predict price movement or direction maybe moderately bullish due 
to dividend announcement, new integration, etc But we need to use reasoning steps we 
need to include prediction section after that need to produce final string with exactly 
those sections Must not add extra lines like assistant: Need to output exactly same 
structure 's do reasoning steps::- March Dogs of the Dow: buy 1; watch 4 - indicates 
AXP included as part of top dogs Good signal for momentum - GARP strategies: AXP 
mentioned as promising [...]"
```

The model is narrating the output format it should produce rather than reasoning about the financial content. Chain-of-thought for financial data is intrinsically difficult: there is no single canonical reasoning chain, causality is hard to verify, and post-hoc rationalisation is indistinguishable from genuine inference.

An important caveat: CoT output is not a faithful trace of internal computation — it is what the model produces as text, which can come apart from what it actually attends to. The sample above is behavioral evidence that something went wrong in the generation; it is not a mechanistic claim about what the model computed.

---

## Answering the Two Questions

**Q1 — Does it forecast?**

Partially. FinGPT (LLaMA-3) extracts genuine financial signal — it beats every ML and LLM baseline on directional accuracy. But its MSE sits above the driftless random walk, meaning a model that predicts zero price change every week outperforms it on magnitude. Directional accuracy of 61% is above chance, but modest given the distribution shift between train and test. The pattern — strong on direction, weak on magnitude — is more consistent with an informed sentiment proxy than a true price forecaster.

**Q2 — What is it using?**

After fine-tuning, both numbers and financial terms are attended to — LoRA on the Dow30 data encoded co-dependence between numerical values and financial vocabulary, reflected in the 25% symmetric degradation under masking. The random number substitution experiment adds a stronger data point: replacing numeric values with different ones — while keeping the prompt structure intact — degrades price magnitude prediction, confirming the model is actively reading the numbers rather than merely reacting to a disrupted prompt format.

But reading numbers is not the same as reasoning over them. The evidence is consistent with the model having learnt a distribution over average price changes during training and using numerical tokens as a weak conditioning signal — not computing from specific values. It has not demonstrably learnt which fundamentals or ratios are predictive for a given stock over a 1-week window. The magnitude predictions cluster too close to the training distribution mean for that interpretation to hold.

---

## Limitations

- **Single time window.** Train and test cover one bullish and one bearish period respectively — regime sensitivity is uncharacterised.
- **Dow 30 only.** Large-cap US equities. Results may not generalise to smaller stocks, other markets, or other asset classes.
- **Tokenizer not varied.** All experiments use the tokenizers shipped with each model. Whether a tokenizer that treats decimals and financial ratios as single units — rather than fragmenting them into digit-level pieces — would change performance is untested, and is a plausible confound in the LLaMA-2 vs LLaMA-3 accuracy gap.
- **Masking confounds.** A masked prompt is out-of-distribution for the model. Token-count asymmetry between masking conditions is partially controlled by the random-substitution experiment, but not fully resolved.
- **CoT trace quality.** Traces were cleaned but not independently verified for genuine financial reasoning.

---

## What Would Strengthen These Findings

- **Multiple market regimes.** Testing across volatile, quiet, and trending periods would establish whether directional accuracy is robust or regime-specific.
- **Richer price data.** More price points and volume data would let classical baselines compete on equal footing and reveal how much sequential structure matters.
- **Percentile-ranked numerics.** Replacing absolute values with peer-group percentile ranks would test whether relative representation mitigates the magnitude problem in representation of tokens.
- **Calibration analysis.** Directional accuracy of 61% says nothing about confidence. ECE and reliability diagrams would reveal whether the model is well-calibrated or confidently wrong in systematic ways.

# Numerical Reasoning in LLMs

Reasoning about ratios takes two skills, and LLMs are much better at one than the other. The first is **picking the right metric** — knowing that P/E, market cap, or EV/EBITDA matter for a forecast. This is really a language task, and models do it well. The second is **reading the number** — knowing what a value means and how much a small change matters. This is where they struggle, because sensitivity depends on the quantity, not the number. A P/E moving from 15 to 16 is only a 6.7% change but shifts the valuation claim entirely, while 23°C to 24°C means nothing for comfort. **How much precision you need is a property of the task, not the number.**

Part of the difficulty is tokenization. Round numbers under 1000 get their own token — "100" is a single, common token, and it beats "123" not because of its value but because round numbers appear across far more contexts involving magnitude, giving them richer and more reliable embeddings. Large integers break differently: `60000001` isn't a rare token, it is **no token at all** — it gets split, and the model must rebuild place value from the fragments. GPT compounds this by chunking left-to-right, so split points shift whenever there's a carry ("1000" becomes `["100","0"]`); forcing right-to-left grouping measurably improves arithmetic (Singh & Strouse, 2024). Decimals are worst: the `.` token mostly means "end of sentence" elsewhere in training, and the digits after it run *backwards* — moving right makes the number smaller. **P/E ratios, the case needing the most precision, sit in the worst tokenization regime.**

But this is more than a tokenization story. There is clear **ordinal structure** — the representation encodes that 6 is greater than 1 and less than 7 — and it would be wrong to conclude magnitude isn't represented at all. For numbers that *are* single tokens, Kadlčík et al. (2025) showed that probing with a **Fourier basis** recovers the actual numeric value to a fair number of digits, near-perfectly. Earlier verdicts of "imprecise representation" were partly a measurement failure: the wrong probe on the right structure.

The right picture is **not a number line but a helix**. Numbers wind around a circle — which handles the digits, since base-10 arithmetic wraps and a carry is just the wraparound — while climbing a **logarithmic z-axis**, which handles overall size. The spacing is therefore uneven: small numbers sit far apart, and the spiral **collapses as magnitude grows**, since 1→2, 10→20, and 100→200 each take the same vertical room. Precision is relative, not absolute — the model separates 6 from 7 easily but cannot reliably separate 6,000,001 from 6,000,002. **Reasoning breaks wherever the precision a task demands falls below the model's resolution at that magnitude** — which is exactly why a P/E of 15.9 versus 16.1, a 1.2% gap, is so hard.

---

## References

<a id="r1">[1]</a> Yang, H. et al. (2023). *FinGPT: Open-Source Financial Large Language Models*. [arXiv:2306.06031](https://arxiv.org/abs/2306.06031)

<a id="r2">[2]</a> Gruver, N. et al. (2023). *Large Language Models Are Zero-Shot Time Series Forecasters*. NeurIPS. 

<a id="r3">[3]</a> Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS.

<a id="r4">[4]</a> Magister, L. et al. (2022). *Teaching Small Language Models to Reason*. arXiv.

<a id="r5">[5]</a> Singh, A. K. & Strouse, D. (2024). *Tokenization Counts: The Impact of Tokenization on Arithmetic in Frontier LLMs*. [arXiv:2402.14903](https://arxiv.org/abs/2402.14903)

<a id="r6">[6]</a> Kadlčík, M. et al. (2025). *Pre-trained Language Models Learn Remarkably Accurate Representations of Numbers*. [arXiv:2506.08966](https://arxiv.org/abs/2506.08966)

<a id="r7">[7]</a> Kantamneni, S. & Tegmark, M. (2025). *Language Models Use Trigonometry to Do Addition*. [arXiv:2502.00873](https://arxiv.org/abs/2502.00873)
