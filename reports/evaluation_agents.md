# LLM Judges and Evaluators: The Case for Automated Quality Infrastructure

## The problem you already know

If you have deployed an LLM in production, you have encountered the same gap: development-time testing passes, but real-world quality is harder to measure. You run a sample of outputs past a human reviewer weekly. You write a few test cases that cover the scenarios you already thought of. You add a prompt instruction when a user complains.

This is not an evaluation strategy. It is reactive quality control with no statistical basis, and it cannot scale.

The alternative most teams reach for first, asking the generating model to score its own outputs, does not work. The data on why is unambiguous.

## Why the generating model cannot judge itself

LLMs show a systematic performance gap between generating a response and detecting errors in one. In controlled benchmarks, models are measurably worse at hallucination detection than at hallucination generation, with multimodal models showing accuracy drops of up to 31% when asked to catch their own errors, even when the correct visual information is present in context [[1]](https://aclanthology.org/2024.acl-long.648.pdf).

Multi-turn settings make this worse. Once a hallucination enters context, it snowballs: subsequent outputs are conditioned on the error, and even a model that could identify the hallucination in isolation often cannot surface it once it has been incorporated. This is the Achilles heel of reasoning models and long-context pipelines specifically.

Running the same model a second time does not fix this. In our HaluBench experiments, single-pass and multi-pass judgment with the same model showed no statistically significant difference: 50.05% vs 50.07% pass rate with fully overlapping confidence intervals (Cohen's h = 0.001, n = 10,586). The ceiling is the model's own blind spot, not the number of passes.

**What you need is an independent evaluator**: one that was not involved in the original generation and is not conditioned on accepting the output as correct.

## What Scorable judges and evaluators do

A note on terminology: in industry literature, "judge" refers to any LLM configured for evaluation (the model doing the judging). In Scorable's API, a **judge** is specifically a pipeline object that groups one or more evaluators; the **evaluator** is what performs the actual judgment. Where this document uses "judge model" or "LLM judge", it refers to the industry concept.

In Scorable, a **judge** is a named, versioned evaluation pipeline that groups one or more **Scorable evaluators** and runs them together against your model outputs. A Scorable evaluator is the callable unit: it takes a request, a response, and optional context, applies a scoring rubric, and returns a score with justification. The model backing each evaluator is your choice.

Scorable evaluators are what the research community calls *evaluation agents*: LLM judge models configured specifically for judgment tasks rather than general-purpose generation. The key properties:

**Rubric inference from examples.** Given a handful of labeled input-output pairs, the evaluator infers scoring criteria automatically. In our pipeline, 3 pairwise preference examples are sufficient to generate 5 rubrics that generalize to unseen inputs. You do not need a pre-specified scoring rubric to get started.

**Calibrated confidence alongside every judgment.** Each evaluation result includes a confidence score. This is not cosmetic; it is load-bearing for how you use the output. High-confidence judgments can be trusted and acted on automatically. Low-confidence judgments are routed for human review. This is how you reduce review volume without reducing quality coverage.

**Batch processing with bootstrap confidence intervals.** Results are persisted per experiment run with bootstrapped 95% CIs on all accuracy metrics (default n=1,000 resamples). This means consecutive runs on the same benchmark are statistically comparable, so you can detect regressions rather than just observing them.

## The accuracy numbers

**Baseline evaluation on SimpleQA (n=200):**

- Evaluator accuracy: **87.4% [95% CI: 82.5%, 91.5%]**
- Confidence vs. accuracy correlation: **r = 0.87 [CI: 0.65, 0.94]**
- Confidence vs. false positive rate correlation: **r = -0.83 [CI: -0.94, -0.42]**

The confidence signal is genuinely informative. Higher-confidence judgments are more accurate, and lower-confidence judgments predict where false positives occur. This enables a tiered approach: automate high-confidence decisions, escalate uncertain ones.

**On harder tasks (SimpleQA full pipeline, n=200):**

| | Accuracy | 95% CI |
|---|---|---|
| Generator model | 14.5% | [9.5%, 19.5%] |
| Scorable evaluator | **89.0%** | [84.5%, 93.0%] |

The Scorable evaluator is not marginally better than the generator at assessing its own outputs; it is categorically better. The generator is correct on roughly 1 in 7 responses. The evaluator catches errors correctly in 9 out of 10 cases.

**Inter-judge-model agreement on HaluBench (n=10,586):**

Claude Sonnet 3.5 judge model agreement rate: **84.1%** across the full HaluBench test set. On ambiguous answer types, evaluator accuracy reaches **94.7% [CI: 94.0%, 95.5%]**.

## Why not just use a frontier API model as a judge model?

You can, and for many tasks it works well. Three reasons to prefer a dedicated Scorable evaluator for production infrastructure:

**Cost and latency at scale.** Evaluation runs happen continuously: on every deployment, every model update, every A/B test. Running a frontier model for each evaluation call at production volume is expensive. Judge models achieve comparable accuracy at FP8 quantization, halving memory requirements and roughly doubling throughput versus bf16 alternatives of the same parameter count.

**Judge-specific training.** General-purpose frontier models are trained on human preference data optimized for helpfulness and harmlessness, not calibrated judgment. Purpose-built evaluators fine-tuned with DPO on preference data consistently outperform same-size general models on instruction-following benchmarks:

| Model | IFEval (loose, inst-level) | Size |
|---|---|---|
| RS-Pairwise-70B DPO | **94.0%** | FP8 ~70GB |
| Llama-3.3-70B (base) | 93.4% | bf16 ~140GB |
| Patronus-Lynx-70B | 83.7% | bf16 ~140GB |
| Nemotron-70B | 85.0% | FP8 ~70GB |

Fine-tuning recovers performance that quantization would otherwise cost, while competitive judge models at bf16 double the memory requirement without matching the benchmark scores.

**Self-enhancement bias.** Using the same model family to evaluate outputs it generated introduces a measurable scoring bias; the judge model favors its own generation style [[2]](https://arxiv.org/html/2404.13076v1). Independent evaluators avoid this.

## What this enables operationally

The practical outcome of calibrated, automated evaluation is not a single accuracy number; it is a changed operational model:

**Continuous regression monitoring.** Performance regressions in AI systems behave like software regressions: they happen on every update, including updates you did not make (cloud model providers ship changes under `latest` tags). Judges run on every deployment and surface regressions with statistical confidence before they reach users.

**Cold-start evaluation from minimal data.** Rubric inference from 3-5 examples eliminates the need to write evaluation criteria from scratch. This matters for organizations deploying LLMs into domains where pre-existing benchmarks do not exist.

**Human review at the right layer.** Calibrated confidence scores route the right cases to human reviewers: those the evaluator is uncertain about, not a random sample. This maintains quality coverage while reducing review volume. Fewer humans are needed; the ones involved focus on genuinely ambiguous cases.

**A/B testing with statistical power.** Paired bootstrap CI on consecutive runs means model comparisons have interpretable error bars. You can tell the difference between a real improvement and sampling noise.

## What this does not claim to solve

Hallucination detection at the level required for fully autonomous error correction (99.99%+) remains out of reach. Our benchmarks show frontier models plateau at 79-87% +/-1.5% on hallucination detection tasks, and this ceiling has not moved materially since GPT-4. Scorable evaluators operating in this regime require human review for uncertain cases; they do not eliminate human oversight.

They also do not substitute for domain-specific validation. A Scorable evaluator gives you a calibrated, scalable signal. For high-stakes domains, that signal should be one input to a review process, not the final arbiter.

The honest framing is: Scorable judges and evaluators eliminate the unscalable, poorly-calibrated quality processes most teams are currently running (spot checking, ad-hoc human review, development-time test suites that never see production distribution) and replace them with a continuous, statistically grounded signal that tells you where to focus human attention.

## References

[1] Leng et al. (2024). The Hallucination Snowball Effect: Evaluation of LLMs under Multi-Turn Hallucination Scenarios. ACL 2024. https://aclanthology.org/2024.acl-long.648.pdf

[2] Panickssery et al. (2024). LLM Evaluators Recognize and Favor Their Own Generations. arXiv:2404.13076. https://arxiv.org/html/2404.13076v1

[3] Min et al. (2023). FActScoring: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. EMNLP 2023. https://aclanthology.org/2024.acl-long.586.pdf

[4] Liu et al. (2023). G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment. arXiv:2303.16634. https://arxiv.org/pdf/2303.16634

[5] Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023. https://arxiv.org/pdf/2306.05685

*Internal benchmark data: HaluBench (n=10,586), SimpleQA subset (n=200), IFEval lm-eval-harness v0.4. Bootstrap CIs computed with 1,000 resamples, seed=42.*
