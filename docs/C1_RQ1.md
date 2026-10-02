# C1 / RQ1: Hierarchy-Aware Polarization Detection — Complete Analysis and Final-Submission Extension

**Project:** Structure over Scale — Hierarchy-Constrained Modeling for English Online Polarization Detection  
**Task:** SemEval-2026 Task 9 (POLAR), English, Subtasks 1–2  
**Scope of this document:** Complete record of C1/RQ1: the original claim, experimental design, definitions of every relevant metric, results, statistical tests, scientific and mathematical interpretation, alternative evaluation, limitations, defensible conclusion, and the proposed end-submission extension.  
**Status:** C1/RQ1 is complete under the frozen mid-submission protocol. The proposed extension is future work and must be reported separately from the frozen RQ1 result.

**Numbering.** This note originally drafted the parent-to-child question as RQ3. The mid-submission report keeps the proposal's RQ3 (label-aware representations) and RQ4 (parameter comparison, not a leaderboard claim). The parent-to-child question is **RQ5** there, and the same name is used below.

---

## 1. Executive conclusion

RQ1 asked whether deriving polarization detection (DET) from polarization-type predictions (TYPE) using a hierarchy-aware noisy-OR model, M4-core, improves DET and TYPE macro-F1 relative to an independent baseline (M1) and a shared multitask baseline (M2).

The answer under the frozen primary evaluation is **no**. M4-core does not improve either primary metric:

- M4-core vs. M1: DET macro-F1 −0.014 and TYPE macro-F1 −0.082; both differences are statistically significant after Holm correction.
- M4-core vs. M2: DET macro-F1 −0.008, which is not significant after Holm correction, and TYPE macro-F1 −0.036, which is significant.
- M4-core is below both baselines in all five seeds for all four RQ1 comparisons.

The hypothesis is therefore **not supported under the frozen protocol**. This wording is more precise than claiming that every corresponding statistical null hypothesis was rejected: the M4-vs.-M2 DET comparison is not significant after multiplicity correction.

However, the rejection does not mean that hierarchy modelling is useless. The error analysis reveals a structured trade-off:

1. M4-core increases polarized-class recall substantially: 0.831 versus 0.761 for M2.
2. It also increases false positives: approximately 472 per seed versus 345 for M2, reducing DET precision from 0.722 to 0.674.
3. The frozen TYPE metric scores only gold-polarized texts and therefore does not penalize M1 and M2 for emitting TYPE labels on neutral texts.
4. When TYPE is scored on all texts, M4-core is best: 0.396, compared with 0.299 for M1 and 0.255 for M2.
5. M4-core has zero internal hierarchy violations, whereas M1 and M2 violate the hierarchy on approximately 62% of texts.

The scientifically correct conclusion is therefore:

> M4-core does not improve the pre-declared DET or gold-polarized TYPE macro-F1 metrics, so the original RQ1 hypothesis is not supported. Nevertheless, its higher DET recall, zero hierarchy violations, and superior all-text TYPE macro-F1 show that hierarchy-aware modelling improves end-to-end structural validity. The result exposes a trade-off between conditional TYPE discrimination and globally consistent prediction rather than a complete failure of hierarchical modelling.

---

## 2. Task formulation

Each input is an English social-media text. The two relevant subtasks are:

- **DET / POLARDETECT:** binary classification of whether a text is polarized.
- **TYPE / POLARTYPE:** multilabel classification over five dimensions:
  - Political
  - Racial/ethnic
  - Religious
  - Gender/sexual identity
  - Other

The gold data satisfy the hierarchy exactly:

```text
DET = 1  if and only if  at least one TYPE label = 1.
```

Equivalently:

```text
TYPE_k = 1  implies  DET = 1, for every type k.
DET = 0     implies  TYPE_k = 0, for every type k.
```

This is stronger than saying that DET and TYPE are merely related tasks. TYPE is nested below DET. A valid end-to-end system should not predict a polarization type while simultaneously predicting that the text is non-polarized.

### 2.1 Data used for the C1/RQ1 conclusions

All conclusions are based on out-of-fold predictions over the official English training set:

| Statistic | Value |
|---|---:|
| Total texts | 3,222 |
| Gold polarized | 1,175 (36.5%) |
| Gold non-polarized | 2,047 |
| Political support | 1,150 |
| Racial/ethnic support | 281 |
| Religious support | 112 |
| Gender/sexual support | 72 |
| Other support | 126 |
| Polarized texts with two or more TYPE labels | 426 |
| Polarized texts with exactly one TYPE label | 749 |

Political occurs in approximately 98% of polarized training texts and is therefore almost a proxy for DET. Religious and Gender/sexual are rare. This imbalance is important for interpreting thresholds, macro-averaged scores, and seed variance.

The separate 160-text development file and 1,452-text test file were not used for these conclusions. The results are internally controlled cross-validation results and are not directly leaderboard-comparable.

---

## 3. What “gold,” “gold-polarized,” and “all-text” mean

### 3.1 Gold label

A **gold label** is the human-provided ground-truth annotation. It is not the model prediction.

For example:

```text
Text: “Those people are destroying our country.”
Gold DET: 1
Gold Political: 1
Gold Racial/ethnic: 0
Gold Religious: 0
Gold Gender/sexual: 0
Gold Other: 0
```

### 3.2 Gold-polarized text

A **gold-polarized text** is any example whose human DET annotation is 1:

```text
G = { i : y_DET(i) = 1 }.
```

The word “gold” matters. Selection is based on the true label, not on whether a model predicts the example as polarized.

### 3.3 Gold-polarized TYPE scoring

In the frozen primary protocol, TYPE metrics are computed only on the 1,175 texts in G. This evaluation asks:

> Assuming that the text is genuinely polarized, how accurately does the model identify its polarization dimensions?

It is therefore a **conditional diagnostic**. It gives the evaluator access to the gold DET decision when deciding which examples enter TYPE scoring.

### 3.4 All-text TYPE scoring

In all-text scoring, TYPE metrics are computed over all 3,222 texts. For every gold non-polarized text, all five TYPE targets are operationally treated as zero. A predicted TYPE label on such a text becomes a false positive.

This asks a different question:

> Can the system assign correct types to polarized texts while appropriately abstaining from type predictions on neutral texts?

This is closer to an end-to-end deployment view because the system is not told in advance which texts are truly polarized.

### 3.5 Why the two TYPE evaluations can rank models differently

Suppose a gold-neutral text has:

```text
Gold DET = 0
Gold TYPE = [0, 0, 0, 0, 0]
```

and M2 predicts:

```text
Predicted DET = 0
Predicted TYPE = [1, 0, 0, 0, 0]
```

Under gold-polarized scoring, this example is excluded and the false political prediction has no effect. Under all-text scoring, it is a TYPE false positive and reduces political precision and F1.

Thus, the two evaluations are not interchangeable:

- Gold-polarized TYPE macro-F1 measures **conditional type discrimination**.
- All-text TYPE macro-F1 measures **end-to-end type prediction and abstention**.

---

## 4. Models in C1/RQ1

All trained conditions use `microsoft/deberta-v3-base`, attention-masked mean pooling, dropout 0.1, and linear prediction heads.

### 4.1 M1: independent models

M1 trains two separate encoders:

- M1-DET: a binary DET model.
- M1-TYPE: a five-label TYPE model.

The predictions are joined without enforcing consistency. M1 uses approximately 368M parameters in total, twice the 184M used by the shared and structured conditions.

### 4.2 M2: shared multitask model

M2 uses one shared encoder with independent DET and TYPE heads:

```text
L_M2 = BCE(y_DET, p_DET) + lambda × L_TYPE,
```

where lambda = 1 and TYPE loss is masked to gold-polarized examples. Sharing allows transfer between tasks, but nothing forces the two output heads to agree.

### 4.3 M3: post-hoc gating

M3 is derived from M2 rather than trained separately. A one-way gate applies:

```text
predicted TYPE_k <- predicted TYPE_k AND predicted DET.
```

The symmetric version additionally applies:

```text
predicted DET <- OR over the gated TYPE predictions.
```

M3 is primarily part of C2/RQ2, but it is useful in C1 interpretation because it shows the effect of forcing consistency after training. Its all-text TYPE result lies between M4 and the unconstrained baselines.

### 4.4 M4-core: training-time noisy-OR structure

M4-core has a TYPE head but no independent DET head. Let z_k be the logit and p_k = sigmoid(z_k) the probability for type k. During training, the DET probability is:

```text
p_DET = 1 − product over k of (1 − p_k).
```

This is the noisy-OR probability that at least one TYPE applies.

The loss is:

```text
L_M4 = BCE(y_DET, p_DET) + lambda × L_TYPE.
```

This allows DET supervision on non-polarized examples to push the TYPE logits downward even though the explicit TYPE loss is masked there.

At inference under the frozen implementation, each TYPE probability is thresholded and DET is decoded as:

```text
predicted TYPE_k = 1[p_k >= tau_k]
predicted DET    = OR over k of predicted TYPE_k.
```

This distinction is important:

- **Training:** DET uses differentiable noisy-OR over probabilities.
- **Frozen inference:** DET is the hard OR of thresholded TYPE decisions.

Consequently, the mathematical accumulation of weak probabilities explains training-time coupling, while the observed test-time over-prediction is especially tied to the hard OR and permissive per-label thresholds.

---

## 5. Original RQ1 claim and hypothesis

### 5.1 Research question

> Does deriving DET from TYPE through a hierarchy-aware noisy-OR model improve DET and TYPE macro-F1 over independent modelling and unconstrained multitask learning?

### 5.2 Directional hypothesis

The expected result was:

```text
DET macro-F1(M4)  > DET macro-F1(M1) and DET macro-F1(M2)
TYPE macro-F1(M4) > TYPE macro-F1(M1) and TYPE macro-F1(M2).
```

The motivation was that M4 encodes a true property of the labels, obtains DET supervision through TYPE, and prevents logically contradictory outputs. The hypothesis assumed that these advantages would improve the two frozen predictive metrics.

### 5.3 What would count as support

Support required positive paired differences for M4 against both baselines on both primary metrics, with uncertainty and multiplicity-corrected significance considered. Merely achieving zero violations was not sufficient because consistency and predictive performance are different outcomes.

---

## 6. Frozen experimental protocol

### 6.1 Cross-validation and seeds

- One fixed five-fold iterative multilabel-stratified split, created with seed 2026.
- Fold sizes: 644–645 texts.
- Rare labels were balanced across folds: 14–15 Gender/sexual and 22–23 Religious positives per fold.
- Training seeds: 13, 21, 42, 87, and 100.
- Within each outer training split, 10% formed an inner multilabel-stratified validation set.
- Early stopping, checkpoint selection, and threshold selection used only the inner validation data.
- Each outer fold was predicted once after choices were frozen.

The sweep contained:

```text
4 trained conditions × 5 folds × 5 seeds = 100 training runs.
```

M1 combinations and M3 derived conditions were produced for every fold/seed pair.

### 6.2 Thresholds

- Search grid: 0.20 to 0.80 in increments of 0.05.
- TYPE thresholds: selected per label to maximize that label’s F1 on inner-validation gold-polarized texts.
- M1/M2 DET thresholds: selected to maximize DET macro-F1 over both DET classes.
- M3 reused M2 thresholds.
- M4 had no separate DET threshold; frozen DET decoding was the hard OR of thresholded TYPE outputs.

### 6.3 Training

- AdamW, learning rate 2e-5, weight decay 0.01.
- Batch size 8 with two gradient-accumulation steps.
- Linear warm-up over 10% of steps.
- Gradient clipping at 1.0.
- FP16 automatic mixed precision.
- Maximum eight epochs and early-stopping patience two.
- M1-DET checkpoint selected using DET macro-F1.
- M1-TYPE, M2, and M4 checkpoints selected using TYPE macro-F1.

### 6.4 Aggregation

For each seed, the five outer-fold prediction sets were concatenated into one 3,222-text out-of-fold set and scored once. The project reports mean ± standard deviation over the five seed-level scores. The 25 fold/seed results are not treated as 25 independent observations.

### 6.5 Statistical inference

Paired comparisons use an example-level paired bootstrap with 2,000 resamples. Every bootstrap resample applies the same sampled multiset of texts to both compared models and to all five seeds. Differences are averaged over seeds, and the 2.5th and 97.5th percentiles form the 95% confidence interval.

The six pre-declared primary RQ1/RQ2 tests are corrected using Holm’s sequential method. Secondary comparisons are exploratory and uncorrected unless explicitly stated otherwise.

---

## 7. Metric definitions

### 7.1 Binary precision, recall, and F1

For the polarized DET class:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
          = 2TP / (2TP + FP + FN).
```

Precision asks: among texts predicted polarized, how many are truly polarized?

Recall asks: among truly polarized texts, how many were detected?

### 7.2 DET macro-F1

DET is evaluated on **all texts**. There is no “gold-polarized-only DET score,” because DET is the decision that separates polarized from non-polarized texts.

The primary DET score is:

```text
DET macro-F1 = [F1_polarized + F1_non-polarized] / 2.
```

This differs from reporting only polarized-class F1. A model can improve polarized-class recall while damaging the non-polarized class through additional false positives, thereby lowering macro-F1.

### 7.3 Gold-polarized TYPE macro-F1

Let G be the set of gold-polarized examples and K = 5 labels. The frozen primary TYPE score is:

```text
TYPE macro-F1_gold = (1/K) × sum over k of F1_k evaluated only on G.
```

Every label contributes equally, regardless of prevalence. Rare Gender/sexual therefore receives the same macro weight as dominant Political.

### 7.4 All-text TYPE macro-F1

The complementary score is:

```text
TYPE macro-F1_all = (1/K) × sum over k of F1_k evaluated on all N texts.
```

For a gold-neutral text, every TYPE target is treated as zero. A TYPE prediction therefore adds a false positive:

```text
FP_k = sum_i 1[y_i,k = 0 and predicted_y_i,k = 1].
```

Because:

```text
Precision_k = TP_k / (TP_k + FP_k),
```

unsupported TYPE predictions lower precision and F1.

True negatives are not directly rewarded by F1. The advantage of abstention appears through avoiding false positives, not through accumulating true-negative counts.

### 7.5 Logical Violation Rate

The symmetric Logical Violation Rate is:

```text
LVR = (1/N) × sum_i 1[predicted_DET(i) != OR_k predicted_TYPE(i,k)].
```

It can be decomposed into:

- **LVR-A:** predicted DET = 0 but at least one TYPE = 1.
- **LVR-B:** predicted DET = 1 but no TYPE = 1.

LVR measures internal logical consistency. It is not the same as correctness. A model can consistently predict DET = 1 and TYPE = Political for a gold-neutral text; that prediction has LVR = 0 but remains wrong.

### 7.6 Neutral TYPE false-positive rate

For future work, a useful complementary measure is:

```text
Neutral TYPE FPR =
number of gold-neutral texts receiving at least one predicted TYPE
divided by number of gold-neutral texts.
```

This measures correctness on neutral examples, whereas LVR measures agreement between a model’s own outputs.

---

## 8. Primary C1/RQ1 results

### 8.1 Main scores

| Model | Parameters | DET macro-F1 | TYPE macro-F1 on gold-polarized texts | TYPE macro-P | TYPE macro-R | LVR total |
|---|---:|---:|---:|---:|---:|---:|
| M1 independent | 368M | **0.799 ± 0.007** | **0.543 ± 0.013** | **0.509 ± 0.024** | **0.600 ± 0.023** | 61.9% |
| M2 shared MTL | 184M | 0.793 ± 0.003 | 0.497 ± 0.015 | 0.444 ± 0.018 | 0.595 ± 0.019 | 61.6% |
| M3 symmetric | 184M | 0.793 ± 0.003 | 0.453 ± 0.022 | 0.469 ± 0.033 | 0.468 ± 0.036 | 0.0% |
| M4-core noisy-OR | 184M | 0.784 ± 0.004 | 0.461 ± 0.007 | 0.465 ± 0.011 | 0.474 ± 0.022 | 0.0% |

M1 is best on both frozen primary metrics. M4 has the lowest DET macro-F1 and trails both unconstrained baselines on gold-polarized TYPE F1.

### 8.2 Pre-declared paired comparisons

| Comparison | Metric | Difference M4 − baseline | 95% paired-bootstrap CI | Raw bootstrap p | Holm-corrected p | Seeds favouring M4 |
|---|---|---:|---:|---:|---:|---:|
| M4 − M1 | DET macro-F1 | **−0.014** | [−0.023, −0.006] | <0.001 | <0.001 | 0/5 |
| M4 − M1 | TYPE macro-F1 | **−0.082** | [−0.102, −0.064] | <0.001 | <0.001 | 0/5 |
| M4 − M2 | DET macro-F1 | −0.008 | [−0.016, −0.001] | 0.024 | 0.072 | 0/5 |
| M4 − M2 | TYPE macro-F1 | **−0.036** | [−0.055, −0.019] | 0.001 | 0.004 | 0/5 |

The percentile confidence intervals shown here are unadjusted paired-bootstrap intervals, while the final p-values are Holm-corrected across the six pre-declared primary tests. This explains why the M4-vs.-M2 DET interval excludes zero while its multiplicity-corrected p-value is 0.072.

### 8.3 Formal decision

The directional RQ1 hypothesis is not supported because M4 did not improve any of the four target comparisons and was worse in all five seeds.

Three differences remain statistically significant after Holm correction, all favouring the baselines. For M4 versus M2 on DET, the evidence is insufficient to declare a corrected significant difference, but it also provides no evidence of the predicted improvement.

The correct language is:

> RQ1 is not supported under the frozen primary protocol.

Avoid the stronger but inaccurate statement:

> Every null hypothesis was rejected and M4 is significantly worse in every comparison.

That is not true for M4 versus M2 on DET after Holm correction.

---

## 9. Why M4 finds more polarized examples

### 9.1 Noisy-OR accumulates evidence

During training, M4 uses:

```text
p_DET = 1 − product_k(1 − p_k).
```

If all five TYPE heads assign probability 0.10:

```text
p_DET = 1 − 0.9^5 = 0.4095.
```

If all assign probability 0.13:

```text
p_DET = 1 − 0.87^5 ≈ 0.502.
```

Thus, several individually weak signals can jointly create strong evidence that some polarization type is present.

The sensitivity to each type probability is:

```text
partial p_DET / partial p_j = product over k != j of (1 − p_k),
```

which is non-negative. Increasing any TYPE probability can only increase the noisy-OR DET probability.

### 9.2 Observed recall gain

The empirical results match this sensitivity:

```text
M2 DET recall = 0.761
M4 DET recall = 0.831
Difference    = +0.070
95% CI        = [+0.059, +0.081]
```

M4 also reduces false negatives:

```text
M2 false negatives ≈ 281 per seed
M4 false negatives ≈ 199 per seed
```

It therefore recovers approximately 82 additional polarized texts per seed relative to M2.

### 9.3 Training probability versus inference decision

The numerical noisy-OR examples describe the differentiable training probability. Under the frozen decoder, M4 does not threshold p_DET directly. It thresholds every TYPE probability and predicts DET as their hard OR. Therefore, final recall is governed by both:

1. training-time coupling through noisy-OR; and
2. whether any TYPE probability crosses its per-label threshold.

This distinction must be preserved in the paper.

---

## 10. Why M4 also marks more neutral examples as polarized

### 10.1 Semantic cause

Neutral texts can contain words and topics associated with political, religious, racial, gender, identity, or conflict discourse without expressing polarization. TYPE heads may respond to these topical or lexical cues even when the rhetoric is neutral.

In M4, weak TYPE activation contributes to DET during training, and the frozen hard-OR decoder needs only one TYPE threshold to fire. A permissive threshold can therefore convert topical similarity into a false-positive DET decision.

### 10.2 Error-profile evidence

| Model | DET false positives per seed | DET false negatives per seed | Predicted polarized per seed |
|---|---:|---:|---:|
| M1 | 325 | 280 | 1,220 |
| M2 | 345 | 281 | 1,239 |
| M4-core | **472** | **199** | **1,448** |

Relative to M2, M4 finds approximately 82 more polarized texts but falsely flags approximately 127 more neutral texts.

The precision–recall change is:

```text
M2 DET precision = 0.722
M4 DET precision = 0.674
Difference       = −0.048
95% CI           = [−0.057, −0.038]

M2 DET recall    = 0.761
M4 DET recall    = 0.831
Difference       = +0.070
95% CI           = [+0.059, +0.081]
```

This is direct evidence of a sensitivity–specificity trade-off rather than uniform deterioration.

### 10.3 Threshold evidence

M4’s Political threshold has:

```text
Mean threshold = 0.216 (approximately 0.22)
Grid floor     = 0.20
Runs at 0.20   = 19 of 25
```

Political is present in 1,150 of 1,175 polarized texts, so it behaves almost like DET. Because the Political threshold is usually at the grid floor, the hard OR frequently predicts DET = 1.

The crucial design issue is that TYPE thresholds were tuned to maximize per-label F1 only on gold-polarized inner-validation examples. The threshold objective never saw the cost of predicting Political or another TYPE on a neutral validation text.

M1 and M2, by contrast, tune dedicated DET thresholds to maximize DET macro-F1 across both DET classes. M1-DET also selects its checkpoint using DET macro-F1, whereas M4 selects using TYPE macro-F1.

Therefore, part of M4’s DET deficit is plausibly a **decoder and calibration artifact**, not necessarily an intrinsic failure of noisy-OR representation learning.

### 10.4 F1 mathematics

Using the positive-class precision and recall:

```text
F1_positive(M4)
= 2 × 0.674 × 0.831 / (0.674 + 0.831)
≈ 0.744.

F1_positive(M2)
= 2 × 0.722 × 0.761 / (0.722 + 0.761)
≈ 0.741.
```

This illustrative calculation shows that M4 can be competitive for the polarized class alone because recall compensates for lower precision.

But the reported primary score is DET macro-F1:

```text
DET macro-F1 = [F1_polarized + F1_non-polarized] / 2.
```

The additional false positives damage the non-polarized class. Consequently, M4’s overall DET macro-F1 is lower even though it detects more polarized examples.

---

## 11. Why M4 is worse under frozen gold-polarized TYPE scoring

### 11.1 The metric does not observe neutral-text TYPE errors

M1 and M2 TYPE heads are trained only on gold-polarized texts, use positive-class weights, and have thresholds tuned only on polarized validation examples. They are never directly taught what a non-polarized TYPE pattern looks like.

This produces extreme inconsistency:

```text
M1: TYPE label on 2,039 of 2,047 gold-neutral texts
M2: TYPE label on 2,047 of 2,047 gold-neutral texts
```

Their overall LVR is approximately 62%. Yet none of these neutral-text TYPE predictions is counted by frozen gold-polarized TYPE F1.

### 11.2 Conditional freedom benefits M1 and M2

M1 can optimize five TYPE decisions independently without requiring them to agree with DET. This additional freedom improves its conditional TYPE recall on examples already known to be polarized.

M4 must solve a joint problem:

1. determine whether any polarization type exists;
2. identify which types exist; and
3. maintain a valid relation between DET and TYPE.

The frozen TYPE metric measures mainly item 2 and does not reward item 3.

### 11.3 Recall loss is the main source of the frozen TYPE gap

Relative to M2, M4 has slightly higher gold-polarized TYPE precision but much lower recall:

```text
M4 − M2 TYPE macro-precision = +0.021
95% CI = [+0.001, +0.039]

M4 − M2 TYPE macro-recall = −0.121
95% CI = [−0.148, −0.098]
```

Thus, M4 is not simply making less reliable positive TYPE predictions. It is making fewer of them, losing recall under a conditional metric that does not charge the baselines for predicting types on neutral inputs.

### 11.4 Per-label evidence

| Label | Support | M1 F1 | M2 F1 | M4 F1 |
|---|---:|---:|---:|---:|
| Political | 1,150 | 0.985 | 0.986 | 0.892 |
| Racial/ethnic | 281 | 0.623 | 0.583 | 0.521 |
| Religious | 112 | 0.548 | 0.477 | 0.450 |
| Gender/sexual | 72 | 0.266 | 0.200 | 0.214 |
| Other | 126 | 0.293 | 0.238 | 0.228 |

M4 trails M1 on all five labels. Relative to M2, M4 loses substantial recall on Political, Racial/ethnic, Religious, and Other, while its Gender/sexual F1 is numerically slightly higher but highly uncertain. Rare-label thresholds are based on very few inner-validation positives, so per-label conclusions for Gender/sexual and Other require caution.

---

## 12. Alternative all-text TYPE evaluation

### 12.1 Results

| Model | All-text TYPE macro-F1 | All-text TYPE macro-P | All-text TYPE macro-R |
|---|---:|---:|---:|
| M1 | 0.299 ± 0.013 | 0.203 | 0.600 |
| M2 | 0.255 ± 0.029 | 0.169 | 0.595 |
| M3 | 0.382 ± 0.017 | 0.339 | 0.468 |
| M4-core | **0.396 ± 0.008** | **0.348** | 0.474 |

The ranking reverses:

```text
M4 > M3 > M1 > M2.
```

### 12.2 Paired differences

```text
M4 − M1 all-text TYPE macro-F1
= +0.097
95% CI = [+0.078, +0.114]
Seeds favouring M4 = 5/5.

M4 − M2 all-text TYPE macro-F1
= +0.141
95% CI = [+0.121, +0.158]
Seeds favouring M4 = 5/5.
```

These are secondary, uncorrected comparisons and should be identified as exploratory even though the effects are large and consistent.

### 12.3 Scientific interpretation

The sign reversal follows from the error definition. Under all-text scoring, the thousands of TYPE labels emitted by M1 and M2 on gold-neutral texts become false positives. Their high recall no longer compensates for very low precision.

M4 prevents the internally inconsistent pattern DET = 0 with some TYPE = 1, so its TYPE precision rises substantially when evaluated end to end. M4 can still make consistent but incorrect predictions on neutral examples—DET = 1 and some TYPE = 1—but it makes far fewer unsupported TYPE predictions than M1 or M2.

### 12.4 What this result does and does not prove

It supports the claim:

> M4 produces better end-to-end TYPE predictions when unsupported TYPE labels on neutral examples count as errors.

It does not support the claim:

> The original frozen RQ1 hypothesis should now be accepted.

The original hypothesis was defined using the frozen primary metrics. Replacing the outcome after seeing results would be post-hoc metric selection. The all-text result must be presented as a pre-declared secondary view or a clearly labelled protocol extension, not as a replacement for the primary analysis.

### 12.5 Official-scorer qualification

The project has not yet verified whether the official SemEval Subtask 2 scorer evaluates only polarized instances or evaluates all submitted instances with all-zero TYPE targets for non-polarized texts. That must be checked before claiming that the all-text analysis is directly equivalent to the leaderboard metric.

Until verified, use the wording:

> “Complementary all-text TYPE evaluation.”

Do not use:

> “Official SemEval TYPE evaluation.”

---

## 13. Hierarchy consistency results

| Model | LVR-A | LVR-B | LVR total |
|---|---:|---:|---:|
| M1 | 61.9% | approximately 0% | 61.9% |
| M2 | 61.5% | approximately 0% | 61.6% |
| M3 symmetric | 0% | 0% | 0% |
| M4-core | 0% | 0% | 0% |

The high M1/M2 LVR is not a mysterious model failure. It follows from three interacting design choices:

1. TYPE loss is masked on non-polarized examples.
2. Positive-class weighting pushes rare TYPE probabilities upward.
3. TYPE thresholds are tuned only on polarized examples.

Nothing in that training and calibration process teaches the unconstrained TYPE heads to abstain on neutral text.

M4’s zero LVR is guaranteed by the hard-OR decoder. This is a real structural advantage, but LVR should never be reported alone because logical consistency does not imply correctness.

---

## 14. Complete scientific explanation of the rejection

The rejected RQ1 hypothesis can be explained through four linked mechanisms.

### 14.1 The hierarchy changes the DET operating point

M4 couples DET to evidence from all TYPE heads. This increases sensitivity and reduces false negatives. The empirical recall gain of approximately seven points is consistent with this mechanism.

### 14.2 The decoder amplifies false alarms

At inference, any thresholded TYPE activates DET. Per-label thresholds are calibrated only on polarized texts, and Political reaches the grid floor in 19/25 runs. Therefore, false TYPE activation on neutral text becomes a false-positive DET decision. The increase from approximately 345 to 472 false positives explains the precision reduction.

### 14.3 DET macro-F1 penalizes both sides of the trade-off

Higher polarized recall helps one class, but additional false positives damage the non-polarized class. Because macro-F1 averages both class F1 scores, the precision/specificity loss outweighs the sensitivity gain in the final DET metric.

### 14.4 Frozen TYPE scoring hides the baseline inconsistency

Gold-polarized TYPE scoring ignores every neutral-text TYPE prediction. M1 and M2 can achieve high conditional recall while assigning types to nearly every neutral input. M4’s structural advantage is invisible under this metric, while its reduced conditional recall remains visible.

Together these mechanisms explain why M4 can simultaneously:

- have lower primary DET macro-F1;
- have lower gold-polarized TYPE macro-F1;
- have higher DET recall;
- have zero hierarchy violations; and
- have higher all-text TYPE macro-F1.

There is no contradiction because the metrics assess different properties.

---

## 15. Alternative explanations and limits on causal claims

The evidence strongly characterizes M4’s error pattern, but the current comparison does not prove that noisy-OR alone caused every difference.

### 15.1 DET decoding is not matched

M1 and M2 optimize a dedicated DET threshold on DET macro-F1. M4 derives hard DET from TYPE thresholds optimized for label F1 on polarized examples. A fair causal test of the representation would require a DET-aware calibration procedure for M4.

### 15.2 Checkpoint objectives differ

M1-DET selects checkpoints using DET macro-F1, while M4 selects using TYPE macro-F1. This can favour M1 on DET.

### 15.3 Parameter count differs

M1 uses two encoders and approximately 368M parameters. M2 and M4 use one encoder and approximately 184M parameters. M1 is therefore not parameter matched.

### 15.4 TYPE training is masked

Because TYPE loss is masked on neutral examples, all abstention learning in M4 comes indirectly through DET/noisy-OR. An alternative training objective that includes neutral all-zero TYPE supervision may change the result.

### 15.5 Calibration and class imbalance

Rare labels have unstable threshold estimates, and the dominant Political label largely determines the hard OR. The observed effect may depend partly on label prevalence and calibration rather than hierarchy alone.

### 15.6 English-only scope

The conclusions cover English training-set cross-validation. SemEval-2026 Task 9 spans 22 languages and multiple cultural and event contexts. The same trade-off may differ in other languages.

### 15.7 Secondary-test multiplicity

The all-text comparisons, per-label analyses, and precision/recall decompositions are secondary. They are scientifically useful for mechanism discovery but must not be presented as multiplicity-controlled confirmatory tests.

---

## 16. Claims that are and are not supported

### 16.1 Supported claims

1. M4 does not outperform M1 or M2 on the frozen RQ1 metrics.
2. M4 significantly trails M1 on DET and TYPE macro-F1.
3. M4 significantly trails M2 on frozen TYPE macro-F1.
4. The corrected evidence is insufficient to call the M4-vs.-M2 DET difference significant.
5. M4 increases DET recall and decreases DET precision relative to M2.
6. Its DET deficit is entirely associated with more false positives; it actually produces fewer false negatives.
7. M1/M2 hierarchy violations are extremely common under masked TYPE training and polarized-only TYPE thresholding.
8. M4 guarantees internal consistency under its hard-OR decoder.
9. M4 is best under complementary all-text TYPE macro-F1.
10. The conclusion about which TYPE model is “best” depends on whether the intended question is conditional classification or end-to-end prediction.

### 16.2 Unsupported or overstated claims

1. “Hierarchy modelling failed completely.” The all-text and LVR results contradict this.
2. “M4 is significantly worse than M2 on DET.” It is directionally worse, but Holm-corrected p = 0.072.
3. “The new metric proves the original hypothesis.” It tests a different property.
4. “Zero LVR means M4 makes no TYPE errors on neutral texts.” Consistent false positives remain possible.
5. “Noisy-OR alone caused all errors.” The decoder, threshold objective, checkpoint criterion, and class imbalance are confounded.
6. “M4 is officially best on SemEval Subtask 2.” Official scoring equivalence has not yet been verified.

---

## 17. ACL-ready answer to RQ1

The following text can be adapted directly into the Results/Discussion section of an ACL-format paper.

### 17.1 Results paragraph

Under the frozen primary evaluation, the hierarchy-aware M4-core model did not outperform either unconstrained baseline. M1 obtained DET and gold-polarized TYPE macro-F1 scores of 0.799 ± 0.007 and 0.543 ± 0.013, respectively, while M2 obtained 0.793 ± 0.003 and 0.497 ± 0.015. M4-core achieved 0.784 ± 0.004 on DET and 0.461 ± 0.007 on TYPE. Relative to M1, M4 reduced DET macro-F1 by 0.014 (95% CI [−0.023, −0.006], Holm-corrected p < 0.001) and TYPE macro-F1 by 0.082 (95% CI [−0.102, −0.064], p < 0.001). Relative to M2, it reduced TYPE macro-F1 by 0.036 (95% CI [−0.055, −0.019], p = 0.004). The DET difference against M2 was −0.008 and negative in all five seeds, but it did not remain significant after Holm correction (p = 0.072). The directional RQ1 hypothesis is therefore not supported under the pre-declared protocol.

### 17.2 Mechanistic explanation paragraph

The DET result reflects a precision–recall trade-off induced by the structured decoder. M4-core increased polarized-class recall from 0.761 for M2 to 0.831, reducing false negatives from approximately 281 to 199 per seed. However, precision decreased from 0.722 to 0.674, while false positives increased from approximately 345 to 472. During training, noisy-OR permits evidence from any TYPE head to increase the DET probability. At inference, DET is the hard OR of five thresholded TYPE decisions. Because the TYPE thresholds were optimized only on gold-polarized validation examples, their objective did not include the cost of neutral-text false alarms. The Political threshold, which nearly acts as a proxy for DET in this dataset, reached the lower grid boundary of 0.20 in 19 of 25 runs. M4 consequently adopted a higher-recall, lower-precision operating point, and the additional errors on the non-polarized class reduced DET macro-F1.

### 17.3 Metric-sensitivity paragraph

The frozen TYPE result must be interpreted in relation to its scoring scope. TYPE macro-F1 was computed only on gold-polarized texts and therefore measured conditional type discrimination rather than end-to-end prediction. This protocol does not penalize M1 and M2 for assigning TYPE labels to neutral examples. Indeed, their hierarchy-violation rates were approximately 62%, while M4-core’s was zero. When the same predictions were evaluated over all 3,222 texts, treating gold-neutral examples as all-negative for TYPE, the ranking reversed: M4-core achieved 0.396 all-text TYPE macro-F1, compared with 0.299 for M1 and 0.255 for M2. The corresponding paired gains were +0.097 over M1 and +0.141 over M2, with both confidence intervals excluding zero and all five seeds favouring M4. These secondary results do not reverse the primary hypothesis decision; instead, they show that the frozen conditional metric does not capture M4’s advantage in global output validity.

### 17.4 Final interpretation paragraph

RQ1 therefore reveals a trade-off between conditional accuracy and structural consistency. M1 is strongest when the evaluator supplies the information that a text is truly polarized and asks only which dimensions apply. M4 is strongest when the complete system must also refrain from assigning polarization dimensions to neutral texts. Hierarchy-aware modelling is consequently useful for end-to-end validity, but the child-to-parent noisy-OR formulation and its polarized-only threshold calibration do not improve the frozen macro-F1 outcomes. The result motivates a parent-to-child probabilistic formulation in which DET remains independently calibrated and softly controls TYPE predictions.

---

## 18. Proposed end-submission extension

The best extension is not to redefine RQ1 after observing the results. It is to formulate a new, prospective research question derived from the discovered failure mechanism.

### 18.1 Proposed research question

> **RQ5:** Can a calibrated parent-to-child hierarchical model improve end-to-end polarization-type classification while preserving polarization-detection performance?

### 18.2 Main hypothesis

> **H5:** A calibrated parent-to-child conditional model will achieve higher all-text TYPE macro-F1 than M2 and M4 while remaining non-inferior to M2 on DET macro-F1.

This separates superiority on the outcome that hierarchy should improve from a guardrail on DET performance.

### 18.3 Confirmatory sub-hypotheses

**H5a — all-text TYPE superiority**

```text
TYPE macro-F1_all(M5) > TYPE macro-F1_all(M2)
TYPE macro-F1_all(M5) > TYPE macro-F1_all(M4).
```

**H5b — DET non-inferiority**

```text
DET macro-F1(M5) − DET macro-F1(M2) > −delta.
```

A reasonable pre-declared margin is delta = 0.01 macro-F1. The margin must be fixed before observing final results and justified as the largest acceptable DET degradation for the gain in end-to-end TYPE validity.

**H5c — conditional TYPE retention**

```text
M5 retains gold-polarized TYPE macro-F1 close to M2
while reducing neutral-text TYPE false positives and LVR.
```

H5c can be treated as secondary unless a formal non-inferiority margin is also pre-declared for gold-polarized TYPE F1.

---

## 19. Proposed M5: parent-to-child conditional factorization

### 19.1 Probabilistic derivation

Because TYPE_k = 1 implies DET = 1:

```text
P(TYPE_k = 1 | x)
= P(DET = 1 | x)
  × P(TYPE_k = 1 | DET = 1, x).
```

Let:

```text
d   = P(DET = 1 | x)
q_k = P(TYPE_k = 1 | DET = 1, x).
```

Then the marginal TYPE probability is:

```text
p_k = d × q_k.
```

This is a soft parent-to-child gate.

It has the useful property:

```text
0 <= p_k <= d.
```

TYPE confidence cannot exceed DET confidence.

### 19.2 Architecture

For encoder representation h:

```text
h   = Encoder(x)
d   = sigmoid(W_D h + b_D)
q_k = sigmoid(W_k h + b_k)
p_k = d × q_k.
```

The independent DET head protects DET from TYPE false activations. Unlike M4, a Political or Religious TYPE score cannot raise the DET probability. Instead, low DET confidence suppresses the final TYPE probability.

### 19.3 Example

For a likely neutral text:

```text
d = 0.15
q_political = 0.80
p_political = 0.15 × 0.80 = 0.12.
```

The conditional type head may recognize political content, but the final TYPE probability remains low because the text is unlikely to be polarized.

For a polarized political text:

```text
d = 0.90
q_political = 0.80
p_political = 0.90 × 0.80 = 0.72.
```

The correct TYPE remains strong.

### 19.4 Training objective

Use:

```text
L_M5 = L_DET + lambda × L_conditional_TYPE,
```

where:

```text
L_DET = BCE(y_DET, d)
```

is evaluated on all examples, and:

```text
L_conditional_TYPE
= I[y_DET = 1] × sum_k weighted_BCE(y_k, q_k)
```

is evaluated on gold-polarized examples because q_k has the explicit interpretation P(TYPE_k | DET = 1, x).

The final p_k values, rather than q_k, are used for end-to-end all-text TYPE evaluation.

### 19.5 Why M5 directly tests the C1 diagnosis

M4 uses child-to-parent dependence:

```text
TYPE -> DET.
```

M5 uses parent-to-child dependence:

```text
DET -> TYPE.
```

If M5 preserves DET precision while improving all-text TYPE F1, this supports the claim that hierarchy was useful but the direction and calibration of M4 were responsible for the primary failure.

If M5 still loses substantial TYPE recall, this suggests that error propagation from DET is an inherent cost of strict hierarchy rather than a noisy-OR-specific problem.

---

## 20. M5 decoding and ablations

At minimum, evaluate two variants.

### 20.1 M5-Soft

```text
p_k = d × q_k
predicted DET = 1[d >= tau_D]
predicted TYPE_k = 1[p_k >= tau_k].
```

This preserves probabilistic ordering p_k <= d but may produce a hard-label violation if tau_k < tau_D.

### 20.2 M5-Constrained

```text
predicted TYPE_k
= 1[d >= tau_D] × 1[p_k >= tau_k].
```

This guarantees that a predicted TYPE requires predicted DET, but DET false negatives can remove correct TYPE labels.

The M5-Soft versus M5-Constrained comparison measures whether perfect discrete consistency is worth the error-propagation cost.

### 20.3 Recommended controlled model ladder

1. M1: independent task-specific models.
2. M2: shared encoder with unconstrained heads.
3. M3: hard post-hoc gate.
4. M4: child-to-parent noisy-OR.
5. M5-Soft: parent-to-child probability factorization.
6. M5-Constrained: factorization plus hard projection.

Use the same encoder, folds, seeds, training budget, and positive weighting so that the hierarchy mechanism is the principal difference.

### 20.4 Optional ablations

If time permits:

- M4 with DET-aware joint threshold calibration on all validation texts.
- M4 threshold grid extended below 0.20.
- M2 trained with explicit all-zero TYPE targets on neutral examples.
- M5 with and without class weighting.
- M5 with and without calibrated probabilities.
- A soft consistency penalty such as mean_k max(0, q_k − d)^2, reported as a separate ablation rather than mixed into the main M5 claim.

These should be prioritized only after the main M5 test is complete.

---

## 21. Threshold and calibration plan for the extension

### 21.1 DET threshold

Choose tau_D on inner-validation data by maximizing DET macro-F1 over all validation texts.

### 21.2 TYPE thresholds

Choose each tau_k using final probabilities p_k = d × q_k over the complete inner-validation set, maximizing all-text per-label F1. This ensures that threshold selection sees neutral-text false positives.

### 21.3 Strict consistency option

One possible constraint is tau_k >= tau_D. Since p_k <= d, this can guarantee that TYPE cannot cross its threshold while DET remains below its own threshold. This may reduce rare-label recall and should therefore be an explicit ablation, not silently imposed.

### 21.4 Calibration metrics

Report at least one probability-quality measure:

- Expected Calibration Error (ECE), or
- Brier score.

Calibration matters because M4’s observed weakness is tied to converting probabilities into hard predictions under label imbalance.

---

## 22. Extension evaluation plan

### 22.1 Primary outcome

All-text TYPE macro-F1.

This is the outcome that directly measures whether M5 improves end-to-end TYPE prediction and neutral abstention.

### 22.2 Guardrail outcome

DET macro-F1, tested for non-inferiority relative to M2 with a pre-declared margin, recommended delta = 0.01.

### 22.3 Secondary outcomes

- Gold-polarized TYPE macro-F1.
- TYPE macro-precision and macro-recall under both scoring scopes.
- Per-label TYPE precision, recall, and F1.
- DET positive-class precision and recall.
- LVR-A, LVR-B, and total LVR.
- Neutral TYPE false-positive rate.
- Number of TYPE labels predicted on gold-neutral texts.
- Multi-label and single-label polarized-text recall.
- ECE or Brier score.
- Seed-level variance.

### 22.4 Statistical tests

For paired run r:

```text
Delta_r = Score(M5, r) − Score(baseline, r).
```

For superiority:

```text
H0: mean Delta <= 0
H1: mean Delta > 0.
```

For DET non-inferiority:

```text
H0: mean Delta <= −delta
H1: mean Delta > −delta.
```

M5 is DET-non-inferior only if the lower confidence bound for M5 − M2 is above −delta.

Example with delta = 0.01:

```text
Difference = −0.002
95% CI = [−0.007, +0.003]
```

This would support non-inferiority because −0.007 > −0.010.

But:

```text
95% CI = [−0.015, +0.003]
```

would not establish non-inferiority.

Use the same paired bootstrap structure as C1 and correct the planned M5 comparisons using Holm’s method. Pre-register which comparisons are primary before inspecting the test outcomes.

---

## 23. Possible extension outcomes and their interpretation

### 23.1 Desired outcome

```text
All-text TYPE: M5 > M4 > M1/M2
DET macro-F1: M5 approximately M2 and above M4
LVR/neutral FPs: M5 substantially below M1/M2
Gold-polarized TYPE: M5 close to M2.
```

Interpretation: hierarchy is beneficial, but parent-to-child factorization and full-set calibration are better suited than child-to-parent noisy-OR.

### 23.2 M5 improves all-text TYPE but loses DET

Interpretation: structural validity still trades against detection quality. The non-inferiority part of H5 fails even if H5a succeeds.

### 23.3 M5 preserves DET but loses conditional TYPE recall

Interpretation: DET uncertainty propagates downward. Soft or hard gating may inherently cap conditional TYPE recall.

### 23.4 M5 does not improve any metric

Interpretation: the main issue may be masked TYPE training, label imbalance, encoder limitations, or annotation ambiguity rather than the direction of factorization.

### 23.5 M4 improves after DET-aware calibration

Interpretation: the mid-submission deficit was primarily a decoding/calibration artifact. This would not change the frozen RQ1 conclusion; it would constitute a successful protocol extension.

---

## 24. Literature review relevant to C1 and the extension

### 24.1 POLAR task

SemEval-2026 Task 9 introduces multilingual, multicultural, and multievent online polarization detection across 22 languages and more than 110,000 annotated instances. It separates polarization presence, polarization type, and rhetorical manifestation into three related subtasks. This nested task structure motivates explicit study of consistency across DET and TYPE rather than treating them as unrelated classifiers.

Resource: Naseem et al. (2026), “SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization.”  
https://aclanthology.org/2026.semeval-1.453/

Task website:  
https://polar-semeval.github.io/

### 24.2 Hierarchical inconsistency in POLAR systems

YEZE observes that modelling subtasks independently can produce hierarchical violations. This motivates measuring output consistency rather than assuming shared representations will automatically learn the hierarchy.

Resource: Guo and Chang (2026), “YEZE at SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization via Heterogeneous Ensembling.”  
https://aclanthology.org/2026.semeval-1.235/

### 24.3 Hard hierarchical conditioning and LVR

Sagarmatha formalizes Logical Violation Rate, audits violations, and applies an inference-time hierarchy rule. It reports that hierarchical correction can slightly limit multilabel recall, providing direct precedent for the consistency–recall trade-off investigated by M3 and M4.

Resource: Maharjan, Shrestha, and Shrestha (2026), “Sagarmatha at SemEval-2026 Task 9: Heterogeneous Ensembling and Hierarchical Task Conditioning.”  
https://aclanthology.org/2026.semeval-1.382/

### 24.4 Binary gatekeeper followed by specialist classifiers

PolaFusion uses a binary gatekeeper before specialist classifiers trained on polarized content. This supports parent-first modelling, but its gate is combined with ensembling, targeted augmentation, and inverse-frequency weighting, making it difficult to isolate the causal contribution of hierarchy.

Resource: Mohammad (2026), “PolaFusion at SemEval-2026 Task 9: Ensemble Transformers with Targeted Augmentation for Multilingual Polarization Detection.”  
https://aclanthology.org/2026.semeval-1.361/

### 24.5 Cascades for reducing multilabel false positives

MINDS introduces a two-stage cascaded architecture to mitigate false positives under severe imbalance. This supports the motivation for all-text TYPE evaluation and explicit neutral-text false-positive analysis.

Resource: Iannielli et al. (2026), “MINDS at SemEval-2026 Task 9: A Multi-Paradigm Approach to Cross-Lingual Polarization Detection.”  
https://aclanthology.org/2026.semeval-1.313/

### 24.6 Hard DET filtering for TYPE

NAMAA applies a Subtask-1 hard filter before TYPE classification for Arabic and reports a TYPE macro-F1 of 0.62. This is additional evidence that DET-first filtering can reduce irrelevant fine-grained predictions, although hard filtering risks propagating DET false negatives.

Resource: Djamai et al. (2026), “NAMAA at SemEval-2026 Task 9: Comparing Generative, Retrieval-Augmented, and Discriminative Methods for Arabic Online Polarization Detection and Type Classification.”  
https://aclanthology.org/2026.semeval-1.439/

### 24.7 Out-of-fold calibration and class-specific thresholds

SMASH combines encoder ensembles with out-of-fold, class-specific threshold tuning and directly optimizes macro-F1. Its results support treating calibration as a core methodological component in imbalanced multilabel polarization classification rather than as a minor post-processing detail.

Resource: Bokaei et al. (2026), “SMASH at SemEval-2026 Task 9: Detecting Multilingual Polarisation with Encoder Ensembles and Calibrated Decision Thresholds.”  
https://aclanthology.org/2026.semeval-1.116/

### 24.8 General hierarchical multilabel learning

Coherent Hierarchical Multi-Label Classification Networks establish the broader principle that model outputs should respect known label hierarchies. This literature supports treating hierarchy consistency as a model property, not only as post-processing.

Resource: Giunchiglia and Lukasiewicz (2020), “Coherent Hierarchical Multi-Label Classification Networks,” NeurIPS 2020.  
https://proceedings.neurips.cc/paper/2020/hash/6dd4e10e3296fa63738371ec0d5df818-Abstract.html

### 24.9 Multitask learning and loss balancing

Classical multitask learning motivates shared representations, while uncertainty-based weighting provides an alternative to a fixed lambda between DET and TYPE losses. These works are relevant if negative transfer between M1 and M2 becomes a final-stage ablation.

Resources:

- Caruana (1997), “Multitask Learning.”
- Kendall, Gal, and Cipolla (2018), “Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.”  
  https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html

### 24.10 Statistical methodology

The project’s paired bootstrap follows the NLP tradition of evaluating paired system differences on resampled examples. Holm correction controls family-wise error across the pre-declared set of comparisons.

Resources:

- Koehn (2004), “Statistical Significance Tests for Machine Translation Evaluation.”  
  https://aclanthology.org/W04-3250/
- Holm (1979), “A Simple Sequentially Rejective Multiple Test Procedure.”
- Sechidis, Tsoumakas, and Vlahavas (2011), “On the Stratification of Multi-Label Data.”

---

## 25. Literature-derived research gap

A defensible gap statement is:

> Existing SemEval-2026 Task 9 systems demonstrate the practical value of binary gatekeepers, cascaded prediction, hierarchical correction, and calibrated thresholds. However, the reviewed studies do not isolate, under a controlled common encoder and evaluation protocol, whether child-to-parent aggregation, hard parent-to-child gating, or probabilistic parent-to-child factorization provides the best trade-off among DET macro-F1, conditional TYPE accuracy, end-to-end TYPE accuracy, and logical consistency.

The proposed contribution is therefore not simply “another polarization classifier.” It is a controlled study of the direction and strength of hierarchy enforcement.

---

## 26. Proposed paper contributions after completing the extension

If M5 is implemented and evaluated correctly, the final paper can claim the following contributions:

1. A controlled comparison of independent, multitask, post-hoc gated, child-to-parent noisy-OR, and parent-to-child conditional models.
2. Evidence that model ranking depends on whether TYPE evaluation is conditional on gold polarization or performed end to end.
3. A detailed error decomposition showing that noisy-OR increases recall but amplifies false-positive detection under polarized-only threshold calibration.
4. A probabilistically motivated factorization:

   ```text
   P(TYPE_k | x) = P(DET | x) × P(TYPE_k | DET, x).
   ```

5. Evaluation of both predictive accuracy and structural validity using DET macro-F1, both TYPE scoring scopes, LVR, and neutral-text TYPE false positives.
6. A statistically paired, seed-aware analysis with bootstrap confidence intervals and multiplicity correction.

---

## 27. Recommended final-report structure for C1/RQ1

For the ACL paper, distribute this material as follows:

### Methods

- Task hierarchy and notation.
- M1, M2, M3, and M4 definitions.
- Noisy-OR training equation and hard-OR inference rule.
- Cross-validation, seeds, threshold tuning, and frozen metrics.
- LVR definition and paired-bootstrap testing.

### Results

- Main frozen metrics.
- Primary paired comparisons and corrected p-values.
- DET precision/recall and FP/FN decomposition.
- Gold-polarized versus all-text TYPE evaluation.
- Hierarchy-violation results.

### Discussion

- Why recall increased.
- Why false positives increased.
- Why the frozen TYPE metric favours unconstrained models.
- Why the all-text ranking reverses.
- Why the secondary metric does not overturn the frozen hypothesis decision.
- Decoder/calibration confounds and limits on causal interpretation.

### Future work / final-submission extension

- RQ5/H5.
- M5 parent-to-child factorization.
- Full-set threshold calibration.
- Superiority and non-inferiority tests.

### Limitations

- English-only OOF results.
- Parameter mismatch.
- Decoder and checkpoint mismatch.
- Rare-label uncertainty.
- Official TYPE scorer not yet verified.
- Secondary comparisons not multiplicity corrected.

---

## 28. One-paragraph final conclusion

C1/RQ1 shows that enforcing the DET–TYPE hierarchy through child-to-parent noisy-OR does not improve the project’s frozen primary metrics. M4-core obtains lower DET and gold-polarized TYPE macro-F1 than the independent M1 baseline and lower gold-polarized TYPE macro-F1 than M2. Its behaviour is nevertheless scientifically coherent: TYPE evidence increases sensitivity to polarization, raising DET recall from 0.761 to 0.831, but the hard OR of TYPE decisions—whose thresholds were calibrated only on polarized texts—also increases neutral-text false alarms, reducing precision from 0.722 to 0.674. The primary TYPE metric further hides thousands of unsupported TYPE predictions made by M1 and M2 on neutral inputs. Once TYPE is evaluated over all texts, M4 ranks first and eliminates internal hierarchy violations. The original hypothesis must remain unsupported because the primary outcomes were frozen, but the secondary results demonstrate a real advantage in end-to-end structural validity. This motivates a new prospective hypothesis: a calibrated parent-to-child probability factorization may preserve M2’s DET precision while retaining M4’s ability to suppress unsupported TYPE predictions.

---

## 29. Project-local evidence and reproducibility resources

All numerical claims in this document should be traceable to the following project artifacts:

- `docs/mid_submission_results.md` — generated summary of the completed experiments.
- `outputs/analysis/model_summary.csv` — model-level mean and standard deviation.
- `outputs/analysis/paired_comparisons.csv` — paired differences, confidence intervals, raw and Holm-corrected p-values.
- `outputs/analysis/per_label_comparisons.csv` — label-level paired comparisons.
- `outputs/analysis/det_error_profile.csv` — false positives, false negatives, and prediction counts.
- `outputs/analysis/gating_audit_per_seed.csv` — consequences of M2-to-M3 gating.
- `outputs/analysis/selected_thresholds.csv` — threshold means and grid-edge counts.
- `outputs/analysis/seed_level_metrics.csv` — seed-level results.
- `outputs/analysis/results.json` — machine-readable aggregate analysis.
- `scripts/analyze_mid_submission.py` — analysis implementation.
- `report/acl_latex.tex` — current ACL-format report source.
- `report/custom.bib` — current bibliography.

The analysis values are computed from saved out-of-fold predictions, not copied from W&B summaries.

---

## 30. Final checklist before using this material in the ACL paper

- [ ] Keep the frozen RQ1 verdict unchanged.
- [ ] Say “not supported under the frozen protocol,” not “hierarchy is useless.”
- [ ] Do not call M4 significantly worse than M2 on DET after Holm correction.
- [ ] Distinguish training-time noisy-OR from hard-OR inference.
- [ ] Distinguish positive-class F1 from DET macro-F1.
- [ ] Define gold-polarized TYPE scoring explicitly.
- [ ] Label all-text TYPE as complementary/secondary unless promoted prospectively for the extension.
- [ ] Verify the official SemEval Subtask 2 scorer before claiming leaderboard equivalence.
- [ ] Report LVR together with correctness metrics.
- [ ] State that zero LVR does not eliminate consistent false positives.
- [ ] Report threshold and checkpoint mismatches as limitations.
- [ ] Pre-declare M5 primary comparisons, delta, and thresholds before running final tests.
- [ ] Keep C1/RQ1 results separate from new M5 extension results.
- [ ] Use the saved CSV/JSON artifacts, not rounded values in prose, when producing final tables.

