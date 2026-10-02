# C2 / RQ2: Post-hoc Gating versus Training-time Structure — Complete Analysis and Final-Submission Extension

**Project:** Structure over Scale — Hierarchy-Constrained Modeling for English Online Polarization Detection  
**Task:** SemEval-2026 Task 9 (POLAR), English, Subtasks 1–2  
**Scope of this document:** Complete record of C2/RQ2: the claim, what makes the M3-versus-M4 comparison controlled, RQ2-specific metrics, results, statistical tests, the gating audit, a mechanistic decomposition of the outcome, limitations, defensible conclusion, and the proposed end-submission extension.  
**Companion document:** [C1_RQ1.md](C1_RQ1.md). Task formulation and data (§2), the meaning of gold / gold-polarized / all-text scoring (§3), model definitions M1–M4 (§4), the frozen protocol (§6) and the shared metric definitions — precision/recall/F1, DET macro-F1, both TYPE scopes, LVR-A/B and neutral TYPE FPR (§7) — are defined there and are **not repeated here**.  
**Status:** C2/RQ2 is complete under the frozen mid-submission protocol. Sections marked *exploratory* were defined after the primary results were known and are reported as uncorrected, mechanism-seeking analyses. The proposed extension is future work and must be reported separately from the frozen RQ2 result.

---

## 1. Executive conclusion

RQ2 asked whether, at equal hard-prediction consistency, a model trained with the hierarchy (M4-core) preserves more TYPE recall and F1 than post-hoc deterministic reconciliation of an unconstrained model (M3 symmetric).

The answer under the frozen primary evaluation is **inconclusive**. Both models reach zero hierarchy violations in every run, and their primary TYPE scores are statistically indistinguishable:

- TYPE macro-F1: M4 0.461 vs. M3 0.453, difference +0.008, 95% CI [−0.009, +0.022], Holm-corrected p = 0.732, 3/5 seeds favour M4.
- TYPE macro-recall: M4 0.474 vs. M3 0.468, difference +0.006, 95% CI [−0.014, +0.023], Holm-corrected p = 0.732, 2/5 seeds favour M4.

The hypothesis that structure preserves more recall than gating is therefore **not supported**, and neither is the reverse. The confidence intervals do rule out large effects: M4 is not worse than M3 by more than ~0.01 TYPE macro-F1, and not better by more than ~0.02.

The null macro result hides a large, consistent structural difference. Under any consistent decoder, a polarized text whose DET decision fails can receive no TYPE label, so TYPE recall factorizes as:

```text
label recall = coverage (share of gold labels on detected texts)
             × conditional recall (share of those labels then predicted)
```

The two models occupy opposite ends of this product *(exploratory)*:

| | Coverage | Conditional recall | Micro label recall |
|---|---:|---:|---:|
| M3 symmetric | 0.768 | **0.853** | 0.655 |
| M4-core | **0.843** | 0.823 | **0.694** |
| M4 − M3 | **+0.075** [+0.064, +0.087], 5/5 | **−0.030** [−0.041, −0.020], 0/5 | **+0.038** [+0.027, +0.049], 5/5 |

- M3 keeps M2's discrimination on the texts it detects but loses every label on the 23% of label-bearing texts that M2's DET head misses.
- M4 detects more texts but discriminates worse among them, especially on minority labels.
- Because the coverage gain falls mostly on the dominant Political label while the discrimination loss falls on Racial/ethnic, Religious and Other, the effects cancel in the label-balanced macro average and survive in the label-frequency-weighted micro average.

A direct look at the predicted probabilities explains why *(exploratory)*: the noisy-OR detection loss drives every TYPE probability on neutral texts to near zero (median ≤ 0.012), but also compresses minority-label probabilities on texts that genuinely carry them (median on gold-positive Gender texts 0.051 vs. 0.354 for M2; Other 0.159 vs. 0.392), while concentrating detection evidence in Political (0.754 vs. 0.577).

The gating audit makes the cost of post-hoc consistency concrete. Per seed, M3 removes 4,379 of M2's 6,647 TYPE labels. 87.7% of these are false alarms on neutral texts, but 338 are correct labels — 22.9% of everything M2 got right — and on 98.7% of the polarized texts that M2's DET head missed, M2 had already predicted at least one correct dimension.

The scientifically correct conclusion is therefore:

> At equal consistency, training-time noisy-OR structure and post-hoc gating achieve statistically indistinguishable primary TYPE scores, so RQ2 is inconclusive. The two methods pay the cost of consistency differently: gating preserves conditional discrimination but inherits the DET head's misses, whereas noisy-OR widens detection coverage at the price of weaker minority-label discrimination and more neutral false alarms. Within this implementation, most of the recall cost belongs to strict consistency itself; where it is enforced determines which labels pay it.

---

## 2. What RQ2 compares and why it is better controlled than RQ1

### 2.1 Research question

> At equal hard-prediction consistency, does a model trained with the hierarchy preserve more TYPE recall and F1 than post-hoc deterministic reconciliation?

### 2.2 Why “equal consistency” matters

RQ1 compared consistent and inconsistent models, so its TYPE outcome was dominated by the scoring scope (C1 §11). RQ2 removes that confound: M3 symmetric and M4-core both have LVR-A = LVR-B = 0 in all 25 fold/seed runs. Any difference in TYPE recall is therefore a difference in *how* consistency is obtained, not in *whether* it is obtained.

### 2.3 What is matched between M3 and M4

| Component | M3 (via M2) | M4-core | Matched? |
|---|---|---|:---:|
| Encoder, pooling, dropout, TYPE head | DeBERTa-v3-base, mean pooling, 0.1, linear | same | yes |
| TYPE loss, positive weights, λ | masked weighted BCE, λ = 1 | same | yes |
| Optimiser, schedule, epochs, patience | frozen protocol (C1 §6.3) | same | yes |
| Checkpoint selection | inner TYPE macro-F1 | inner TYPE macro-F1 | **yes** |
| TYPE threshold search | per-label F1 on inner gold-polarized texts | same | yes |
| Folds, seeds, inner splits | fixed | same | yes |
| Source of DET | independent DET head, threshold tuned for DET macro-F1 | noisy-OR in training; hard OR of TYPE at inference | **no — this is the manipulated factor** |
| Where consistency is enforced | after decoding | in the loss and the decoder | **no — manipulated** |

Unlike the RQ1 comparison against M1-DET, checkpoint selection is identical here, and both models use one 184M-parameter encoder. The remaining non-matched element — M3 inherits a DET threshold tuned for DET macro-F1, while M4's DET operating point is set implicitly by the TYPE thresholds — is part of what the two mechanisms are, but it also means the comparison is not *operating-point matched* (§12.1).

### 2.4 The RQ2 hypothesis as stated in the proposal

The proposal predicted:

1. M3 would reproduce the recall loss reported by Maharjan et al. (2026) for post-hoc gating.
2. M4 would avoid that loss because it “discards no predictions.”
3. If M4 matched M3 on consistency and M1/M2 on recall, the consistency–recall trade-off would come from *where* the constraint is applied, not from the constraint itself.

Formally, the pre-declared primary tests were:

```text
TYPE macro-F1(M4)     > TYPE macro-F1(M3 symmetric)
TYPE macro-recall(M4) > TYPE macro-recall(M3 symmetric)
```

with both tests included in the six-test Holm family shared with RQ1 (C1 §6.5).

### 2.5 What would count as support

Support required positive paired differences on both primary metrics, with Holm-corrected significance. Zero LVR for both models was a precondition, not evidence: both are consistent by construction, so consistency cannot distinguish them.

---

## 3. RQ2-specific quantities

All shared metrics are defined in C1 §7. RQ2 additionally uses the following. Let G be the gold-polarized texts, D̂ the texts a model predicts polarized, and for each gold label (i, k) with y_ik = 1, let ŷ_ik be the prediction.

### 3.1 Coverage

```text
coverage = #{gold labels (i,k) : i ∈ G ∩ D̂} / #{gold labels (i,k) : i ∈ G}
```

The share of gold TYPE labels whose text the model detects. Under a consistent decoder, a gold label on an undetected text is necessarily missed, so coverage is an upper bound on label-level recall.

### 3.2 Conditional recall

```text
conditional recall = #{(i,k) : i ∈ G ∩ D̂, ŷ_ik = 1} / #{gold labels (i,k) : i ∈ G ∩ D̂}
```

The share of gold labels on detected texts that the model then predicts. This isolates type discrimination from detection.

### 3.3 The recall identity

For any model whose decoder satisfies ŷ_ik = 1 ⇒ predicted DET = 1:

```text
micro label recall = coverage × conditional recall.
```

This is exact, not an approximation, because the numerator of micro recall contains only labels on detected texts. It holds for M3 and M4. It does **not** hold for M1/M2, which can predict labels on texts they classify as non-polarized.

Note that the identity applies to **micro** (label-frequency-weighted) recall. The primary metric is **macro** recall, an unweighted mean over the five labels, so Political — 1,150 of the 1,741 gold labels (66%) — dominates micro recall but contributes only one fifth of macro recall. This difference in weighting is central to §8.

### 3.4 Gating-audit quantities

For M2 → M3 symmetric, per seed over all 3,222 texts: TYPE labels removed by gating, split into wrong labels on neutral texts, wrong labels on polarized texts, and correct labels on polarized texts; polarized texts gated (equal to M2's DET false negatives); and, among those, texts on which M2 had at least one correct TYPE label.

### 3.5 Label recall on single- and multi-label texts

Label-level recall restricted to the 749 polarized texts with exactly one gold TYPE and the 426 with two or more. These test whether gating's recall cost is specific to multi-label texts, as Maharjan et al. (2026) suggest.

---

## 4. Primary RQ2 results

### 4.1 Scores

| Model | TYPE macro-F1 | TYPE macro-P | TYPE macro-R | DET macro-F1 | LVR total |
|---|---:|---:|---:|---:|---:|
| M2 (reference, unconstrained) | 0.497 ± 0.015 | 0.444 ± 0.018 | 0.595 ± 0.019 | 0.793 ± 0.003 | 61.6% |
| M3 one-way | 0.453 ± 0.022 | 0.469 ± 0.033 | 0.468 ± 0.036 | 0.793 ± 0.003 | 0.0% |
| M3 symmetric | 0.453 ± 0.022 | 0.469 ± 0.033 | 0.468 ± 0.036 | 0.793 ± 0.003 | 0.0% |
| M4-core | 0.461 ± 0.007 | 0.465 ± 0.011 | 0.474 ± 0.022 | 0.784 ± 0.004 | 0.0% |

TYPE scores are on gold-polarized texts (frozen primary scope). Mean ± sd over five seed-level out-of-fold scores.

### 4.2 Pre-declared paired comparisons

| Comparison | Metric | M4 | M3 sym. | Δ (M4 − M3) | 95% paired-bootstrap CI | Raw p | Holm p | Seeds favouring M4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M4 − M3 | TYPE macro-F1 | 0.461 | 0.453 | +0.008 | [−0.009, +0.022] | 0.366 | 0.732 | 3/5 |
| M4 − M3 | TYPE macro-R | 0.474 | 0.468 | +0.006 | [−0.014, +0.023] | 0.561 | 0.732 | 2/5 |

### 4.3 Formal decision

Neither primary difference is significant, before or after Holm correction, and the seeds split in both directions. The directional RQ2 hypothesis is **not supported**.

The correct language is:

> RQ2 is inconclusive under the frozen primary protocol: M4-core and post-hoc gating are statistically indistinguishable on TYPE macro-F1 and macro-recall at equal consistency.

Avoid:

> “M4 and M3 are equivalent.”

Equivalence requires a pre-declared margin and a two one-sided test (§13.1). Without it, the defensible statement is about what the intervals exclude.

### 4.4 What the intervals rule out

The 95% intervals bound the plausible effect sizes:

```text
TYPE macro-F1:     −0.009 ≤ Δ ≤ +0.022
TYPE macro-recall: −0.014 ≤ Δ ≤ +0.023
```

So the data are inconsistent with M4 being substantially worse than gating (more than ~0.01 F1) and with M4 recovering most of gating's recall cost. The recall that consistency costs relative to M2 is −0.127 (§6.2); the most favourable bound for M4, +0.023, would recover less than a fifth of it. This is the strongest quantitative statement RQ2 supports: **training-time structure did not recover the recall that consistency costs**.

### 4.5 Seed-level picture

| Seed | M4 TYPE F1 | M3 TYPE F1 | Δ F1 | M4 TYPE R | M3 TYPE R | Δ R | Δ TYPE P | Δ DET F1 | Δ multi-label recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 0.469 | 0.450 | +0.019 | 0.496 | 0.442 | +0.053 | −0.022 | −0.016 | +0.055 |
| 21 | 0.453 | 0.474 | −0.020 | 0.456 | 0.477 | −0.021 | −0.055 | −0.000 | +0.019 |
| 42 | 0.466 | 0.465 | +0.001 | 0.497 | 0.514 | −0.017 | −0.005 | −0.010 | +0.008 |
| 87 | 0.455 | 0.416 | +0.039 | 0.475 | 0.423 | +0.051 | +0.041 | −0.012 | +0.047 |
| 100 | 0.462 | 0.462 | −0.001 | 0.448 | 0.484 | −0.036 | +0.019 | −0.002 | −0.005 |

Two observations:

1. **The seed variation comes mostly from M3.** M3's TYPE recall ranges from 0.423 to 0.514 across seeds, against 0.448–0.497 for M4 (TYPE F1 sd 0.022 vs. 0.007). M3 inherits the variance of M2's DET head: M2's DET false negatives range from 239 to 303 per seed, and every one of them removes all of that text's labels.
2. **No single seed-level factor predicts the sign.** In seeds 13 and 87 M4 wins recall by ~0.05, and in seed 42, where M2's DET head missed the fewest polarized texts (239), M3 wins. But seed 100 has the most M2 misses (303) and M3 still wins recall by 0.036. With five seeds, the per-seed differences are best read as scatter around a near-zero mean, which is what the primary test concludes.

---

## 5. Consistency verification

### 5.1 LVR

| Model | LVR-A | LVR-B | LVR total |
|---|---:|---:|---:|
| M2 | 61.5% | ≈ 0% | 61.6% |
| M3 one-way | 0.0% | 0.01% (0.4 texts per seed) | 0.01% |
| M3 symmetric | 0.0% | 0.0% | 0.0% |
| M4-core | 0.0% | 0.0% | 0.0% |

The RQ2 precondition holds in every run: M3 symmetric and M4-core never produce a contradictory output.

### 5.2 One-way versus symmetric gating

The two M3 variants are practically the same model in this study:

- Their TYPE predictions are **identical** in all five seeds, by construction (the symmetric OR step changes only DET).
- Their DET predictions differ on **2 texts in seed 100** and on none in the other four seeds.

The reason is M2's low Political threshold (mean 0.27): whenever M2's DET head says “polarized,” at least one TYPE head almost always fires, so LVR-B — the only violation the symmetric step repairs — almost never occurs. Consequently, the one-way variant's guarantee “LVR-A = 0, LVR-B may be non-zero” is realised here as LVR-B ≈ 0. This is a property of these thresholds, not of one-way gating in general; with stricter TYPE thresholds the two variants would diverge.

---

## 6. The cost of post-hoc consistency: gating audit

### 6.1 What gating removes

| Seed | M2 TYPE labels | Removed | Wrong, neutral texts | Wrong, polarized texts | **Correct, polarized texts** | M2 DET misses | …with ≥1 correct TYPE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 7,257 | 5,167 | 4,578 | 241 | 348 | 281 | 272 |
| 21 | 6,872 | 4,608 | 4,041 | 222 | 345 | 291 | 290 |
| 42 | 7,067 | 4,510 | 4,037 | 186 | 287 | 239 | 234 |
| 87 | 6,732 | 4,515 | 3,914 | 243 | 358 | 290 | 288 |
| 100 | 5,307 | 3,097 | 2,623 | 120 | 354 | 303 | 302 |
| **Mean** | **6,647** | **4,379** | **3,839 (87.7%)** | **202 (4.6%)** | **338 (7.7%)** | **281** | **277 (98.7%)** |

The 338 correct labels removed per seed are **22.9%** of the 1,479 correct TYPE labels that M2 predicts on polarized texts.

### 6.2 Effect on M2's metrics

| Metric | M2 | M3 symmetric | M3 − M2 | 95% CI | Seeds |
|---|---:|---:|---:|---|---:|
| TYPE macro-F1 (gold-polarized) | 0.497 | 0.453 | −0.043 | [−0.056, −0.032] | 0/5 |
| TYPE macro-R (gold-polarized) | 0.595 | 0.468 | −0.127 | [−0.145, −0.110] | 0/5 |
| TYPE macro-P (gold-polarized) | 0.444 | 0.469 | +0.025 | [+0.013, +0.038] | 5/5 |
| TYPE macro-F1 (all texts) | 0.255 | 0.382 | +0.127 | [+0.113, +0.142] | 5/5 |
| Label recall, multi-label texts | 0.746 | 0.585 | −0.161 | [−0.184, −0.137] | 0/5 |
| Label recall, single-label texts | 0.987 | 0.748 | −0.239 | [−0.265, −0.215] | 0/5 |
| Neutral texts with ≥1 TYPE | 100% | 16.8% | −83.2 pts | [−84.6, −81.7] | 0/5 |
| DET macro-F1 | 0.793 | 0.793 | −0.000 | [−0.000, +0.000] | — |

All comparisons in this table are secondary and uncorrected.

### 6.3 Interpretation

1. **Gating works as designed on neutral texts.** It removes 3,839 false alarms per seed and reduces the share of neutral texts receiving a TYPE label from 100% to 16.8%; the residual 16.8% is exactly M2's DET false-positive rate, which gating cannot touch. Under all-text scoring this is worth +0.127 TYPE macro-F1.
2. **Its cost is error propagation from the parent decision.** Every correct label it removes sits on a polarized text that M2's DET head missed, and on 98.7% of those texts M2 had already identified at least one correct dimension. Gating converts a single DET error into the loss of every dimension on that text — the canonical failure mode of top-down hierarchical classification (Silla and Freitas, 2011).
3. **Gating does not change discrimination.** On detected texts, M3's conditional recall equals M2's (0.853 for both; CI of the difference [0.000, +0.001]). Gating's entire recall cost is coverage.
4. **The cost is not specific to multi-label texts.** Maharjan et al. (2026) describe gating as limiting multi-label recall. Here the absolute loss is larger on single-label texts (−0.239) than on multi-label texts (−0.161): a missed single-label text loses its only label, whereas a multi-label text has some chance of being detected through one of several cues. The mechanism is detection misses, not label multiplicity.
5. **The cost is much larger than the published figure.** Maharjan et al. (2026) report English TYPE moving from 0.511 to 0.506 under gating. Here gating costs −0.043 TYPE F1 on the primary scope while gaining +0.127 on the all-text scope. Their scoring scope may differ from ours, so the two numbers should not be compared directly until the official scorer is verified (C1 §12.5).

### 6.4 Per-label cost of gating (M3 symmetric − M2, gold-polarized texts)

| Label | Δ recall | Δ precision | Δ F1 |
|---|---:|---:|---:|
| Political | −0.238 | −0.000 | −0.134 |
| Racial/ethnic | −0.145 | +0.037 | −0.036 |
| Religious | −0.082 | +0.015 | −0.021 |
| Gender/sexual | −0.122 | +0.065 | −0.025 |
| Other | −0.048 | +0.009 | −0.001 |

Seed means; exploratory. Political pays most because nearly every polarized text carries it, so nearly every DET miss removes a correct Political label. The rare labels gain some precision because gating also removes their false alarms on polarized texts that M2's DET head rejected.

---

## 7. Per-label RQ2 results

### 7.1 Scores (gold-polarized texts, mean over seeds)

| Label (support) | M3 P | M4 P | M3 R | M4 R | M3 F1 | M4 F1 |
|---|---:|---:|---:|---:|---:|---:|
| Political (1,150) | 0.979 | 0.978 | 0.754 | **0.821** | 0.852 | **0.892** |
| Racial/ethnic (281) | **0.532** | 0.490 | 0.569 | 0.557 | **0.547** | 0.521 |
| Religious (112) | 0.424 | 0.425 | 0.502 | 0.486 | 0.456 | 0.450 |
| Gender/sexual (72) | 0.235 | 0.241 | 0.147 | **0.211** | 0.175 | 0.214 |
| Other (126) | 0.178 | 0.191 | **0.370** | 0.297 | 0.237 | 0.228 |

### 7.2 Paired differences (M4 − M3 symmetric)

| Label | Δ recall [95% CI] | Δ precision [95% CI] | Δ F1 [95% CI] |
|---|---|---|---|
| Political | **+0.067** [+0.055, +0.078] | −0.001 [−0.004, +0.001] | **+0.040** [+0.033, +0.048] |
| Racial/ethnic | −0.011 [−0.041, +0.019] | **−0.042** [−0.068, −0.016] | **−0.026** [−0.051, −0.003] |
| Religious | −0.016 [−0.059, +0.026] | +0.001 [−0.035, +0.039] | −0.006 [−0.041, +0.030] |
| Gender/sexual | **+0.064** [+0.009, +0.116] | +0.006 [−0.061, +0.067] | +0.039 [−0.014, +0.089] |
| Other | **−0.073** [−0.113, −0.034] | +0.013 [−0.013, +0.040] | −0.009 [−0.039, +0.019] |

Bold: interval excludes zero. Secondary and uncorrected; ten label-level tests on two statistics should be read as a pattern, not as individual confirmatory findings.

### 7.3 Interpretation

The null macro result is not the absence of effects but the cancellation of opposite ones:

- **Political: M4 better (+0.040 F1).** Its higher coverage recovers Political labels on texts M2's DET head missed. Political is almost a proxy for DET, so detection coverage translates almost one-to-one into Political recall.
- **Gender/sexual: M4 better on recall (+0.064).** M4's per-label threshold for this label (mean 0.50) is lower than M2's (0.57), and its coverage of Gender texts is higher (0.831 vs. 0.733). Support is only 72, so the F1 interval still crosses zero.
- **Other: M4 worse on recall (−0.073).** See §8.3.
- **Racial/ethnic: M4 worse on precision and F1.** M4 detects more polarized texts, giving its Racial/ethnic head more opportunities for false alarms on polarized texts without that label.

Because macro averaging weights each label equally, one strong gain (Political) and one rare-label gain (Gender) are offset by Other, Racial/ethnic and Religious. The macro tie is a composition effect.

### 7.4 Single- versus multi-label texts (M4 − M3 symmetric)

| Label recall on | M4 | M3 | Δ | 95% CI | Seeds M4 > M3 |
|---|---:|---:|---:|---|---:|
| Single-label polarized texts (749) | 0.805 | 0.748 | **+0.056** | [+0.043, +0.070] | 5/5 |
| Multi-label polarized texts (426) | 0.610 | 0.585 | **+0.025** | [+0.009, +0.041] | 4/5 |

Secondary (multi-label) and exploratory (single-label); uncorrected. M4 recovers more labels than gating on both kinds of text, and more on single-label texts. Its advantage is therefore not a multi-label effect: it is the coverage effect of §8, which matters most where a single detection miss removes a text's only label.

---

## 8. Mechanism: coverage versus discrimination *(exploratory)*

### 8.1 The decomposition

| Quantity | M3 symmetric | M4-core | M4 − M3 | 95% CI | Seeds M4 > M3 |
|---|---:|---:|---:|---|---:|
| Coverage | 0.768 ± 0.019 | 0.843 ± 0.017 | **+0.075** | [+0.064, +0.087] | 5/5 |
| Conditional recall | 0.853 ± 0.014 | 0.823 ± 0.011 | **−0.030** | [−0.041, −0.020] | 0/5 |
| Micro label recall | 0.655 ± 0.025 | 0.694 ± 0.012 | **+0.038** | [+0.027, +0.049] | 5/5 |
| Macro recall (primary) | 0.468 ± 0.036 | 0.474 ± 0.022 | +0.006 | [−0.014, +0.023] | 2/5 |

Check of the identity (§3.3): 0.768 × 0.853 = 0.655 and 0.843 × 0.823 = 0.694.

Unlike the macro comparison, all three decomposition effects are large relative to their intervals and consistent in sign across all five seeds. What looks like “no difference” on the primary metric is two substantial, opposing differences.

### 8.2 Per-label decomposition (mean over seeds)

| Label | M3 coverage | M4 coverage | M3 conditional recall | M4 conditional recall | M3 recall | M4 recall |
|---|---:|---:|---:|---:|---:|---:|
| Political | 0.760 | 0.830 | 0.992 | 0.989 | 0.754 | 0.821 |
| Racial/ethnic | 0.788 | 0.870 | 0.721 | 0.640 | 0.569 | 0.557 |
| Religious | 0.796 | 0.884 | 0.630 | 0.550 | 0.502 | 0.486 |
| Gender/sexual | 0.733 | 0.831 | 0.197 | 0.254 | 0.147 | 0.211 |
| Other | 0.783 | 0.870 | 0.473 | 0.341 | 0.370 | 0.297 |

M4's coverage advantage is uniform (+0.07 to +0.10 on every label). Its conditional recall is essentially unchanged on Political, higher on Gender, and substantially lower on Racial/ethnic (−0.08), Religious (−0.08) and Other (−0.13). Whether a label gains or loses overall depends on which effect is larger:

```text
Political:      coverage gain dominates          → recall +0.067
Gender/sexual:  both effects favour M4           → recall +0.064
Racial/ethnic:  roughly balanced                 → recall −0.011
Religious:      roughly balanced                 → recall −0.016
Other:          discrimination loss dominates    → recall −0.073
```

### 8.3 Why M4 discriminates minority labels worse

Median TYPE probability, pooled over all folds and seeds:

| Label | M2: texts with label | M4: texts with label | M2: polarized, without label | M4: polarized, without label | M2: neutral texts | M4: neutral texts |
|---|---:|---:|---:|---:|---:|---:|
| Political | 0.577 | **0.754** | 0.475 | 0.607 | 0.548 | **0.012** |
| Racial/ethnic | 0.726 | 0.614 | 0.239 | 0.127 | 0.404 | **0.005** |
| Religious | 0.772 | **0.419** | 0.129 | 0.014 | 0.302 | **0.004** |
| Gender/sexual | 0.354 | **0.051** | 0.182 | 0.020 | 0.505 | **0.004** |
| Other | 0.392 | **0.159** | 0.269 | 0.062 | 0.269 | **0.003** |

(M3 uses M2's probabilities unchanged.)

Three effects are visible:

1. **Abstention on neutral text works.** M4 drives every TYPE probability on neutral texts to ~0.01 or below, whereas M2 assigns Political a median of 0.548 on neutral text. This is exactly the supervision the noisy-OR loss was designed to provide where the TYPE loss is masked.
2. **Detection evidence concentrates in Political.** The DET target on 1,175 polarized texts can be satisfied by Political alone, which is present on 98% of them. The noisy-OR loss therefore rewards raising Political (median 0.754 on texts with the label) and gives little incentive to raise the other heads, while the 2,047 neutral texts push *all* heads down.
3. **Minority heads are compressed even on true positives.** The median Gender probability on texts that actually carry Gender falls from 0.354 to 0.051, Other from 0.392 to 0.159, and Religious from 0.772 to 0.419. Their positive-class weights in the masked TYPE loss do not fully counteract the downward pressure from the DET loss.

This is a credit-assignment effect of the noisy-OR: since p_DET = 1 − ∏(1 − p_k), a single dominant head can explain the DET label, and the gradient of the DET loss on a polarized text with respect to head j is scaled by ∏_{k≠j}(1 − p_k). Once Political is confident, that factor is small for every other head, so polarized texts stop pushing minority heads up, while neutral texts keep pushing them down. The per-label thresholds partly compensate — and M4's Gender threshold is lower than M2's — but a threshold cannot restore the ranking information lost when positive and negative probabilities are compressed together.

### 8.4 What this means for RQ2's question

The proposal framed RQ2 as: is the recall cost of consistency caused by the constraint itself, or by where it is applied? The decomposition gives a more precise answer:

- **Strict consistency imposes the bound.** Under any consistent decoder, label recall cannot exceed detection coverage. Both consistent models lose about 0.12 macro recall relative to M2, which is not bound by its own detection.
- **Placement changes which factor pays.** Post-hoc gating keeps the unconstrained model's discrimination but accepts its detector's coverage. Training-time noisy-OR raises coverage but, through the credit-assignment effect, degrades minority-label discrimination.
- **In this implementation the two costs are about the same size in macro terms.** Hence the primary null.

The trade-off is therefore not eliminated by moving the constraint into training — at least not with a child-to-parent noisy-OR and polarized-only threshold calibration.

---

## 9. The other side of the trade-off: detection and neutral texts

### 9.1 DET

| Metric | M3 symmetric | M4-core | Δ | 95% CI | Seeds |
|---|---:|---:|---:|---|---:|
| DET macro-F1 | 0.793 | 0.784 | −0.008 | [−0.016, −0.001] | 0/5 |
| DET precision (polarized) | 0.722 | 0.674 | −0.048 | [−0.057, −0.038] | 0/5 |
| DET recall (polarized) | 0.761 | 0.831 | +0.070 | [+0.059, +0.081] | 5/5 |

M3's DET equals M2's except for two texts, so this is the same precision–recall trade-off analysed in C1 §10, with the same cause: M4's DET operating point is set by TYPE thresholds tuned without neutral texts (Political at the 0.20 grid floor in 19/25 runs).

### 9.2 Neutral texts

| Quantity (per seed) | M3 symmetric | M4-core | Δ | 95% CI | Seeds |
|---|---:|---:|---:|---|---:|
| Neutral texts with ≥1 TYPE (neutral TYPE FPR) | 16.8% (345 texts) | 23.0% (472 texts) | +6.2 pts | [+5.3, +7.1] | 5/5 against M4 |
| TYPE labels on neutral texts | 617 | 643 | +26 | [−3, +53] | 4/5 against M4 |

M4 abstains well in probability (§8.3), but its hard-OR decoder plus low Political threshold turns the remaining borderline Political scores into DET false positives — each of which, by consistency, also carries a TYPE label. Its zero LVR therefore coexists with more consistent-but-wrong predictions on neutral texts than M3. LVR alone would hide this (C1 §7.5).

### 9.3 All-text TYPE

| Metric | M3 symmetric | M4-core | Δ | 95% CI | Seeds |
|---|---:|---:|---:|---|---:|
| TYPE macro-F1, all texts | 0.382 ± 0.017 | 0.396 ± 0.008 | +0.014 | [−0.001, +0.026] | 4/5 |

Despite more neutral false alarms, M4 is marginally ahead on all-text TYPE F1 because its higher label recall on polarized texts outweighs them; the interval just touches zero. This is secondary, uncorrected, and should not be presented as evidence for RQ2.

### 9.4 Overall operating points

| | M2 | M3 symmetric | M4-core |
|---|---|---|---|
| Hierarchy violations | 61.6% | 0% | 0% |
| DET recall / precision | 0.761 / 0.722 | 0.761 / 0.722 | 0.831 / 0.674 |
| Detection coverage of gold labels | — | 0.768 | 0.843 |
| Conditional TYPE recall | 0.853 | 0.853 | 0.823 |
| Neutral TYPE FPR | 100% | 16.8% | 23.0% |
| TYPE macro-F1 (gold-polarized / all) | 0.497 / 0.255 | 0.453 / 0.382 | 0.461 / 0.396 |
| Seed sd of TYPE macro-F1 | 0.015 | 0.022 | 0.007 |

M3 and M4 are two different points on the same consistency-constrained frontier: M3 is the higher-precision, lower-coverage point; M4 is the higher-coverage, lower-precision point. The frozen primary metric does not separate them.

---

## 10. Stability

M4 is the most stable consistent model:

```text
TYPE macro-F1 seed sd:   M4 0.007   M3 0.022   (M2 0.015)
TYPE macro-recall sd:    M4 0.022   M3 0.036
Micro label recall sd:   M4 0.012   M3 0.025
Multi-label recall sd:   M4 0.018   M3 0.033
```

M3 compounds two sources of seed variance — M2's TYPE head and M2's DET head — because every DET miss removes all of a text's labels. M4's DET decision is derived from its TYPE head, so there is no separate detector whose errors multiply the TYPE errors.

With five seeds, these standard deviations are themselves imprecise, and no formal variance test was pre-declared. The stability difference is consistent across every RQ2 metric but should be reported descriptively.

---

## 11. Complete scientific explanation of the RQ2 outcome

The inconclusive primary result follows from four linked mechanisms.

### 11.1 Consistency caps TYPE recall at detection coverage

For both consistent models, a gold label can only be recovered on a detected text. M2, which is not consistent, is not subject to this cap and reaches 0.595 macro recall; both consistent models fall to ~0.47.

### 11.2 Gating inherits its detector's coverage unchanged

M3 cannot improve on M2's DET head. Its conditional recall equals M2's (0.853), and its entire recall loss — 338 correct labels per seed — comes from the 23% of gold labels on texts M2's detector misses.

### 11.3 Noisy-OR raises coverage but redistributes evidence

M4's TYPE-derived detection covers 84% of gold labels. But the noisy-OR loss concentrates detection evidence in Political and suppresses minority-label probabilities on both neutral and positive texts, lowering conditional recall on Racial/ethnic, Religious and Other.

### 11.4 Macro averaging cancels the two effects

The coverage gain is concentrated in the most frequent label; the discrimination loss is concentrated in minority labels. Label-frequency-weighted (micro) recall favours M4 by +0.038 with all seeds agreeing, while label-balanced (macro) recall shows +0.006 with seeds split.

Together these explain why M4 can simultaneously:

- tie M3 on the primary TYPE metrics;
- have higher micro label recall and detection coverage;
- have lower conditional recall on minority labels;
- have more neutral false alarms but higher all-text TYPE F1; and
- have lower seed variance.

There is no contradiction: each metric weights the coverage and discrimination effects differently.

---

## 12. Alternative explanations and limits on causal claims

### 12.1 The comparison is not operating-point matched

M3 inherits M2's DET threshold, tuned for DET macro-F1; M4's DET operating point is fixed implicitly by TYPE thresholds tuned on polarized texts only. Part of M4's coverage gain may simply be a more permissive operating point rather than a property of training-time structure. A matched comparison would hold DET precision (or recall) equal and compare TYPE recall at that point (§15.2).

### 12.2 M3 is a re-implementation, not the published system

M3 applies the deterministic rule of Maharjan et al. (2026) to our M2. It does not reproduce their architecture, including the stop-gradient that detaches child from parent logits, nor their ensemble. The RQ2 comparison isolates *post-hoc rule versus training-time structure on a common encoder*; it does not test the published Sagarmatha system.

### 12.3 One training-time structure was tested

M4 is a child-to-parent noisy-OR with hard-OR decoding. The credit-assignment effect in §8.3 is specific to that form. Other training-time constraints — a soft consistency penalty, a semantic-loss formulation (Xu et al., 2018), a leaky noisy-OR (Pearl, 1988), or a parent-to-child factorization (C1 §19) — could place the trade-off elsewhere.

### 12.4 Thresholds were tuned without neutral texts

Both models' TYPE thresholds were chosen on inner-validation gold-polarized texts. This matters more for M4, whose DET decision depends on them, and is the same confound documented for RQ1 (C1 §15.1).

### 12.5 The mechanism analyses are post hoc

Coverage, conditional recall, the probability profile and the per-label decompositions were designed after the primary result was observed. Their intervals are uncorrected and they were not part of the Holm family. They explain the primary result; they do not replace it.

### 12.6 Rare-label uncertainty

Gender/sexual (72 texts) and Religious (112) thresholds are tuned on roughly 6 and 9 inner-validation positives per run. Label-level conclusions for these labels carry wide intervals.

### 12.7 Scope

English training-set cross-validation only, one encoder, five seeds, one consumer GPU. The development and test sets were not scored.

---

## 13. Statistical notes specific to RQ2

### 13.1 Inconclusive is not equivalent

A non-significant difference does not establish equivalence. An equivalence claim requires a margin δ fixed in advance and a two one-sided test showing the whole CI lies within (−δ, +δ). No margin was pre-declared for RQ2, so none can be applied now without post-hoc selection. A margin would have to be pre-registered for the final submission (§15.3).

### 13.2 Precision of the primary estimate

The 95% interval half-widths are ~0.016 for TYPE macro-F1 and ~0.019 for macro-recall. Differences smaller than about 0.02 could not have been reliably detected with 3,222 texts and five seeds. The bootstrap resamples texts; training randomness enters only through the five seeds, whose split (3/5 and 2/5) independently indicates no consistent primary effect.

### 13.3 Why micro and macro disagree without contradiction

Macro recall = (1/5) Σ_k recall_k; micro recall = Σ_k w_k recall_k with w_k = support_k / Σ support. With w_Political = 0.66, a +0.067 Political recall gain contributes +0.044 to micro recall but only +0.013 to macro recall, while the −0.073 Other loss contributes −0.005 to micro but −0.015 to macro. The primary metric's label balance — an appropriate choice for an imbalanced task — is exactly what makes the two mechanisms cancel.

---

## 14. Claims that are and are not supported

### 14.1 Supported claims

1. M3 symmetric and M4-core both achieve zero hierarchy violations in every run.
2. At equal consistency, their TYPE macro-F1 and macro-recall are not significantly different (Holm p = 0.732 for both).
3. The intervals rule out M4 being more than ~0.01 TYPE macro-F1 worse than gating, and rule out M4 recovering most of the recall gating costs.
4. Post-hoc gating removes 22.9% of M2's correct TYPE labels, entirely through DET misses, and nearly all gated polarized texts (98.7%) already had a correct dimension.
5. Gating improves TYPE precision and all-text TYPE F1 but costs 0.127 macro recall on the primary scope.
6. Gating's recall cost is not specific to multi-label texts; it is larger in absolute terms on single-label texts.
7. One-way and symmetric gating are practically identical under M2's thresholds.
8. *(Exploratory)* M4 has higher detection coverage and micro label recall but lower conditional recall than M3, consistently across seeds.
9. *(Exploratory)* M4 compresses minority-label probabilities, including on texts that carry those labels, while concentrating detection evidence in Political.
10. M4 has lower seed variance than M3 on every RQ2 metric (descriptive).

### 14.2 Unsupported or overstated claims

1. “Training-time structure preserves more recall than gating.” Not supported on the primary metrics.
2. “M3 and M4 are equivalent.” No equivalence margin was pre-declared.
3. “The recall cost of consistency is due to post-hoc placement.” Both placements pay a comparable macro cost.
4. “M4 improves multi-label recall” as a confirmatory finding. The +0.025 gain is secondary, uncorrected, and smaller than M4's single-label gain (+0.056), so it is not multi-label-specific.
5. “Zero LVR means M4 makes fewer errors on neutral texts than M3.” M4 flags more neutral texts (23.0% vs. 16.8%).
6. “This replicates Sagarmatha.” M3 re-implements their inference rule on our model, not their system.
7. “Noisy-OR is inherently worse at minority labels.” Shown for one noisy-OR formulation with polarized-only thresholds; alternatives are untested.

---

## 15. Proposed end-submission extension

The extension must not redefine RQ2 after seeing the result. The analyses above motivate new, prospectively stated questions. The parent-to-child factorization (M5) proposed in C1 §18–23 is the natural joint continuation of RQ1 and RQ2 and is not repeated here; this section lists only the RQ2-specific additions.

### 15.1 Proposed research question

> **RQ2-ext:** At a matched detection operating point, does training-time hierarchy enforcement retain more TYPE recall than post-hoc gating, and does a credit-assignment correction remove the minority-label discrimination loss?

### 15.2 H4 — operating-point-matched comparison

The saved predictions already contain M2's DET probabilities and M4's noisy-OR DET probabilities, so this comparison needs no retraining:

1. Sweep M3's DET threshold and M4's TYPE threshold scale on inner-validation data.
2. Select the points at which both reach the same DET precision (e.g., M2's 0.722) and, separately, the same DET recall.
3. Compare TYPE macro-recall and macro-F1 at each matched point on the outer folds.

```text
H4: TYPE macro-recall(M4 | matched DET precision) > TYPE macro-recall(M3 | matched DET precision).
```

If H4 holds, training-time structure has a genuine advantage that the frozen protocol masked by placing the two models at different operating points. If it fails, the frozen null is not an operating-point artifact.

### 15.3 H5 — equivalence with a pre-declared margin

If H4 is null, test equivalence rather than reporting another non-significant difference:

```text
H5: |TYPE macro-F1(M4) − TYPE macro-F1(M3)| < δ,  with δ = 0.02 fixed before the run,
```

using a two one-sided paired-bootstrap test. δ = 0.02 is roughly the precision of the current design (§13.2) and is smaller than the gating cost (−0.043 F1) the comparison is meant to address.

### 15.4 H6 — credit-assignment correction for noisy-OR

The §8.3 mechanism suggests modifications that keep M4's coverage while protecting minority-label discrimination:

- **Leaky noisy-OR** (Pearl, 1988): `p_DET = 1 − (1 − ℓ) ∏_k (1 − p_k)`, with a learned leak ℓ that absorbs polarization not explained by any head, so the DET loss stops forcing Political to explain everything.
- **Per-label DET-gradient balancing:** rescale the noisy-OR DET gradient reaching each head by inverse label frequency.
- **Neutral-aware threshold calibration:** choose TYPE thresholds on all inner-validation texts with a joint DET + TYPE objective (shared with C1 §21).

```text
H6: conditional recall(M4-corrected) ≥ conditional recall(M3) on Racial/ethnic, Religious and Other,
    while coverage(M4-corrected) > coverage(M3).
```

### 15.5 Pre-registration requirements

- Fix δ, the matched operating points, and the primary comparisons before running.
- Keep coverage and conditional recall as pre-declared secondary outcomes, since they are now known to explain the primary metric.
- Correct the new primary family with Holm, separately from the frozen mid-submission family.
- Report the frozen RQ2 verdict unchanged alongside any extension result.

### 15.6 Possible outcomes

| Outcome | Interpretation |
|---|---|
| H4 holds | Training-time structure helps recall once operating points are matched; the frozen null was an operating-point artifact. |
| H4 fails, H5 holds | Placement genuinely does not matter for macro TYPE quality at equal consistency in this setting. |
| H6 holds, macro recall improves | The frozen null was caused by noisy-OR credit assignment, not by training-time enforcement in general. |
| H6 holds, macro recall unchanged | Coverage and discrimination trade off even after correction; the bound in §11.1 dominates. |
| M5-Constrained ≈ M3 and M5-Soft > both | Strict discrete consistency, not its placement, is the binding cost (see C1 §20). |

---

## 16. Literature specific to RQ2

Shared POLAR, hierarchy and statistics references are reviewed in C1 §24. The works below bear specifically on the post-hoc-versus-training-time question.

### 16.1 Post-hoc hierarchical correction in POLAR

Maharjan et al. (2026) formalise LVR and enforce the parent-to-child rule at inference while detaching the child from the parent during training. They report that the rule “slightly limits raw multi-label recall.” Our M3 re-implements their rule; our audit (§6) quantifies its cost and finds the loss is driven by detection misses rather than label multiplicity.  
https://aclanthology.org/2026.semeval-1.382/

Mohammad (2026) uses a binary gatekeeper that zeroes child labels for non-polarized texts and forces an argmax when a polarized text receives no label — a hard-gating design whose second step corresponds to the LVR-B repair of symmetric M3.  
https://aclanthology.org/2026.semeval-1.361/

Lin et al. (2026) apply an inference-time check of the DET prediction before assigning TYPE labels, another instance of post-hoc consistency.  
https://aclanthology.org/2026.semeval-1.166/

### 16.2 Error propagation in top-down hierarchical classification

Silla and Freitas (2011), “A survey of hierarchical classification across different application domains,” *Data Mining and Knowledge Discovery* 22(1–2), describe how top-down (local, parent-first) classifiers propagate errors made at a parent node to all its descendants. The gating audit is a direct measurement of this effect at the DET → TYPE level.

### 16.3 Enforcing constraints during training versus at inference

Giunchiglia and Lukasiewicz (2020), “Coherent Hierarchical Multi-Label Classification Networks” (NeurIPS), build hierarchy consistency into the network output through a max-constraint module and a matching loss, rather than repairing outputs afterwards — the closest general analogue of M4's child-to-parent aggregation.

Xu et al. (2018), “A Semantic Loss Function for Deep Learning with Symbolic Knowledge” (ICML), train networks with a loss derived from logical constraints, a softer alternative to architectural enforcement relevant to §12.3.

Giunchiglia, Stoian and Lukasiewicz (2022), “Deep Learning with Logical Constraints” (IJCAI survey), review the design space of constraint enforcement — in the architecture, in the loss, or at inference — which frames RQ2's comparison of placements.

### 16.4 Noisy-OR and the leak parameter

Pearl (1988), *Probabilistic Reasoning in Intelligent Systems*, introduces the noisy-OR model of independent causes and its leaky variant, in which a leak probability accounts for causes outside the modelled set. The leak motivates H6 (§15.4): without it, the model must attribute every polarized text to one of the five heads.

---

## 17. ACL-ready answer to RQ2

The following text can be adapted directly into the Results/Discussion section of an ACL-format paper.

### 17.1 Results paragraph

At equal hard-prediction consistency, training-time structure did not outperform post-hoc gating. Both M3 symmetric and M4-core produced zero hierarchy violations in every run. M4-core obtained 0.461 ± 0.007 TYPE macro-F1 and 0.474 ± 0.022 macro-recall on gold-polarized texts, against 0.453 ± 0.022 and 0.468 ± 0.036 for M3. The paired differences, +0.008 (95% CI [−0.009, +0.022]) and +0.006 ([−0.014, +0.023]), were not significant (Holm-corrected p = 0.732 for both), and the seeds split 3/5 and 2/5. RQ2 is therefore inconclusive under the pre-declared protocol. The intervals nonetheless exclude M4 recovering most of the 0.127 macro recall that gating costs relative to the unconstrained M2.

### 17.2 Gating-audit paragraph

Gating removed 4,379 of the 6,647 TYPE labels M2 predicted per seed. Most removals (87.7%) were false alarms on neutral texts, which raised all-text TYPE macro-F1 by 0.127, but 338 were correct labels — 22.9% of M2's correct TYPE predictions. Every removed correct label sat on a polarized text that M2's detection head had missed, and on 98.7% of those texts M2 had already predicted at least one correct dimension. The loss was larger on single-label (−0.239) than on multi-label texts (−0.161), indicating that the cost of gating is driven by detection errors propagating downward rather than by label multiplicity.

### 17.3 Mechanism paragraph

The macro-level tie conceals two opposing effects. Under any consistent decoder, label recall equals detection coverage times recall among detected texts. M4-core covered more gold labels than M3 (0.843 vs. 0.768; +0.075, 95% CI [+0.064, +0.087]) but recovered fewer of them once detected (0.823 vs. 0.853; −0.030, [−0.041, −0.020]); both differences held in all five seeds. The coverage gain concentrated in Political (+0.067 recall), which appears in 98% of polarized texts, while the discrimination loss concentrated in Other, Racial/ethnic and Religious. Predicted probabilities show why: the noisy-OR detection loss drove all TYPE probabilities on neutral texts below 0.013 in median, but also compressed minority-label probabilities on texts that carried them (e.g., Gender/sexual from 0.354 to 0.051), as detection evidence concentrated in Political. Micro recall, which weights labels by frequency, favoured M4-core by 0.038; macro recall, which weights them equally, did not. These decomposition analyses are exploratory.

### 17.4 Final interpretation paragraph

RQ2 therefore suggests that, in this setting, most of the recall cost of hierarchical consistency belongs to strict consistency itself: no consistent decoder can recover a label on a text it does not detect. Where the constraint is enforced determines how the cost is paid. Post-hoc gating preserves the unconstrained model's discrimination but inherits its detector's misses; child-to-parent noisy-OR training widens detection but redistributes evidence away from minority labels. Neither placement eliminated the trade-off. Operating-point-matched comparisons and constraint formulations that avoid noisy-OR credit concentration are the natural next tests.

---

## 18. Recommended final-report structure for C2/RQ2

### Methods

- M3 one-way and symmetric rules (inherited thresholds).
- The matched/unmatched components table (§2.3).
- Coverage, conditional recall and the recall identity (§3), introduced as analysis tools.

### Results

- LVR confirmation of equal consistency.
- Primary paired comparisons with Holm-corrected p-values and intervals.
- Gating audit table and figure.
- Per-label M4 − M3 differences.

### Discussion

- The coverage × conditional-recall decomposition and why macro and micro disagree.
- The probability profile and the noisy-OR credit-assignment effect.
- Single- versus multi-label gating cost and the comparison with Maharjan et al. (2026).
- Operating-point confound.

### Future work / final-submission extension

- RQ2-ext with H4 (matched operating point), H5 (equivalence margin), H6 (credit-assignment correction).
- M5-Soft versus M5-Constrained (C1 §20) as the parent-to-child test of the same question.

### Limitations

- Not operating-point matched.
- M3 re-implements a rule, not the published system.
- Post-hoc mechanism analyses.
- Rare-label uncertainty; English-only; five seeds.

---

## 19. One-paragraph final conclusion

C2/RQ2 compared two ways of making polarization-type predictions consistent with polarization detection: repairing an unconstrained model's outputs after decoding (M3) and training a model whose detection is derived from its type predictions (M4-core). Both achieved zero hierarchy violations, and their primary TYPE macro-F1 and macro-recall were statistically indistinguishable, so the hypothesis that training-time structure preserves more recall is not supported. The gating audit shows that post-hoc consistency removes nearly a quarter of the correct type labels, all on polarized texts its detector missed and almost all of which already carried a correct dimension. An exploratory decomposition shows that the primary tie is the cancellation of two consistent effects: M4 detects more label-bearing texts but discriminates minority labels less well among them, because its noisy-OR objective concentrates detection evidence in the dominant Political label and compresses rare-label probabilities. Within this implementation, the recall cost of consistency belongs mainly to strict consistency itself, and its placement decides which labels pay. Operating-point-matched comparisons and alternative training-time constraints are needed before concluding that placement cannot remove the trade-off.

---

## 20. Project-local evidence and reproducibility resources

Primary results (shared with C1, see C1 §29):

- `outputs/analysis/model_summary.csv` — model-level means and standard deviations.
- `outputs/analysis/paired_comparisons.csv` — primary RQ2 tests (rows with `tier = primary`, `rq = RQ2`) and secondary M4 − M3 and M3 − M2 comparisons.
- `outputs/analysis/per_label_comparisons.csv` — per-label M4 − M3 differences with CIs.
- `outputs/analysis/gating_audit_per_seed.csv` — the gating audit.
- `outputs/analysis/seed_level_metrics.csv` — seed-level scores, including per-label precision/recall/F1 and multi/single-label recall.
- `outputs/analysis/selected_thresholds.csv` — inherited M2 thresholds and M4 thresholds.
- `scripts/analyze_mid_submission.py` — primary analysis.

RQ2-specific exploratory analyses:

- `outputs/analysis/rq2_decomposition.csv` — coverage, conditional recall, micro/single/multi-label recall and neutral false positives, M4 − M3 and M3 − M2, with paired-bootstrap CIs.
- `outputs/analysis/rq2_per_label_decomposition.csv` — per-label coverage and conditional recall (§8.2).
- `outputs/analysis/rq2_seed_differences.csv` — per-seed differences (§4.5).
- `outputs/analysis/rq2_probability_profile.csv` — median TYPE probabilities (§8.3).
- `scripts/analyze_rq2_decomposition.py` — produces the four files above from the saved out-of-fold predictions (`python scripts/analyze_rq2_decomposition.py`, after `scripts/postprocess_m1_m3.sh`).

All values are computed from saved out-of-fold predictions, not from W&B summaries.

---

## 21. Final checklist before using this material in the ACL paper

- [ ] Keep the frozen RQ2 verdict unchanged: inconclusive, not supported.
- [ ] Do not describe M3 and M4 as equivalent without a pre-declared margin.
- [ ] State that both models have zero LVR before comparing TYPE scores.
- [ ] Report what the intervals exclude, not only that p > 0.05.
- [ ] Label coverage, conditional recall and the probability profile as exploratory.
- [ ] Distinguish micro from macro recall whenever the decomposition is cited.
- [ ] Present gating's cost as detection-error propagation, with the single- vs. multi-label evidence.
- [ ] Do not claim M4 makes fewer neutral-text errors; it makes more (23.0% vs. 16.8%).
- [ ] Note that one-way and symmetric gating coincide under these thresholds.
- [ ] Report the operating-point mismatch as a limitation.
- [ ] Say M3 re-implements the Sagarmatha rule, not their system.
- [ ] Verify the official SemEval Subtask 2 scorer before comparing gating costs with Maharjan et al. (2026).
- [ ] Pre-register δ, matched operating points and primary comparisons before running the RQ2 extension.
