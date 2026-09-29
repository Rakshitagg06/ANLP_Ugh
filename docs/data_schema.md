# Data schema

## Canonical training table

The training pipeline expects one row per example with these columns:

| Column | Type | Meaning |
|---|---|---|
| `example_id` | string | Stable unique identifier |
| `text` | string | Input English text |
| `gold_det` | integer | DET label, 0 or 1 |
| `type_political` | integer | Political label, 0 or 1 |
| `type_racial_ethnic` | integer | Racial/ethnic label, 0 or 1 |
| `type_religious` | integer | Religious label, 0 or 1 |
| `type_gender_sexual_identity` | integer | Gender/sexual-identity label, 0 or 1 |
| `type_other` | integer | Other label, 0 or 1 |
| `fold` | integer | Fixed outer fold from 0 through 4 |

CSV, JSON, JSONL, and Parquet are supported. Parquet requires the optional `pyarrow` dependency.

## Mapping official columns

If the official dataset uses different names, change only the `data` mapping in the YAML files. Do not rename fields independently in model code.

Example:

```yaml
data:
  id_column: post_id
  text_column: sentence
  det_label_column: is_polarized
  fold_column: fold
  type_label_columns:
    political: political
    racial_ethnic: racial_or_ethnic
    religious: religious
    gender_sexual_identity: gender_or_sexual_identity
    other: other
```

For the supplied English CSV files, the committed `configs/create_folds.yaml` mapping is:

```yaml
id_column: id
text_column: text
det_label_column: polarization
type_label_columns:
  political: political
  racial_ethnic: racial/ethnic
  religious: religious
  gender_sexual_identity: gender/sexual
  other: other
```

The supplied data shows that `other` can co-occur with named TYPE labels; it is not an exclusive residual label.

## Required validation before training

Produce and review:

- dataset size and duplicate-ID count;
- missing-text count;
- DET class counts;
- every TYPE label count;
- label counts by fold;
- count of DET-negative examples with a positive TYPE label;
- count of DET-positive examples with no TYPE label;
- co-occurrence counts involving `Other`;
- duplicate and near-duplicate text audit.

Do not silently convert missing TYPE labels into negatives until the official annotation semantics are confirmed.

## TYPE evaluation mask

The current configuration calculates TYPE scores on gold DET-positive examples because the proposal describes TYPE labels as undefined for non-polarized examples. This is controlled by:

```yaml
type_metrics_on_gold_det_positive_only: true
```

Confirm this choice against the official scorer before final experiments.
