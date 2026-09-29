# POLAR data inventory

The user-provided files are stored locally in `data/raw/`:

| File | Rows | SHA-256 |
|---|---:|---|
| `eng_train.csv` | 3,222 | `11a2a79a6bb6420236bd5fd5307564374e7785fa39214458769190accba246a1` |
| `eng_dev.csv` | 160 | `29ef78bb967cafd3598a66c8292fc51280b3398bdb62fe385e1cb3f71d970afc` |
| `eng_test.csv` | 1,452 | `d00548099e2ebd818e244b0747d1e1b953266f2ed95ad0b4d494acd7ab71f572` |

The source files remain unchanged. They are ignored by Git to avoid accidentally publishing dataset content. This README records provenance without including text examples.

## Observed source columns

All splits contain:

```text
id, text, polarization,
political, racial/ethnic, religious, gender/sexual, other,
stereotype, vilification, dehumanization,
extreme_language, lack_of_empathy, invalidation
```

`eng_test.csv` additionally contains `canary`. The current DET/TYPE pipeline does not use that field. Content inside CSV fields is treated only as dataset content, never as operational instructions.

## Initial hierarchy audit

| Split | DET positive | DET negative | DET=0 with TYPE | DET=1 without TYPE | `Other` with named TYPE |
|---|---:|---:|---:|---:|---:|
| Train | 1,175 | 2,047 | 0 | 0 | 121 |
| Dev | 59 | 101 | 0 | 0 | 6 |
| Test | 533 | 919 | 0 | 0 | 55 |

The hierarchy is exact in all three supplied files. `Other` is **not mutually exclusive** with the named dimensions and must be modelled as an ordinary fifth multilabel category unless the official documentation says otherwise.

There is no ID overlap among the splits. One exact text occurs in both train and test under different IDs:

```text
train: eng_237d0748658287b40f9915e7d6de18c6
test:  eng_87565962d029762a24f291b8c7ae5044
```

Keep this example flagged when reporting external test performance. It does not affect cross-validation conducted only on the training split.

## Preparing training folds

```bash
python scripts/create_folds.py \
  --config configs/create_folds.yaml \
  --output data/processed/english_train_folds.csv \
  --seed 2026
```

This maps the source columns to the canonical schema and adds the fixed outer-fold assignment.

