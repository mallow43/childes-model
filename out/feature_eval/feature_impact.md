# Feature Impact Summary

## Methodology
- All configs trained with `bin/classify` on `data/split` using features from `extract_features.py -e`
- Each row averages 3 runs to smooth optimizer variance

## Key Findings
- Lexical/length baseline: ~46% accuracy
- Adding function words, morphology, intelligibility, word-class props: marginal gains (<0.5 pts)
- Adding extended syntax: **+5 pts → ~51% accuracy**
- **Bigrams are the key driver** within extended syntax (+4.9 pts alone)
- Trigrams provide secondary benefit (+3 pts alone)
- POS tags, POS n-grams, and markers contribute negligibly on their own

---

## Additive Builds (Coarse Groups)

- **lexical_only** (lexical_length)
  - Dev: 46.13 | Test: 46.47

- **lexical_function** (+ function_words)
  - Dev: 46.15 | Test: 46.32

- **+morphology** (+ morphology_inflection)
  - Dev: 46.14 | Test: 46.36

- **+intelligibility** (+ intelligibility)
  - Dev: 46.20 | Test: 46.31

- **baseline_no_extended** (+ word_class_props)
  - Dev: 46.21 | Test: 46.35

- **full_extended** (+ extended_syntax)
  - Dev: **51.15** | Test: **51.08**

---

## Leave-One-Out Ablations (Coarse Groups)

- **full_minus_extended** (removed: extended_syntax)
  - Dev: 46.21 | Test: 46.35 | Δ: **-4.73**

- **full_minus_lexical** (removed: lexical_length)
  - Dev: 47.00 | Test: 46.28 | Δ: **-4.80**

- **full_minus_function_words** (removed: function_words)
  - Dev: 51.56 | Test: 51.24 | Δ: +0.16

- **full_minus_morphology** (removed: morphology_inflection)
  - Dev: 51.34 | Test: 51.32 | Δ: +0.24

- **full_minus_intelligibility** (removed: intelligibility)
  - Dev: 51.07 | Test: 51.01 | Δ: -0.07

- **full_minus_word_class_props** (removed: word_class_props)
  - Dev: 51.08 | Test: 51.07 | Δ: -0.01

---

## Extended Syntax Sub-Group Analysis

### Adding Each Sub-Group to Baseline

- **baseline+bigrams** (word bigrams)
  - Dev: 51.45 | Test: 51.24 | Δ: **+4.89**

- **baseline+trigrams** (word trigrams)
  - Dev: 49.18 | Test: 49.32 | Δ: **+2.97**

- **baseline+pos** (POS unigrams)
  - Dev: 46.21 | Test: 46.35 | Δ: +0.00

- **baseline+pos_bigrams** (POS bigrams)
  - Dev: 46.21 | Test: 46.35 | Δ: +0.00

- **baseline+pos_trigrams** (POS trigrams)
  - Dev: 46.21 | Test: 46.35 | Δ: +0.00

- **baseline+markers** (syntactic markers)
  - Dev: 46.26 | Test: 46.33 | Δ: -0.02

- **full_detailed** (all sub-groups)
  - Dev: 51.15 | Test: 51.08

### Removing Each Sub-Group from Full

- **full_minus_bigrams** (removed: word bigrams)
  - Dev: 49.18 | Test: 49.23 | Δ: **-1.85**

- **full_minus_trigrams** (removed: word trigrams)
  - Dev: 51.32 | Test: 51.16 | Δ: +0.08

- **full_minus_pos** (removed: POS unigrams)
  - Dev: 51.15 | Test: 51.08 | Δ: +0.00

- **full_minus_pos_bigrams** (removed: POS bigrams)
  - Dev: 51.15 | Test: 51.08 | Δ: +0.00

- **full_minus_pos_trigrams** (removed: POS trigrams)
  - Dev: 51.15 | Test: 51.08 | Δ: +0.00

- **full_minus_markers** (removed: syntactic markers)
  - Dev: 51.06 | Test: 51.18 | Δ: +0.10

---

## Summary: Extended Syntax Feature Contributions

| Sub-Group       | Add to Baseline (Δ) | Remove from Full (Δ) | Verdict              |
|-----------------|:-------------------:|:--------------------:|----------------------|
| Word bigrams    | **+4.89**           | **-1.85**            | Primary driver       |
| Word trigrams   | **+2.97**           | +0.08                | Helps alone, redundant with bigrams |
| POS unigrams    | +0.00               | +0.00                | No impact            |
| POS bigrams     | +0.00               | +0.00                | No impact            |
| POS trigrams    | +0.00               | +0.00                | No impact            |
| Markers         | -0.02               | +0.10                | No impact            |

> Δ = change in test accuracy vs baseline (additive) or full model (ablation)
