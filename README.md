# Child Age Prediction from Spontaneous Speech

This project explores whether computational models can accurately predict a child’s age
group based solely on their spontaneous speech utterances. We focus on which linguistic
features—lexical diversity, morphological complexity, or utterance length—are most
predictive of developmental stage.

## Research Questions

- Can we classify a child’s age group from individual utterances?
- Does vocabulary diversity increase predictably with age?
- Are syntactic features (e.g., embedded clauses, question formation) more predictive
  than lexical features?

## Data

We use transcripts from the CHILDES corpus, focusing initially on the MacWhinney
subcorpus (longitudinal, English). Depending on progress, we may extend to the Brown and
Providence corpora to test cross-corpus generalization.

- Corpus archives: Brown (`https://childes.talkbank.org/data/Eng-NA/Brown.zip`),
  MacWhinney (`https://childes.talkbank.org/data/Eng-NA/MacWhinney.zip`),
  Providence (`https://childes.talkbank.org/data/Eng-NA/Providence.zip`)
- Primary focus: English child speech, spontaneous utterances

## Data Setup

1. Download corpora (archives cached in `data/downloads`, extracted to `data/raw`):

   ```
   python scripts/download_childes.py --corpora Brown MacWhinney
   ```

   Add `--force` to refresh an existing corpus or `--skip-download` to reuse cached zips.

2. Parse transcripts into a single CSV of child utterances (recursively scans `.cha` files):

   ```
   python parse_childes.py --data-dir data/raw --output data/processed/utterances.csv --corpora Brown MacWhinney
   ```

   The output includes columns for `utterance`, `age_months`, `corpus`, `file`, `speaker`, and the relative `path` to the source file.

3. (Optional) Clean the utterances and add word counts:

   ```
   python clean_data.py
   ```

## Models

Our modeling strategy is incremental:

- Baseline: Logistic Regression
  - We vary the feature sets (lexical, morphological, n-gram-based) to compare
    predictive power.
- Optional extensions (time permitting):
  - Simple neural network classifier
  - N-gram probabilistic language model to capture sentence grammaticality and
    structure

## Features

We experiment with several types of features:

- Lexical features
  - Type–Token Ratio (TTR) for vocabulary diversity
  - Proportion of different word classes (nouns, verbs, function words)
- Morphological features
  - Mean Length of Utterance (MLU) in words and morphemes
  - Use of verb inflections (-ing, -ed, -s)
  - Plural and possessive marking
- N-gram features
  - Bi-grams and tri-grams of POS tags
  - Common word combinations

These features are used individually and in combination within our models to see which
sets best predict age.

## Simplifying Assumptions

To keep the project tractable, we make the following assumptions:

- Treat each utterance as independent (ignore speaker-level variance).
- Use only clean, transcribed speech; unintelligible or heavily garbled segments are
  excluded.
- Represent age in discrete bins (e.g., 12 months, 2 years, 3 years).
- Do not use word intelligibility as a feature to avoid interpreter-dependent
  variability.

## Evaluation

We evaluate our models along several dimensions:

- Metrics
  - Classification accuracy: % of utterances correctly assigned to the correct age
    group
  - Mean Absolute Error (in months): average deviation from the true age
- Baselines
  - Random guessing (e.g., 4 classes → 25% accuracy)
  - Simple MLU-only classifier (morphemes per utterance)
- Analysis
  - Feature importance: which features drive correct predictions?
  - Neighboring-age variation: where do models confuse adjacent age groups, and are
    certain age boundaries less clear?

## Related Work

- Sagae, K. (2021). Tracking Child Language Development With Neural Network Language
  Models. Frontiers in Psychology, 12, 674402.
- Alhama, R. G., Foushee, R., Byrne, D., Ettinger, A., Alishahi, A., & Goldin-Meadow, S.
  (2024). Using computational modeling to validate the onset of productive determiner-
  noun combinations in English-learning children. PNAS, 121(50), e2316527121.
- Novotný, M., Cmejla, R., & Tykalová, T. (2023). Automated prediction of children's age
  from voice acoustics. Biomedical Signal Processing and Control, 81, 104490.
