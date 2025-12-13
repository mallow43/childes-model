# Child Age Prediction from Spontaneous Speech

This project explores whether computational models can accurately predict a child’s age
group based solely on their spontaneous speech utterances. We focus on which linguistic
features—lexical diversity, morphological complexity, or utterance length—are most
predictive of developmental stage.

## Current Features

- Lexical features: word count, unique word count, character length, Type–Token Ratio/TTR (ratio of unique words to total number of words in utterance), first word/last word, function word counts and proportions, function word type count, content-to-function word ratio, common lexical phrases 
- Morphological features: MLU (the ratio of the number of morphemes to the number of words in the utterance), verb inflection indicators (-ing, -ed, third person singular/plural -s), possessive markings, proportion of nouns and verbs, presence of past tense
- Syntactic/n-gram features: bigrams and trigrams, syntactic cue markers (because, when, if, that, so), presence of plural nouns, presence of negation (not, -n’t), presence of question words (who, what, when, where, why, how), parts of speech, including POS bigrams and trigrams
- Other features tested: Unintelligibility (Check if it has unintelligible markers, unintelligibility count, unintelligibility proportion, unintelligible binning)


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

## Folder Layout 

Below is the expected structure for the project, specifically noting the structure of the `data` folder. Make sure to `mkdir` any folders necessary.

```
childes-model/
├── bin/                      # Contains the binary to train/evaluate the model
│  
├── data/
│   ├── raw/
│   │   ├── Brown/             # Raw CHILDES transcripts (Brown corpus)
│   │   └── MacWhinney/        # Raw CHILDES transcripts (MacWhinney corpus)
│   │
│   ├── processed/
│   │   ├── utterances.csv          # Extracted utterances with metadata
│   │   └── utterances_clean.csv    # Cleaned utterances after preprocessing
│   │
│   └── split/
│       ├── train.csv          # Training split
│       ├── dev.csv            # Developmentsplit
│       └── test.csv           # Test split
│
├── lib/                       # Contains the logistic regression model 
│
└── out/
    ├── ext.train              # Extracted features (train)
    ├── ext.dev                # Extracted features (dev)
    ├── ext.test               # Extracted features (test)

```

## Setup/Instructions 

1. Download corpora to `data/raw`:
  - Brown:(https://git.talkbank.org/childes/data/Eng-NA/Brown.zip)
  - MacWhinney: (https://git.talkbank.org/childes/data/Eng-NA/MacWhinney.zip)

2. Parse transcripts into a single CSV of child utterances (recursively scans `.cha` files):

   ```
   python parse_childes.py
   ```

   The output includes columns for `utterance`, `age_months`, `corpus`, `file`, `speaker`, and the relative `path` to the source file.

3. Clean the utterances and add word counts:

   ```
   python clean_data.py
   ```

4. Split the data into train/test split: 

   ```
   python split_data.py
   ```

5. Extract features from each of the split dataset, using extended features `-e` as well
   
   ```
   python extract_features.py -i data/split/(train/dev/test).csv -e > out/ext.(train/dev/test)
   ```

6. Train the logistic regression model on the 'train' dataset 

   ```
   bin/classify train out/ext.train out/childes.model > /dev/null
   ```

7. Evaluate the model on the development set (additionally, can use `-s n` flag to see some of the utterances, along with their gold and predicted labels)

   ```
   bin/classify apply out/childes.model out/ext.dev | ./score.py -g out/ext.dev
   ```

8. Once satisfied with feature set and hyperparameters, evaluate the model on the test set (do not forget the smoothing value)

   ```
   bin/classify apply out/childes.model out/ext.test | ./score.py -g out/ext.test
   ```
   
Note: if NLTK has trouble installing is POS tagger, we have included a script to install the requirements if you were to use POS as a feature. You can install these requirements using `python setup_nltk.py`.

## Models

Our modeling strategy is incremental:

- Baseline: Logistic Regression
  - We vary the feature sets (lexical, morphological, n-gram-based) to compare
    predictive power.
- Optional extensions (time permitting):
  - Simple neural network classifier
  - N-gram probabilistic language model to capture sentence grammaticality and
    structure

## Simplifying Assumptions

To keep the project tractable, we make the following assumptions:

- Treat each utterance as independent (ignore speaker-level variance).
- Use only clean, transcribed speech, removing any unnecessary or non-lingustic markers.
- Represent age in seven coarse bins (0yo, 1yo, 2yo, 3yo, 4yo, 5yo, 6yo_plus).
- Do not solely use word intelligibility as a feature to avoid interpreter-dependent
  variability.

## Evaluation

We evaluate our models along several dimensions:

- Metrics
  - Classification accuracy: % of utterances correctly assigned to the correct age
    group
  - Within-1-bin accuracy: The percentage of utterances for which the predicted age bin is within one bin of the gold age bin (either 6 or 12-month bin)
  - Macro-average recall: The average recall across all age bins: Of utterances in a certain bin, how many were correctly identified? 
  - Mean absolute error (bins): The average distance, in terms of bins, between the predicted and gold age bins
  - Mean absolute error (months): The average distance, in terms of months, between the predicted and gold age bins. For each bin, the number of months is calculated as the midpoint between the possible months that the bin represent (eg 18 months for 1 year—12 to 23 months) 

- Baselines
  - Random guessing (e.g., 7 classes → 14% accuracy)
  - Simple utterance-only classifier

- Analysis
  - Feature importance: which features drive correct predictions?
  - Neighboring-age variation: where do models confuse adjacent age groups, and are
    certain age boundaries less clear?

## Related Work

- Sagae K. (2021). Tracking Child Language Development With Neural Network Language Models. Frontiers in psychology, 12, 674402. https://doi.org/10.3389/fpsyg.2021.674402
- Alhama, R. G., Foushee, R., Byrne, D., Ettinger, A., Alishahi, A., & Goldin-Meadow, S. (2024). Using computational modeling to validate the onset of productive determiner-noun combinations in English-learning children. Proceedings of the National Academy of Sciences of the United States of America, 121(50), e2316527121.
- Alhama, R. G., Foushee, R., Byrne, D., Ettinger, A., Alishahi, A., & Goldin-Meadow, S. (2024). Using computational modeling to validate the onset of productive determiner-noun combinations in English-learning children. Proceedings of the National Academy of Sciences of the United States of America, 121(50), e2316527121.
- Novotný, Michal & Cmejla, Roman & Tykalová, Tereza. (2023). Automated prediction of children's age from voice acoustics. Biomedical Signal Processing and Control. 81. 104490. 10.1016/j.bspc.2022.104490. 

