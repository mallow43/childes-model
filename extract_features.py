#!/usr/bin/env python3
# NOTE: TAKEN FROM PSET 2b (ppa_features.py), and modified to fit our features for CHILDES


#############################################################################
# Copyright 2011 Jason Baldridge
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#############################################################################

import sys
from classify_util import *
import pandas as pd
import re
import nltk 
import string
from feature_constants import FUNCTION_WORDS, CONTRACTION_MAP

## To run (pipeline):
# 1: python extract_features.py -i data/split/train.csv -e > out/ext.train (Extract all features from each set)
# 2: Train the logistic regression model: bin/classify train out/ext.train out/childes.model > /dev/null
# 3: Apply the model to dev/test set: bin/classify apply out/childes.model out/ext.dev | ./score.py -g out/ext.dev

#############################################################################
# Set up the options
parser = get_feature_extractor_option_parser()
(options, args) = parser.parse_args()
check_mandatory_options(parser, options, ['input'])

#############################################################################
# Use the options to set the input and output files appropriately
input_file = options.input
output_file = sys.stdout
if options.out != None:
    output_file = open(options.out, "w")

verbose = options.verbose
errmsg("Showing verbose output.", verbose)

#############################################################################
# Helper function: Convert months to age category
# NOTE: WE CAN CHANGE TO HOWEVER WE WANT HERE 
def bucket_age(age_months):
    """Convert continuous age in months into labels."""
    if age_months is None or not isinstance(age_months, int):
        return "UNK"
    if age_months < 24:
        return "1yo"
    elif age_months < 36:
        return "2yo"
    elif age_months < 48:
        return "3yo"
    elif age_months < 60:
        return "4yo"
    elif age_months < 72:
        return "5yo"
    else:
        return "6yo_plus"

#############################################################################
# Lightweight lexical/morphological helpers

# Closed-class items we care about for proportions and POS heuristics
PRONOUNS = {"i","you","he","she","we","they","it","me","him","her","us","them","my","your","his","her","our","their"}
COMMON_VERBS = {"go","goes","went","gone","see","saw","seen","get","got","gotten","have","has","had","want","wants","wanted","like","likes","liked","make","makes","made","say","says","said","think","thinks","thought","come","comes","came","take","takes","took","need","needs","needed"}
COMMON_NOUNS = {"dog","dogs","cat","cats","ball","balls","mom","mommy","dad","daddy","car","cars","toy","toys","book","books","house","houses","baby","babies","friend","friends","boy","boys","girl","girls"}
UNINTELLIGIBLE_MARKERS = {"xxx", "yyy"}
PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("'", ""))

def is_verb(tok):
    if tok in COMMON_VERBS:
        return True
    return tok.endswith(("ing","ed")) or tok.endswith("s") and tok[:-1] in COMMON_VERBS

def is_noun(tok):
    if tok in COMMON_NOUNS:
        return True
    # Plural/possessive cues
    return tok.endswith(("s","'s"))

def morpheme_count(tok):
    t = re.sub(r"^[^A-Za-z']+|[^A-Za-z']+$", "", tok.lower())
    if not t:
        return 0
    count = 1
    if t.endswith("'s"):
        count += 1
    elif t.endswith("s") and len(t) > 3:
        count += 1
    if t.endswith("ing") and len(t) > 4:
        count += 1
    elif t.endswith("ed") and len(t) > 3:
        count += 1
    return count

# POS tagging via NLTK
def get_pos_tags(tokens):
    if tokens:
        return [tag for _, tag in nltk.tag.pos_tag(tokens, tagset="universal")]
    return []

def normalize_token(tok):
    """
    Normalize tokens:
    - lowercase
    - strip surrounding punctuation while preserving internal apostrophes
    """
    cleaned = tok.lower()
    # Strip leading/trailing punctuation but keep internal apostrophes (can't -> can't)
    cleaned = re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", cleaned)
    cleaned = cleaned.translate(PUNCT_TABLE)
    cleaned = cleaned.strip("'")
    return cleaned

def bin_unintelligible(prop, count):
    """
    Bucket unintelligible proportion into coarse bins
    """
    if count == 0:
        return "none"
    if prop < 0.1:
        return "low"
    if prop < 0.5:
        return "mid"
    return "high"

def is_alpha_token(tok):
    return bool(re.fullmatch(r"[a-z]+", tok))

def expand_for_function_words(tok):
    """
    Return the token itself or expanded pieces for contractions
    so function-word counting recognizes hidden auxiliaries/negation.
    """
    return CONTRACTION_MAP.get(tok, [tok])

#############################################################################
# Load the cleaned CSV (note in pandas dataframe)
df = pd.read_csv(input_file)

# Feature extraction
# TODO HERE: ADD ALL NORMAL AND EXTENDED FEATURES 
for idx, row in df.iterrows():

    utter = row["clean_utterance"]
    age_m = row["age_months"]
    label = bucket_age(age_m)

    # Basic tokenization
    raw_tokens = utter.split()
    # Normalize tokens (lowercase, strip punctuation)
    tokens = []
    for tok in raw_tokens:
        norm = normalize_token(tok)
        if norm:
            tokens.append(norm)
    lower_tokens = [t.lower() for t in tokens]
    num_tokens = len(tokens)

    features = []

    # Basic lexical features
    features.append("word_count=" + str(num_tokens))
    features.append("unique_words=" + str(len(set(lower_tokens))))
    if num_tokens > 0:
        ttr = len(set(lower_tokens)) / num_tokens
        features.append(f"ttr={ttr:.3f}")
    else:
        features.append("ttr=0.000")

    # First/last word
    if num_tokens > 0:
        features.append("first_word=" + tokens[0].lower())
        features.append("last_word=" + tokens[-1].lower())

    # Character length
    features.append("char_len=" + str(len(utter)))

    # Function word aggregates (counts + ratios; contraction-aware)
    function_word_hits = []
    expanded_len = 0
    for tok in lower_tokens:
        expanded = expand_for_function_words(tok)
        expanded_len += len(expanded)
        for piece in expanded:
            if piece in FUNCTION_WORDS:
                function_word_hits.append(piece)
    func_count = len(function_word_hits)
    func_types = len(set(function_word_hits))
    func_prop = func_count / expanded_len if expanded_len > 0 else 0.0
    content_tokens = max(expanded_len - func_count, 0)
    features.append("function_word_count=" + str(func_count))
    features.append(f"function_word_prop={func_prop:.3f}")
    features.append("function_word_types=" + str(func_types))
    if func_count > 0:
        features.append(f"content_to_function_ratio={content_tokens/func_count:.3f}")
    else:
        features.append("content_to_function_ratio=0.000")

    # MLU approximations (words and morphemes)
    morph_count = sum(morpheme_count(t) for t in tokens)
    features.append("morpheme_count=" + str(morph_count))
    if num_tokens > 0:
        features.append(f"mlu_morphemes={morph_count/num_tokens:.3f}")
    else:
        features.append("mlu_morphemes=0.000")

    # Intelligibility markers (xxx/yyy kept from cleaning step)
    unintelligible_count = sum(1 for t in lower_tokens if t in UNINTELLIGIBLE_MARKERS)
    features.append("unintelligible_count=" + str(unintelligible_count))
    if num_tokens > 0:
        features.append(f"unintelligible_prop={unintelligible_count/num_tokens:.3f}")
    else:
        features.append("unintelligible_prop=0.000")
    if unintelligible_count > 0:
        features.append("has_unintelligible")
    # Binned version to reduce variance
    prop_val = unintelligible_count/num_tokens if num_tokens > 0 else 0.0
    features.append("unintelligible_bin=" + bin_unintelligible(prop_val, unintelligible_count))

    # Inflectional cues
    if any(t.endswith("ing") for t in lower_tokens):
        features.append("has_ing")
    if any(t.endswith("ed") for t in lower_tokens):
        features.append("has_ed")
    if any(t.endswith("s") and len(t) > 1 for t in lower_tokens):
        features.append("has_3sg_or_plural")
    if any(t.endswith("'s") for t in lower_tokens):
        features.append("has_possessive")

    # Word class proportions (heuristic)
    noun_count = sum(1 for t in lower_tokens if is_noun(t))
    verb_count = sum(1 for t in lower_tokens if is_verb(t))
    if num_tokens > 0:
        features.append(f"prop_nouns={noun_count/num_tokens:.3f}")
        features.append(f"prop_verbs={verb_count/num_tokens:.3f}")
    else:
        features.append("prop_nouns=0.000")
        features.append("prop_verbs=0.000")

    # Extended features if requested
    if options.extended_features:
        pos_tags = get_pos_tags(tokens)

        # Bigrams
        for i in range(num_tokens-1):
            bigram = lower_tokens[i] + "_" + lower_tokens[i+1]
            features.append("bigram=" + bigram)

        # Trigrams
        for i in range(num_tokens-2):
            trigram = lower_tokens[i] + "_" + lower_tokens[i+1] + "_" + lower_tokens[i+2]
            features.append("trigram=" + trigram)

        # Syntactic cue words
        syntactic_markers = ["because", "when", "that", "if", "so"]
        for sm in syntactic_markers:
            if sm in lower_tokens:
                features.append("has_marker_" + sm)

        # Presence of plural nouns (proxy for morphological development)
        for t in lower_tokens:
            if t.endswith("s") and len(t) > 1:
                features.append("has_plural")

        # Check for negation (older children = more negation forms)
        if "not" in lower_tokens or "don't" in lower_tokens:
            features.append("has_negation")

        # POS tags and n-grams (NLTK); if tagging fails, skip POS features
        for tag in pos_tags:
            features.append("pos=" + tag)
        for i in range(len(pos_tags)-1):
            features.append("pos_bigram=" + pos_tags[i] + "_" + pos_tags[i+1])
        for i in range(len(pos_tags)-2):
            features.append("pos_trigram=" + pos_tags[i] + "_" + pos_tags[i+1] + "_" + pos_tags[i+2])

    # Label at the end (must be last)
    features.append(label)

    # Output line
    output_file.write(",".join(features) + "\n")
