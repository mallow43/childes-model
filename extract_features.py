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
# Load the cleaned CSV (note in pandas dataframe)
df = pd.read_csv(input_file)

# Feature extraction
# TODO HERE: ADD ALL NORMAL AND EXTENDED FEATURES 
for idx, row in df.iterrows():

    utter = row["clean_utterance"]
    age_m = row["age_months"]
    label = bucket_age(age_m)

    # Basic tokenization
    tokens = utter.split()
    lower_tokens = [t.lower() for t in tokens]

    features = []

    # Basic lexical features
    features.append("word_count=" + str(len(tokens)))
    features.append("unique_words=" + str(len(set(lower_tokens))))

    # First/last word
    if len(tokens) > 0:
        features.append("first_word=" + tokens[0].lower())
        features.append("last_word=" + tokens[-1].lower())

    # Character length
    features.append("char_len=" + str(len(utter)))

    # Presence of function words
    function_words = {"the","a","an","and","but","or","because","if","when","that"}
    for fw in function_words:
        if fw in lower_tokens:
            features.append("has_" + fw)

    # MLU (approx)
    # (This is proxy MLU for single utterance: words / 1)
    features.append("mlu=" + str(len(tokens)))

    # Extended features if requested
    if options.extended_features:

        # Bigrams
        for i in range(len(lower_tokens)-1):
            bigram = lower_tokens[i] + "_" + lower_tokens[i+1]
            features.append("bigram=" + bigram)

        # Trigrams
        for i in range(len(lower_tokens)-2):
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

    # Label at the end (must be last)
    features.append(label)

    # Output line
    output_file.write(",".join(features) + "\n")