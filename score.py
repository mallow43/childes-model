#!/usr/bin/env python3

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

from __future__ import division
import sys
from optparse import OptionParser
from classify_util import *
from collections import defaultdict

#############################################################################
# Age bin definitions - must match extract_features.py

# Ordered list of age bins (for ordinal distance calculation)
AGE_BIN_ORDER = ["0yo", "1yo", "2yo", "3yo", "4yo", "5yo", "6yo_plus"]

# Map each bin to its ordinal index (for calculating bin distance)
AGE_BIN_TO_INDEX = {label: i for i, label in enumerate(AGE_BIN_ORDER)}

# Map each bin to its midpoint in months (for MAE in months)
AGE_BIN_TO_MONTHS = {
    "0yo": 6,        # Midpoint of 0-11 months
    "1yo": 18,       # Midpoint of 12-23 months
    "2yo": 30,       # Midpoint of 24-35 months
    "3yo": 42,       # Midpoint of 36-47 months
    "4yo": 54,       # Midpoint of 48-59 months
    "5yo": 66,       # Midpoint of 60-71 months
    "6yo_plus": 78,  # Estimate for 72+ months
}

#############################################################################
# Set up the options

parser = OptionParser()
parser.add_option("-g", "--gold", dest="gold",
                  help="use gold labels in FILE", metavar="FILE")
parser.add_option("-p", "--predict", dest="predict",
                  help="score predicted labels in FILE", metavar="FILE")
parser.add_option("-s", "--show-examples", dest="show_examples", type="int", default=0,
                  help="show examples of predictions versus gold labels (default 10 utterances)")

(options, args) = parser.parse_args()
check_mandatory_options(parser, options, ['gold'])

#############################################################################
# Use the options to set the input and output files appropriately

gold_file = open(options.gold)
prediction_file = sys.stdin
if options.predict != None:
    prediction_file = file(options.predict)

#############################################################################
# Do the scoring.

# Slurp in the labels from each file as lists
#gold = [x.strip().split(',')[-1] for x in gold_file.readlines()]
gold_lines = [x.strip() for x in gold_file.readlines()]
gold = [line.split(',')[-1] for line in gold_lines]
predicted = [x.strip().split(' ')[0] for x in prediction_file.readlines()]

if len(gold) != len(predicted):
    print("ERROR: Different number of gold and predicted labels!")
    print("\tNum gold labels:", len(gold))
    print("\tNum predicted labels:", len(predicted))
    print("Exiting.")
    sys.exit

#############################################################################
# Calculate metrics

n = len(gold)

# 1. Exact accuracy
num_correct = sum([x[0]==x[1] for x in zip(gold, predicted)])
accuracy = num_correct / n * 100.0

# 2. Within-1-bin accuracy (correct if prediction is at most 1 bin away)
within_1_correct = 0
for g, p in zip(gold, predicted):
    g_idx = AGE_BIN_TO_INDEX.get(g, -1)
    p_idx = AGE_BIN_TO_INDEX.get(p, -1)
    if g_idx >= 0 and p_idx >= 0 and abs(g_idx - p_idx) <= 1:
        within_1_correct += 1
within_1_accuracy = within_1_correct / n * 100.0

# 3. Macro-averaged recall

gold_counts = defaultdict(int)
true_positive_counts = defaultdict(int)

for g, p in zip(gold, predicted):
    gold_counts[g] += 1
    if g == p:
        true_positive_counts[g] += 1

macro_recall_sum = 0
num_classes = 0

for label in gold_counts:
    if gold_counts[label] > 0:
        recall = true_positive_counts[label] / gold_counts[label]
        macro_recall_sum += recall
        num_classes += 1

macro_recall = (macro_recall_sum / num_classes) * 100 if num_classes > 0 else 0

# 4. Mean Absolute Error in bins (ordinal distance)
total_bin_error = 0
for g, p in zip(gold, predicted):
    g_idx = AGE_BIN_TO_INDEX.get(g, -1)
    p_idx = AGE_BIN_TO_INDEX.get(p, -1)
    if g_idx >= 0 and p_idx >= 0:
        total_bin_error += abs(g_idx - p_idx)
mae_bins = total_bin_error / n

# 5. Mean Absolute Error in months (using bin midpoints)
total_month_error = 0
for g, p in zip(gold, predicted):
    g_months = AGE_BIN_TO_MONTHS.get(g, 0)
    p_months = AGE_BIN_TO_MONTHS.get(p, 0)
    total_month_error += abs(g_months - p_months)
mae_months = total_month_error / n

#############################################################################
# Print results

print("=" * 50)
print("EVALUATION METRICS")
print("=" * 50)
print(f"Exact Accuracy:       {accuracy:.2f}%")
print(f"Within-1-Bin Acc:     {within_1_accuracy:.2f}%")
print(f"Macro Recall:         {macro_recall:.2f}%")
print(f"MAE (bins):           {mae_bins:.3f}")
print(f"MAE (months):         {mae_months:.2f}")
print("=" * 50)

# Show sample predictions if flag pressed
if options.show_examples:
    print("\nSAMPLE PREDICTIONS (gold -> predicted):")
    print("=" * 50)

    shown = 0
    max_show = options.show_examples

    for line, g, p in zip(gold_lines, gold, predicted):
        #if g != p: ADD THIS IF ONLY WANT TO LOOK AT ERRORS
        parts = line.split(',')
        # Create iterator that will turn feature back into the clean utterance
        utt = next((x.replace("utter=", "").replace("<COMMA>", ",") for x in parts if x.startswith("utter=")),None)
        # Only show valid utterances
        if utt:
            context = f'"{utt}"'
            print(f"{context:40s}  GOLD={g:6s}  PRED={p:6s}")
            shown += 1
        if shown >= max_show:
            break
