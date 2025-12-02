#!/usr/bin/env python3
"""
Run additive and ablation studies to measure how each feature group affects
classification accuracy. Generates a superset of features (with -e), filters
them by group, trains/evaluates with bin/classify, and writes a summary table.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple
from feature_constants import FUNCTION_WORDS

# Paths
BASE_DIR = Path(__file__).parent
DATA_SPLIT_DIR = BASE_DIR / "data" / "split"
EXTRACT_SCRIPT = BASE_DIR / "extract_features.py"
CLASSIFY_BIN = BASE_DIR / "bin" / "classify"
SCORE_SCRIPT = BASE_DIR / "score.py"

# Output locations
FEATURE_EVAL_DIR = BASE_DIR / "out" / "feature_eval"
FULL_FEATURE_DIR = FEATURE_EVAL_DIR / "full"
FILTERED_DIR = FEATURE_EVAL_DIR / "filtered"
MODEL_DIR = FEATURE_EVAL_DIR / "models"
RESULTS_PATH = FEATURE_EVAL_DIR / "feature_impact.tsv"

SPLITS: Sequence[str] = ("train", "dev", "test")
RUNS_PER_CONFIG = 1  # set to 3 for final results, 1 for quick iteration

GROUP_LEXICAL = "lexical_length"
GROUP_FUNCTION = "function_words"
GROUP_MORPH = "morphology_inflection"
GROUP_INTEL = "intelligibility"
GROUP_CLASS_PROP = "word_class_props"
GROUP_EXTENDED = "extended_syntax"

# Extended syntax sub-groups for fine-grained analysis
GROUP_EXT_BIGRAMS = "ext_bigrams"
GROUP_EXT_TRIGRAMS = "ext_trigrams"
GROUP_EXT_POS = "ext_pos"
GROUP_EXT_POS_BIGRAMS = "ext_pos_bigrams"
GROUP_EXT_POS_TRIGRAMS = "ext_pos_trigrams"
GROUP_EXT_MARKERS = "ext_markers"

EXTENDED_SUBGROUPS: Set[str] = {
    GROUP_EXT_BIGRAMS,
    GROUP_EXT_TRIGRAMS,
    GROUP_EXT_POS,
    GROUP_EXT_POS_BIGRAMS,
    GROUP_EXT_POS_TRIGRAMS,
    GROUP_EXT_MARKERS,
}

ALL_GROUPS: Set[str] = {
    GROUP_LEXICAL,
    GROUP_FUNCTION,
    GROUP_MORPH,
    GROUP_INTEL,
    GROUP_CLASS_PROP,
    GROUP_EXTENDED,
}

# All groups with extended broken into sub-groups
ALL_GROUPS_DETAILED: Set[str] = {
    GROUP_LEXICAL,
    GROUP_FUNCTION,
    GROUP_MORPH,
    GROUP_INTEL,
    GROUP_CLASS_PROP,
} | EXTENDED_SUBGROUPS

# Additive path: build up from lexical to full extended model
ADDITIVE_CONFIGS: List[Tuple[str, Set[str]]] = [
    ("lexical_only", {GROUP_LEXICAL}),
    ("lexical_function", {GROUP_LEXICAL, GROUP_FUNCTION}),
    ("+morphology", {GROUP_LEXICAL, GROUP_FUNCTION, GROUP_MORPH}),
    ("+intelligibility", {GROUP_LEXICAL, GROUP_FUNCTION, GROUP_MORPH, GROUP_INTEL}),
    (
        "baseline_no_extended",
        {GROUP_LEXICAL, GROUP_FUNCTION, GROUP_MORPH, GROUP_INTEL, GROUP_CLASS_PROP},
    ),
    ("full_extended", ALL_GROUPS),
]

# Leave-one-out ablations from the full model
ABLATION_CONFIGS: List[Tuple[str, Set[str]]] = [
    ("full_minus_lexical", ALL_GROUPS - {GROUP_LEXICAL}),
    ("full_minus_function_words", ALL_GROUPS - {GROUP_FUNCTION}),
    ("full_minus_morphology", ALL_GROUPS - {GROUP_MORPH}),
    ("full_minus_intelligibility", ALL_GROUPS - {GROUP_INTEL}),
    ("full_minus_word_class_props", ALL_GROUPS - {GROUP_CLASS_PROP}),
    ("full_minus_extended", ALL_GROUPS - {GROUP_EXTENDED}),
]

# Baseline groups (everything except extended syntax)
BASELINE_GROUPS: Set[str] = {GROUP_LEXICAL, GROUP_FUNCTION, GROUP_MORPH, GROUP_INTEL, GROUP_CLASS_PROP}

# Extended syntax sub-group analysis: add each sub-group to baseline
EXTENDED_ADDITIVE_CONFIGS: List[Tuple[str, Set[str]]] = [
    ("baseline+bigrams", BASELINE_GROUPS | {GROUP_EXT_BIGRAMS}),
    ("baseline+trigrams", BASELINE_GROUPS | {GROUP_EXT_TRIGRAMS}),
    ("baseline+pos", BASELINE_GROUPS | {GROUP_EXT_POS}),
    ("baseline+pos_bigrams", BASELINE_GROUPS | {GROUP_EXT_POS_BIGRAMS}),
    ("baseline+pos_trigrams", BASELINE_GROUPS | {GROUP_EXT_POS_TRIGRAMS}),
    ("baseline+markers", BASELINE_GROUPS | {GROUP_EXT_MARKERS}),
    ("full_detailed", ALL_GROUPS_DETAILED),
]

# Extended syntax sub-group analysis: remove each sub-group from full
EXTENDED_ABLATION_CONFIGS: List[Tuple[str, Set[str]]] = [
    ("full_minus_bigrams", ALL_GROUPS_DETAILED - {GROUP_EXT_BIGRAMS}),
    ("full_minus_trigrams", ALL_GROUPS_DETAILED - {GROUP_EXT_TRIGRAMS}),
    ("full_minus_pos", ALL_GROUPS_DETAILED - {GROUP_EXT_POS}),
    ("full_minus_pos_bigrams", ALL_GROUPS_DETAILED - {GROUP_EXT_POS_BIGRAMS}),
    ("full_minus_pos_trigrams", ALL_GROUPS_DETAILED - {GROUP_EXT_POS_TRIGRAMS}),
    ("full_minus_markers", ALL_GROUPS_DETAILED - {GROUP_EXT_MARKERS}),
]


def run_cmd(args: Sequence[str], desc: str) -> subprocess.CompletedProcess:
    """Execute a command and raise a helpful error on failure."""
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed ({result.returncode}): {result.stderr}")
    return result


def build_full_features() -> Dict[str, Path]:
    """Extract the superset of features (with -e) for each split."""
    FULL_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for split in SPLITS:
        out_path = FULL_FEATURE_DIR / f"{split}.events"
        cmd = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "-i",
            str(DATA_SPLIT_DIR / f"{split}.csv"),
            "-e",
            "-o",
            str(out_path),
        ]
        run_cmd(cmd, f"feature extraction for {split}")
        paths[split] = out_path
    return paths


def group_for_feature(feat: str, detailed: bool = False) -> str | None:
    """Map a raw feature string to its group name.

    If detailed=True, returns fine-grained extended syntax sub-groups.
    If detailed=False, returns coarse GROUP_EXTENDED for all extended syntax features.
    """
    if feat.startswith(("word_count=", "unique_words=", "ttr=", "first_word=", "last_word=", "char_len=")):
        return GROUP_LEXICAL
    if feat.startswith(
        (
            "function_word_count=",
            "function_word_prop=",
            "function_word_types=",
            "content_to_function_ratio=",
        )
    ):
        return GROUP_FUNCTION
    if feat.startswith(("mlu_words=", "morpheme_count=", "mlu_morphemes=")) or feat in {
        "has_ing",
        "has_ed",
        "has_3sg_or_plural",
        "has_possessive",
    }:
        return GROUP_MORPH
    if feat.startswith(
        ("unintelligible_count=", "unintelligible_prop=", "unintelligible_bin=")
    ) or feat == "has_unintelligible":
        return GROUP_INTEL
    if feat.startswith(("prop_nouns=", "prop_verbs=", "prop_function_words=")):
        return GROUP_CLASS_PROP

    # Extended syntax features - return sub-group if detailed, else coarse group
    if feat.startswith("bigram="):
        return GROUP_EXT_BIGRAMS if detailed else GROUP_EXTENDED
    if feat.startswith("trigram="):
        return GROUP_EXT_TRIGRAMS if detailed else GROUP_EXTENDED
    if feat.startswith("pos="):
        return GROUP_EXT_POS if detailed else GROUP_EXTENDED
    if feat.startswith("pos_bigram="):
        return GROUP_EXT_POS_BIGRAMS if detailed else GROUP_EXTENDED
    if feat.startswith("pos_trigram="):
        return GROUP_EXT_POS_TRIGRAMS if detailed else GROUP_EXTENDED
    if feat.startswith("has_marker_") or feat in {"has_plural", "has_negation"}:
        return GROUP_EXT_MARKERS if detailed else GROUP_EXTENDED

    return None


def assert_no_unknown_features(full_paths: Iterable[Path]) -> None:
    """Guard against silently dropping new/renamed features."""
    unknown: Set[str] = set()
    for path in full_paths:
        with path.open() as fh:
            for line in fh:
                feats = line.strip().split(",")[:-1]
                for feat in feats:
                    # Check both coarse and detailed mapping
                    if group_for_feature(feat, detailed=False) is None:
                        unknown.add(feat.split("=")[0])
    if unknown:
        raise RuntimeError(f"Unmapped features found: {sorted(unknown)}")


def filter_features(full_path: Path, dest_path: Path, allowed_groups: Set[str]) -> None:
    """Write a filtered feature file keeping only the requested groups."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # Use detailed mapping if any extended sub-group is in allowed_groups
    use_detailed = bool(allowed_groups & EXTENDED_SUBGROUPS)
    with full_path.open() as src, dest_path.open("w") as dst:
        for line in src:
            parts = line.strip().split(",")
            feats, label = parts[:-1], parts[-1]
            kept = [f for f in feats if group_for_feature(f, detailed=use_detailed) in allowed_groups]
            dst.write(",".join(kept + [label]) + "\n")


def score_model(model_path: Path, eval_path: Path) -> Dict[str, float]:
    """Apply a trained model and return all metrics parsed from score.py output."""
    apply_res = run_cmd(
        [str(CLASSIFY_BIN), "apply", str(model_path), str(eval_path)],
        f"model application for {eval_path.name}",
    )
    score_res = subprocess.run(
        [sys.executable, str(SCORE_SCRIPT), "-g", str(eval_path)],
        input=apply_res.stdout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if score_res.returncode != 0:
        raise RuntimeError(f"Scoring failed for {eval_path}: {score_res.stderr}")

    # Parse all metrics from score.py output
    metrics: Dict[str, float] = {}

    acc_match = re.search(r"Exact Accuracy:\s+([0-9.]+)", score_res.stdout)
    if not acc_match:
        raise RuntimeError(f"Could not parse accuracy from: {score_res.stdout}")
    metrics["accuracy"] = float(acc_match.group(1))

    within1_match = re.search(r"Within-1-Bin Acc:\s+([0-9.]+)", score_res.stdout)
    if within1_match:
        metrics["within_1_acc"] = float(within1_match.group(1))

    mae_bins_match = re.search(r"MAE \(bins\):\s+([0-9.]+)", score_res.stdout)
    if mae_bins_match:
        metrics["mae_bins"] = float(mae_bins_match.group(1))

    mae_months_match = re.search(r"MAE \(months\):\s+([0-9.]+)", score_res.stdout)
    if mae_months_match:
        metrics["mae_months"] = float(mae_months_match.group(1))

    return metrics


def train_and_eval(config_name: str, allowed_groups: Set[str], full_paths: Dict[str, Path]) -> Dict[str, float]:
    """Filter features, train RUNS_PER_CONFIG models, and return averaged dev metrics."""
    filtered_paths = {
        split: FILTERED_DIR / f"{config_name}.{split}" for split in SPLITS
    }
    for split in SPLITS:
        filter_features(full_paths[split], filtered_paths[split], allowed_groups)

    dev_metrics_list: List[Dict[str, float]] = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for run_idx in range(RUNS_PER_CONFIG):
        model_path = MODEL_DIR / f"{config_name}.model"
        train_desc = f"training {config_name} (run {run_idx + 1}/{RUNS_PER_CONFIG})"
        run_cmd([str(CLASSIFY_BIN), "train", str(filtered_paths["train"]), str(model_path)], train_desc)
        dev_metrics_list.append(score_model(model_path, filtered_paths["dev"]))
        model_path.unlink(missing_ok=True)

    # Average all metrics across runs
    def avg_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}

    return avg_metrics(dev_metrics_list)


def write_results(rows: List[Dict[str, str]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([
            "config", "type", "groups_included", "runs",
            "accuracy", "within_1_acc", "mae_bins", "mae_months"
        ])
        for row in rows:
            writer.writerow(
                [
                    row["config"],
                    row["type"],
                    row["groups"],
                    row["runs"],
                    f'{float(row["accuracy"]):.2f}',
                    f'{float(row["within_1_acc"]):.2f}',
                    f'{float(row["mae_bins"]):.3f}',
                    f'{float(row["mae_months"]):.2f}',
                ]
            )


def main() -> None:
    FEATURE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    full_paths = build_full_features()
    assert_no_unknown_features(full_paths.values())

    experiments: List[Tuple[str, str, Set[str]]] = []
    experiments += [("additive", name, groups) for name, groups in ADDITIVE_CONFIGS]
    experiments += [("ablation", name, groups) for name, groups in ABLATION_CONFIGS]
    # Extended syntax sub-group experiments
    experiments += [("ext_additive", name, groups) for name, groups in EXTENDED_ADDITIVE_CONFIGS]
    experiments += [("ext_ablation", name, groups) for name, groups in EXTENDED_ABLATION_CONFIGS]

    results: List[Dict[str, str]] = []
    for exp_type, name, groups in experiments:
        metrics = train_and_eval(name, groups, full_paths)
        results.append(
            {
                "config": name,
                "type": exp_type,
                "groups": ",".join(sorted(groups)),
                "runs": str(RUNS_PER_CONFIG),
                "accuracy": metrics["accuracy"],
                "within_1_acc": metrics.get("within_1_acc", 0),
                "mae_bins": metrics.get("mae_bins", 0),
                "mae_months": metrics.get("mae_months", 0),
            }
        )

    write_results(results)
    print(f"Wrote {len(results)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
