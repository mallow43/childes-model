import os
import re
import pandas as pd

DATA_DIR = "data/raw"
OUTPUT_FILE = "data/processed/utterances.csv"


def age_to_months(age_str):
    """
    Convert CHILDES age format from Y;MM.DD → months.
    """
    try:
        years, rest = age_str.split(";")
        months = rest.split(".")[0]
        return int(years) * 12 + int(months)
    except:
        return None

def extract_utterances_from_file(filepath, corpus_name):
    """Extract all *CHI: utterances and age from one .cha file."""
    utterances = []
    age_months = None

    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    # 1. find age
    for line in lines:
        if line.startswith("@Age:"):
            age_str = line.replace("@Age:", "").strip()
            age_months = age_to_months(age_str)
            break

    # 2. extract child utterances
    for line in lines:
        if line.startswith("*CHI:"):
            utt = line.replace("*CHI:", "").strip()
            utterances.append((utt, age_months, corpus_name, os.path.basename(filepath)))

    return utterances


def extract_all():
    all_rows = []

    for corpus in ["Brown", "MacWhinney"]:
        folder = os.path.join(DATA_DIR, corpus)
        print(f"Processing: {folder}")

        for filename in os.listdir(folder):
            if filename.endswith(".cha"):
                filepath = os.path.join(folder, filename)
                rows = extract_utterances_from_file(filepath, corpus)
                all_rows.extend(rows)

    # dataframe
    df = pd.DataFrame(all_rows, columns=["utterance", "age_months", "corpus", "file"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} utterances → {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_all()
