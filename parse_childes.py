import os
import re
import pandas as pd

DATA_DIR = "data/raw"
OUTPUT_FILE = "data/processed/utterances.csv"


def iter_cha_files(root_dir):
    """Yield full paths to .cha files under root_dir (recursively)."""
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".cha"):
                yield os.path.join(dirpath, fname)


def age_to_months(age_str):
    """
    Convert CHILDES age format from Y;MM.DD to months
    """
    # Here use try/except in case any of these lines fail due to bad parsing
    try:
        years, rest = age_str.split(";")
        months = rest.split(".")[0]
        return int(years) * 12 + int(months)
    except:
        return None


def extract_speaker_ages(lines):
    """
    Build a dictionary mapping speaker codes to age_months
    by reading @ID lines.
    """
    speaker_age = {}

    for raw in lines:
        line = raw.strip().replace("\ufeff", "")

        if line.startswith("@ID:"):
            parts = line.split("|")
            
            # Skip all IDs that are not parsed correctly or are not formatted correctly
            if len(parts) < 5:
                continue

            # Parse components
            speaker = parts[2].strip() # "CHI", "MAR"
            age_str = parts[3].strip() # "5;06.24", etc
            role = parts[7].strip() if len(parts) > 7 else ""

            # Only assign ages for target children
            if "Target_Child" in role and ";" in age_str:
                speaker_age[speaker] = age_to_months(age_str)

    return speaker_age


def extract_utterances_from_file(filepath, corpus_name):
    """Extract child utterances and ages from one .cha file."""
    utterances = []

    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    # 1. Map speaker to age
    speaker_age = extract_speaker_ages(lines)

    # 2. Extract utterances for all child speakers found
    for raw in lines:
        if raw.startswith("*"):
            line = raw.strip()
            match = re.match(r"\*(\w+):\s*(.*)", line)

            if match:
                speaker = match.group(1)
                utt = match.group(2).strip()

                # Only include target children with known age
                if speaker in speaker_age:
                    age_months = speaker_age[speaker]
                    utterances.append((utt, age_months, corpus_name, os.path.basename(filepath)))

    return utterances


def extract_all():
    all_rows = []

    for corpus in ["Brown", "MacWhinney"]: # This is assuming we are only using Brown and MacWhinney
        folder = os.path.join(DATA_DIR, corpus)
        if not os.path.isdir(folder):
            continue

        print(f"Processing: {folder}")
        file_count = 0

        for filepath in iter_cha_files(folder):
            file_count += 1
            rows = extract_utterances_from_file(filepath, corpus)
            all_rows.extend(rows)

        print(f"Scanned {file_count} .cha files, collected {len(all_rows)} rows so far.")

    df = pd.DataFrame(all_rows, columns=["utterance", "age_months", "corpus", "file"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} utterances to {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_all()
