from pathlib import Path

import pandas as pd

from models.label import Labels,Label

CSV_PATH = Path(__file__).resolve().parents[1] / "languages.csv"


def read_language_labels() -> pd.DataFrame:
    records = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        header = next(f, None)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",", 3)
            if len(parts) != 4:
                raise ValueError(f"Malformed languages.csv row: {line}")
            records.append({
                "language": parts[0],
                "component": parts[1],
                "label": parts[2],
                "text": parts[3],
            })

    return pd.DataFrame.from_records(records, columns=["language", "component", "label", "text"])


def get_labels_for_component(components: list[str], language: str):
    df = read_language_labels()
    filtered = df[(df["language"] == language) & (df["component"].isin(components))]
    if filtered.empty:
        return None

    labels = [
        Label(
            language=row["language"],
            component=row["component"],
            label=row["label"],
            text=row["text"],
        )
        for _, row in filtered.iterrows()
    ]

    return Labels(labels=labels)