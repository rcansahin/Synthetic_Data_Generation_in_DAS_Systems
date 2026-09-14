from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ClassInfo:
    """Container for dataset specific class metadata."""

    classes: list[str]
    cls2id: dict[str, int]
    id2cls: dict[int, str]

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def _normalize_folder(data_folder: str | Path) -> Path:
    folder = Path(data_folder).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")
    return folder


def _load_unique_events(train_csv: Path) -> Iterable[str]:
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found under: {train_csv.parent}")

    df = pd.read_csv(train_csv, usecols=["event"])
    events = df["event"].dropna().astype(str).unique().tolist()
    return sorted(events)


@lru_cache(maxsize=None)
def load_class_info(data_folder: str | Path) -> ClassInfo:
    """
    Read <data_folder>/train.csv and derive class metadata.
    The mapping order follows pandas/LabelEncoder behavior (sorted alphabetically).
    """
    folder = _normalize_folder(data_folder)
    classes = _load_unique_events(folder / "train.csv")
    cls2id = {c: i for i, c in enumerate(classes)}
    id2cls = {i: c for c, i in cls2id.items()}
    return ClassInfo(classes=classes, cls2id=cls2id, id2cls=id2cls)


def get_class_names(data_folder: str | Path) -> list[str]:
    return load_class_info(data_folder).classes


def get_num_classes(data_folder: str | Path) -> int:
    return load_class_info(data_folder).n_classes


def get_class_to_id_map(data_folder: str | Path) -> dict[str, int]:
    return load_class_info(data_folder).cls2id


def get_id_to_class_map(data_folder: str | Path) -> dict[int, str]:
    return load_class_info(data_folder).id2cls
