"""TF-IDF feature preparation for Model 2A."""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from dataset_generation.model2.text_audit import build_combined_text, load_and_validate_profiles


@dataclass
class TextFeatureArtifacts:
    vectorizer: TfidfVectorizer
    fit_source: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame | None
    test_df: pd.DataFrame | None
    split_dir: str | None


def _load_split(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if os.path.exists(path) else None


def load_split_or_full(input_csv: str) -> TextFeatureArtifacts:
    base_dir = os.path.dirname(input_csv)
    split_dir = os.path.join(base_dir, "splits")
    train_path = os.path.join(split_dir, "train.csv")
    val_path = os.path.join(split_dir, "val.csv")
    test_path = os.path.join(split_dir, "test.csv")

    train = _load_split(train_path)
    val = _load_split(val_path)
    test = _load_split(test_path)
    if train is not None and val is not None and test is not None:
        return TextFeatureArtifacts(TfidfVectorizer(), "train_split", train, val, test, split_dir)

    df = load_and_validate_profiles(input_csv)
    print("WARNING: split files not found; fitting TF-IDF on the full dataset. This run is exploratory and not leakage-safe.")
    return TextFeatureArtifacts(TfidfVectorizer(), "full_dataset", df, None, None, None)


def fit_tfidf_vectorizer(train_df: pd.DataFrame, max_features: int, seed: int = 42, text_fields=None) -> tuple[TfidfVectorizer, any]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )
    corpus = build_combined_text(train_df, text_fields)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def transform_text(df: pd.DataFrame, vectorizer: TfidfVectorizer, text_fields=None):
    return vectorizer.transform(build_combined_text(df, text_fields))


def save_vectorizer(vectorizer: TfidfVectorizer, output_dir: str) -> str:
    path = os.path.join(output_dir, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, path)
    return path
