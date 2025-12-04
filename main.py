"""
Network Intrusion Detection using Random Forest.

This module provides functionality for loading network intrusion data,
training a Random Forest classifier, and calculating Precision/Recall
metrics to handle unbalanced classes.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load network intrusion data from a CSV file.

    Args:
        filepath: Path to the CSV file containing network intrusion data.

    Returns:
        DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be parsed as CSV.
    """
    logger.info(f"Loading data from {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}") from e


def preprocess_data(
    df: pd.DataFrame, target_column: str = "label"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the data for training.

    Args:
        df: Raw DataFrame containing network intrusion data.
        target_column: Name of the column containing the target labels.

    Returns:
        Tuple of (features DataFrame, target Series).

    Raises:
        ValueError: If target column is not found in DataFrame.
    """
    logger.info("Preprocessing data")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical target if needed
    if y.dtype == "object":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
        logger.info(f"Encoded {len(le.classes_)} target classes")

    # Handle categorical features
    categorical_cols = X.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Scale numerical features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    logger.info(f"Preprocessed {len(X)} samples with {len(X.columns)} features")
    return X, y


def handle_imbalance(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using SMOTE oversampling.

    Args:
        X: Feature DataFrame.
        y: Target Series.

    Returns:
        Tuple of (resampled features, resampled targets).
    """
    logger.info("Handling class imbalance with SMOTE")
    logger.info(f"Original class distribution: {dict(y.value_counts())}")

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled)

    logger.info(f"Resampled class distribution: {dict(y_resampled.value_counts())}")
    return X_resampled, y_resampled


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train: Training features.
        y_train: Training targets.
        n_estimators: Number of trees in the forest.
        random_state: Random seed for reproducibility.

    Returns:
        Trained RandomForestClassifier.
    """
    logger.info(f"Training Random Forest with {n_estimators} estimators")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    logger.info("Random Forest training completed")
    return model


def evaluate_model(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """
    Evaluate the model and calculate Precision/Recall metrics.

    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: Test targets.

    Returns:
        Dictionary containing precision and recall metrics.
    """
    logger.info("Evaluating model performance")

    y_pred = model.predict(X_test)

    # Calculate metrics for handling unbalanced classes
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    # Also calculate macro averages (treats all classes equally)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

    metrics = {
        "precision_weighted": precision,
        "recall_weighted": recall,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
    }

    logger.info(f"Precision (weighted): {precision:.4f}")
    logger.info(f"Recall (weighted): {recall:.4f}")
    logger.info(f"Precision (macro): {precision_macro:.4f}")
    logger.info(f"Recall (macro): {recall_macro:.4f}")

    # Print full classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info(f"\nClassification Report:\n{report}")

    return metrics


def main(data_path: str = "data/raw/network_data.csv") -> dict:
    """
    Main function to run the network intrusion detection pipeline.

    Args:
        data_path: Path to the network intrusion data file.

    Returns:
        Dictionary containing evaluation metrics.
    """
    logger.info("Starting Network Intrusion Detection pipeline")

    # Load data
    df = load_data(data_path)

    # Preprocess data
    X, y = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Handle class imbalance on training data only
    X_train_resampled, y_train_resampled = handle_imbalance(X_train, y_train)

    # Train model
    model = train_random_forest(X_train_resampled, y_train_resampled)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    logger.info("Pipeline completed successfully")
    return metrics


if __name__ == "__main__":
    import sys

    # Use command line argument for data path if provided
    data_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/network_data.csv"
    main(data_file)
