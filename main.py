"""
Network Intrusion Detection using Random Forest.

This module provides functionality for loading network intrusion data,
training a Random Forest classifier, and calculating Precision/Recall
metrics to handle unbalanced classes.
"""
import joblib
import logging
import os
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

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
    df: pd.DataFrame,
    target_column: str = "label",
    target_encoder: LabelEncoder | None = None,
    feature_encoders: dict[str, LabelEncoder] | None = None,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Preprocess the data for training or inference.

    Args:
        df: Raw DataFrame containing network intrusion data.
        target_column: Name of the column containing the target labels.
        target_encoder: Pre-fitted LabelEncoder for target (for inference).
        feature_encoders: Pre-fitted encoders for categorical features (for inference).
        scaler: Pre-fitted StandardScaler (for inference).
        fit: If True, fit new encoders/scaler; if False, use provided ones.

    Returns:
        Tuple of (features DataFrame, target Series, preprocessing artifacts dict).

    Raises:
        ValueError: If target column is not found in DataFrame.
    """
    logger.info("Preprocessing data")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    # Initialize preprocessing artifacts
    artifacts = {
        "target_encoder": target_encoder,
        "feature_encoders": feature_encoders if feature_encoders else {},
        "scaler": scaler,
    }

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical target if needed
    if y.dtype == "object":
        if fit:
            artifacts["target_encoder"] = LabelEncoder()
            y = pd.Series(artifacts["target_encoder"].fit_transform(y), index=y.index)
            logger.info(f"Encoded {len(artifacts['target_encoder'].classes_)} target classes")
        else:
            # Handle unknown target categories during inference
            encoder = artifacts["target_encoder"]
            known_classes = set(encoder.classes_)
            unknown_mask = ~y.isin(known_classes)
            if unknown_mask.any():
                logger.warning(
                    f"Found {unknown_mask.sum()} samples with unknown target categories, "
                    "mapping to -1"
                )
                # Create mapping dict and handle unknowns
                class_to_idx = {c: i for i, c in enumerate(encoder.classes_)}
                y = pd.Series(
                    [class_to_idx.get(val, -1) for val in y], index=y.index
                )
            else:
                y = pd.Series(encoder.transform(y), index=y.index)

    # Handle categorical features
    categorical_cols = X.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if fit:
            artifacts["feature_encoders"][col] = LabelEncoder()
            X[col] = artifacts["feature_encoders"][col].fit_transform(X[col].astype(str))
        else:
            # Handle unknown categories during inference using vectorized mapping
            encoder = artifacts["feature_encoders"][col]
            col_values = X[col].astype(str)

            # Create mapping dict for efficient lookup
            class_to_idx = {c: i for i, c in enumerate(encoder.classes_)}

            # Vectorized mapping with -1 for unknown categories
            X[col] = col_values.map(class_to_idx).fillna(-1).astype(int)

            # Log warning if unknowns were found
            unknown_count = (X[col] == -1).sum()
            if unknown_count > 0:
                logger.warning(
                    f"Found {unknown_count} unknown categories in column '{col}'"
                )

    # Scale numerical features
    if fit:
        artifacts["scaler"] = StandardScaler()
        X = pd.DataFrame(artifacts["scaler"].fit_transform(X), columns=X.columns, index=X.index)
    else:
        X = pd.DataFrame(artifacts["scaler"].transform(X), columns=X.columns, index=X.index)

    logger.info(f"Preprocessed {len(X)} samples with {len(X.columns)} features")
    return X, y, artifacts


def handle_imbalance(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using SMOTE oversampling.
    Adjusts k_neighbors dynamically for rare classes.
    Args:
        X: Feature DataFrame.
        y: Target Series.
    Returns:
        Tuple of (resampled features, resampled targets).
    """
    logger.info("Handling class imbalance with SMOTE")
    class_counts = y.value_counts()
    logger.info(f"Original class distribution: {dict(class_counts)}")

    # Find the smallest class size to set k_neighbors safely
    min_class_size = class_counts.min()

    # SMOTE default is k=5. We need at least k+1 samples in the class.
    # Logic: If smallest class has 2 samples, k can be max 1.
    k_neighbors = min(5, min_class_size - 1)
    
    # Safety net: k must be at least 1
    if k_neighbors < 1:
        k_neighbors = 1
        
    logger.info(f"Adjusting SMOTE k_neighbors to {k_neighbors} due to rare classes")

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
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

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.
    Args:
        X_train: Training features.
        y_train: Training targets.
        max_iter: Maximum number of iterations for convergence.
        random_state: Random seed.
    Returns:
        Trained LogisticRegression model.
    """
    logger.info(f"Training Logistic Regression (max_iter={max_iter})")

    # Logistic Regression is sensitive to scale, but we already scaled in preprocess_data!
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced", # Critical for our imbalance
        solver='lbfgs' # Standard solver
    )
    model.fit(X_train, y_train)

    logger.info("Logistic Regression training completed")
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
        Dictionary containing evaluation metrics and preprocessing artifacts.
    """
    logger.info("Starting Network Intrusion Detection pipeline")

    # Load data
    df = load_data(data_path)

    # Preprocess data (fit=True to train encoders and scaler)
    X, y, artifacts = preprocess_data(df, fit=True)

    # --- FIX START: Remove classes with fewer than 2 instances ---
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    
    # Check if we have rare classes to drop
    if len(valid_classes) < len(class_counts):
        diff = len(class_counts) - len(valid_classes)
        logger.warning(f"Dropping {diff} rare classes with only 1 sample to allow stratified split.")
        
        # Filter mask
        mask = y.isin(valid_classes)
        X = X[mask]
        y = y[mask]
    # --- FIX END ---

    # Split data
    # Now this won't crash because all classes have at least 2 samples
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

    # Store model and artifacts for future inference
# Store model and artifacts for future inference
    result = {
        "metrics": metrics,
        "model": model,
        "artifacts": artifacts,
    }

    # --- NEW: Save the model to disk ---
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "nids_model.pkl"
    artifacts_path = models_dir / "preprocessing_artifacts.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(artifacts, artifacts_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Artifacts saved to {artifacts_path}")
    # -----------------------------------

    logger.info("Pipeline completed successfully")
    return result

if __name__ == "__main__":
    import sys

    # Use command line argument for data path if provided
    data_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/network_data.csv"
    main(data_file)
