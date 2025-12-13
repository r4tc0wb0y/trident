# -*- coding: utf-8 -*-
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Import our local 'trident' package
# (Make sure you installed it with 'pip install -e .' first)
from trident import config, data, features, models

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Trident Training Pipeline...")

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    logger.info("Loading dataset...")
    try:
        df = data.load_data()
        logger.info(f"   Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # ---------------------------------------------------------
    # 2. SPLIT DATA (Stratified)
    # ---------------------------------------------------------
    logger.info("Splitting data (80% Train / 20% Test)...")
    
    if config.TARGET_COL not in df.columns:
        logger.error(f"Target column '{config.TARGET_COL}' not found.")
        return

    # --- FIX: Handle Rare Classes ---
    # We must filter out classes with fewer than 2 samples because 
    # stratify requires at least 2 members per class.
    class_counts = df[config.TARGET_COL].value_counts()
    rare_classes = class_counts[class_counts < 2].index
    
    if len(rare_classes) > 0:
        logger.warning(f"   Dropping classes with only 1 sample: {list(rare_classes)}")
        # Keep only rows where the label is NOT in rare_classes
        df = df[~df[config.TARGET_COL].isin(rare_classes)]
    # --------------------------------

    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=config.RANDOM_STATE, 
        stratify=df[config.TARGET_COL]
    )
    logger.info(f"   Train set: {train_df.shape[0]} samples")
    logger.info(f"   Test set:  {test_df.shape[0]} samples")

    # ---------------------------------------------------------
    # 3. PREPROCESSING (Fit on Train, Transform Test)
    # ---------------------------------------------------------
    logger.info("Preprocessing Training Data (Fitting Encoders)...")
    # 'fit=True' learns the means/categories from the Train Set
    X_train, y_train, artifacts = features.preprocess_data(train_df, fit=True)
    
    logger.info("Preprocessing Test Data (Using Saved Encoders)...")
    # 'fit=False' uses the artifacts learned above (Simulating production)
    X_test, y_test, _ = features.preprocess_data(test_df, fit=False, artifacts=artifacts)

    # ---------------------------------------------------------
    # 4. HANDLE IMBALANCE (SMOTE)
    # ---------------------------------------------------------
    logger.info("Applying SMOTE to Training Data...")
    
    # SMOTE is applied ONLY to the training set to prevent data leakage
    X_train_bal, y_train_bal = features.handle_imbalance(X_train, y_train)
    
    # Optional: Log class distribution after SMOTE
    # logger.info(f"   After SMOTE: {pd.Series(y_train_bal).value_counts().to_dict()}")

    # ---------------------------------------------------------
    # 5. TRAIN MODEL (Random Forest)
    # ---------------------------------------------------------
    logger.info("Training Random Forest Model...")
    model = models.train_random_forest(X_train_bal, y_train_bal)

    # ---------------------------------------------------------
    # 6. EVALUATION
    # ---------------------------------------------------------
    logger.info("Evaluating on Test Set...")
    accuracy = models.evaluate_model(model, X_test, y_test)
    logger.info(f"   Test Accuracy: {accuracy:.4%}")

    # ---------------------------------------------------------
    # 7. SAVE ARTIFACTS
    # ---------------------------------------------------------
    logger.info("Saving Model and Artifacts...")
    
    # Ensure the directory exists
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the model
    model_path = config.MODELS_DIR / "best_model_rf.pkl"
    joblib.dump(model, model_path)
    
    # Save preprocessing artifacts (encoders, scalers) for the Dashboard
    artifacts_path = config.MODELS_DIR / "preprocessing_artifacts.pkl"
    joblib.dump(artifacts, artifacts_path)
    
    logger.info(f"   Model saved to: {model_path}")
    logger.info("Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()