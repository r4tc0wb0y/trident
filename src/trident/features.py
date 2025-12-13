import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from . import config

# Standard NSL-KDD categorical columns
CAT_COLS = ['protocol_type', 'service', 'flag']

def preprocess_data(df: pd.DataFrame, fit: bool = True, artifacts: dict = None):
    """
    Preprocessing Pipeline:
    1. Separates Features (X) and Target (y).
    2. Applies OneHotEncoder to categorical columns (handle_unknown='ignore').
    3. Applies StandardScaler to numerical columns.
    4. Encodes the Target if it exists.
    """
    df = df.copy()
    
    # 1. Separate X and y
    if config.TARGET_COL in df.columns:
        y = df[config.TARGET_COL]
        X = df.drop(columns=[config.TARGET_COL])
    else:
        y = None
        X = df

    # Dynamically identify numerical columns
    num_cols = [c for c in X.columns if c not in CAT_COLS]

    # 2. Configure or Load ColumnTransformer (Encoder + Scaler)
    if fit:
        # Define transformer for mixed columns
        # handle_unknown='ignore' is VITAL to prevent failure if a new service appears in Test
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS)
            ],
            verbose_feature_names_out=False
        )
        
        # Fit (learn means and categories)
        X_processed = preprocessor.fit_transform(X)
        
        # Save feature names for the dashboard
        feature_names = list(preprocessor.get_feature_names_out())
        
        # Label Encoder for the Target
        target_encoder = LabelEncoder()
        if y is not None:
            y_encoded = target_encoder.fit_transform(y)
        else:
            y_encoded = None
            
        # Save everything in the artifacts dictionary
        new_artifacts = {
            "preprocessor": preprocessor,
            "target_encoder": target_encoder,
            "feature_names": feature_names
        }
        
    else:
        # INFERENCE / TEST MODE
        if artifacts is None:
            raise ValueError("Artifacts are required when fit=False")
        
        preprocessor = artifacts["preprocessor"]
        target_encoder = artifacts["target_encoder"]
        
        # Transform using what was learned in Train
        X_processed = preprocessor.transform(X)
        
        # Transform the target if it exists (for evaluation)
        if y is not None:
            # Use a trick to avoid errors if there are new classes in test not found in train
            # (Although we filter rare classes in train.py, this is double security)
            y_encoded = np.zeros(len(y), dtype=int)
            known_labels = set(target_encoder.classes_)
            
            for i, label in enumerate(y):
                if label in known_labels:
                    y_encoded[i] = target_encoder.transform([label])[0]
                else:
                    y_encoded[i] = -1 # Unknown marker
        else:
            y_encoded = None
            
        new_artifacts = artifacts

    # Convert to DataFrame to keep column names (useful for debug)
    # Note: X_processed is already a numpy array, we leave it like that for the model, 
    # but if you wanted a dataframe: pd.DataFrame(X_processed, columns=new_artifacts['feature_names'])
    
    return X_processed, y_encoded, new_artifacts

def handle_imbalance(X, y):
    """Applies SMOTE to balance classes in the training set."""
    # k_neighbors=1 prevents errors if a class has very few examples (e.g., 2 or 3)
    smote = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res