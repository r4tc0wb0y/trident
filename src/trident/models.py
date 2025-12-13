from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
from . import config

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Classifier.
    
    Args:
        X_train: Feature matrix (balanced).
        y_train: Target vector (balanced).
        
    Returns:
        Trained RandomForestClassifier model.
    """
    # n_jobs=-1 uses all CPU cores for faster training
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=config.RANDOM_STATE, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and prints a report.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test targets.
        
    Returns:
        float: Accuracy score.
    """
    preds = model.predict(X_test)
    
    # Generate textual report
    report = classification_report(y_test, preds)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    print("="*60 + "\n")
    
    return accuracy_score(y_test, preds)