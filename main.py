"""
main.py
-------
Entry point for the Trident Backend Pipeline.
This script aligns with Section 5.1 of the project report, serving as the 
command-line interface to trigger data loading, preprocessing, and model training.

Implementation details are modularized in the 'trident' package (src/trident).
"""

import sys
from trident import config
from scripts.train import main as run_training_pipeline

if __name__ == "__main__":
    print("Trident NIDS - Backend Pipeline Initiated")
    print(f"Project Root: {config.PROJECT_ROOT}")
    
    # Execute the training logic defined in scripts/train.py
    try:
        run_training_pipeline()
        print("\n✅ Execution Complete. Model artifacts ready for Dashboard.")
    except Exception as e:
        print(f"\nCritical Error: {e}")
        sys.exit(1)