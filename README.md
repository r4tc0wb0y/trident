<img width="3264" height="1312" alt="Gemini_Generated_Image_6har4l6har4l6har" src="https://github.com/user-attachments/assets/29dcf0b7-b20b-40e3-8029-5e88728dddd7" />
🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱

# 🔱🌊🌊🌊🌊🌊🌊🌊🔱TRIDENT🔱🌊🌊🌊🌊🌊🌊🌊🔱
🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱
### **T**emporal **R**esearch **I**n **D**etection on **E**volving **N**etwork **T**opologies
🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱🔱

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)](https://scikit-learn.org/)

> **Baseline Phase:** Establishing robust non-spatial Machine Learning baselines (Random Forest) on intrusion detection datasets to serve as a benchmark for future spatiotemporal models.

---

## Overview

**Trident** is a modular Network Intrusion Detection System (NIDS) built to analyze network traffic patterns and classify them as normal behavior or specific attacks (e.g., Smurf, Neptune).

Currently in its **Baseline Phase**, Trident implements a complete MLOps pipeline including:
* **Data Pipeline:** Automated cleaning, scaling, and encoding of NSL-KDD network traffic data.
* **Class Imbalance Handling:** Uses SMOTE (Synthetic Minority Over-sampling Technique) to effectively learn rare attack signatures.
* **Model:** A tuned **Random Forest Classifier** achieving >99% accuracy.
* **Dashboard:** An interactive **Streamlit** interface for real-time traffic simulation and threat analysis.

---

## Project Structure

The project follows a professional Python package structure for reproducibility and scalability:

```text
trident/
├── dashboard.py         # Interactive Web Interface (Streamlit)
├── scripts/
│   └── train.py         # Main training pipeline script
├── src/
│   └── trident/         # Main Python Package
│       ├── config.py    # Central configuration (paths, constants)
│       ├── data.py      # Data loading logic
│       ├── features.py  # Preprocessing & FE (SMOTE, Scaling)
│       └── models.py    # Model definitions (Random Forest)
├── models/              # Saved artifacts (.pkl files)
├── data/                # Raw and processed datasets
└── requirements.txt     # Project dependencies