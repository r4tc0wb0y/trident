# src/dashboard.py

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# 1. Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Trident NIDS",
    page_icon="",
    layout="wide"
)

# ----------------------------------------------------------------------
# 2. Import preprocessing function from main.py
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../trident/src
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                # .../trident
sys.path.append(PROJECT_ROOT)

try:
    from main import preprocess_data
except ImportError as e:
    st.error(
        "Could not import 'preprocess_data' from 'main.py'. "
        "Make sure 'main.py' is in the project root and defines this function.\n"
        f"Details: {e}"
    )
    st.stop()

# ----------------------------------------------------------------------
# 3. Load model and preprocessing artifacts
# ----------------------------------------------------------------------
@st.cache_resource
def load_assets():
    """
    Load the trained Random Forest model and preprocessing artifacts.
    Cached so it is only loaded once per session.
    """
    model_path = os.path.join(CURRENT_DIR, "..", "models", "best_model_rf.pkl")
    artifacts_path = os.path.join(CURRENT_DIR, "..", "models", "preprocessing_artifacts.pkl")

    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None, None

    if not os.path.exists(artifacts_path):
        st.error(f"Artifacts file not found at: {artifacts_path}")
        return None, None

    try:
        model = joblib.load(model_path)
        artifacts = joblib.load(artifacts_path)
        return model, artifacts
    except Exception as e:
        st.error(f"Error loading model or artifacts: {e}")
        return None, None


model, artifacts = load_assets()
if model is None or artifacts is None:
    st.stop()

# ----------------------------------------------------------------------
# 4. User Interface
# ----------------------------------------------------------------------
st.title("Trident: Network Intrusion Detection System")
st.markdown("---")

col_left, col_right = st.columns([1, 2])

# ----------------------------------------------------------------------
# 4.1 Left column – packet / flow configuration
# ----------------------------------------------------------------------
with col_left:
    st.subheader("Traffic Configuration")
    st.info("Simulate a network connection by adjusting key parameters.")

    # Example values are roughly based on normal NSL-KDD traffic
    duration = st.number_input(
        "Duration (seconds)", min_value=0.0, value=0.0, step=0.1
    )

    protocol_type = st.selectbox("Protocol", ["tcp", "udp", "icmp"])
    service = st.selectbox(
        "Service",
        ["http", "private", "ecr_i", "smtp", "ftp_data", "other"]
    )
    flag = st.selectbox("TCP Flag", ["SF", "S0", "REJ", "RSTR", "SH"])

    src_bytes = st.number_input(
        "Source bytes (src_bytes)", min_value=0, value=200, step=50
    )
    dst_bytes = st.number_input(
        "Destination bytes (dst_bytes)", min_value=0, value=4000, step=100
    )

    count = st.slider(
        "Connections to same host (count)", min_value=0, max_value=511, value=2
    )
    srv_count = st.slider(
        "Connections to same service (srv_count)", min_value=0, max_value=511, value=2
    )

    serror_rate = st.slider(
        "SYN error rate (serror_rate)", min_value=0.0, max_value=1.0, value=0.0
    )
    dst_host_srv_count = st.slider(
        "Destination host service count (dst_host_srv_count)",
        min_value=0,
        max_value=255,
        value=100
    )

# ----------------------------------------------------------------------
# 4.2 Right column – analysis and output
# ----------------------------------------------------------------------
with col_right:
    st.subheader("Real-Time Analysis")

    if st.button("Analyze Connection", type="primary"):
        try:
            # ------------------------------------------------------------------
            # 5. Build a full feature dictionary (41 NSL-KDD features)
            #    Non-controlled features are filled with default values.
            # ------------------------------------------------------------------
            input_dict = {
                "duration": duration,
                "protocol_type": protocol_type,
                "service": service,
                "flag": flag,
                "src_bytes": src_bytes,
                "dst_bytes": dst_bytes,
                "land": 0,
                "wrong_fragment": 0,
                "urgent": 0,
                "hot": 0,
                "num_failed_logins": 0,
                "logged_in": 1,
                "num_compromised": 0,
                "root_shell": 0,
                "su_attempted": 0,
                "num_root": 0,
                "num_file_creations": 0,
                "num_shells": 0,
                "num_access_files": 0,
                "num_outbound_cmds": 0,
                "is_host_login": 0,
                "is_guest_login": 0,
                "count": count,
                "srv_count": srv_count,
                "serror_rate": serror_rate,
                "srv_serror_rate": serror_rate,
                "rerror_rate": 0.0,
                "srv_rerror_rate": 0.0,
                "same_srv_rate": 1.0,
                "diff_srv_rate": 0.0,
                "srv_diff_host_rate": 0.0,
                "dst_host_count": 1,
                "dst_host_srv_count": dst_host_srv_count,
                "dst_host_same_srv_rate": 1.0,
                "dst_host_diff_srv_rate": 0.0,
                "dst_host_same_src_port_rate": 0.0,
                "dst_host_srv_diff_host_rate": 0.0,
                "dst_host_serror_rate": 0.0,
                "dst_host_srv_serror_rate": 0.0,
                "dst_host_rerror_rate": 0.0,
                "dst_host_srv_rerror_rate": 0.0,
                # Dummy label so preprocess_data does not raise an error
                "label": "unknown"
            }

            raw_df = pd.DataFrame([input_dict])

            # ------------------------------------------------------------------
            # 6. Preprocess using the training artifacts
            #    We pass the dummy target column and use fit=False so that
            #    encoders and scaler are not refit.
            # ------------------------------------------------------------------
            X_processed, _, _ = preprocess_data(
                raw_df,
                target_column="label",
                fit=False,
                target_encoder=None,
                feature_encoders=artifacts["feature_encoders"],
                scaler=artifacts["scaler"],
            )

            # ------------------------------------------------------------------
            # 7. Prediction
            # ------------------------------------------------------------------
            prediction = model.predict(X_processed)[0]
            prob_vector = model.predict_proba(X_processed)[0]
            confidence = float(np.max(prob_vector))

            # Decode predicted label
            pred_label = artifacts["target_encoder"].inverse_transform([prediction])[0]
            is_normal = pred_label.lower() in {"normal", "benign"}

            # ------------------------------------------------------------------
            # 8. Display result
            # ------------------------------------------------------------------
            if is_normal:
                st.success("Normal traffic detected.")
                st.metric("Model confidence", f"{confidence:.2%}")
            else:
                st.error(f"Intrusion detected: {pred_label.upper()}")
                st.metric("Model confidence", f"{confidence:.2%}")
                st.progress(confidence)
                st.warning("Suggested action: investigate and block the source if confirmed malicious.")

            # Optional: show processed feature vector for debugging
            with st.expander("Show processed feature vector (debug)"):
                # X_processed is already a DataFrame with its own columns
                st.write(X_processed)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.info("Tip: verify that 'main.py' is in the 'src' folder and matches the training pipeline.")

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption("Trident Project – CS3315 – Francisco Maldonado")