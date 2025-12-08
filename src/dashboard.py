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
    page_icon=None,
    layout="wide"
)

# ----------------------------------------------------------------------
# 2. Import preprocessing function from main.py
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

try:
    from main import preprocess_data
except ImportError:
    st.error(
        "Could not import 'preprocess_data' from 'main.py'. "
        "Make sure 'main.py' exists in the 'src' folder."
    )
    st.stop()

# ----------------------------------------------------------------------
# 3. Load model and preprocessing artifacts
# ----------------------------------------------------------------------
@st.cache_resource
def load_assets():
    """
    Load the trained Random Forest model and preprocessing artifacts.
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
        st.error(f"Error loading assets: {e}")
        return None, None


model, artifacts = load_assets()
if model is None:
    st.stop()

# ----------------------------------------------------------------------
# 4. User Interface
# ----------------------------------------------------------------------
st.title("Trident: Network Intrusion Detection System")
st.markdown("---")

col_left, col_right = st.columns([1, 2])

# 4.1 Left column – Configuration
with col_left:
    st.subheader("Traffic Configuration")
    st.info("Simulate a network connection by adjusting key parameters.")

    duration = st.number_input("Duration (seconds)", min_value=0.0, value=0.0, step=0.1)
    protocol_type = st.selectbox("Protocol", ["tcp", "udp", "icmp"])
    service = st.selectbox("Service", ["http", "private", "ecr_i", "smtp", "ftp_data", "other"])
    flag = st.selectbox("TCP Flag", ["SF", "S0", "REJ", "RSTR", "SH"])
    src_bytes = st.number_input("Source bytes", min_value=0, value=200, step=50)
    dst_bytes = st.number_input("Destination bytes", min_value=0, value=4000, step=100)
    count = st.slider("Connections to same host (count)", 0, 511, 2)
    srv_count = st.slider("Connections to same service (srv_count)", 0, 511, 2)
    serror_rate = st.slider("SYN error rate", 0.0, 1.0, 0.0)
    dst_host_srv_count = st.slider("Destination host service count", 0, 255, 100)

# 4.2 Right column – Analysis
with col_right:
    st.subheader("Real-Time Analysis")

    if st.button("Analyze Connection", type="primary"):
        try:
            # 5. Build feature dictionary (dummy label included)
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
                # dummy target; required by preprocess_data but not used for prediction
                "label": "normal"
            }

            raw_df = pd.DataFrame([input_dict])

            # 6. Preprocess using the same artifacts as training
            X_processed, _, _ = preprocess_data(
                raw_df,
                target_column="label",
                fit=False,
                target_encoder=artifacts["target_encoder"],
                feature_encoders=artifacts["feature_encoders"],
                scaler=artifacts["scaler"],
            )

            # 7. Prediction
            prediction = model.predict(X_processed)[0]
            prob_vector = model.predict_proba(X_processed)[0]
            confidence = float(np.max(prob_vector))

            # Decode label
            pred_label = artifacts["target_encoder"].inverse_transform([prediction])[0]
            is_normal = pred_label.lower() in ["normal", "benign", "benign."]

            # 8. Display result with confidence threshold
            threshold = 0.70

            if is_normal and confidence >= threshold:
                st.success("Traffic classified as NORMAL (no intrusion detected).")
                st.metric("Model confidence", f"{confidence:.2%}")
                st.info("No immediate action required. Monitoring continues.")
            elif is_normal and confidence < threshold:
                st.warning("Traffic classified as NORMAL, but with LOW confidence.")
                st.metric("Model confidence", f"{confidence:.2%}")
                st.info(
                    "Suggested action: keep monitoring this source. "
                    "Consider deeper inspection if unusual patterns persist."
                )
            else:
                st.error(f"INTRUSION DETECTED: {pred_label.upper()}")
                st.metric("Model confidence", f"{confidence:.2%}")
                st.progress(confidence)
                st.warning("Suggested action: isolate source IP and investigate logs.")

            # Debug: processed feature vector
            with st.expander("View processed feature vector (debug)"):
                st.write(X_processed)

            # Debug: class probabilities
            with st.expander("View class probabilities (debug)"):
                # Classes that the model actually outputs probabilities for
                model_class_indices = model.classes_  # integer labels used by the model
                class_labels = artifacts["target_encoder"].inverse_transform(model_class_indices)
            
                probs_df = pd.DataFrame(
                    {
                        "Class": class_labels,
                        "Probability": prob_vector
                    }
                ).sort_values("Probability", ascending=False)
            
                st.write(probs_df.reset_index(drop=True))

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Check your model, artifacts and main.py consistency.")

# Footer
st.markdown("---")
st.caption("Trident Project – CS3315 – Francisco Maldonado")