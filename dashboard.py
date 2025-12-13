import streamlit as st
import pandas as pd
import numpy as np
import joblib
from trident import config
from trident.features import preprocess_data

# Page Configuration
st.set_page_config(
    page_title="Trident NIDS",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Cyberpunk/Security Style ---
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-family: 'Courier New', monospace;
        color: #00ff41;
        text-align: center;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #d63031;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff7675;
        color: black;
    }
    .metric-container {
        background-color: #1e272e;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #00d2d3;
    }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🔱 Trident: Network Intrusion Detection System")
st.markdown("---")

# --- Model Loading Function (Cached) ---
@st.cache_resource
def load_artifacts():
    # Use config.MODELS_DIR to ensure path safety
    model_path = config.MODELS_DIR / "best_model_rf.pkl"
    artifacts_path = config.MODELS_DIR / "preprocessing_artifacts.pkl"
    
    if not model_path.exists():
        st.error(f"Model not found at: {model_path}")
        st.stop()
        
    if not artifacts_path.exists():
        st.error(f"Artifacts not found at: {artifacts_path}")
        st.stop()

    model = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)
    return model, artifacts

# Load assets
try:
    model, artifacts = load_artifacts()
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# --- Sidebar: Traffic Configuration ---
st.sidebar.header("Traffic Configuration")
st.sidebar.markdown("Simulate a network connection by adjusting key parameters.")

# User Inputs (Simulating a network packet)
# Default values represent 'normal' HTTP traffic
duration = st.sidebar.number_input("Duration (seconds)", min_value=0.0, value=0.0)
protocol_type = st.sidebar.selectbox("Protocol", ["tcp", "udp", "icmp"])
service = st.sidebar.selectbox("Service", ["http", "private", "ecr_i", "smtp", "ftp_data", "other"])
flag = st.sidebar.selectbox("TCP Flag", ["SF", "S0", "REJ", "RSTR", "SH", "RSTO"])
src_bytes = st.sidebar.number_input("Source Bytes", min_value=0, value=200)
dst_bytes = st.sidebar.number_input("Destination Bytes", min_value=0, value=4000)

st.sidebar.markdown("---")
count = st.sidebar.slider("Connections to same host (count)", 0, 511, 2)
srv_count = st.sidebar.slider("Connections to same service (srv_count)", 0, 511, 2)
serror_rate = st.sidebar.slider("SYN error rate", 0.0, 1.0, 0.0)
dst_host_srv_count = st.sidebar.slider("Destination host service count", 0, 255, 100)

# --- Main Panel ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Real-Time Analysis")
    
    # Analysis Button
    if st.button("Analyze Connection"):
        # 1. Construct DataFrame with input data
        # We must include ALL columns the model expects, even if we hardcode defaults for hidden ones
        feature_names = artifacts.get("feature_names", [])
        
        # Base dictionary with user inputs and safe defaults for others
        input_data = {
            'duration': [duration],
            'protocol_type': [protocol_type],
            'service': [service],
            'flag': [flag],
            'src_bytes': [src_bytes],
            'dst_bytes': [dst_bytes],
            'count': [count],
            'srv_count': [srv_count],
            'serror_rate': [serror_rate],
            'dst_host_srv_count': [dst_host_srv_count],
            # Defaults for features not exposed in the simplified UI
            'wrong_fragment': [0],
            'urgent': [0],
            'hot': [0],
            'num_failed_logins': [0],
            'logged_in': [1 if service == 'http' else 0],
            'num_compromised': [0],
            'root_shell': [0],
            'su_attempted': [0],
            'num_root': [0],
            'num_file_creations': [0],
            'num_shells': [0],
            'num_access_files': [0],
            'num_outbound_cmds': [0],
            'is_host_login': [0],
            'is_guest_login': [0],
            'srv_serror_rate': [serror_rate], # Assumed similar to serror_rate
            'rerror_rate': [0.0],
            'srv_rerror_rate': [0.0],
            'same_srv_rate': [1.0],
            'diff_srv_rate': [0.0],
            'srv_diff_host_rate': [0.0],
            'dst_host_count': [255],
            'dst_host_same_srv_rate': [1.0],
            'dst_host_diff_srv_rate': [0.0],
            'dst_host_same_src_port_rate': [0.0],
            'dst_host_srv_diff_host_rate': [0.0],
            'dst_host_serror_rate': [serror_rate],
            'dst_host_srv_serror_rate': [serror_rate], # FIXED: Added Missing Column
            'dst_host_rerror_rate': [0.0],
            'dst_host_srv_rerror_rate': [0.0],         # FIXED: Added Missing Column
            'land': [0]
        }
        
        # Create DataFrame
        raw_df = pd.DataFrame(input_data)
        
        # 2. Preprocess (Using our trident package)
        try:
            # Dashboard is inference only, so fit=False
            X_processed, _, _ = preprocess_data(raw_df, fit=False, artifacts=artifacts)
            
            # 3. Prediction
            prediction_idx = model.predict(X_processed)[0]
            probabilities = model.predict_proba(X_processed)[0]
            
            # Decode the label
            target_encoder = artifacts["target_encoder"]
            pred_label = target_encoder.inverse_transform([prediction_idx])[0]
            confidence = np.max(probabilities)
            
            # 4. Display Results
            st.markdown(f"### Model Confidence: **{confidence:.2%}**")
            
            # Check if prediction is normal (accounting for potential variations in label naming)
            if pred_label in ['normal', 'normal.']:
                st.success(f"Traffic classified as **NORMAL**. No intrusion detected.")
                st.info("No immediate action required. Monitoring continues.")
            else:
                st.error(f"**INTRUSION DETECTED**: {pred_label.upper()}")
                st.warning(f"Suggested action: Isolate source IP immediately and investigate logs for {pred_label} signature.")
            
            # Debugging Data Expanders
            with st.expander("View processed feature vector (debug)"):
                st.write(X_processed)
                
            with st.expander("View class probabilities (debug)"):
                # Create a readable dataframe for probabilities
                probs_df = pd.DataFrame(probabilities, index=target_encoder.classes_, columns=["Probability"])
                # Sort by probability descending
                probs_df = probs_df.sort_values(by="Probability", ascending=False)
                st.dataframe(probs_df.style.format("{:.4f}"))

        except Exception as e:
            st.error(f"Processing Error: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Trident - CS3315 Final Project")