import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web.bridge import SimulationRunner, set_stop_simulation

# Set page config for a widescreen layout
st.set_page_config(
    page_title="Self-Driving Car Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium, minimalist, and modern look
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    :root {
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
        --accent-color: #FF4B4B;
    }

    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: transparent;
    }

    /* Glassmorphism containers */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.05);
    }

    /* Metric styling */
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.1);
        padding: 20px;
        border-radius: 12px;
        transition: transform 0.2s ease-in-out;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: var(--accent-color);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        font-weight: 300 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
    }

    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 600 !important;
    }

    /* Sidebar improvements */
    section[data-testid="stSidebar"] {
        background-color: var(--background-color);
        border-right: 1px solid var(--glass-border);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF7B7B 100%) !important;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3) !important;
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4) !important;
        transform: scale(1.02);
    }

    /* Status Pill */
    .status-pill {
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 10px;
    }

    .status-running {
        background: rgba(0, 255, 0, 0.1);
        color: #00FF00;
        border: 1px solid rgba(0, 255, 0, 0.2);
    }

    .status-standby {
        background: rgba(255, 165, 0, 0.1);
        color: #FFA500;
        border: 1px solid rgba(255, 165, 0, 0.2);
    }
    
    /* Header styling */
    h1 {
        font-weight: 600 !important;
        letter-spacing: -1.5px !important;
        margin-bottom: 0 !important;
    }
    
    h3 {
        font-weight: 300 !important;
        letter-spacing: -0.5px !important;
        opacity: 0.8;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: var(--glass-bg);
        border-radius: 8px 8px 0 0;
        border: none;
        padding: 10px 20px;
        font-weight: 400;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--accent-color) !important;
        color: white !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header Section
st.title("Self-Driving Dashboard")
st.markdown("### NEAT Neural Evolution Simulation")

# Sidebar Implementation
with st.sidebar:
    st.markdown("## Control Panel")
    st.markdown("Monitor and control the evolution of the self-driving AI.")

    col_start, col_stop = st.columns(2)

    if st.button("▶ START", type="primary", use_container_width=True):
        st.session_state["running"] = True
        set_stop_simulation(False)

    if st.button("⏹ STOP", type="secondary", use_container_width=True):
        st.session_state["running"] = False
        set_stop_simulation(True)

    st.divider()
    st.info(
        "The AI uses NEAT (NeuroEvolution of Augmenting Topologies) to learn how to drive on the track."
    )

st.markdown("---")

# Metrics Section
m1, m2, m3, m4 = st.columns(4)
with m1:
    curr_gen = st.empty()
    curr_gen.metric("Generation", "0")
with m2:
    max_fit = st.empty()
    max_fit.metric("Max Fitness", "0.0")
with m3:
    avg_fit = st.empty()
    avg_fit.metric("Avg Fitness", "0.0")
with m4:
    status_ind = st.empty()
    # status_ind will be updated later with pill logic

st.markdown("---")

# Main Content Layout
col_feed, col_data = st.columns([2, 1], gap="large")

with col_feed:
    st.subheader("Live Simulation Feed")
    frame_placeholder = st.empty()

    if not st.session_state.get("running"):
        frame_placeholder.info(
            "Simulation is currently in Standby. Click 'START' to begin."
        )

with col_data:
    st.subheader("Real-time Analytics")
    tab_chart, tab_raw = st.tabs(["📈 Fitness Growth", "📋 Statistics"])

    with tab_chart:
        chart_placeholder = st.empty()

    with tab_raw:
        stats_table = st.empty()

# Simulation Execution Logic
if st.session_state.get("running"):
    status_ind.markdown(
        '<div class="status-pill status-running">● Running</div>',
        unsafe_allow_html=True,
    )

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "neat.cfg")

    runner = SimulationRunner(config_path)

    try:
        runner_iterator = runner.run()

        for msg_type, data in runner_iterator:
            if not st.session_state.get("running"):
                break

            if msg_type == "frame":
                # Pygame surfaces use col-major, Streamlit needs row-major
                frame = np.transpose(data, (1, 0, 2))
                frame_placeholder.image(frame, channels="RGB", use_container_width=True)

            elif msg_type == "stats":
                if data:
                    latest = data[-1]
                    curr_gen.metric("Generation", latest["Generation"])
                    max_fit.metric("Max Fitness", f"{latest['Max Fitness']:.2f}")
                    avg_fit.metric("Avg Fitness", f"{latest['Average Fitness']:.2f}")

                    df = pd.DataFrame(data)
                    chart_placeholder.line_chart(
                        df.set_index("Generation")[["Max Fitness", "Average Fitness"]],
                        color=["#FF4B4B", "#808080"],
                    )
                    stats_table.dataframe(
                        df.sort_values(by="Generation", ascending=False).head(10),
                        use_container_width=True,
                    )

    except KeyboardInterrupt:
        st.warning("Simulation Stopped.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state["running"] = False
else:
    status_ind.markdown(
        '<div class="status-pill status-standby">● Standby</div>',
        unsafe_allow_html=True,
    )
