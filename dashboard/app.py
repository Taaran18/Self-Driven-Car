import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.simulation import SimulationRunner, set_stop_simulation

st.set_page_config(page_title="Self-Driving Car Dashboard", layout="wide")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Card styling for metrics */
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #363945;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Improve headers */
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Custom button styling */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Self-Driving Car Simulation")
st.markdown("### Neural Evolution of Augmenting Topologies (NEAT)")

# Sidebar controls
st.sidebar.header("Control Panel")

if st.sidebar.button("▶ Start Simulation", type="primary"):
    st.session_state["running"] = True
    set_stop_simulation(False)

if st.sidebar.button("⏹ Stop Simulation", type="secondary"):
    st.session_state["running"] = False
    set_stop_simulation(True)

st.markdown("---")

# Layout: Metrics at the top
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
    status_ind.metric("Status", "Standby")

st.markdown("---")

# Main content
col1, col2 = st.columns([1.8, 1.2], gap="large")

with col1:
    st.subheader("Live Feed")
    frame_placeholder = st.empty()
    # Placeholder for the start image or waiting state
    if not st.session_state.get("running"):
        frame_placeholder.info("Click 'Start Simulation' in the sidebar to begin.")

with col2:
    st.subheader("Analytics")
    tab1, tab2 = st.tabs(["fitness_chart", "raw_data"])

    with tab1:
        st.caption("Max (Red) vs Average (Gray) Fitness over Generations")
        chart_placeholder = st.empty()

    with tab2:
        st.caption("Detailed Statistics")
        stats_table = st.empty()

# Run logic
if st.session_state.get("running"):
    status_ind.metric("Status", "Running")

    # Path to config
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "config_file.txt"
    )

    runner = SimulationRunner(config_path)

    # We iterate through the simulation generator
    try:
        runner_iterator = runner.run()

        for msg_type, data in runner_iterator:
            if not st.session_state.get("running"):
                break

            if msg_type == "frame":
                # Data is (W, H, 3) from surfarray.
                # Need to rotate/flip because surfarray is usually transposed relative to standard image
                # pygame.surfarray.array3d returns (width, height, 3) where x is width.
                # Matrix representation (row, col) = (y, x).
                # So we usually transpose (1, 0, 2) to get (Height, Width, 3)
                frame = np.transpose(data, (1, 0, 2))
                frame_placeholder.image(frame, channels="RGB", use_container_width=True)

            elif msg_type == "stats":
                # Data is list of dicts
                if data:
                    latest = data[-1]
                    curr_gen.metric("Generation", latest["Generation"])
                    max_fit.metric("Max Fitness", f"{latest['Max Fitness']:.2f}")
                    avg_fit.metric("Avg Fitness", f"{latest['Average Fitness']:.2f}")

                    df = pd.DataFrame(data)
                    chart_placeholder.line_chart(
                        df.set_index("Generation")[["Max Fitness", "Average Fitness"]],
                        color=["#ff4b4b", "#808080"],
                    )
                    stats_table.dataframe(
                        df.sort_values(by="Generation", ascending=False).head(10),
                        use_container_width=True,
                    )

    except KeyboardInterrupt:
        st.warning("Simulation Stopped.")
        status_ind.metric("Status", "Stopped")
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    status_ind.metric("Status", "Stopped")
