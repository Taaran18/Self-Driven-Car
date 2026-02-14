# Self-Driving Car Simulation 🏎️🧠

A high-performance self-driving car simulation powered by **NEAT** (NeuroEvolution of Augmenting Topologies). This project demonstrates how neural networks can evolve to master complex navigation tasks through genetic algorithms.

---

## 🌟 Key Features

- **Premium Dashboard**: A minimalist, glassmorphism-based web interface built with Streamlit.
- **NEAT Evolution**: Full implementation of neuroevolution including mutation, crossover, and speciation.
- **Headless Engine**: Physics core runs independently of display, optimized for real-time web streaming.
- **Modular Design**: Professionally refactored architecture for clarity and extensibility.
- **Theme-Aware**: Dashboard automatically adapts to light and dark browser modes.

## 📂 Project Structure

```text
├── assets/          # Sprites and visual resources
├── config/          # Centralized constants and NEAT configuration
├── engine/          # Core physics, simulation logic, and NN visualizers
├── web/             # Modern dashboard interface and simulation bridge
└── main.py          # Standalone CLI simulation entry point (Legacy)
```

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Taaranjain/Self-Driven-Car.git
cd Self-Driven-Car
pip install -r requirements.txt
```

### 2. Launch the Dashboard

Experience the simulation with real-time analytics in your browser:

```bash
streamlit run web/dashboard.py
```

> [!TIP]
> Use the **Control Panel** in the sidebar to START/STOP the evolution process.

### 3. CLI Mode

Run the simulation in a native window:

```bash
python main.py
```

## 🧠 Technical Overview

### Neural Network Inputs

- **8x Ray-cast Sensors**: Measuring distance to road boundaries.
- **Velocity**: Current speed of the vehicle.

### Outputs (Decisions)

- **Accelerate / Brake**
- **Turn Left / Turn Right**

---
Built with Python, Pygame, and Streamlit.
