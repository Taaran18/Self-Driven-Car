# Self-Driven Car using NEAT 🚗🧠

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png" height="60" alt="Python">
  <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" height="60" alt="Streamlit">
  <img src="https://www.pygame.org/docs/_images/pygame_logo.png" height="60" alt="Pygame">
  <img src="https://numpy.org/images/logo.svg" height="60" alt="NumPy">
</div>

<br>

This project is a simulation of self-driving cars controlled by neural networks. It uses the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm to evolve a population of cars that learn to navigate a randomly generated track. The simulation is built with **Pygame** for the physics engine and features a modern **Streamlit** dashboard for monitoring and control.

## ✨ Features

*   **Beautful Dashboard**: A modern, minimalistic web interface built with Streamlit to control and view the simulation.
*   **Headless Simulation**: The car physics runs in the background without needing a native window, making it perfect for web deployments.
*   **Real-time Analytics**: Stick charts and metrics showing fitness progression, generation count, and population stats.
*   **Modular Codebase**: Clean architecture with separated logic (`src/`), configuration (`config/`), and UI (`dashboard/`).
*   **NEAT Integration**: Full implementation of genetic algorithms (mutation, crossover, speciation).

## 📂 Project Structure

```text
├── assets/          # Images and resources
├── config/          # Configuration files (NEAT config, variables)
├── dashboard/       # Streamlit dashboard code (app.py, reporter.py)
├── src/             # Core simulation logic (car, road, world, etc.)
├── main.py          # Legacy entry point for command line
└── requirements.txt # Project dependencies
```

## 🚀 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Taaran18/Self-Driven-Car.git
    cd Self-Driven-Car
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

### Option 1: Modern Dashboard (Recommended)
Run the Streamlit app to view the simulation in your browser with real-time stats.

```bash
streamlit run dashboard/app.py
```
*   Click **▶ Start Simulation** in the sidebar.
*   Monitor improvement in the **Analytics** tab.

### Option 2: Command Line
Run the simulation in a native Pygame window (Legacy mode).

```bash
python main.py
```

## 🧠 Neural Network inputs
- 8 Ray-cast sensors measuring distance to road borders.
- Current velocity.

**Outputs:**
- Accelerate, Brake, Turn Left, Turn Right.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License
This project is licensed under the [MIT License](LICENSE).
