# Self-Driven Car using NEAT 🚗🧠

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png" height="100">
  <img src="https://www.pygame.org/docs/_images/pygame_logo.png" height="100">
  <img src="https://numpy.org/images/logo.svg" height="100">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/SCIPY_2.svg/1200px-SCIPY_2.svg.png" height="100">
  <img src="https://thumbs.dreamstime.com/b/machine-learning-icon-two-color-design-red-black-style-elements-icons-collection-creative-web-apps-software-print-144659464.jpg" height="100">
</div>

This project is a simulation of self-driving cars controlled by neural networks, implemented using the Python programming language and the Pygame library for graphics rendering. The cars navigate through a randomly generated road, and their neural networks are trained using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm to learn how to drive successfully.

<div align="center">
  <img src="https://user-images.githubusercontent.com/16992394/205456901-c6330d6a-a96c-4f1f-a98e-8b7616b71df7.gif" width="500">
</div>

## Features ✨

- Simulation of self-driving cars controlled by neural networks
- Randomly generated road with varying curvatures and slopes
- Neural network training using the NEAT algorithm
- Visualization of the neural network structure and connections
- Real-time display of car positions, velocities, and sensor inputs

## Neural Network Architecture 🧠

The neural network architecture used in this simulation is a feed-forward network with the following structure:

- **Input Layer**: 9 nodes
  - 8 nodes representing the distances from the car to the road boundaries in 8 different directions
  - 1 node representing the car's current velocity
- **Hidden Layer**: Variable number of nodes, determined by the NEAT algorithm
- **Output Layer**: 4 nodes
  - 1 node for accelerating
  - 1 node for braking
  - 1 node for turning left
  - 1 node for turning right

The neural network takes the sensor inputs and the car's velocity as input and outputs four values representing the actions to be taken (accelerate, brake, turn left, or turn right).

## Neural Network Calculation 🧮

The neural network is calculated using the following steps:

1. The sensor inputs and the car's velocity are fed into the input layer of the neural network.
2. The input values are propagated through the hidden layer(s) using the weights and biases determined by the NEAT algorithm.
3. The outputs of the hidden layer(s) are propagated to the output layer, producing four output values.
4. The output values are interpreted as follows:
   - The accelerate output determines whether the car should accelerate or not.
   - The brake output determines whether the car should brake or not.
   - The turn left and turn right outputs determine the direction the car should turn.
5. The car's actions (accelerate, brake, turn left, or turn right) are executed based on the output values.

The neural network is trained over multiple generations using the NEAT algorithm, which evolves the network's topology and weights to optimize the car's performance on the road.

## Requirements 📋

- Python 3.x (https://www.python.org/downloads/)
- Pygame (https://www.pygame.org/news)
- NumPy (https://numpy.org/)
- SciPy (https://scipy.org/)
- NEAT-Python (https://neat-python.readthedocs.io/en/latest/)

## Installation 🚀

1. Clone the repository or download the source code.

```bash
git clone https://github.com/Taaran18/Self-Driven-Car.git
```

2. Install the required dependencies using pip:

```bash
pip install pygame numpy scipy neat-python
```

## Usage 🏃‍♂️

1. Navigate to the project directory.

```bash
cd self-driven-car
```

2. Run the main script:

```bash
python main.py
```

The simulation will start, and you can observe the self-driving cars navigating through the road. The neural networks will be trained over multiple generations, and the best-performing car's neural network structure will be visualized.

## Configuration ⚙️

The simulation parameters and neural network configurations can be modified in the `config_variables.py` and `config_file.txt` files, respectively. Refer to the comments in these files for more information on the available settings.

## Contributing 🤝

Contributions to this project are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

### Project Contributors

- [Taran Jain](https://github.com/Taaran18)
- [Vikas Sharma](https://github.com/vikasharma005)
- [Tanishq Soni](https://github.com/TanishqSoni2003)

## License 📄

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments 🙏

This project was inspired by various neural network car simulations and tutorials available online. Special thanks to the creators of the Pygame library, NumPy, SciPy, and NEAT-Python for providing the necessary tools and libraries.

## Support 🌟

If you find this project useful, please consider giving it a ⭐️ on GitHub!
