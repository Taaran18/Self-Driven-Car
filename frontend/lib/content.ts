export const DECIDE_SOURCE = `def decide(outputs: np.ndarray) -> Decisions:
    accelerate, brake, left, right = outputs.T
    active = outputs > ACTIVATION_THRESHOLD
    return Decisions(
        accelerate=active[:, 0] & (accelerate > brake),
        brake=active[:, 1] & (brake > accelerate),
        turn_left=active[:, 2] & (left > right),
        turn_right=active[:, 3] & (right > left),
    )`

export const FAQ = [
  {
    question: "Do I Need an Account?",
    answer:
      "No. Open the simulator and press Start Training. Your runs are linked to an anonymous trial ID that lives in your browser, so there is nothing to sign up for.",
  },
  {
    question: "How Many Runs Can I Start?",
    answer:
      "Each network (IP address) can start 5 training runs per day and 20 per week. The daily count resets at midnight UTC and the weekly count resets on Monday. A run that fails because of a server error before finishing its first generation is not counted.",
  },
  {
    question: "Is This Real Machine Learning or an Animation?",
    answer:
      "It is real. Every car is driven by its own neural network, evaluated on the server with the NEAT-Python library. The code panel shows the actual source that runs, together with the real numbers from the leading car.",
  },
  {
    question: "What Is NEAT?",
    answer:
      "NEAT stands for NeuroEvolution of Augmenting Topologies. It is a genetic algorithm introduced by Kenneth Stanley and Risto Miikkulainen in 2002 that evolves both the weights and the structure of neural networks, starting simple and adding neurons and connections only when they help.",
  },
  {
    question: "Why Do the Cars Crash So Much at First?",
    answer:
      "The first generation's networks are random, so most cars steer into a wall within seconds. Evolution keeps the few that got a little further and breeds from them. Within a handful of generations you usually see cars holding the road.",
  },
  {
    question: "Why Does Starting a Run Sometimes Take a Few Seconds?",
    answer:
      "The simulation server goes to sleep when nobody is using it, which keeps this free project affordable. Pressing Start Training wakes it up, which usually takes 5 to 30 seconds.",
  },
  {
    question: "What Data Do You Keep?",
    answer:
      "Your trial ID, your IP address, when you started each run, and the results of your runs. That is what the limits and your run history need. You can export or delete your runs at any time in Settings.",
  },
]
