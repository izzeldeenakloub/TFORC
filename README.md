# AI-Based Traffic Flow Optimization System for Real-Time Control

An AI-based traffic signal control system that uses **Reinforcement Learning (RL)** and **Multi-Agent Reinforcement Learning (MARL)** to optimize traffic flow at connected signalized intersections in **Khalda, Amman, Jordan**.

The project models two real-world intersections — **Al-Assaf (Node2)** and **Arab Bank (Node_AB)** — in the **SUMO microscopic traffic simulator** and uses **Python + TraCI** to control traffic lights dynamically based on real-time traffic conditions.

---

## Overview

Fixed-time traffic signals use predetermined phase durations regardless of current traffic conditions. This can result in inefficient green-time allocation, unnecessary queues, and increased delay.

This project investigates whether reinforcement learning can provide better adaptive control and, more importantly, how much **inter-agent coordination** is actually beneficial when controlling connected intersections under a limited training budget.

The system progresses through:

1. A fixed-time **static baseline**
2. **Q-Learning** for single-intersection control
3. **Actor-Critic** for single-intersection control
4. Four two-agent **Multi-Agent Actor-Critic** architectures with increasing levels of coordination

The main finding is that **more communication does not necessarily produce better performance**. The Shared State architecture, which provides only a discretized representation of the neighboring intersection's congestion, achieved the best overall performance and stability.

---

## Project Architecture

```text
                   Real-World Khalda Network
                            │
                            ▼
                    OpenStreetMap (OSM)
                            │
                            ▼
                    SUMO Road Network
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
          Al-Assaf Node2        Arab Bank Node_AB
          4 approaches           3 approaches
                 │                     │
                 └──────────┬──────────┘
                            │
                         TraCI
                            │
                            ▼
                    Python Controller
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Traffic State                Traffic Signal
         Detection                    Control
              │                           │
              └─────────────┬─────────────┘
                            ▼
                   Reinforcement Learning
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
       ▼                    ▼                    ▼
 Q-Learning          Actor-Critic             MARL
                                             │
                     ┌───────────────────────┼───────────────────────┐
                     │           │           │           │
                     ▼           ▼           ▼           ▼
                    SEP         SS          SSA         SSAR
```

---

## Simulation Environment

The traffic network represents two connected signalized intersections in Khalda, Amman:

### Al-Assaf Intersection — Node2

* Four approaches: West, East, North, South
* Four selectable green phases
* Action space:

```text
{0, 2, 4, 6}
```

### Arab Bank Intersection — Node_AB

* Three approaches: West, East, South
* Three selectable green phases
* Action space:

```text
{0, 2, 4}
```

The two intersections are connected through a shared east-west corridor, meaning traffic discharged from one intersection can directly affect congestion at the other.

---

## Traffic Simulation

The network was created from **OpenStreetMap** data and converted into a SUMO network.

The simulation uses:

* **Eclipse SUMO**
* **Netedit**
* **OpenStreetMap**
* **JOSM**
* **Python**
* **TraCI**
* **E2 lane-area detectors**

The project used SUMO **1.24.0** for simulation, with Netedit versions in the **1.25.0–1.26.0** range for network editing and detector placement.

### Traffic Demand

The simulation covers six hours divided into three demand periods:

| Period  | Duration | Representative West Flow |
| ------- | -------: | -----------------------: |
| Morning |  2 hours |              2100 veh/hr |
| Midday  |  2 hours |              2760 veh/hr |
| Evening |  2 hours |              3350 veh/hr |

This increasing demand allows the controllers to be evaluated under progressively heavier traffic conditions.

---

## Traffic State Representation

Traffic conditions are measured using SUMO **E2 lane-area detectors**.

For Al-Assaf:

```text
s_main = (W, E, N, S, φ_main)
```

For Arab Bank:

```text
s_AB = (W, E, S, φ_AB)
```

where:

* `W` = West approach queue
* `E` = East approach queue
* `N` = North approach queue
* `S` = South approach queue
* `φ` = currently active traffic signal phase

The Actor-Critic architectures use semantic discretization to reduce the state-space size and allow the tabular agents to learn effectively within the available training budget.

---

## Control Algorithms

### 1. Static Baseline

The static controller represents conventional fixed-time traffic signal operation.

The signal follows the phase timings already defined in the SUMO network without receiving traffic observations or making adaptive decisions.

It is used as the primary baseline for evaluating the RL controllers.

---

### 2. Q-Learning

Q-Learning was initially implemented for single-intersection control at Al-Assaf.

The agent uses:

* Queue-based state representation
* ε-greedy action selection
* Q-table
* Temporal-difference updates
* Minimum green-time constraint

However, raw integer queue counts create a rapidly growing state space. Many states are visited too infrequently during training, resulting in slow convergence and high variance.

This motivated the transition from Q-Learning to Actor-Critic.

---

### 3. Actor-Critic

The Actor-Critic controller combines:

**Actor**

* Maintains a policy over available signal phases
* Uses a stochastic softmax policy

**Critic**

* Estimates the value of the current state
* Provides the temporal-difference learning signal

The Actor-Critic implementation uses semantic state discretization and a **20-second minimum green time** to prevent excessive signal switching.

---

## Multi-Agent Reinforcement Learning

The main contribution of the project is the comparison of four two-agent Actor-Critic architectures.

### Separated Agents — SEP

Each intersection operates independently.

```text
Agent 1 → Local State → Local Action
Agent 2 → Local State → Local Action
```

No information is exchanged between agents.

Each agent receives a reward based on its own queue.

---

### Shared State — SS

Each agent receives a compact representation of the neighboring intersection's congestion.

```text
Local State + Neighbor Congestion
              │
              ▼
            Agent
              │
              ▼
           Action
```

The neighboring congestion is discretized into semantic levels, keeping the state space bounded.

The reward remains primarily local while including a lightweight queue-imbalance penalty.

---

### Shared State and Action — SSA

SSA extends Shared State by additionally providing information about the neighboring agent's recent signal behavior.

The state includes:

* Local traffic conditions
* Neighbor congestion
* Neighbor's last selected phase
* Time since the neighbor's last phase switch

This provides richer coordination but also significantly increases the effective state space.

---

### Shared State, Action, and Reward — SSAR

SSAR provides the highest level of information sharing.

Both agents receive:

* Shared traffic information
* Neighbor action information
* A common team reward

The team reward is based on the total network queue plus a load-imbalance penalty.

This aligns both agents toward a network-wide objective, but introduces a credit-assignment problem because the intersections are structurally asymmetric.

---

## Experimental Methodology

The controllers were evaluated using the same SUMO environment and traffic demand.

The main evaluation metric is:

> **Average queue length**

Additional metrics include:

* Cumulative delay
* Episode-to-episode variance
* Best episode performance
* Worst episode performance
* Queue imbalance
* Demand-period performance
* Convergence behaviour

Each MARL architecture was trained for **100 episodes**.

---

## Results

### Single-Agent Comparison

The Actor-Critic controller significantly outperformed Q-Learning.

| Controller   | Average Queue | Improvement |
| ------------ | ------------: | ----------: |
| Static       |          71.5 |    Baseline |
| Q-Learning   |          59.6 |       16.6% |
| Actor-Critic |          40.2 |       43.8% |

Actor-Critic also reduced cumulative delay to approximately **0.93 million vehicle-seconds**, compared with approximately **1.66 million** for the static controller.

---

## MARL Results

The final comparison used a static total network queue of **109.6 vehicles**.

| Architecture                   | Best Episode | Worst Episode | Std. Dev. | Improvement vs Static |
| ------------------------------ | -----------: | ------------: | --------: | --------------------: |
| Static                         |        109.6 |             — |         — |                  0.0% |
| Separated                      |         77.2 |         135.7 |      11.8 |                +29.6% |
| **Shared State**               |     **55.9** |      **84.0** |   **3.4** |            **+49.0%** |
| Shared State + Action          |         71.1 |         145.8 |      13.6 |                +35.1% |
| Shared State + Action + Reward |         70.5 |         158.9 |      17.7 |                +35.6% |

### Best Architecture: Shared State

The **Shared State architecture** achieved the best results across every evaluated metric.

Key results:

* **49.0%** reduction in best-episode total queue compared with static control
* **45.7%** mean improvement across the 100 training episodes
* Only **3.4 vehicles** standard deviation
* Best episode: **55.9 vehicles**
* Worst episode: **84.0 vehicles**
* Every training episode remained below the static baseline
* Convergence was achieved by approximately **episode 11**

Most importantly, Shared State maintained positive improvement across both intersections and all three demand periods, including the evening peak.

---

## Why More Coordination Did Not Perform Better

One of the central findings of the project is that increasing the amount of information shared between agents can actually hurt performance when the training budget is limited.

### Shared State

Uses a compact discretized neighbor-congestion signal.

Result:

```text
Small state space
      ↓
Frequent state visits
      ↓
Reliable learning
      ↓
Fast convergence
      ↓
Stable performance
```

### SSA

Adds the neighbor's action information.

This expands the effective state space to approximately **300 combinations per agent**.

```text
More information
      ↓
Larger state space
      ↓
Fewer visits per state
      ↓
Slower convergence
      ↓
Higher variance
```

### SSAR

Adds a shared team reward.

Although this creates a common network-level objective, the two intersections are asymmetric. The larger Al-Assaf intersection can dominate the reward signal, making it difficult for the Arab Bank agent to distinguish its own contribution from the network-wide outcome.

This resulted in the highest observed variance and the worst single episode.

---

## Key Findings

The experiments demonstrate several important conclusions:

1. **Actor-Critic outperformed Q-Learning** under the project's training conditions.
2. **Tabular semantic discretization** is effective for small corridor-scale traffic networks.
3. **Some inter-agent communication is better than no communication.**
4. **More communication is not necessarily better.**
5. The **Shared State** architecture provided the best balance between:

   * Performance
   * Stability
   * Convergence speed
   * State-space complexity
6. A lightweight representation of neighboring congestion was sufficient to coordinate the two intersections.
7. Richer action sharing increased the state-space complexity without providing a corresponding performance improvement within the 100-episode training budget.
8. A shared team reward introduced a credit-assignment problem in the asymmetric two-intersection network.
9. Evening peak demand was the strongest discriminator between the architectures.

---

## Project Structure

The repository is organized around the SUMO simulation environment, traffic demand files, detectors, and Python controllers.

A typical project setup contains:

```text
.
├── *.py                     # Python controllers and experiments
│
├── khalda.net.xml           # SUMO road network
├── khalda.sumocfg           # Main SUMO configuration
├── khalda_add.xml           # E2 detector configuration
├── khalda.netecfg           # Netedit configuration
│
├── demand_morning.rou.xml   # Morning traffic demand
├── demand_midday.rou.xml    # Midday traffic demand
├── demand_evening.rou.xml   # Evening traffic demand
│
├── results/                 # Experimental results
├── plots/                   # Generated visualizations
└── README.md
```

> The exact Python filenames and repository layout depend on the version of the source code included in this repository.

---

## Requirements

The implementation requires:

* Python 3
* Eclipse SUMO
* SUMO Python tools
* TraCI
* NumPy
* Matplotlib

The simulation was developed using SUMO 1.24.0.

---

## Running the Simulation

Make sure SUMO is installed and that the SUMO tools are accessible from Python.

The main simulation configuration is:

```text
khalda.sumocfg
```

The configuration loads:

```text
khalda.net.xml
khalda_add.xml
demand_morning.rou.xml
demand_midday.rou.xml
demand_evening.rou.xml
```

The Python controller connects to SUMO through TraCI and performs the following loop:

```text
Read simulation time
       ↓
Read E2 detector queue lengths
       ↓
Read current traffic-light phase
       ↓
Construct state
       ↓
Select RL action
       ↓
Check minimum green-time constraint
       ↓
Apply signal phase
       ↓
Advance SUMO by one timestep
       ↓
Calculate reward
       ↓
Update agent
       ↓
Repeat
```

---

## Simulation Constraints

To make the experiments consistent and realistic:

* Simulation timestep: **1 second**
* Actor-Critic minimum green time: **20 seconds**
* Training episodes: **100**
* Demand periods: Morning, Midday, Evening
* Evaluation baseline: Fixed-time static controller
* Queue measurements: SUMO E2 lane-area detectors

---

## Applications

Although this project focuses on two intersections in Khalda, the methodology is not limited to this specific network.

Potential applications include:

* Adaptive traffic signal control
* Connected intersection coordination
* Smart-city traffic management
* Reinforcement learning research
* Multi-agent systems
* Traffic simulation
* Edge-based intelligent transportation systems
* Testing traffic-control strategies before real-world deployment

---

## Future Work

The project identifies several directions for future development.

### Larger Intersection Networks

Extend the two-agent system to a larger network of interconnected intersections, allowing agents to receive information from upstream and downstream neighbors.

### Improved Scalability

Investigate how the Shared State architecture behaves as the number of controlled intersections increases.

### Longer Training

Evaluate whether richer coordination mechanisms such as SSA and SSAR can outperform Shared State when provided with a substantially larger training budget.

### Advanced RL Models

Explore neural-network-based approaches for larger state spaces while preserving the stability advantages observed with compact state representations.

### Real-World Deployment

The simulation framework can serve as a foundation for eventually connecting adaptive traffic-control algorithms to real traffic infrastructure after appropriate validation and safety testing.

---

## Project Deliverables

The project includes:

* SUMO road network model
* Traffic demand and route files
* E2 lane-area detector configuration
* Static traffic signal baseline
* Q-Learning controller
* Actor-Critic controller
* Four MARL coordination architectures
* Experimental evaluation
* Performance plots
* Comparative analysis
* Final project documentation
* Simulation demonstration

---

## Authors

**Izzeldeen Alkloub**
Department of Computer Engineering
The University of Jordan

**Roaa Mousa AbdaAlqader**
Department of Computer Engineering
The University of Jordan

### Supervisor

**Dr. Ashraf Al-Suyyagh**

---

## Acknowledgements

This project was developed as a senior-year graduation project in the **Department of Computer Engineering, School of Engineering, The University of Jordan**.

Special thanks to **Dr. Ashraf Al-Suyyagh** for his supervision and guidance throughout the project.

---

## License

This project was developed as an academic graduation project.

If you intend to reuse, modify, or redistribute the source code and simulation files, please refer to the license included in this repository.

---

## References

The complete academic references used in the project are available in the final project report.

Key technologies and concepts include:

* Simulation of Urban MObility (SUMO)
* Traffic Control Interface (TraCI)
* OpenStreetMap (OSM)
* Reinforcement Learning
* Q-Learning
* Actor-Critic
* Multi-Agent Reinforcement Learning (MARL)
* Traffic Signal Control
* E2 Lane-Area Detectors
