import os
import sys
import csv
import numpy as np
import random
import itertools

# -----------------------------
# SUMO Setup
# -----------------------------
if "SUMO_HOME" not in os.environ:
    sys.exit("Please set SUMO_HOME first.")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

import traci

# -----------------------------
# CONFIG
# -----------------------------
SUMO_BINARY = "sumo"   # use sumo-gui if you want visualization
SUMO_CONFIG = "../network/khalda.sumocfg"
TLS_ID = "Node2"
STEP_LENGTH = 0.1
MIN_GREEN = 5.0

APPROACHES = ["W", "E", "N", "S"]

# -----------------------------
# SEARCH SPACE (EDIT HERE)
# -----------------------------
GAMMAS = [0.8, 0.9, 0.95]
ACTOR_LRS = [0.01, 0.05, 0.1]
CRITIC_LRS = [0.05, 0.1]
TEMPERATURES = [0.5, 1.0]
EPSILONS = [0.01, 0.05]

RUNS_PER_CONFIG = 2   # increase for more stable results

# -----------------------------
# Detector Logic
# -----------------------------
def det_id(app, lane, seg):
    return f"{app}_L{lane}_S{seg}"

def arm_sum(app):
    total = 0

    if app == "W":
        for lane in range(3):
            for seg in range(2):
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app, lane, seg))
        total += traci.lanearea.getLastStepVehicleNumber("W_add")

    elif app == "N":
        for lane in range(3):
            for seg in range(2):
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app, lane, seg))
        total += traci.lanearea.getLastStepVehicleNumber("N_add")

    elif app == "E":
        for lane in range(4):
            for seg in range(3):
                if det_id(app, lane, seg) == "E_L3_S2":
                    continue
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app, lane, seg))

    elif app == "S":
        for lane in range(2):
            for seg in range(2):
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app, lane, seg))
        total += traci.lanearea.getLastStepVehicleNumber("S_add")

    return total

def get_state():
    w = arm_sum("W")
    e = arm_sum("E")
    n = arm_sum("N")
    s = arm_sum("S")
    phase = traci.trafficlight.getPhase(TLS_ID)
    return (w, e, n, s, phase)

def total_queue(s):
    return s[0] + s[1] + s[2] + s[3]

# -----------------------------
# Phase Detection
# -----------------------------
def detect_main_green_phases():
    prog = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
    phases = prog.phases
    controlled_links = traci.trafficlight.getControlledLinks(TLS_ID)

    phase_scores = []

    for p_idx, ph in enumerate(phases):
        state = ph.state
        counts = {a: 0 for a in APPROACHES}

        for sig_i, sig_char in enumerate(state):
            if sig_char not in ("G", "g"):
                continue
            if sig_i >= len(controlled_links) or controlled_links[sig_i] is None:
                continue

            for conn in controlled_links[sig_i]:
                inLane = conn[0]
                for a in APPROACHES:
                    if inLane.startswith(a):
                        counts[a] += 1

        phase_scores.append(counts)

    main = {}
    used = set()

    for app in APPROACHES:
        ranked = sorted(
            [(i, phase_scores[i][app]) for i in range(len(phases))],
            key=lambda x: x[1],
            reverse=True
        )

        for idx, score in ranked:
            if score > 0 and idx not in used:
                main[app] = idx
                used.add(idx)
                break

    return main

# -----------------------------
# Actor-Critic Agent
# -----------------------------
class TabularActorCritic:

    def __init__(self, n_actions, gamma, actor_lr, critic_lr, temperature, epsilon):
        self.n_actions = n_actions
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.temperature = temperature
        self.epsilon = epsilon
        self.prefs = {}
        self.V = {}

    def ensure(self, s):
        if s not in self.prefs:
            self.prefs[s] = np.zeros(self.n_actions)
        if s not in self.V:
            self.V[s] = 0.0

    def softmax(self, x):
        x = np.array(x) / max(1e-9, self.temperature)
        m = np.max(x)
        e = np.exp(x - m)
        return e / np.sum(e)

    def choose_action(self, s):
        self.ensure(s)

        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        pi = self.softmax(self.prefs[s])
        return int(np.random.choice(np.arange(self.n_actions), p=pi))

    def update(self, s, a, r, s2):
        self.ensure(s)
        self.ensure(s2)

        delta = r + self.gamma * self.V[s2] - self.V[s]
        self.V[s] += self.critic_lr * delta

        pi = self.softmax(self.prefs[s])

        for k in range(self.n_actions):
            if k == a:
                self.prefs[s][k] += self.actor_lr * delta * (1 - pi[k])
            else:
                self.prefs[s][k] -= self.actor_lr * delta * pi[k]

# -----------------------------
# Single AC Run
# -----------------------------
def run_ac(gamma, actor_lr, critic_lr, temp, eps):

    traci.start([
        SUMO_BINARY,
        "-c", SUMO_CONFIG,
        "--step-length", str(STEP_LENGTH),
        "--start"
    ])

    main_map = detect_main_green_phases()
    MAIN_ORDER = ["W", "E", "N", "S"]
    actions = [main_map[a] for a in MAIN_ORDER]

    agent = TabularActorCritic(
        n_actions=len(actions),
        gamma=gamma,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        temperature=temp,
        epsilon=eps
    )

    min_green_steps = int(MIN_GREEN / STEP_LENGTH)
    step = 0
    last_switch = 0
    cum_delay = 0

    traci.trafficlight.setPhase(TLS_ID, actions[0])

    while traci.simulation.getMinExpectedNumber() > 0:

        s = get_state()
        a_idx = agent.choose_action(s)

        if step > 0 and (step - last_switch) >= min_green_steps:
            traci.trafficlight.setPhase(TLS_ID, actions[a_idx])
            last_switch = step

        traci.simulationStep()
        step += 1

        s2 = get_state()
        r = -float(total_queue(s2))

        agent.update(s, a_idx, r, s2)

        cum_delay += total_queue(s2) * STEP_LENGTH

    traci.close()
    return cum_delay

# -----------------------------
# Tuning Loop
# -----------------------------
def tune():

    results = []

    for gamma, actor_lr, critic_lr, temp, eps in itertools.product(
            GAMMAS, ACTOR_LRS, CRITIC_LRS, TEMPERATURES, EPSILONS):

        print("\nTesting:", gamma, actor_lr, critic_lr, temp, eps)

        delays = []
        for _ in range(RUNS_PER_CONFIG):
            delay = run_ac(gamma, actor_lr, critic_lr, temp, eps)
            delays.append(delay)

        avg_delay = sum(delays) / len(delays)

        print("Average delay:", avg_delay)

        results.append([gamma, actor_lr, critic_lr, temp, eps, avg_delay])

    with open("ac_tuning_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma", "actor_lr", "critic_lr", "temperature", "epsilon", "avg_delay"])
        writer.writerows(results)

    print("\nTuning complete.")
    print("Best configuration:")

    best = min(results, key=lambda x: x[-1])
    print(best)


if __name__ == "__main__":
    tune()