# compare_static_vs_ql_ac_FULL.py
# Runs SUMO three times:
#  1) Static controller: fixed 90s per detected main green phase
#  2) Q-Learning controller
#  3) Actor-Critic controller (tabular softmax policy + tabular V critic)
# Then plots (or saves CSV if matplotlib is not available).

import os, sys, random, csv
import numpy as np

# ----------------------------
# OPTIONAL plotting (won't crash if matplotlib missing)
# ----------------------------
HAS_PLOT = True
try:
    import matplotlib.pyplot as plt
except Exception as e:
    HAS_PLOT = False
    PLOT_IMPORT_ERROR = str(e)

# ----------------------------
# SUMO / TraCI setup
# ----------------------------
if "SUMO_HOME" not in os.environ:
    sys.exit("Please set SUMO_HOME environment variable first (export SUMO_HOME=...).")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

import traci

# ----------------------------
# EDIT THESE FOR YOUR PROJECT
# ----------------------------
SUMO_BINARY = "sumo-gui"         # or "sumo" (faster)
SUMO_CONFIG = "../network/khalda.sumocfg"   # your .sumocfg path
TLS_ID = "Node2"                # your traffic light id
STEP_LENGTH = 0.1               # must match SUMO command
MAX_SIM_TIME = None             # seconds; set None to stop when no vehicles

# Static timing
STATIC_PHASE_TIME = 90.0        # seconds per phase (green)

# Q-Learning hyperparams
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.15                  # exploration
MIN_GREEN = 5.0                 # seconds minimum before changing phase

# Actor-Critic hyperparams
ACTOR_LR = 0.01                 # policy learning rate
CRITIC_LR = 0.05                 # value learning rate
AC_TEMPERATURE = 0.5            # softmax temperature (lower => more greedy)
AC_EPSILON = 0.05               # optional exploration (keeps it from getting stuck)

# Output files (in case matplotlib isn't installed)
OUT_STATIC_CSV = "static_run.csv"
OUT_QL_CSV = "ql_run.csv"
OUT_AC_CSV = "ac_run.csv"

# ----------------------------
# Your detector logic
# ----------------------------
APPROACHES = ["W", "E", "N", "S"]

def det_id(app, lane, seg):
    return f"{app}_L{lane}_S{seg}"

def arm_sum(app):
    total = 0

    if app == "W":
        for lane in range(3):
            for seg in range(2):
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app, lane, seg))
        total += traci.lanearea.getLastStepVehicleNumber("W_add")
        return total

    if app == "N":
        for lane in range(3):
            for seg in range(2):
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app, lane, seg))
        total += traci.lanearea.getLastStepVehicleNumber("N_add")
        return total

    if app == "E":
        for lane in range(4):
            for seg in range(3):
                if det_id(app, lane, seg) == "E_L3_S2":
                    continue
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app, lane, seg))
        return total

    if app == "S":
        for lane in range(2):
            for seg in range(2):
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app, lane, seg))
        total += traci.lanearea.getLastStepVehicleNumber("S_add")
        return total

    return total

def get_state():
    # state = (Wq, Eq, Nq, Sq, phase)
    w, e, n, s = (arm_sum("W"), arm_sum("E"), arm_sum("N"), arm_sum("S"))
    phase = traci.trafficlight.getPhase(TLS_ID)
    return (w, e, n, s, phase)

def total_queue_from_state(s):
    return s[0] + s[1] + s[2] + s[3]

# ----------------------------
# SUMO start/stop helpers
# ----------------------------
def start_sumo():
    cmd = [
        SUMO_BINARY, "-c", SUMO_CONFIG,
        "--step-length", str(STEP_LENGTH),
        "--start"
    ]
    traci.start(cmd)

def stop_condition():
    if MAX_SIM_TIME is not None:
        return traci.simulation.getTime() >= MAX_SIM_TIME
    return traci.simulation.getMinExpectedNumber() <= 0

# ----------------------------
# Phase detection: auto detect main phases per approach
# ----------------------------
def approach_from_lane_id(lane_id: str):
    if not lane_id:
        return None
    for a in APPROACHES:
        if lane_id.startswith(a + "_"):
            return a
        if lane_id.startswith(a):
            return a
    return None

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
                app = approach_from_lane_id(inLane)
                if app is not None:
                    counts[app] += 1

        phase_scores.append(counts)

    main = {}
    used = set()
    for app in APPROACHES:
        ranked = sorted(
            [(p_idx, phase_scores[p_idx][app]) for p_idx in range(len(phases))],
            key=lambda x: x[1],
            reverse=True
        )
        chosen = None
        for p_idx, score in ranked:
            if score <= 0:
                continue
            if p_idx not in used:
                chosen = p_idx
                break
        if chosen is None and ranked and ranked[0][1] > 0:
            chosen = ranked[0][0]
        if chosen is None:
            chosen = 0
        main[app] = chosen
        used.add(chosen)

    return main, phase_scores

# def print_phase_debug(main_phases, phase_scores):
#     prog = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
#     print("\n=== TLS Phase Debug ===")
#     print("TLS_ID:", TLS_ID)
#     print("Total phases:", len(prog.phases))
#     print("Detected main phases:", main_phases)
#     for i, ph in enumerate(prog.phases):
#         print(f"Phase {i:2d}  dur={ph.duration:>5}  greens={phase_scores[i]}  state={ph.state}")
#     print("=== End Debug ===\n")

# ----------------------------
# Run 1: Static controller
# ----------------------------
def run_static(main_phase_order):
    start_sumo()

    main_map, phase_scores = detect_main_green_phases()
    print_phase_debug(main_map, phase_scores)

    cycle = [main_map[a] for a in main_phase_order]
    phase_steps = int(round(STATIC_PHASE_TIME / STEP_LENGTH))

    t_list, q_list, cum_delay_list = [], [], []
    cum_delay = 0.0
    step = 0
    idx = 0

    traci.trafficlight.setPhase(TLS_ID, cycle[idx])
    last_switch_step = 0

    while not stop_condition():
        if step > 0 and (step - last_switch_step) >= phase_steps:
            idx = (idx + 1) % len(cycle)
            traci.trafficlight.setPhase(TLS_ID, cycle[idx])
            last_switch_step = step

        traci.simulationStep()
        step += 1

        s = get_state()
        tq = total_queue_from_state(s)
        cum_delay += tq * STEP_LENGTH

        t_list.append(traci.simulation.getTime())
        q_list.append(tq)
        cum_delay_list.append(cum_delay)

    traci.close()
    return np.array(t_list), np.array(q_list), np.array(cum_delay_list)

# ----------------------------
# Run 2: Q-Learning controller
# ----------------------------
def run_ql(main_phase_order):
    start_sumo()

    main_map, phase_scores = detect_main_green_phases()
    # print_phase_debug(main_map, phase_scores)

    actions = [main_map[a] for a in main_phase_order]
    n_actions = len(actions)

    min_green_steps = int(round(MIN_GREEN / STEP_LENGTH))
    t_list, q_list, cum_delay_list = [], [], []

    Q = {}  # state -> Q-values

    def ensure(s):
        if s not in Q:
            Q[s] = np.zeros(n_actions, dtype=float)

    def choose_action(s):
        ensure(s)
        if random.random() < EPSILON:
            return random.randrange(n_actions)
        return int(np.argmax(Q[s]))

    def reward(s_new):
        return -float(total_queue_from_state(s_new))

    cum_delay = 0.0
    step = 0

    traci.trafficlight.setPhase(TLS_ID, actions[0])
    last_switch_step = 0

    while not stop_condition():
        s = get_state()
        ensure(s)

        a_idx = choose_action(s)

        if step > 0 and (step - last_switch_step) >= min_green_steps:
            traci.trafficlight.setPhase(TLS_ID, actions[a_idx])
            last_switch_step = step

        traci.simulationStep()
        step += 1

        s2 = get_state()
        ensure(s2)
        r = reward(s2)

        best_future = np.max(Q[s2])
        Q[s][a_idx] = Q[s][a_idx] + ALPHA * (r + GAMMA * best_future - Q[s][a_idx])

        tq = total_queue_from_state(s2)
        cum_delay += tq * STEP_LENGTH

        t_list.append(traci.simulation.getTime())
        q_list.append(tq)
        cum_delay_list.append(cum_delay)

    traci.close()
    return np.array(t_list), np.array(q_list), np.array(cum_delay_list)

# ----------------------------
# Run 3: Actor-Critic controller (tabular)
# Actor: softmax policy π(a|s) with preferences
# Critic: V(s)
# Update using TD error: δ = r + γV(s') - V(s)
# ----------------------------
def run_actor_critic(main_phase_order):
    start_sumo()

    main_map, phase_scores = detect_main_green_phases()
    # print_phase_debug(main_map, phase_scores)

    actions = [main_map[a] for a in main_phase_order]
    n_actions = len(actions)

    min_green_steps = int(round(MIN_GREEN / STEP_LENGTH))
    t_list, q_list, cum_delay_list = [], [], []

    # Actor params: prefs[s] = preferences (one per action)
    prefs = {}
    # Critic params: V[s] = value estimate
    V = {}

    def ensure(s):
        if s not in prefs:
            prefs[s] = np.zeros(n_actions, dtype=float)
        if s not in V:
            V[s] = 0.0

    def softmax(x):
        # stable softmax with temperature
        x = np.array(x, dtype=float) / max(1e-9, AC_TEMPERATURE)
        m = np.max(x)
        e = np.exp(x - m)
        return e / np.sum(e)

    def choose_action(s):
        ensure(s)
        if random.random() < AC_EPSILON:
            return random.randrange(n_actions)
        pi = softmax(prefs[s])
        # sample from policy
        return int(np.random.choice(np.arange(n_actions), p=pi))

    def reward(s_new):
        return -float(total_queue_from_state(s_new))

    cum_delay = 0.0
    step = 0

    # init phase
    traci.trafficlight.setPhase(TLS_ID, actions[0])
    last_switch_step = 0

    while not stop_condition():
        s = get_state()
        ensure(s)

        a_idx = choose_action(s)

        # apply action respecting MIN_GREEN
        if step > 0 and (step - last_switch_step) >= min_green_steps:
            traci.trafficlight.setPhase(TLS_ID, actions[a_idx])
            last_switch_step = step

        traci.simulationStep()
        step += 1

        s2 = get_state()
        ensure(s2)
        r = reward(s2)

        # TD error
        delta = r + GAMMA * V[s2] - V[s]

        # Critic update
        V[s] = V[s] + CRITIC_LR * delta

        # Actor update (policy gradient with baseline)
        pi = softmax(prefs[s])

        # Update preferences
        # prefs[a] += lr * delta * (1 - pi[a]) for chosen action
        # prefs[other] -= lr * delta * pi[other]
        for k in range(n_actions):
            if k == a_idx:
                prefs[s][k] += ACTOR_LR * delta * (1.0 - pi[k])
            else:
                prefs[s][k] -= ACTOR_LR * delta * (pi[k])

        tq = total_queue_from_state(s2)
        cum_delay += tq * STEP_LENGTH

        t_list.append(traci.simulation.getTime())
        q_list.append(tq)
        cum_delay_list.append(cum_delay)

    traci.close()
    return np.array(t_list), np.array(q_list), np.array(cum_delay_list)

# ----------------------------
# Save CSV
# ----------------------------
def save_csv(path, t, q, d):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "total_queue", "cum_delay_proxy"])
        for i in range(len(t)):
            w.writerow([float(t[i]), float(q[i]), float(d[i])])

# ----------------------------
# Plot comparison
# ----------------------------
def plot_results(t_static, q_static, d_static, t_ql, q_ql, d_ql, t_ac, q_ac, d_ac):
    if not HAS_PLOT:
        print("\nmatplotlib is not available on your machine right now.")
        print("Reason:", PLOT_IMPORT_ERROR)
        print(f"Saved CSV instead: {OUT_STATIC_CSV}, {OUT_QL_CSV}, {OUT_AC_CSV}")
        return

    plt.figure()
    plt.plot(t_static, q_static, label="Static 90s/phase")
    plt.plot(t_ql, q_ql, label="Q-Learning")
    plt.plot(t_ac, q_ac, label="Actor-Critic")
    plt.xlabel("Simulation time (s)")
    plt.ylabel("Total queue (vehicles)")
    plt.title("Total Queue vs Time")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(t_static, d_static, label="Static 90s/phase")
    plt.plot(t_ql, d_ql, label="Q-Learning")
    plt.plot(t_ac, d_ac, label="Actor-Critic")
    plt.xlabel("Simulation time (s)")
    plt.ylabel("Cumulative queue-area (veh·s)  (delay proxy)")
    plt.title("Cumulative Delay Proxy vs Time")
    plt.legend()
    plt.grid(True)

    plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    MAIN_ORDER = ["W", "E", "N", "S"]

    print("=== Running Actor-Critic only ===")
    t_a, q_a, d_a = run_actor_critic(MAIN_ORDER)
    save_csv(OUT_AC_CSV, t_a, q_a, d_a)

    print("\n--- Summary ---")
    print(f"Actor-Critic final delay proxy: {d_a[-1]:.2f}")
    print(f"CSV saved: {OUT_AC_CSV}")

    if HAS_PLOT:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t_a, q_a, label="Actor-Critic")
        plt.xlabel("Simulation time (s)")
        plt.ylabel("Total queue (vehicles)")
        plt.title("Actor-Critic Queue vs Time")
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.plot(t_a, d_a, label="Actor-Critic")
        plt.xlabel("Simulation time (s)")
        plt.ylabel("Cumulative queue-area (veh·s)")
        plt.title("Actor-Critic Delay Proxy vs Time")
        plt.legend()
        plt.grid(True)

        plt.show()

if __name__ == "__main__":
    main()