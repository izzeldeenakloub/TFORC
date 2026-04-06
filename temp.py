import os , sys , random , csv
import numpy as np

import matplotlib.pyplot as plt

# SUMO / TRACI setup
tools = os.path.join(os.environ["SUMO_HOME"] , "tools")
sys.path.append(tools)

import traci

SUMO_BINARY = "sumo"
SUMO_CONFIG = "network/khalda.sumocfg"
TLS_ID = "Node2"
STEP_LENGTH = 0.1
MAX_SIM_TIME = None
MIN_GREEN = 5.0

# ACTOR-Critic hyperparams
GAMMA = 0.95
ACTOR_LR = 0.01  # ploicy learning rate
CRITIC_LR = 0.05 # value learning rate
AC_TEMPERATURE = 0.5 # softmax temperature 
AC_EPSILON = 0.05 # optional exploration

APPROACHES = ["W", "E", "N", "S"]

# output files
OUT_AC_CSV = "ac_run.csv"
#-------------------
#  ###############detector logic############################ 
# it's specific to the lane area detector spread over our map
#-------------------
def det_id(app , lane , seg):
    return f"{app}_L{lane}_S{seg}"

def arm_sum(app):
    total = 0

    if app  == "W":
        for lane in range(3):
            for seg in range(2):
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app,lane,seg))
        total +=traci.lanearea.getLastStepVehicleNumber("W_add")
        return total

    if app == "N":
        for lane in range(3):
            for seg in range(2):
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app,lane,seg))
        total +=traci.lanearea.getLastStepVehicleNumber("N_add")
        return total

    if app == "E":
        for lane in range(4):
            for seg in range(3):
                if det_id(app, lane , seg) == "E_L3_S2":
                    continue
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app,lane,seg))
        return total

    if app == "S":
        for lane in range(2):
            for seg in range(2):
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app,lane,seg))
        total +=traci.lanearea.getLastStepVehicleNumber("S_add")
        return total

    return total

# state of the current phase and the # of cars on each arm 
def get_state():
    w, e, n, s = (arm_sum("W"),arm_sum("E"),arm_sum("N"),arm_sum("S"))
    phase = traci.trafficlight.getPhase(TLS_ID)
    return (w , e , n , s , phase)
def total_queue_from_state(s):
    return s[0] + s[1] + s[2] + s[3]

def start_sumo():
    cmd = [
        SUMO_BINARY , "-c" , SUMO_CONFIG,
        "--step-length" , str(STEP_LENGTH),
        "--start"
    ]
    traci.start(cmd)

def stop_condition():
    if MAX_SIM_TIME is not None:
        return traci.simulation.getTime() >= MAX_SIM_TIME
    return traci.simulation.getMinExpectedNumber() <= 0

def approach_from_lane_id(lane_id):
    if not lane_id:
        return None
    for a in APPROACHES:
        if lane_id.startswith(a + "_"):
            return a
        if lane_id.startswith(a):
            return a
    return None

def run_actor_critic(main_phase_order):
    start_sumo()

    main_map = {
    "W": 0,
    "E": 2,
    "N": 4,
    "S": 6
    }

    actions = [main_map[a] for a in main_phase_order]
    n_actions = len(actions)

    min_green_steps = int(round(MIN_GREEN/STEP_LENGTH))
    t_list , q_list , cum_delay_list , state_list =[], [] , [] ,[]

    prefs = {}
    V = {}

    def ensure(s):
        if s not in prefs:
            prefs[s] = np.zeros(n_actions , dtype=float)
        if s not in V:
            V[s] = 0.0
    
    def softmax(x):
        x = np.array(x , dtype=float) / AC_TEMPERATURE
        m = np.max(x)
        e = np.exp(x-m)
        return e/np.sum(e)

    def choose_action(s):
        ensure(s)
        if random.random() < AC_EPSILON:
            return random.randrange(n_actions)
        pi = softmax(prefs[s])
        return int(np.random.choice(np.arange(n_actions),p = pi))

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
            traci.trafficlight.setPhase(TLS_ID , actions[a_idx])
            last_switch_step = step
        
        traci.simulationStep()
        step += 1

        s2 = get_state()
        ensure(s2)
        r = reward(s2)

        delta = r + GAMMA * V[s2] - V[s]

        V[s] = V[s] + CRITIC_LR * delta

        pi = softmax(prefs[s])

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
        state_list.append(s2)

    traci.close()
    return np.array(t_list), np.array(q_list) , np.array(cum_delay_list) , np.array(state_list)


# ----------------------------
# Save CSV
# ----------------------------
def save_csv(path, t, q, d, states):
    with open(path, "w") as f:

        header = (
            f"{'time':>8} "
            f"{'W':>5} "
            f"{'E':>5} "
            f"{'N':>5} "
            f"{'S':>5} "
            f"{'phase':>6} "
            f"{'queue':>8} "
            f"{'delay':>10}\n"
        )

        f.write(header)
        f.write("-"*60 + "\n")

        for i in range(len(t)):
            W,E,N,S,phase = states[i]

            line = (
                f"{int(t[i]):8d} "
                f"{int(W):5d} "
                f"{int(E):5d} "
                f"{int(N):5d} "
                f"{int(S):5d} "
                f"{int(phase):6d} "
                f"{int(q[i]):8d} "
                f"{int(d[i]):10d}\n"
            )

            f.write(line)

# ----------------------------
# Plot comparison
# ----------------------------

# ----------------------------
# Main
# ----------------------------
def main():
    MAIN_ORDER = ["W", "E", "N", "S"]

    print("=== Running Actor-Critic only ===")
    t_a, q_a, d_a  , s_a= run_actor_critic(MAIN_ORDER)
    save_csv(OUT_AC_CSV, t_a, q_a, d_a , s_a)

    print("\n--- Summary ---")
    print(f"Actor-Critic final delay proxy: {d_a[-1]:.2f}")
    print(f"CSV saved: {OUT_AC_CSV}")

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
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            traci.close()
        except:
            pass


