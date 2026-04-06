import os, sys, csv, random
import numpy as np
import matplotlib.pyplot as plt
import pickle

if "SUMO_HOME" not in os.environ:
    sys.exit("Please set SUMO_HOME")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

import traci

# ---------------------------
# CONFIG
# ---------------------------
SUMO_BINARY = "sumo"
SUMO_CONFIG = "../network/khalda.sumocfg"
STEP_LENGTH = 1.0

TLS_MAIN = "Node2"
TLS_AB   = "Node_AB"

STATIC_PHASE_TIME = 90.0
MIN_GREEN = 20.0

# AC params
ACTOR_LR = 0.05
CRITIC_LR = 0.1
GAMMA = 0.9
AC_TEMPERATURE = 1.0
AC_EPSILON = 0.05

# ---------------------------
# DETECTORS (ASSAF)
# ---------------------------
def det_id(app, lane, seg):
    return f"{app}_L{lane}_S{seg}"

def arm_main(app):
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

def get_main():
    # state = (Wq, Eq, Nq, Sq, phase)
    w, e, n, s = (arm_main("W"), arm_main("E"), arm_main("N"), arm_main("S"))
    phase = traci.trafficlight.getPhase(TLS_MAIN)
    return (w, e, n, s, phase)

def total_main(s):
    return s[0]+s[1]+s[2]+s[3]

# ---------------------------
# DETECTORS (AB)
# ---------------------------
def det_ab(app,lane,seg):
    return f"AB_{app}_L{lane}_S{seg}"

def arm_ab(app):
    total=0
    if app=="W":
        for lane in range(3):
            for seg in range(3):
                if det_ab(app,lane,seg)=="AB_W_L2_S2":
                    continue
                try:
                    total+=traci.lanearea.getLastStepVehicleNumber(det_ab(app,lane,seg))
                except:
                    pass
    if app=="E":
        for lane in range(2):
            for seg in range(2):
                try:
                    total+=traci.lanearea.getLastStepVehicleNumber(det_ab(app,lane,seg))
                except:
                    pass
    if app=="S":
        for lane in range(2):
            for seg in range(3):
                try:
                    total+=traci.lanearea.getLastStepVehicleNumber(det_ab(app,lane,seg))
                except:
                    pass
    return total

def get_ab():
    return (arm_ab("W"), arm_ab("E"), arm_ab("S"),
            traci.trafficlight.getPhase(TLS_AB))

def total_ab(s):
    return s[0]+s[1]+s[2]

# ---------------------------
# SUMO
# ---------------------------
def start():
    traci.start([SUMO_BINARY,"-c",SUMO_CONFIG,"--step-length",str(STEP_LENGTH),"--start"])

def stop():
    return traci.simulation.getMinExpectedNumber()<=0

# ---------------------------
# STATIC RUN
# ---------------------------
def run_static():
    start()

    t=[]; qm=[]; qa=[]; qt=[]
    dm=[]; da=[]; dt=[]

    cdm=cda=cdt=0

    step=0
    phase_steps=int(STATIC_PHASE_TIME/STEP_LENGTH)

    while not stop():
        traci.simulationStep()
        step+=1

        s1=get_main()
        s2=get_ab()

        q1=total_main(s1)
        q2=total_ab(s2)
        q=q1+q2

        cdm+=q1*STEP_LENGTH
        cda+=q2*STEP_LENGTH
        cdt+=q*STEP_LENGTH

        t.append(traci.simulation.getTime())
        qm.append(q1); qa.append(q2); qt.append(q)
        dm.append(cdm); da.append(cda); dt.append(cdt)

    traci.close()
    return np.array(t),np.array(qm),np.array(qa),np.array(qt),np.array(dm),np.array(da),np.array(dt)
# ---------------------------
# AC RUN (2 agents)
# ---------------------------
class IntersectionAgent:
    def __init__(self, name, tls_id, phases, min_green=20.0, epsilon=0.2, temp=2.0):
        self.name = name
        self.tls_id = tls_id
        self.phases = phases

        self.min_green = min_green
        self.epsilon = epsilon
        self.temp = temp

        self.prefs = {}
        self.V = {}
        self.last_switch = 0.0

    def ensure_state(self, state):
        if state not in self.prefs:
            self.prefs[state] = np.zeros(len(self.phases))
        if state not in self.V:
            self.V[state] = 0.0

    def softmax(self, x):
        x = np.array(x, dtype=float) / max(1e-9, self.temp)
        e = np.exp(x - np.max(x))
        p = e / np.sum(e)
        p = 0.05 + 0.95 * p
        return p / np.sum(p)

    def choose_action(self, state, time):
        self.ensure_state(state)
        pi = self.softmax(self.prefs[state])

        if time - self.last_switch >= self.min_green:
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.phases)
            else:
                idx = np.random.choice(len(self.phases), p=pi)
                action = self.phases[idx]

            traci.trafficlight.setPhase(self.tls_id, int(action))
            self.last_switch = time
        else:
            action = traci.trafficlight.getPhase(self.tls_id)

        return int(action), pi

    def update(self, s, s_next, action, pi, reward):
        self.ensure_state(s_next)

        td = reward + GAMMA * self.V[s_next] - self.V[s]

        # critic
        self.V[s] += CRITIC_LR * td

        # actor
        for k in range(len(self.phases)):
            if self.phases[k] == action:
                self.prefs[s][k] += ACTOR_LR * td * (1 - pi[k])
            else:
                self.prefs[s][k] -= ACTOR_LR * td * pi[k]

        self.prefs[s] = np.clip(self.prefs[s], -5, 5)

        return td


# =========================================================
# SAVE / LOAD
# =========================================================
def save_model(path, main_agent, ab_agent):
    with open(path, "wb") as f:
        pickle.dump((main_agent.prefs, main_agent.V,
                     ab_agent.prefs, ab_agent.V), f)

def load_model(path, main_agent, ab_agent):
    if os.path.exists(path):
        with open(path, "rb") as f:
            p1, V1, p2, V2 = pickle.load(f)
        main_agent.prefs, main_agent.V = p1, V1
        ab_agent.prefs, ab_agent.V = p2, V2
        print("Loaded model")
    else:
        print("Starting fresh")


# =========================================================
# MAIN RUN FUNCTION (CONNECTED VERSION)
# =========================================================
def run_ac_connected():

    MODEL_PATH = "ac_connected.pkl"

    main_agent = IntersectionAgent("MAIN", TLS_MAIN, [0,2,4,6])
    ab_agent   = IntersectionAgent("AB",   TLS_AB,   [0,2,4])

    start()
    load_model(MODEL_PATH, main_agent, ab_agent)

    t, qm, qa, qt = [], [], [], []
    dm, da, dt = [], [], []

    cdm = cda = cdt = 0.0

    while not stop():

        time = traci.simulation.getTime()

        # =====================================================
        # LOCAL STATES
        # =====================================================
        s_main_local = get_main()
        s_ab_local   = get_ab()

        # =====================================================
        # 🔥 SHARED STATE (COMMUNICATION)
        # =====================================================
        s_main = (s_main_local, s_ab_local)
        s_ab   = (s_ab_local, s_main_local)

        # =====================================================
        # ACTIONS
        # =====================================================
        a_main, pi_main = main_agent.choose_action(s_main, time)
        a_ab,   pi_ab   = ab_agent.choose_action(s_ab, time)

        # =====================================================
        # STEP
        # =====================================================
        traci.simulationStep()

        # =====================================================
        # NEXT STATES
        # =====================================================
        s_main_next_local = get_main()
        s_ab_next_local   = get_ab()

        s_main_next = (s_main_next_local, s_ab_next_local)
        s_ab_next   = (s_ab_next_local, s_main_next_local)

        # =====================================================
        # REWARD (WITH COORDINATION)
        # =====================================================
        q_main = total_main(s_main_next_local)
        q_ab   = total_ab(s_ab_next_local)

        # 🔥 coordinated reward
        r_main = -(q_main + 0.3 * abs(q_main - q_ab))
        r_ab   = -(q_ab   + 0.3 * abs(q_ab - q_main))

        # =====================================================
        # UPDATE
        # =====================================================
        main_agent.update(s_main, s_main_next, a_main, pi_main, r_main)
        ab_agent.update(s_ab,     s_ab_next,   a_ab,   pi_ab,   r_ab)

        # =====================================================
        # METRICS
        # =====================================================
        q_total = q_main + q_ab

        cdm += q_main * STEP_LENGTH
        cda += q_ab * STEP_LENGTH
        cdt += q_total * STEP_LENGTH

        t.append(time)
        qm.append(q_main)
        qa.append(q_ab)
        qt.append(q_total)

        dm.append(cdm)
        da.append(cda)
        dt.append(cdt)

        # debug
        # print(f"t={time:.1f} MAIN={q_main} AB={q_ab}")

    traci.close()
    save_model(MODEL_PATH, main_agent, ab_agent)

    print("Connected model saved")

    return (
        np.array(t),
        np.array(qm),
        np.array(qa),
        np.array(qt),
        np.array(dm),
        np.array(da),
        np.array(dt),
    )
# MAIN
# ---------------------------
def main():

    runs = 100   # number of runs

    for i in range(runs):

        print(f"\n=== Run {i} ===")

        # =========================
        # RUN SIMULATION
        # =========================
        t_a, qm_a, qa_a, qt_a, dm_a, da_a, dt_a = run_ac_connected()

        # =========================
        # PLOT (2 SUBPLOTS IN ONE FIGURE)
        # =========================
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # ---------------------------
        # AB INTERSECTION (LEFT)
        # ---------------------------
        axs[0].plot(t_a, qa_a, label="AC")

        avg_ab = np.mean(qa_a)
        axs[0].axhline(avg_ab, linestyle='--', label=f"Avg = {avg_ab:.1f}")

        axs[0].set_title(f"AB Intersection (Run {i})")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Queue")
        axs[0].legend()
        axs[0].grid()

        # ---------------------------
        # ASSAF INTERSECTION (RIGHT)
        # ---------------------------
        axs[1].plot(t_a, qm_a, label="AC")

        avg_main = np.mean(qm_a)
        axs[1].axhline(avg_main, linestyle='--', label=f"Avg = {avg_main:.1f}")

        axs[1].set_title(f"Assaf Intersection (Run {i})")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Queue")
        axs[1].legend()
        axs[1].grid()

        # =========================
        # SAVE
        # =========================
        plt.tight_layout()

        filename = f"exp_connected/ex{i+1}.png"
        plt.savefig(filename, dpi=300)

        print(f"Saved: {filename}")

        # IMPORTANT → clear figure
        plt.clf()
if __name__ == "__main__":
    main()