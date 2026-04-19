import os
import sys
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
SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "../network/khalda.sumocfg"
STEP_LENGTH = 1.0

TLS_MAIN = "Node2"
TLS_AB   = "Node_AB"

STATIC_PHASE_TIME = 90.0
MIN_GREEN = 20.0

# AC params
ACTOR_LR = 0.01
CRITIC_LR = 0.05
GAMMA = 0.9
AC_TEMPERATURE = 1.0
AC_EPSILON = 0.05

MODEL_PATH = "ac_model_separated.pkl"
OUT_DIR = "exp_separated"

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
    w = arm_main("W")
    e = arm_main("E")
    n = arm_main("N")
    s = arm_main("S")
    phase = traci.trafficlight.getPhase(TLS_MAIN)
    return (w, e, n, s, phase)

def total_main(state):
    return state[0] + state[1] + state[2] + state[3]

# ---------------------------
# DETECTORS (AB)
# ---------------------------
def det_ab(app, lane, seg):
    return f"AB_{app}_L{lane}_S{seg}"

def arm_ab(app):
    total = 0

    if app == "W":
        for lane in range(3):
            for seg in range(3):
                if det_ab(app, lane, seg) == "AB_W_L2_S2":
                    continue
                try:
                    total += traci.lanearea.getLastStepVehicleNumber(det_ab(app, lane, seg))
                except:
                    pass

    if app == "E":
        for lane in range(2):
            for seg in range(2):
                try:
                    total += traci.lanearea.getLastStepVehicleNumber(det_ab(app, lane, seg))
                except:
                    pass

    if app == "S":
        for lane in range(2):
            for seg in range(3):
                try:
                    total += traci.lanearea.getLastStepVehicleNumber(det_ab(app, lane, seg))
                except:
                    pass

    return total

def get_ab():
    return (
        arm_ab("W"),
        arm_ab("E"),
        arm_ab("S"),
        traci.trafficlight.getPhase(TLS_AB),
    )

def total_ab(state):
    return state[0] + state[1] + state[2]

# ---------------------------
# SUMO
# ---------------------------
def start():
    traci.start([
        SUMO_BINARY,
        "-c", SUMO_CONFIG,
        "--step-length", str(STEP_LENGTH),
        "--start"
    ])

def stop():
    return traci.simulation.getMinExpectedNumber() <= 0

# ---------------------------
# STATIC RUN
# ---------------------------
def run_static():
    start()

    t, qm, qa, qt = [], [], [], []
    dm, da, dt = [], [], []

    cdm = 0.0
    cda = 0.0
    cdt = 0.0

    while not stop():
        traci.simulationStep()

        s1 = get_main()
        s2 = get_ab()

        q1 = total_main(s1)
        q2 = total_ab(s2)
        q = q1 + q2

        cdm += q1 * STEP_LENGTH
        cda += q2 * STEP_LENGTH
        cdt += q * STEP_LENGTH

        t.append(traci.simulation.getTime())
        qm.append(q1)
        qa.append(q2)
        qt.append(q)

        dm.append(cdm)
        da.append(cda)
        dt.append(cdt)

    traci.close()

    return (
        np.array(t),
        np.array(qm),
        np.array(qa),
        np.array(qt),
        np.array(dm),
        np.array(da),
        np.array(dt),
    )

# ---------------------------
# AC RUN (2 separated agents)
# ---------------------------
class IntersectionAgent:
    def __init__(self, name, tls_id, phases, min_green=20.0, epsilon=0.05, temp=1.0):
        self.name = name
        self.tls_id = tls_id
        self.phases = phases
        self.min_green = min_green
        self.epsilon = epsilon
        self.temp = temp

        self.prefs = {}
        self.V = {}
        self.last_switch = -min_green  # allow immediate action at t=0

    def ensure_state(self, state):
        if state not in self.prefs:
            self.prefs[state] = np.zeros(len(self.phases), dtype=float)
        if state not in self.V:
            self.V[state] = 0.0

    def softmax(self, x):
        x = np.array(x, dtype=float) / max(1e-9, self.temp)
        e = np.exp(x - np.max(x))
        p = e / np.sum(e)
        p = 0.05 + 0.95 * p
        return p / np.sum(p)

    def choose_action(self, state, current_time):
        self.ensure_state(state)
        pi = self.softmax(self.prefs[state])

        if current_time - self.last_switch >= self.min_green:
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.phases)
            else:
                idx = np.random.choice(len(self.phases), p=pi)
                action = self.phases[idx]

            traci.trafficlight.setPhase(self.tls_id, int(action))
            self.last_switch = current_time
        else:
            action = traci.trafficlight.getPhase(self.tls_id)

        return int(action), pi

    def update(self, state, next_state, action, pi, reward, gamma, actor_lr, critic_lr):
        self.ensure_state(state)
        self.ensure_state(next_state)

        td_error = reward + gamma * self.V[next_state] - self.V[state]

        self.V[state] += critic_lr * td_error

        for k in range(len(self.phases)):
            phase_k = self.phases[k]
            if phase_k == action:
                self.prefs[state][k] += actor_lr * td_error * (1.0 - pi[k])
            else:
                self.prefs[state][k] -= actor_lr * td_error * pi[k]

        self.prefs[state] = np.clip(self.prefs[state], -3.0, 3.0)
        return td_error

    def export_model(self):
        return {
            "prefs": self.prefs,
            "V": self.V,
        }

    def load_model(self, data):
        self.prefs = data.get("prefs", {})
        self.V = data.get("V", {})
        self.last_switch = -self.min_green

def save_agents(model_path, main_agent, ab_agent):
    data = {
        "main_agent": main_agent.export_model(),
        "ab_agent": ab_agent.export_model(),
    }
    with open(model_path, "wb") as f:
        pickle.dump(data, f)

def load_agents(model_path, main_agent, ab_agent):
    if not os.path.exists(model_path):
        print("Starting fresh model")
        return

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    if "main_agent" in data:
        main_agent.load_model(data["main_agent"])
    if "ab_agent" in data:
        ab_agent.load_model(data["ab_agent"])

    print("Loaded existing model")

def run_ac_separated(train=True):

    main_agent = IntersectionAgent(
        name="MAIN",
        tls_id=TLS_MAIN,
        phases=[0, 2, 4, 6],
        min_green=MIN_GREEN,
        epsilon=AC_EPSILON if train else 0.0,
        temp=AC_TEMPERATURE,
    )

    ab_agent = IntersectionAgent(
        name="AB",
        tls_id=TLS_AB,
        phases=[0, 2, 4],
        min_green=MIN_GREEN,
        epsilon=AC_EPSILON if train else 0.0,
        temp=AC_TEMPERATURE,
    )

    start()
    load_agents(MODEL_PATH, main_agent, ab_agent)

    main_agent.last_switch = -main_agent.min_green
    ab_agent.last_switch   = -ab_agent.min_green

    # totals
    t, qm, qa, qt = [], [], [], []

    # delays
    dm, da, dt = [], [], []

    # MAIN arms
    mw, me, mn, ms = [], [], [], []

    # AB arms
    aw, ae, ass = [], [], []

    cdm = cda = cdt = 0.0

    while not stop():

        current_time = traci.simulation.getTime()

        s_main = get_main()   # [W,E,N,S]
        s_ab   = get_ab()     # [W,E,S]

        a_main, pi_main = main_agent.choose_action(s_main, current_time)
        a_ab, pi_ab     = ab_agent.choose_action(s_ab, current_time)

        traci.simulationStep()

        s_main_next = get_main()
        s_ab_next   = get_ab()

        q_main = total_main(s_main_next)
        q_ab   = total_ab(s_ab_next)
        q_total = q_main + q_ab

        r_main = -q_main
        r_ab   = -q_ab

        if train:

            main_agent.update(
                state=s_main,
                next_state=s_main_next,
                action=a_main,
                pi=pi_main,
                reward=r_main,
                gamma=GAMMA,
                actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR,
            )

            ab_agent.update(
                state=s_ab,
                next_state=s_ab_next,
                action=a_ab,
                pi=pi_ab,
                reward=r_ab,
                gamma=GAMMA,
                actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR,
            )

        # totals
        cdm += q_main * STEP_LENGTH
        cda += q_ab   * STEP_LENGTH
        cdt += q_total * STEP_LENGTH

        t.append(current_time)
        qm.append(q_main)
        qa.append(q_ab)
        qt.append(q_total)

        dm.append(cdm)
        da.append(cda)
        dt.append(cdt)

        # MAIN arms
        mw.append(s_main_next[0])
        me.append(s_main_next[1])
        mn.append(s_main_next[2])
        ms.append(s_main_next[3])

        # AB arms
        aw.append(s_ab_next[0])
        ae.append(s_ab_next[1])
        ass.append(s_ab_next[2])

    traci.close()

    if train:
        save_agents(MODEL_PATH, main_agent, ab_agent)
        print("Separated model saved")

    return (
        np.array(t),
        np.array(qm),
        np.array(qa),
        np.array(qt),

        np.array(mw),
        np.array(me),
        np.array(mn),
        np.array(ms),

        np.array(aw),
        np.array(ae),
        np.array(ass),

        np.array(dm),
        np.array(da),
        np.array(dt),
    )


# ==========================================================
# MAIN
# ==========================================================
def main():

    os.makedirs("exp_separated_HD", exist_ok=True)

    runs = 100

    for i in range(runs):

        print(f"=== Run {i+1} ===")

        # ===============================================
        # RUN AC
        # ===============================================
        (
            t_a,
            qm_a, qa_a, qt_a,

            mw, me, mn, ms,
            aw, ae, ass,

            dm_a, da_a, dt_a

        ) = run_ac_separated(train=True)

        # ===============================================
        # RUN STATIC
        # ===============================================
        t_s, qm_s, qa_s, qt_s, dm_s, da_s, dt_s = run_static()

        # ===============================================
        # SUM OF ARMS
        # ===============================================
        main_sum = mw + me + mn + ms
        ab_sum   = aw + ae + ass

        # ===============================================
        # FIGURE LIKE YOUR IMAGE
        # ===============================================
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # ------------------------------------------------
        # AB INTERSECTION
        # ------------------------------------------------
        axs[0].plot(t_a, ab_sum, label="AC")
        axs[0].plot(t_s, qa_s, label="Static")

        avg_ac_ab = np.mean(ab_sum)
        avg_st_ab = np.mean(qa_s)

        axs[0].axhline(avg_ac_ab, linestyle='--',
                       label=f"AC Avg = {avg_ac_ab:.1f}")

        axs[0].axhline(avg_st_ab, linestyle='--',
                       label=f"Static Avg = {avg_st_ab:.1f}")

        axs[0].set_title("AB Intersection")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Queue")
        axs[0].grid()
        axs[0].legend()

        # ------------------------------------------------
        # MAIN INTERSECTION
        # ------------------------------------------------
        axs[1].plot(t_a, main_sum, label="AC")
        axs[1].plot(t_s, qm_s, label="Static")

        avg_ac_main = np.mean(main_sum)
        avg_st_main = np.mean(qm_s)

        axs[1].axhline(avg_ac_main, linestyle='--',
                       label=f"AC Avg = {avg_ac_main:.1f}")

        axs[1].axhline(avg_st_main, linestyle='--',
                       label=f"Static Avg = {avg_st_main:.1f}")

        axs[1].set_title("Assaf Intersection")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Queue")
        axs[1].grid()
        axs[1].legend()

        plt.tight_layout()

        plt.savefig(
            f"exp_separated_HD/sum_compare_run_{i+1}.png",
            dpi=300
        )

        plt.close()

        print("Saved")


if __name__ == "__main__":
    main()