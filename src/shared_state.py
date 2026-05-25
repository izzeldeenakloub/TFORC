import os, sys
import shutil
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
SUMO_CONFIG = "/home/kloub/Desktop/college/graduation_project/GP_Core/TFORC/network/khalda.sumocfg"
STEP_LENGTH = 1.0
SIM_END = 8000;
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
BALANCE_WEIGHT = 0.1

MODEL_PATH      = "ac_connected_v2.pkl"
BEST_MODEL_PATH = "ac_connected.pkl"
OUT_DIR         = "exp_connected"


# ---------------------------
# DISCRETIZATION
# ---------------------------

def discretize_queue(q):
    if q == 0:    return 0
    elif q <= 5:  return 1
    elif q <= 15: return 2
    elif q <= 30: return 3
    else:         return 4

def discretize_main(s):
    return (discretize_queue(s[0]),
            discretize_queue(s[1]),
            discretize_queue(s[2]),
            discretize_queue(s[3]),
            s[4])

def discretize_ab(s):
    return (discretize_queue(s[0]),
            discretize_queue(s[1]),
            discretize_queue(s[2]),
            s[3])


def get_neighbor_signal(total):
    return discretize_queue(total)
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
    traci.start([SUMO_BINARY,"-c"
                 ,SUMO_CONFIG,
                 "--step-length",str(STEP_LENGTH),
                # "--end", str(SIM_END),
                 "--start"])

def stop():
    return (traci.simulation.getMinExpectedNumber() <= 0 #or
           # traci.simulation.getTime() >= SIM_END)
    )
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
    def __init__(self, name, tls_id, phases, min_green=20.0,
                 epsilon=AC_EPSILON, temp=AC_TEMPERATURE,
                 actor_lr=ACTOR_LR, critic_lr=CRITIC_LR):
        self.name = name
        self.tls_id = tls_id
        self.phases = phases

        self.min_green = min_green
        self.epsilon = epsilon
        self.temp = temp
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.prefs = {}
        self.V = {}
        self.last_switch = -min_green

    def ensure_state(self, state):
        if state not in self.prefs:
            self.prefs[state] = np.zeros(len(self.phases))
        if state not in self.V:
            self.V[state] = -100.0  # pessimistic init: rewards are always negative (~-100)

    def softmax(self, x):
        x = np.array(x, dtype=float) / max(1e-9, self.temp)
        e = np.exp(x - np.max(x))
        p = e / np.sum(e)
        p = 0.01 + 0.99 * p
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
        self.ensure_state(s)
        self.ensure_state(s_next)
        
        td = reward + GAMMA * self.V[s_next] - self.V[s]

        # critic
        self.V[s] += self.critic_lr * td

        # actor
        for k in range(len(self.phases)):
            if self.phases[k] == action:
                self.prefs[s][k] += self.actor_lr * td * (1 - pi[k])
            else:
                self.prefs[s][k] -= self.actor_lr * td * pi[k]

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
def run_ac_connected(epsilon=AC_EPSILON, temp=AC_TEMPERATURE,
                     actor_lr=ACTOR_LR, critic_lr=CRITIC_LR):

    main_agent = IntersectionAgent("MAIN", TLS_MAIN, [0,2,4,6],
                                   epsilon=epsilon, temp=temp,
                                   actor_lr=actor_lr, critic_lr=critic_lr)
    ab_agent   = IntersectionAgent("AB",   TLS_AB,   [0,2,4],
                                   epsilon=epsilon, temp=temp,
                                   actor_lr=actor_lr, critic_lr=critic_lr)

    start()
    # always start from the best known policy, not the last saved one
    load_model(BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH,
               main_agent, ab_agent)

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

        time = traci.simulation.getTime()

        # =====================================================
        # STATES
        # =====================================================
        s_main_local = get_main()     # [W,E,N,S]
        s_ab_local   = get_ab()       # [W,E,S]
        neighbor_signal_for_main = get_neighbor_signal(total_ab(s_ab_local))
        neighbor_signal_for_ab   = get_neighbor_signal(total_main(s_main_local))

        s_main = (discretize_main(s_main_local), neighbor_signal_for_main)
        s_ab   = (discretize_ab(s_ab_local),     neighbor_signal_for_ab)

        # =====================================================
        # ACTIONS
        # =====================================================
        a_main, pi_main = main_agent.choose_action(s_main, time)
        a_ab, pi_ab     = ab_agent.choose_action(s_ab, time)

        # =====================================================
        # STEP
        # =====================================================
        traci.simulationStep()

        # =====================================================
        # NEXT STATE
        # =====================================================
        s_main_next_local = get_main()
        s_ab_next_local   = get_ab()
        s_main_next = (discretize_main(s_main_next_local),
               get_neighbor_signal(total_ab(s_ab_next_local)))    # ← match s_main structure

        s_ab_next   = (discretize_ab(s_ab_next_local),
               get_neighbor_signal(total_main(s_main_next_local)))

        # =====================================================
        # QUEUES
        # =====================================================
        q_main = total_main(s_main_next_local)
        q_ab   = total_ab(s_ab_next_local)
        q_total = q_main + q_ab

        # =====================================================
        # REWARD
        # =====================================================
        r_main = -(q_main + BALANCE_WEIGHT * abs(q_main - q_ab))
        r_ab   = -(q_ab   + BALANCE_WEIGHT * abs(q_ab - q_main))

        # =====================================================
        # UPDATE
        # =====================================================
        main_agent.update(s_main, s_main_next, a_main, pi_main, r_main)
        ab_agent.update(s_ab, s_ab_next, a_ab, pi_ab, r_ab)

        # =====================================================
        # SAVE TOTALS
        # =====================================================
        cdm += q_main * STEP_LENGTH
        cda += q_ab   * STEP_LENGTH
        cdt += q_total * STEP_LENGTH

        t.append(time)
        qm.append(q_main)
        qa.append(q_ab)
        qt.append(q_total)

        dm.append(cdm)
        da.append(cda)
        dt.append(cdt)

        # =====================================================
        # SAVE MAIN ARMS
        # =====================================================
        mw.append(s_main_next_local[0])
        me.append(s_main_next_local[1])
        mn.append(s_main_next_local[2])
        ms.append(s_main_next_local[3])

        # =====================================================
        # SAVE AB ARMS
        # =====================================================
        aw.append(s_ab_next_local[0])
        ae.append(s_ab_next_local[1])
        ass.append(s_ab_next_local[2])

    traci.close()

    return (main_agent, ab_agent,
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

    os.makedirs("exp_connected", exist_ok=True)
    os.makedirs("exp_connected/per_arm", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # run static baseline once — it's deterministic, no need to repeat
    print("Running static baseline...")
    # t_s, qm_s, qa_s, *_ = run_static()
    # avg_st_main = float(np.mean(qm_s))
    # avg_st_ab   = float(np.mean(qa_s))
    # avg_st_total = float(np.mean(qm_s + qa_s))
    # print(f"  Static avg queue: {avg_st_total:.1f}  (main={avg_st_main:.1f}, ab={avg_st_ab:.1f})")

    runs = 5
    episode_stats = []
    best_avg = float('inf')

    for i in range(runs):

        print(f"=== Run {i+1} ===")

        # decay hyperparameters each episode
        epsilon   = max(0.01, 0.20 * (0.96 ** i))
        temp      = max(0.30, 2.00 * (0.97 ** i))
        actor_lr  = max(0.001, ACTOR_LR  * (0.98 ** i))
        critic_lr = max(0.005, CRITIC_LR * (0.98 ** i))
        print(f"  eps={epsilon:.4f}  temp={temp:.3f}  alr={actor_lr:.5f}  clr={critic_lr:.5f}")

        # ================================================
        # RUN AC CONNECTED
        # ================================================
        (
            main_agent, ab_agent,
            t_a,
            qm_a, qa_a, qt_a,
            mw, me, mn, ms,
            aw, ae, ass,
            dm_a, da_a, dt_a

        ) = run_ac_connected(epsilon=0.0, temp=temp,
                             actor_lr=actor_lr, critic_lr=critic_lr)

        # ================================================
        # SUM OF ARMS
        # ================================================
        main_sum = mw + me + mn + ms
        ab_sum   = aw + ae + ass

        # ================================================
        # FIGURE 1 — COMPARISON PLOT
        # ================================================
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # --- AB subplot ---
        avg_ac_ab = np.mean(ab_sum)

        axs[0].plot(t_a, ab_sum, label="AC Connected")
        # axs[0].plot(t_s, qa_s,   label="Static")
        axs[0].axhline(avg_ac_ab, linestyle='--',
                       label=f"AC Avg = {avg_ac_ab:.1f}")
        # axs[0].axhline(avg_st_ab, linestyle='--',
                    #    label=f"Static Avg = {avg_st_ab:.1f}")
        axs[0].set_title("AB Intersection")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Queue")
        axs[0].grid()
        axs[0].legend()

        # --- Main subplot ---
        avg_ac_main = np.mean(main_sum)

        axs[1].plot(t_a, main_sum, label="AC Connected")
        # axs[1].plot(t_s, qm_s,     label="Static")
        axs[1].axhline(avg_ac_main, linestyle='--',
                       label=f"AC Avg = {avg_ac_main:.1f}")
        # axs[1].axhline(avg_st_main, linestyle='--',
                    #    label=f"Static Avg = {avg_st_main:.1f}")
        axs[1].set_title("Assaf Intersection")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Queue")
        axs[1].grid()
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(f"exp_shared_state_v2/shared_state_v2_(test)_{i+1}.png", dpi=300)
        plt.close()

    #     # ================================================
    #     # FIGURE 2 — MAIN PER ARM
    #     # ================================================
    #     fig1, axs1 = plt.subplots(2, 2, figsize=(14, 10),
    #                                sharex=True, sharey=True)
    #     axs1 = axs1.flatten()

    #     main_arms_ac     = [mw,   me,   mn,   ms]
    #     main_arms_labels = ["West","East","North","South"]

    #     for k in range(4):
    #         avg_ac = np.mean(main_arms_ac[k])
    #         axs1[k].plot(t_a, main_arms_ac[k], label="AC Connected")
    #         axs1[k].axhline(avg_ac, linestyle='--',
    #                         label=f"AC Avg = {avg_ac:.1f}")
    #         axs1[k].set_title(main_arms_labels[k])
    #         axs1[k].set_xlabel("Time (s)")
    #         axs1[k].set_ylabel("Queue")
    #         axs1[k].grid()
    #         axs1[k].legend()

    #     fig1.suptitle(f"Main Intersection — Per Arm (Run {i+1})")
    #     plt.tight_layout()
    #     plt.savefig(f"exp_connected/per_arm/main_arms_run_{i+1}_v2.png", dpi=300)
    #     plt.close(fig1)

    #     # ================================================
    #     # FIGURE 3 — AB PER ARM (3x1 vertical)
    #     # ================================================
    #     fig2, axs2 = plt.subplots(3, 1, figsize=(12, 12),
    #                                sharex=True, sharey=True)

    #     ab_arms_ac     = [aw,    ae,    ass]
    #     ab_arms_labels = ["West","East","South"]

    #     for k in range(3):
    #         avg_ac = np.mean(ab_arms_ac[k])
    #         axs2[k].plot(t_a, ab_arms_ac[k], label="AC Connected")
    #         axs2[k].axhline(avg_ac, linestyle='--',
    #                         label=f"AC Avg = {avg_ac:.1f}")
    #         axs2[k].set_title(ab_arms_labels[k])
    #         axs2[k].set_xlabel("Time (s)")
    #         axs2[k].set_ylabel("Queue")
    #         axs2[k].grid()
    #         axs2[k].legend()

    #     fig2.suptitle(f"AB Intersection — Per Arm (Run {i+1})")
    #     plt.tight_layout()
    #     plt.savefig(f"exp_connected/per_arm/ab_arms_run_{i+1}_v2.png", dpi=300)
    #     plt.close(fig2)

    #     # ================================================
    #     # TRACK LEARNING + CHECKPOINT
    #     # ================================================
    #     avg_total = float(np.mean(qt_a))
    #     episode_stats.append(avg_total)

    #     print(f"  AC avg queue:     {avg_total:.1f}")
    #     print(f"  Static avg queue: {avg_st_total:.1f}")

    #     # always persist this episode's model so checkpoints are current
    #     save_model(MODEL_PATH, main_agent, ab_agent)

    #     # keep a copy of the best model found so far
    #     if avg_total < best_avg:
    #         best_avg = avg_total
    #         save_model(BEST_MODEL_PATH, main_agent, ab_agent)
    #         print(f"  *** New best: {best_avg:.1f} ***")

    #     # checkpoint every 10 episodes
    #     if (i + 1) % 10 == 0:
    #         shutil.copy(MODEL_PATH,
    #                     f"checkpoints/model_ep_{i+1}_v2.pkl")
    #         print(f"  Checkpoint saved at episode {i+1}")

    # # ====================================================
    # # AFTER ALL 100 RUNS — LEARNING CURVE
    # # ====================================================
    # window   = 10
    # smoothed = np.convolve(episode_stats,
    #                        np.ones(window) / window,
    #                        mode='valid')

    # plt.figure(figsize=(10, 4))
    # plt.plot(episode_stats,
    #          alpha=0.4, color='steelblue',
    #          label="Per episode avg queue")
    # plt.plot(range(window - 1, runs), smoothed,
    #          linewidth=2, color='steelblue',
    #          label=f"{window}-episode rolling average")
    # plt.xlabel("Episode")
    # plt.ylabel("Average Total Queue")
    # plt.title("Learning Curve — AC Connected (100 Episodes)")
    # plt.legend()
    # plt.grid()
    # plt.savefig("exp_connected/learning_curve_v2.png", dpi=300)
    # plt.close()

    # print(f"\nAll done. Best avg queue achieved: {best_avg:.1f}")
    # print("Learning curve saved to exp_connected/learning_curve_v2.png")


if __name__ == "__main__":
    main()
