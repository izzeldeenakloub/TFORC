"""
CTDE-MAAC — Fixed Training Loop & Plotting
Fixes:
  1. Each episode uses a different SUMO random seed → unique traffic patterns
  2. Agent is NOT recreated each episode — weights persist and accumulate
  3. plot_episode() draws avg lines on ALL 4 subplots (queue + waiting time)
  4. learning_curves.png shows per-episode averages trending over time
  5. Average waiting time is tracked and plotted across episodes
"""

import os, sys, random, pickle
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ── SUMO ──────────────────────────────────────────────────────────────────────
if "SUMO_HOME" not in os.environ:
    sys.exit("Please set SUMO_HOME")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SUMO_BINARY  = "sumo"
SUMO_CONFIG  = "/home/kloub/Desktop/college/graduation_project/GP_Core/TFORC/network/khalda.sumocfg"
STEP_LENGTH  = 1.0
TLS_MAIN     = "Node2"
TLS_AB       = "Node_AB"
MIN_GREEN    = 20.0

LAMBDA_LOCAL  = 0.7
LAMBDA_GLOBAL = 0.3

N_STEPS          = 10
BUFFER_CAPACITY  = 20_000
BATCH_SIZE       = 128
WARMUP_STEPS     = 500

HIDDEN_DIM   = 128
ACTOR_LR     = 3e-4
CRITIC_LR    = 1e-3
GAMMA        = 0.97
TAU          = 0.005
GRAD_CLIP    = 1.0

EPSILON_START = 0.20
EPSILON_MIN   = 0.01
EPSILON_DECAY = 0.90          # aggressive decay so agent exploits sooner

NUM_EPISODES    = 50
MODEL_PATH      = "ctde_maac.pt"
BEST_MODEL_PATH = "ctde_maac_best.pt"
OUT_DIR         = "exp_ctde_maac"

MAIN_STATE_DIM  = 7
AB_STATE_DIM    = 6
JOINT_STATE_DIM = MAIN_STATE_DIM + AB_STATE_DIM

MAIN_N_ACTIONS  = 4
AB_N_ACTIONS    = 3
MAX_QUEUE       = 60.0


# ══════════════════════════════════════════════════════════════════════════════
# NETWORKS
# ══════════════════════════════════════════════════════════════════════════════

class ActorNet(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class CentralizedCriticNet(nn.Module):
    def __init__(self, joint_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_dim, hidden * 2), nn.LayerNorm(hidden * 2), nn.ReLU(),
            nn.Linear(hidden * 2, hidden),    nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY + N-STEP BUFFERS
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(args)

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        sm, sa, am, aa, rm, ra, sm_n, sa_n, d = zip(*batch)
        to_t = lambda x: torch.FloatTensor(np.array(x))
        to_l = lambda x: torch.LongTensor(np.array(x))
        return (to_t(sm), to_t(sa),
                to_l(am), to_l(aa),
                to_t(rm).unsqueeze(1), to_t(ra).unsqueeze(1),
                to_t(sm_n), to_t(sa_n),
                to_t(d).unsqueeze(1))

    def __len__(self):
        return len(self.buf)


class NStepBuffer:
    def __init__(self, n=N_STEPS, gamma=GAMMA):
        self.n = n; self.gamma = gamma
        self.buf = deque()

    def add(self, transition):
        self.buf.append(transition)

    def ready(self):
        return len(self.buf) >= self.n

    def get(self):
        R_main = R_ab = 0.0
        for i, tr in enumerate(self.buf):
            R_main += (self.gamma ** i) * tr[4]
            R_ab   += (self.gamma ** i) * tr[5]
        s0 = self.buf[0]; sn = self.buf[-1]
        result = (s0[0], s0[1], s0[2], s0[3],
                  R_main, R_ab, sn[6], sn[7], sn[8])
        self.buf.popleft()
        return result

    def clear(self):
        self.buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
# STATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def encode_phase(phase, n_phases):
    angle = 2 * np.pi * phase / max(n_phases, 1)
    return np.sin(angle), np.cos(angle)

def state_vector_main(local, neighbor_load):
    w, e, n, s, phase = local
    ps, pc = encode_phase(phase, MAIN_N_ACTIONS * 2)
    return np.array([w, e, n, s, ps, pc, neighbor_load],
                    dtype=np.float32) / np.array(
                    [MAX_QUEUE]*4 + [1, 1, MAX_QUEUE], dtype=np.float32)

def state_vector_ab(local, neighbor_load):
    w, e, sq, phase = local
    ps, pc = encode_phase(phase, AB_N_ACTIONS * 2)
    return np.array([w, e, sq, ps, pc, neighbor_load],
                    dtype=np.float32) / np.array(
                    [MAX_QUEUE]*3 + [1, 1, MAX_QUEUE], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTORS
# ══════════════════════════════════════════════════════════════════════════════

def det_id(app, lane, seg): return f"{app}_L{lane}_S{seg}"

def arm_main(app):
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
                if det_id(app, lane, seg) == "E_L3_S2": continue
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app, lane, seg))
    elif app == "S":
        for lane in range(2):
            for seg in range(2):
                total += traci.lanearea.getLastStepVehicleNumber(det_id(app, lane, seg))
        total += traci.lanearea.getLastStepVehicleNumber("S_add")
    return total

def get_main():
    return (arm_main("W"), arm_main("E"), arm_main("N"), arm_main("S"),
            traci.trafficlight.getPhase(TLS_MAIN))

def total_main(s): return s[0]+s[1]+s[2]+s[3]

def det_ab(app, lane, seg): return f"AB_{app}_L{lane}_S{seg}"

def arm_ab(app):
    total = 0
    cfg = {"W": (3,3,"AB_W_L2_S2"), "E": (2,2,None), "S": (2,3,None)}
    lanes, segs, skip = cfg[app]
    for lane in range(lanes):
        for seg in range(segs):
            did = det_ab(app, lane, seg)
            if did == skip: continue
            try: total += traci.lanearea.getLastStepVehicleNumber(did)
            except: pass
    return total

def get_ab():
    return (arm_ab("W"), arm_ab("E"), arm_ab("S"),
            traci.trafficlight.getPhase(TLS_AB))

def total_ab(s): return s[0]+s[1]+s[2]


# ══════════════════════════════════════════════════════════════════════════════
# REWARD
# ══════════════════════════════════════════════════════════════════════════════

def get_waiting_time(tls_id):
    total = 0.0
    for lane in traci.trafficlight.getControlledLanes(tls_id):
        total += traci.lane.getWaitingTime(lane)
    return total

def compute_rewards(wt_main, wt_ab):
    wt_global = wt_main + wt_ab
    r_main = -(LAMBDA_LOCAL * wt_main + LAMBDA_GLOBAL * wt_global)
    r_ab   = -(LAMBDA_LOCAL * wt_ab   + LAMBDA_GLOBAL * wt_global)
    scale  = max(1.0, wt_global)
    return r_main / scale, r_ab / scale


# ══════════════════════════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════════════════════════

class CTDEMaacAgent:
    def __init__(self):
        self.actor_main    = ActorNet(MAIN_STATE_DIM, MAIN_N_ACTIONS)
        self.actor_ab      = ActorNet(AB_STATE_DIM,   AB_N_ACTIONS)
        self.critic        = CentralizedCriticNet(JOINT_STATE_DIM)
        self.critic_target = CentralizedCriticNet(JOINT_STATE_DIM)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.opt_actor_main = optim.Adam(self.actor_main.parameters(), lr=ACTOR_LR)
        self.opt_actor_ab   = optim.Adam(self.actor_ab.parameters(),   lr=ACTOR_LR)
        self.opt_critic     = optim.Adam(self.critic.parameters(),     lr=CRITIC_LR)

        self.epsilon          = EPSILON_START
        self.last_switch_main = -MIN_GREEN
        self.last_switch_ab   = -MIN_GREEN
        self.actions_main     = [0, 2, 4, 6]
        self.actions_ab       = [0, 2, 4]

    def _choose(self, actor, sv, actions, last_switch, time):
        if time - last_switch < MIN_GREEN:
            return None, None
        t = torch.FloatTensor(sv).unsqueeze(0)
        with torch.no_grad():
            probs = actor(t).squeeze(0).numpy()
        idx = random.randrange(len(actions)) if random.random() < self.epsilon \
              else int(np.argmax(probs))
        return actions[idx], probs

    def choose_action_main(self, sv, time):
        phase, probs = self._choose(self.actor_main, sv,
                                    self.actions_main, self.last_switch_main, time)
        if phase is None:
            return traci.trafficlight.getPhase(TLS_MAIN), None
        traci.trafficlight.setPhase(TLS_MAIN, int(phase))
        self.last_switch_main = time
        return phase, probs

    def choose_action_ab(self, sv, time):
        phase, probs = self._choose(self.actor_ab, sv,
                                    self.actions_ab, self.last_switch_ab, time)
        if phase is None:
            return traci.trafficlight.getPhase(TLS_AB), None
        traci.trafficlight.setPhase(TLS_AB, int(phase))
        self.last_switch_ab = time
        return phase, probs

    def update(self, batch):
        sm, sa, am, aa, rm, ra, sm_n, sa_n, done = batch
        joint_s  = torch.cat([sm,   sa],   dim=1)
        joint_sn = torch.cat([sm_n, sa_n], dim=1)

        with torch.no_grad():
            V_next    = self.critic_target(joint_sn)
            r_joint   = (rm + ra) / 2.0
            td_target = r_joint + GAMMA ** N_STEPS * V_next * (1 - done)

        V_pred    = self.critic(joint_s)
        td_error  = (td_target - V_pred).detach()
        c_loss    = F.mse_loss(V_pred, td_target)
        self.opt_critic.zero_grad(); c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()

        def actor_loss(actor, opt, sv, actions_taken, n_actions):
            probs    = actor(sv)
            log_p    = torch.log(probs + 1e-8)
            chosen   = log_p.gather(1, actions_taken.view(-1,1) % n_actions)
            entropy  = -(probs * log_p).sum(dim=1, keepdim=True)
            loss     = -(chosen * td_error + 0.01 * entropy).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
            opt.step()
            return loss.item()

        lam = actor_loss(self.actor_main, self.opt_actor_main,
                         sm, am, MAIN_N_ACTIONS)
        lab = actor_loss(self.actor_ab,   self.opt_actor_ab,
                         sa, aa, AB_N_ACTIONS)

        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        return c_loss.item(), lam, lab

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def reset_switches(self):
        """Call at the start of each episode so timing is fresh."""
        self.last_switch_main = -MIN_GREEN
        self.last_switch_ab   = -MIN_GREEN

    def save(self, path):
        torch.save({
            "actor_main":    self.actor_main.state_dict(),
            "actor_ab":      self.actor_ab.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "epsilon":       self.epsilon,
        }, path)
        print(f"  [SAVE] {path}")

    def load(self, path):
        if not os.path.exists(path):
            print(f"  [INIT] No checkpoint at {path} — fresh start"); return
        ckpt = torch.load(path, map_location="cpu")
        self.actor_main.load_state_dict(ckpt["actor_main"])
        self.actor_ab.load_state_dict(ckpt["actor_ab"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.epsilon = ckpt.get("epsilon", EPSILON_START)
        print(f"  [LOAD] {path}  (ε={self.epsilon:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# SUMO
# ══════════════════════════════════════════════════════════════════════════════

def start(seed: int):
    """
    FIX 1: Pass a different seed every episode so SUMO generates
    different departure times → each episode is unique.
    """
    traci.start([
        SUMO_BINARY, "-c", SUMO_CONFIG,
        "--step-length", str(STEP_LENGTH),
        "--seed", str(seed),       # ← KEY FIX
        "--start",
    ])

def sim_done():
    return traci.simulation.getMinExpectedNumber() <= 0


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(agent, replay, nstep, seed, train=True):
    start(seed)
    agent.reset_switches()   # FIX 2: reset timing, not weights
    nstep.clear()

    t_arr, qm_arr, qa_arr = [], [], []
    wm_arr, wa_arr        = [], []
    loss_c, loss_am, loss_ab = [], [], []

    while not sim_done():
        time = traci.simulation.getTime()

        s_main_loc = get_main()
        s_ab_loc   = get_ab()
        sv_main    = state_vector_main(s_main_loc, total_ab(s_ab_loc))
        sv_ab      = state_vector_ab(s_ab_loc, total_main(s_main_loc))

        a_main, _ = agent.choose_action_main(sv_main, time)
        a_ab,   _ = agent.choose_action_ab(sv_ab,   time)

        traci.simulationStep()

        s_main_next = get_main()
        s_ab_next   = get_ab()
        sv_main_n   = state_vector_main(s_main_next, total_ab(s_ab_next))
        sv_ab_n     = state_vector_ab(s_ab_next,   total_main(s_main_next))

        wt_main = get_waiting_time(TLS_MAIN)
        wt_ab   = get_waiting_time(TLS_AB)
        r_main, r_ab = compute_rewards(wt_main, wt_ab)

        done_flag = float(sim_done())

        nstep.add((sv_main, sv_ab, a_main, a_ab,
                   r_main, r_ab, sv_main_n, sv_ab_n, done_flag))
        if nstep.ready():
            replay.push(*nstep.get())

        if train and len(replay) >= WARMUP_STEPS:
            lc, lam, lab = agent.update(replay.sample(BATCH_SIZE))
            loss_c.append(lc); loss_am.append(lam); loss_ab.append(lab)

        t_arr.append(time)
        qm_arr.append(total_main(s_main_next))
        qa_arr.append(total_ab(s_ab_next))
        wm_arr.append(wt_main)
        wa_arr.append(wt_ab)

    traci.close()

    return {
        "t":       np.array(t_arr),
        "q_main":  np.array(qm_arr),
        "q_ab":    np.array(qa_arr),
        "wt_main": np.array(wm_arr),
        "wt_ab":   np.array(wa_arr),
        "loss_c":  float(np.mean(loss_c))  if loss_c  else 0.0,
        "loss_am": float(np.mean(loss_am)) if loss_am else 0.0,
        "loss_ab": float(np.mean(loss_ab)) if loss_ab else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING  (FIX 3: avg lines on all 4 subplots, including waiting time)
# ══════════════════════════════════════════════════════════════════════════════

def plot_episode(ep, res, out_dir):
    """
    4 subplots, each with its own computed average dashed line.
    Queue (top) and Waiting Time (bottom) for each intersection.
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    avg_qm  = np.mean(res["q_main"])
    avg_qa  = np.mean(res["q_ab"])
    avg_wtm = np.mean(res["wt_main"])
    avg_wta = np.mean(res["wt_ab"])

    # ── Top-left: Assaf Queue ──────────────────────────────────────────────
    axs[0,0].plot(res["t"], res["q_main"], color="steelblue", alpha=0.6, lw=0.8)
    axs[0,0].axhline(avg_qm, linestyle="--", color="steelblue", lw=1.8,
                     label=f"Avg = {avg_qm:.1f} veh")
    axs[0,0].set_title("Assaf Intersection — Queue")
    axs[0,0].set_xlabel("Time (s)"); axs[0,0].set_ylabel("Vehicles")
    axs[0,0].legend(fontsize=10); axs[0,0].grid(True, alpha=0.3)

    # ── Top-right: AB Queue ────────────────────────────────────────────────
    axs[0,1].plot(res["t"], res["q_ab"], color="darkorange", alpha=0.6, lw=0.8)
    axs[0,1].axhline(avg_qa, linestyle="--", color="darkorange", lw=1.8,
                     label=f"Avg = {avg_qa:.1f} veh")
    axs[0,1].set_title("AB Intersection — Queue")
    axs[0,1].set_xlabel("Time (s)"); axs[0,1].set_ylabel("Vehicles")
    axs[0,1].legend(fontsize=10); axs[0,1].grid(True, alpha=0.3)

    # ── Bottom-left: Assaf Waiting Time  ← FIX: avg line added ───────────
    axs[1,0].plot(res["t"], res["wt_main"], color="steelblue", alpha=0.6, lw=0.8)
    axs[1,0].axhline(avg_wtm, linestyle="--", color="steelblue", lw=1.8,
                     label=f"Avg = {avg_wtm:.1f} s")
    axs[1,0].set_title("Assaf Intersection — Waiting Time")
    axs[1,0].set_xlabel("Time (s)"); axs[1,0].set_ylabel("Total Wait (s)")
    axs[1,0].legend(fontsize=10); axs[1,0].grid(True, alpha=0.3)

    # ── Bottom-right: AB Waiting Time  ← FIX: avg line added ─────────────
    axs[1,1].plot(res["t"], res["wt_ab"], color="darkorange", alpha=0.6, lw=0.8)
    axs[1,1].axhline(avg_wta, linestyle="--", color="darkorange", lw=1.8,
                     label=f"Avg = {avg_wta:.1f} s")
    axs[1,1].set_title("AB Intersection — Waiting Time")
    axs[1,1].set_xlabel("Time (s)"); axs[1,1].set_ylabel("Total Wait (s)")
    axs[1,1].legend(fontsize=10); axs[1,1].grid(True, alpha=0.3)

    fig.suptitle(f"CTDE-MAAC — Episode {ep+1}  |  ε decay applied",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"episode_{ep+1:03d}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curves(stats, out_dir):
    """
    FIX 4: Shows per-episode averages for queue AND waiting time,
    so you can see the learning trend across all episodes.
    """
    eps      = [s["ep"]          for s in stats]
    avg_qm   = [s["avg_q_main"]  for s in stats]
    avg_qa   = [s["avg_q_ab"]    for s in stats]
    avg_qt   = [s["avg_q_total"] for s in stats]
    avg_wtm  = [s["avg_wt_main"] for s in stats]
    avg_wta  = [s["avg_wt_ab"]   for s in stats]
    avg_wtt  = [s["avg_wt_total"]for s in stats]
    lc       = [s["loss_c"]      for s in stats]

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    def smooth(arr, w=5):
        if len(arr) < w: return arr
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    def add_smooth(ax, x, y, color, label, w=5):
        ax.plot(x, y, color=color, alpha=0.3, lw=1, label="Raw")
        sx = x[w-1:] if len(x) >= w else x
        ax.plot(sx, smooth(y, w), color=color, lw=2.5, label=f"Smoothed ({label})")

    # Queue — Main
    add_smooth(axs[0,0], eps, avg_qm, "steelblue", "Assaf")
    axs[0,0].set_title("Avg Queue — Assaf"); axs[0,0].set_xlabel("Episode")
    axs[0,0].set_ylabel("Vehicles"); axs[0,0].legend(); axs[0,0].grid(True, alpha=0.3)

    # Queue — AB
    add_smooth(axs[0,1], eps, avg_qa, "darkorange", "AB")
    axs[0,1].set_title("Avg Queue — AB"); axs[0,1].set_xlabel("Episode")
    axs[0,1].set_ylabel("Vehicles"); axs[0,1].legend(); axs[0,1].grid(True, alpha=0.3)

    # Queue — Total
    add_smooth(axs[0,2], eps, avg_qt, "purple", "Total")
    axs[0,2].set_title("Avg Total Queue"); axs[0,2].set_xlabel("Episode")
    axs[0,2].set_ylabel("Vehicles"); axs[0,2].legend(); axs[0,2].grid(True, alpha=0.3)

    # Waiting Time — Main
    add_smooth(axs[1,0], eps, avg_wtm, "steelblue", "Assaf")
    axs[1,0].set_title("Avg Waiting Time — Assaf"); axs[1,0].set_xlabel("Episode")
    axs[1,0].set_ylabel("Total Wait (s)"); axs[1,0].legend(); axs[1,0].grid(True, alpha=0.3)

    # Waiting Time — AB
    add_smooth(axs[1,1], eps, avg_wta, "darkorange", "AB")
    axs[1,1].set_title("Avg Waiting Time — AB"); axs[1,1].set_xlabel("Episode")
    axs[1,1].set_ylabel("Total Wait (s)"); axs[1,1].legend(); axs[1,1].grid(True, alpha=0.3)

    # Critic Loss
    axs[1,2].plot(eps, lc, color="crimson", lw=1.5)
    axs[1,2].set_title("Critic Loss"); axs[1,2].set_xlabel("Episode")
    axs[1,2].set_ylabel("MSE"); axs[1,2].grid(True, alpha=0.3)

    fig.suptitle("CTDE-MAAC — Learning Curves (Queue + Waiting Time)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "learning_curves.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Learning curves → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # FIX: agent, replay, nstep created ONCE and persist across all episodes
    agent  = CTDEMaacAgent()
    replay = ReplayBuffer(BUFFER_CAPACITY)
    nstep  = NStepBuffer(N_STEPS, GAMMA)

    for path in [BEST_MODEL_PATH, MODEL_PATH]:
        if os.path.exists(path):
            agent.load(path); break

    stats    = []
    best_wt  = float("inf")

    for ep in range(NUM_EPISODES):
        # FIX: unique seed per episode → different traffic each time
        seed = 42 + ep * 17
        print(f"\n{'='*60}")
        print(f"  Episode {ep+1}/{NUM_EPISODES}  seed={seed}  ε={agent.epsilon:.4f}")
        print(f"{'='*60}")

        res = run_episode(agent, replay, nstep, seed=seed, train=True)

        avg_qm   = float(np.mean(res["q_main"]))
        avg_qa   = float(np.mean(res["q_ab"]))
        avg_qt   = avg_qm + avg_qa
        avg_wtm  = float(np.mean(res["wt_main"]))
        avg_wta  = float(np.mean(res["wt_ab"]))
        avg_wtt  = avg_wtm + avg_wta

        print(f"  Queue  — Assaf: {avg_qm:.1f}  AB: {avg_qa:.1f}  Total: {avg_qt:.1f}")
        print(f"  Wait   — Assaf: {avg_wtm:.1f}s  AB: {avg_wta:.1f}s  Total: {avg_wtt:.1f}s")
        print(f"  Loss   — Critic: {res['loss_c']:.5f}  |  Replay: {len(replay)}")

        stats.append({
            "ep": ep + 1,
            "avg_q_main":   avg_qm,
            "avg_q_ab":     avg_qa,
            "avg_q_total":  avg_qt,
            "avg_wt_main":  avg_wtm,
            "avg_wt_ab":    avg_wta,
            "avg_wt_total": avg_wtt,
            "loss_c":       res["loss_c"],
        })

        agent.save(MODEL_PATH)
        if avg_wtt < best_wt:
            best_wt = avg_wtt
            agent.save(BEST_MODEL_PATH)
            print(f"  *** New best total wait: {best_wt:.1f}s ***")

        plot_episode(ep, res, OUT_DIR)
        agent.decay_epsilon()

        # save running stats after every episode
        with open(os.path.join(OUT_DIR, "stats.pkl"), "wb") as f:
            pickle.dump(stats, f)

    plot_learning_curves(stats, OUT_DIR)
    print(f"\nDone. Best total waiting time: {best_wt:.1f}s")


if __name__ == "__main__":
    main()