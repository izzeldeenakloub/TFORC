import csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv(path):
    t, q, d = [], [], []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row["time_s"]))
            q.append(float(row["total_queue"]))
            d.append(float(row["cum_delay_proxy"]))
    return np.array(t), np.array(q), np.array(d)

# Load saved runs
t_s, q_s, d_s = load_csv("static_run.csv")
t_q, q_q, d_q = load_csv("ql_run.csv")

# Optional Actor-Critic
has_ac = False
t_a = q_a = d_a = None
try:
    t_a, q_a, d_a = load_csv("ac_run.csv")
    has_ac = True
except Exception as e:
    print("No ac_run.csv found (or failed to read it). Skipping Actor-Critic.")

# -------------------------
# Averages (overall mean)
# -------------------------
avg_s = np.mean(q_s)
avg_q = np.mean(q_q)
avg_a = np.mean(q_a) if has_ac else None

# -------------------------
# Plot 1: Queue + average lines
# -------------------------
plt.figure()
plt.plot(t_s, q_s, label="Static 90s/phase")
plt.plot(t_q, q_q, label="Q-Learning")
if has_ac:
    plt.plot(t_a, q_a, label="Actor-Critic")

# average lines (dashed)
plt.axhline(avg_s, linestyle="--",color="blue", linewidth=2, label=f"Static Avg = {avg_s:.1f}")
plt.axhline(avg_q, linestyle="--",color="orange", linewidth=2, label=f"QL Avg = {avg_q:.1f}")
if has_ac:
    plt.axhline(avg_a, linestyle="--",color="green", linewidth=2, label=f"AC Avg = {avg_a:.1f}")

plt.xlabel("Simulation time (s)")
plt.ylabel("Total queue (vehicles)")
plt.title("Total Queue vs Time")
plt.legend()
plt.grid(True)

# -------------------------
# Plot 2: Cumulative delay
# -------------------------
plt.figure()
plt.plot(t_s, d_s, label="Static 90s/phase")
plt.plot(t_q, d_q, label="Q-Learning")
if has_ac:
    plt.plot(t_a, d_a, label="Actor-Critic")

plt.xlabel("Simulation time (s)")
plt.ylabel("Cumulative queue-area (veh·s)")
plt.title("Cumulative Delay Proxy vs Time")
plt.legend()
plt.grid(True)

# Print averages to terminal too
print(f"Average queue (Static): {avg_s:.2f}")
print(f"Average queue (QL):     {avg_q:.2f}")
if has_ac:
    print(f"Average queue (AC):     {avg_a:.2f}")

plt.show()