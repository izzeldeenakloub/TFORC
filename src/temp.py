import csv
import numpy as np

def load_csv(path):
    t, q, d = [], [], []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["time_s"]))
            q.append(float(row["total_queue"]))
            d.append(float(row["cum_delay_proxy"]))

    return np.array(t), np.array(q), np.array(d)


t, q, d = load_csv("ac_run.csv")

avg_queue = np.mean(q)
final_delay = d[-1]
sim_time = t[-1]
avg_delay_per_sec = final_delay / sim_time

print("===== AC Results =====")
print(f"Simulation time: {sim_time:.2f} s")
print(f"Average queue: {avg_queue:.2f} vehicles")
print(f"Final cumulative delay proxy: {final_delay:.2f}")
print(f"Average delay proxy per second: {avg_delay_per_sec:.2f}")