import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import csv
import numpy as np
from sumo_env import SumoTrafficEnv
from stable_baselines3 import DQN, PPO

ALGO = "DQN"   # change to "PPO"

env = SumoTrafficEnv()

if ALGO == "DQN":
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        verbose=1
    )

elif ALGO == "PPO":
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1
    )

print("Training:", ALGO)
model.learn(total_timesteps=30000)

# -----------------------------
# Evaluation run
# -----------------------------
obs, _ = env.reset()
done = False

t_list = []
q_list = []
d_list = []

cum_delay = 0
time = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

    total_queue = np.sum(obs) * 100.0  # rescale back
    cum_delay += total_queue * 0.1

    t_list.append(time)
    q_list.append(total_queue)
    d_list.append(cum_delay)

    time += 0.1

env.close()

filename = ALGO.lower() + "_run.csv"

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "total_queue", "cum_delay_proxy"])
    for i in range(len(t_list)):
        writer.writerow([t_list[i], q_list[i], d_list[i]])

print("Saved:", filename)