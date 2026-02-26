import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci

SUMO_BINARY = "sumo"          # or "sumo-gui"
SUMO_CONFIG = "khalda.sumocfg"
TLS_ID = "Node2"
STEP_LENGTH = 0.1
MIN_GREEN = 5.0


class SumoTrafficEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        self.min_green_steps = int(MIN_GREEN / STEP_LENGTH)
        self.current_step = 0
        self.last_switch = 0

    # -----------------------------
    # Detector helpers
    # -----------------------------
    def det_id(self, app, lane, seg):
        return f"{app}_L{lane}_S{seg}"

    def arm_sum(self, app):
        total = 0

        if app == "W":
            for lane in range(3):
                for seg in range(2):
                    total += traci.lanearea.getLastStepVehicleNumber(
                        self.det_id(app, lane, seg)
                    )
            total += traci.lanearea.getLastStepVehicleNumber("W_add")
            return total

        if app == "N":
            for lane in range(3):
                for seg in range(2):
                    total += traci.lanearea.getLastStepVehicleNumber(
                        self.det_id(app, lane, seg)
                    )
            total += traci.lanearea.getLastStepVehicleNumber("N_add")
            return total

        if app == "E":
            for lane in range(4):
                for seg in range(3):
                    if self.det_id(app, lane, seg) == "E_L3_S2":
                        continue
                    total += traci.lanearea.getLastStepVehicleNumber(
                        self.det_id(app, lane, seg)
                    )
            return total

        if app == "S":
            for lane in range(2):
                for seg in range(2):
                    total += traci.lanearea.getLastStepVehicleNumber(
                        self.det_id(app, lane, seg)
                    )
            total += traci.lanearea.getLastStepVehicleNumber("S_add")
            return total

        return total

    # -----------------------------
    # Environment methods
    # -----------------------------
    def reset(self, seed=None, options=None):

        if traci.isLoaded():
            traci.close()

        traci.start([
            SUMO_BINARY,
            "-c", SUMO_CONFIG,
            "--step-length", str(STEP_LENGTH),
            "--start"
        ])

        self.current_step = 0
        self.last_switch = 0

        return self._get_state(), {}

    def _get_state(self):

        w = self.arm_sum("W")
        e = self.arm_sum("E")
        n = self.arm_sum("N")
        s = self.arm_sum("S")

        # Normalize queues (VERY important for deep RL)
        max_q = 100.0

        state = np.array([
            w / max_q,
            e / max_q,
            n / max_q,
            s / max_q
        ], dtype=np.float32)

        return state

    def step(self, action):

        # Enforce minimum green time
        if (self.current_step - self.last_switch) >= self.min_green_steps:
            traci.trafficlight.setPhase(TLS_ID, action)
            self.last_switch = self.current_step

        traci.simulationStep()
        self.current_step += 1

        state = self._get_state()

        # Reward = negative total queue
        reward = -np.sum(state)

        done = traci.simulation.getMinExpectedNumber() <= 0

        return state, reward, done, False, {}

    def close(self):
        traci.close()