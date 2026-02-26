import os
import sys
import random

# -------------------------
# SUMO / TraCI setup
# -------------------------
if "SUMO_HOME" not in os.environ:
    sys.exit("Please set SUMO_HOME")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

import traci

SUMO_BINARY = "sumo-gui"   # or "sumo"
SUMO_CONFIG = "khalda.sumocfg"

SUMO_CMD = [
    SUMO_BINARY,
    "-c", SUMO_CONFIG,
    "--step-length", "0.1",
    "--start"
]

# -------------------------
# Crazy behavior setup
# -------------------------
VEH_BEHAVIOR = {}
BEHAVIORS = ["slowdown", "stop_near_junction", "lane_change"]

BEHAVIOR_COLORS = {
    "slowdown":           (0, 0, 255, 255),   # blue
    "stop_near_junction": (255, 0, 0, 255),   # red
    "lane_change":        (0, 255, 0, 255),   # green
}

STUCK_STEPS = {}
STOP_SPEED_EPS = 0.2
STUCK_MIN_STEPS = 15
LEADER_LOOKAHEAD = 30
BLOCK_GAP_MAX = 10

# -------------------------
# Detector setup
# -------------------------
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

def get_state_arm_level():
    return tuple(arm_sum(app) for app in APPROACHES)

# -------------------------
# Crazy behavior functions
# -------------------------
def make_some_vehicles_behave_weird(step):
    veh_ids = traci.vehicle.getIDList()

    for vid in veh_ids:
        if traci.vehicle.getTypeID(vid) != "crazyDriver":
            continue

        if vid not in VEH_BEHAVIOR:
            VEH_BEHAVIOR[vid] = random.choice(BEHAVIORS)

        behavior = VEH_BEHAVIOR[vid]
        traci.vehicle.setColor(vid, BEHAVIOR_COLORS[behavior])
        apply_abnormal_behavior(vid, step, behavior)

def apply_abnormal_behavior(veh_id, step, behavior):
    if behavior == "slowdown":
        abnormal_slowdown(veh_id, step)
    elif behavior == "stop_near_junction":
        abnormal_stop_near_junction(veh_id, step)
    elif behavior == "lane_change":
        abnormal_lane_change(veh_id, step)

def help_normal_cars_overtake():
    veh_ids = traci.vehicle.getIDList()

    for vid in veh_ids:
        if traci.vehicle.getTypeID(vid) == "crazyDriver":
            continue

        speed = traci.vehicle.getSpeed(vid)

        if speed < STOP_SPEED_EPS:
            STUCK_STEPS[vid] = STUCK_STEPS.get(vid, 0) + 1
        else:
            STUCK_STEPS[vid] = 0
            continue

        if STUCK_STEPS[vid] < STUCK_MIN_STEPS:
            continue

        leader = traci.vehicle.getLeader(vid, LEADER_LOOKAHEAD)
        if not leader:
            continue

        leader_id, gap = leader

        if traci.vehicle.getTypeID(leader_id) != "crazyDriver":
            continue
        if gap > BLOCK_GAP_MAX:
            continue
        if traci.vehicle.getSpeed(leader_id) > STOP_SPEED_EPS:
            continue

        edge_id = traci.vehicle.getRoadID(vid)
        if edge_id == "" or edge_id[0] == ":":
            continue

        lane_count = traci.edge.getLaneNumber(edge_id)
        if lane_count <= 1:
            continue

        cur_lane = traci.vehicle.getLaneIndex(vid)
        for target_lane in [cur_lane + 1, cur_lane - 1]:
            if 0 <= target_lane < lane_count:
                try:
                    traci.vehicle.changeLane(vid, target_lane, 5.0)
                    STUCK_STEPS[vid] = 0
                    break
                except traci.TraCIException:
                    pass

def abnormal_slowdown(veh_id, step):
    if step % 50 != 0:
        return
    if random.random() < 0.4:
        current_speed = traci.vehicle.getSpeed(veh_id)
        new_speed = max(0.0, current_speed * 0.3)
        traci.vehicle.slowDown(veh_id, new_speed, 5.0)

def abnormal_stop_near_junction(veh_id, step):
    lane_id = traci.vehicle.getLaneID(veh_id)
    if lane_id == "":
        return

    pos = traci.vehicle.getLanePosition(veh_id)
    lane_length = traci.lane.getLength(lane_id)

    if lane_length - pos < 10:
        if random.random() < 0.5:
            traci.vehicle.slowDown(veh_id, 0.0, 3.0)

def abnormal_lane_change(veh_id, step):
    if step % 20 != 0:
        return

    edge_id = traci.vehicle.getRoadID(veh_id)
    if edge_id == "" or edge_id[0] == ":":
        return

    lane_count = traci.edge.getLaneNumber(edge_id)
    if lane_count <= 1:
        return

    current_lane_index = traci.vehicle.getLaneIndex(veh_id)
    target_lane = current_lane_index + random.choice([-1, 1])

    if 0 <= target_lane < lane_count:
        traci.vehicle.changeLane(veh_id, target_lane, 5.0)

# -------------------------
# Main
# -------------------------
def run():
    step = 0
    print("Running simulation: crazy behaviors + detector state...\n")

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        # 1) apply crazy behaviors
        make_some_vehicles_behave_weird(step)

        # 2) help normal cars bypass
        help_normal_cars_overtake()

        # 3) read detectors (print every 10 steps to reduce spam)
        if step % 10 == 0:
            t = traci.simulation.getTime()
            print(f"t={t:.1f}  state={get_state_arm_level()}")

    traci.close()

if __name__ == "__main__":
    traci.start(SUMO_CMD)
    run()
