import os
import sys
import random

# ---- SUMO / TraCI setup ----
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set the SUMO_HOME environment variable.")

import traci  # import AFTER adding tools to path

SUMO_BINARY = "sumo-gui"  # or "sumo"
SUMO_CONFIG = "khalda.sumocfg"

# ---- Global behavior mapping: vehicle_id -> behavior_name ----
VEH_BEHAVIOR = {}  # e.g. {"veh_0": "slowdown", "veh_1": "lane_change", ...}

BEHAVIORS = ["slowdown", "stop_near_junction", "lane_change"]

BEHAVIOR_COLORS = {
    "slowdown":           (0, 0, 255, 255),   # blue
    "stop_near_junction": (255, 0, 0, 255),   # red
    "lane_change":        (0, 255, 0, 255),   # green
}

# ---- Memory for "stuck" tracking for normal cars ----
# How many consecutive steps a normal car has been nearly stopped
STUCK_STEPS = {}

# Tune these thresholds if needed
STOP_SPEED_EPS = 0.2          # below this is "stopped"
STUCK_MIN_STEPS = 15          # must be stopped this many steps to trigger help
LEADER_LOOKAHEAD = 30         # meters ahead to look for leader
BLOCK_GAP_MAX = 10            # leader gap smaller than this means "blocked"


def run():
    step = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        make_some_vehicles_behave_weird(step)

        # NEW: help normal cars bypass stopped crazy drivers
        help_normal_cars_overtake(step)

    traci.close()


def make_some_vehicles_behave_weird(step):
    veh_ids = traci.vehicle.getIDList()

    for vid in veh_ids:
        vtype = traci.vehicle.getTypeID(vid)

        # only affect "crazyDriver" vehicles (adjust if your vType name is different)
        if vtype != "crazyDriver":
            continue

        # If this vehicle doesn't have a behavior yet, assign one randomly
        if vid not in VEH_BEHAVIOR:
            VEH_BEHAVIOR[vid] = random.choice(BEHAVIORS)

        behavior = VEH_BEHAVIOR[vid]

        # Set color based on behavior
        color = BEHAVIOR_COLORS[behavior]
        traci.vehicle.setColor(vid, color)

        # Apply the corresponding abnormal behavior
        apply_abnormal_behavior(vid, step, behavior)


def apply_abnormal_behavior(veh_id, step, behavior):
    """Choose and apply abnormal behavior based on the assigned type."""
    if behavior == "slowdown":
        abnormal_slowdown(veh_id, step)
    elif behavior == "stop_near_junction":
        abnormal_stop_near_junction(veh_id, step)
    elif behavior == "lane_change":
        abnormal_lane_change(veh_id, step)


# ---- NEW: Normal-car bypass logic ----
def help_normal_cars_overtake(step):
    """
    If a normal car is stuck behind a stopped crazyDriver, try to move it to another lane.
    This only works if:
      - current edge has multiple lanes
      - lane change is possible (SUMO will ignore if forbidden/unsafe)
    """

    veh_ids = traci.vehicle.getIDList()

    for vid in veh_ids:
        # Skip the abnormal vehicles
        if traci.vehicle.getTypeID(vid) == "crazyDriver":
            continue

        # Track how long this normal car has been nearly stopped
        speed = traci.vehicle.getSpeed(vid)

        if speed < STOP_SPEED_EPS:
            STUCK_STEPS[vid] = STUCK_STEPS.get(vid, 0) + 1
        else:
            STUCK_STEPS[vid] = 0
            continue

        # Not stuck long enough? do nothing
        if STUCK_STEPS[vid] < STUCK_MIN_STEPS:
            continue

        # Look at the leader (vehicle directly ahead)
        leader = traci.vehicle.getLeader(vid, LEADER_LOOKAHEAD)  # (leaderID, gap) or None
        if not leader:
            continue

        leader_id, gap = leader

        # We only care if the leader is a "crazyDriver" AND currently stopped/slow
        if traci.vehicle.getTypeID(leader_id) != "crazyDriver":
            continue

        if gap > BLOCK_GAP_MAX:
            continue

        if traci.vehicle.getSpeed(leader_id) > STOP_SPEED_EPS:
            continue

        # Try to change lane on the current edge
        edge_id = traci.vehicle.getRoadID(vid)
        if edge_id == "" or edge_id[0] == ":":
            continue  # internal edge / junction

        lane_count = traci.edge.getLaneNumber(edge_id)
        if lane_count <= 1:
            # No lane to bypass on. (Optional) You could reroute here instead.
            continue

        cur_lane = traci.vehicle.getLaneIndex(vid)

        # Try left then right (you can swap order if you want)
        candidate_lanes = [cur_lane + 1, cur_lane - 1]

        for target_lane in candidate_lanes:
            if 0 <= target_lane < lane_count:
                try:
                    traci.vehicle.changeLane(vid, target_lane, 5.0)
                    # After issuing a change, reset stuck counter to avoid spamming
                    STUCK_STEPS[vid] = 0
                    break
                except traci.TraCIException:
                    # If SUMO rejects the change, try the other side
                    pass


# ---- Behavior implementations ----

def abnormal_slowdown(veh_id, step):
    """Random slowdowns: the car sometimes drops its speed a lot."""
    if step % 50 != 0:
        return

    if random.random() < 0.4:  # 40% chance when condition is met
        current_speed = traci.vehicle.getSpeed(veh_id)
        new_speed = max(0.0, current_speed * 0.3)  # cut speed to ~30%
        traci.vehicle.slowDown(veh_id, new_speed, 5.0)


def abnormal_stop_near_junction(veh_id, step):
    """If the vehicle is close to the end of its lane, it may suddenly stop."""
    lane_id = traci.vehicle.getLaneID(veh_id)
    if lane_id == "":
        return  # sometimes vehicles are not on a lane yet (e.g. teleport)

    pos = traci.vehicle.getLanePosition(veh_id)
    lane_length = traci.lane.getLength(lane_id)

    # If within last 10 meters of the lane, sometimes stop
    if lane_length - pos < 10:
        if random.random() < 0.5:
            traci.vehicle.slowDown(veh_id, 0.0, 3.0)


def abnormal_lane_change(veh_id, step):
    """Aggressive lane changing: vehicle tries to switch lanes frequently."""
    if step % 20 != 0:
        return

    edge_id = traci.vehicle.getRoadID(veh_id)
    if edge_id == "" or edge_id[0] == ":":
        # empty edge id or internal junction edge – skip
        return

    lane_count = traci.edge.getLaneNumber(edge_id)
    if lane_count <= 1:
        return  # nothing to change to

    current_lane_index = traci.vehicle.getLaneIndex(veh_id)
    direction = random.choice([-1, 1])
    target_lane = current_lane_index + direction

    if 0 <= target_lane < lane_count:
        # changeLane(vehID, laneIndex, duration)
        traci.vehicle.changeLane(veh_id, target_lane, 5.0)


if __name__ == "__main__":
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])
    run()
