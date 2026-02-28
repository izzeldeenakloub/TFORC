import random

# ---------------------------
# FEEDS (your exact ranges)
# ---------------------------
west_feeds  = [f"westFeed{i}"  for i in range(1, 5)]  # westFeed2..westFeed10
south_feeds = [f"southFeed{i}" for i in range(1, 4)]   # southFeed1..southFeed4
east_feeds  = [f"eastFeed{i}"  for i in range(1, 4)]   # eastFeed1..eastFeed4
north_feeds = [f"northFeed{i}" for i in range(1, 3)]   # northFeed1..northFeed3

feeds = west_feeds + south_feeds + east_feeds + north_feeds

# ---------------------------
# SINKS (same for all feeds)
# ---------------------------
sinks = [f"Sink{i}" for i in range(1, 5)]  # Sink1..Sink4

# ---------------------------
# VIA mapping (based on feed prefix)
# ---------------------------
VIA_BY_PREFIX = {
    "westFeed":  "W_in",
    "eastFeed":  "E_in",
    "northFeed": "N_in",
    "southFeed": "S_in",
}

def via_for_feed(feed_id: str) -> str:
    for prefix, via_edge in VIA_BY_PREFIX.items():
        if feed_id.startswith(prefix):
            return via_edge
    raise ValueError(f"Unknown feed '{feed_id}'. Add its prefix to VIA_BY_PREFIX.")

# ---------------------------
# Time-of-day profiles
# ---------------------------
PROFILES = {
    "morning": (0,     7200,  90),
    "midday":  (7200,  18000, 20),
    "evening": (18000, 25200, 80),
}

# How many random OD flows per file
NUM_FLOWS = 20

# Reproducible randomness (optional)
RANDOM_SEED = 42


def write_file(name, begin, end, vph):
    random.seed(RANDOM_SEED + (hash(name) % 100000))

    fname = f"demand_{name}.rou.xml"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("<routes>\n")

        # Keep your vTypes logic as you requested
        if name == "evening":
            f.write('  <vType id="normalDriver" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.9"/>\n')
            f.write('  <vType id="crazyDriver"  vClass="passenger" accel="2.6" decel="6.0" sigma="1.0" length="5" maxSpeed="13.9"/>\n\n')

        for k in range(1, NUM_FLOWS + 1):
            feed = random.choice(feeds)
            sink = random.choice(sinks)
            via_edge = via_for_feed(feed)

            f.write(
                f'  <flow id="{name}_{k}" type="normalDriver" begin="{begin}" end="{end}" '
                f'vehsPerHour="{vph}" from="{feed}" to="{sink}" via="{via_edge}" '
                f'departLane="best" departSpeed="max"/>\n'
            )

        f.write("</routes>\n")

    print(f"Saved {fname} with {NUM_FLOWS} random flows. (via depends on feed side)")


if __name__ == "__main__":
    for name, (b, e, vph) in PROFILES.items():
        write_file(name, b, e, vph)
