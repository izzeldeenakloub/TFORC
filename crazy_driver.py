import os
import random
import xml.etree.ElementTree as ET

INPUT_FILE = "routes.rou.xml"            # original route file
OUTPUT_FILE = "routes_crazy.rou.xml"     # new file with types added
CRAZY_PERCENT = 0.30                     # 30%


def assign_crazy_driver_to_routes():
    input_path = os.path.abspath(INPUT_FILE)
    print(f"Reading: {input_path}")

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File {INPUT_FILE} not found in this folder.")
        return

    tree = ET.parse(INPUT_FILE)
    root = tree.getroot()

    # This will find ALL <vehicle> tags anywhere in the file
    vehicles = list(root.iter("vehicle"))
    total = len(vehicles)

    print(f"Found {total} <vehicle> elements.")

    if total == 0:
        print("No vehicles found. Check that you're using the correct routes.rou.xml file.")
        return

    crazy_count = int(total * CRAZY_PERCENT)
    print(f"Assigning 'crazyDriver' to {crazy_count} vehicles ({CRAZY_PERCENT*100:.0f}%).")

    # Randomly select indices for crazy drivers
    selected_indices = set(random.sample(range(total), crazy_count))

    for idx, veh in enumerate(vehicles):
        if idx in selected_indices:
            veh.set("type", "crazyDriver")
        else:
            veh.set("type", "normalDriver")  # optional default

    # Save to a NEW file so you keep the original safe
    output_path = os.path.abspath(OUTPUT_FILE)
    tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
    print(f"✔ Done! Saved modified routes to: {output_path}")


if __name__ == "__main__":
    assign_crazy_driver_to_routes()
