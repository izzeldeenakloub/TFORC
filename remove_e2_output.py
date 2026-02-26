#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import shutil
import os

# Ask user for the add file path
add_file_path = input("Enter path to khalda.add.xml: ").strip()

# Backup the original file
backup_path = add_file_path + ".backup"
shutil.copy(add_file_path, backup_path)
print(f"[+] Backup created: {backup_path}")

# Parse XML
tree = ET.parse(add_file_path)
root = tree.getroot()

# Ensure detectors folder exists
os.makedirs("detectors", exist_ok=True)
output_file = "detectors/allDetectors.xml"

# Update all laneAreaDetector elements
count = 0
for detector in root.findall("laneAreaDetector"):
    detector.set("file", output_file)
    count += 1

# Save changes back to the original file
tree.write(add_file_path, encoding="utf-8", xml_declaration=True)

print(f"[✓] Redirected {count} detector outputs → {output_file}")
print("[✓] SUMO will now produce only ONE file instead of hundreds.")
