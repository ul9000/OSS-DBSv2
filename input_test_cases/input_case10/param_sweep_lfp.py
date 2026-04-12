import json
import subprocess
import os

start = 10 
stop = 12
step = 2

path = "/home/ulrike/OSS-DBSv2/input_test_cases/input_case10/"

# Simulate E1C1
for i in range(start, stop, step):
    filename = f"{path}input_ratlfp_stn_{i}.json"

    with open(f"{path}input_ratlfp_stn.json", "r") as file:
        data = json.load(file)

    data["StimulationSignal"]["LFP_radius[mm]"] = i / 10 # convert to mm

    for electrode in data["Electrodes"]:
        for contact in electrode["Contacts"]:
            if contact["Contact_ID"] == 1:
                contact["Active"] = True
                contact["Current[A]"] = 1.0
                contact["Voltage[V]"] = 1.0
                contact["Floating"] = False
            if contact["Contact_ID"] == 2:
                contact["Active"] = False
                contact["Current[A]"] = 0.0
                contact["Voltage[V]"] = 0.0
                contact["Floating"] = False

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    subprocess.run(["ossdbs", filename])
    delete_command = f"rm {path}input_ratlfp_stn_{i}.json"
    subprocess.run(delete_command, shell=True)

# Simulate E1C2
for i in range(start, stop, step):
    filename = f"{path}input_ratlfp_stn_{i}.json"

    with open(f"{path}input_ratlfp_stn.json", "r") as file:
        data = json.load(file)

    data["StimulationSignal"]["LFP_radius[mm]"] = 0.1 * i


    for electrode in data["Electrodes"]:
        for contact in electrode["Contacts"]:
            if contact["Contact_ID"] == 2:
                contact["Active"] = True
                contact["Current[A]"] = 1.0
                contact["Voltage[V]"] = 1.0
                contact["Floating"] = False
            if contact["Contact_ID"] == 1:
                contact["Active"] = False
                contact["Current[A]"] = 0.0
                contact["Voltage[V]"] = 0.0
                contact["Floating"] = False

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    subprocess.run(["ossdbs", filename])
    delete_command = f"rm {path}input_ratlfp_stn_{i}.json"
    subprocess.run(delete_command, shell=True)

    os.system('beep')
