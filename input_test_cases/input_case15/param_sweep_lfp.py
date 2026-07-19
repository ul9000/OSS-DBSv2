"""Bipolar-LFP radius sweep, followed directly by the RMS-vs-radius plot.

For each radius in [start, stop) (mm = value / 10), ossdbs is run twice via
reciprocity: once with contact 1 as the active source (E1C1) and once with
contact 2 (E1C2). Each run writes c<contact>_lfp_at_contact_in_time_<radius>.csv
into the model's OutputPath. plotrms_bipolar_lfp.plot_rms_bipolar() then
differences the two contacts' traces to get the bipolar LFP and plots RMS vs.
radius, reusing the same OutputPath and radius range so the two stages can
never drift out of sync.

The point-model file, encapsulation layer thickness, and DTI flag together
determine the result folder name, e.g. for
  --pointmodel-file ../../input_files/I_filtered_1000_350_sigma6_myGilliesSTNwithAIS.h5
  --encap 0 --dti true
the model's OutputPath becomes:
  ./Results_1000_350_sigma6_myGilliesSTNwithAIS_Encap0_DTI

Run with:
  nohup python3 param_sweep_lfp.py \\
      --pointmodel-file ../../input_files/I_filtered_1000_350_sigma6_myGilliesSTNwithAIS.h5 \\
      --encap 0 --dti true --radius-start 5 --radius-stop 16 --radius-step 1 \\
      > param_sweep_lfp.log 2>&1 &
"""

import argparse
import json
import subprocess
from pathlib import Path

from plotrms_bipolar_lfp import plot_rms_bipolar

CASE_DIR = Path(__file__).resolve().parent
TEMPLATE = CASE_DIR / "input_ratlfp_stn.json"

dt = 0.0004  # sampling interval [s], must match StimulationSignal Timestep[s]


def str2bool(value):
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pointmodel-file",
        default=None,
        help="Path for PointModel->Pathway->FileName, as it should appear in the "
        "input JSON (e.g. ../../input_files/I_filtered_1000_350_sigma6_myGilliesSTNwithAIS.h5). "
        "Defaults to the value already in input_ratlfp_stn.json.",
    )
    parser.add_argument(
        "--encap",
        type=int,
        default=0,
        help="Encapsulation layer thickness in micrometers (um). Default: 0.",
    )
    parser.add_argument(
        "--dti",
        type=str2bool,
        default=False,
        help="Whether the diffusion tensor (DTI) is active: true or false. Default: false.",
    )
    # Radius of the neurons to include in the LFP calculation, in tenths of a millimeter.
    parser.add_argument("--radius-start", type=int, default=5)
    parser.add_argument("--radius-stop", type=int, default=16)
    parser.add_argument("--radius-step", type=int, default=1)
    return parser.parse_args()


def _pointmodel_tag(pointmodel_file):
    """input_files/I_filtered_1000_350_sigma6_myGilliesSTNwithAIS.h5 -> 1000_350_sigma6_myGilliesSTNwithAIS"""
    return Path(pointmodel_file).stem.removeprefix("I_filtered_")


def build_output_path(pointmodel_file, encap_um, dti_active):
    tag = _pointmodel_tag(pointmodel_file)
    dti_tag = "DTI" if dti_active else "noDTI"
    return f"./Results_{tag}_Encap{encap_um}_{dti_tag}"


def _activate_contact(data, active_id):
    for electrode in data["Electrodes"]:
        for contact in electrode["Contacts"]:
            is_active = contact["Contact_ID"] == active_id
            contact["Active"] = is_active
            contact["Current[A]"] = 1.0 if is_active else 0.0
            contact["Voltage[V]"] = 1.0 if is_active else 0.0
            contact["Floating"] = False


def run_sweep(active_contact_id, pointmodel_file, encap_um, dti_active, output_path, start, stop, step):
    for i in range(start, stop, step):
        filename = CASE_DIR / f"input_ratlfp_stn_{i}.json"

        with open(TEMPLATE) as file:
            data = json.load(file)

        data["StimulationSignal"]["LFP_radius[mm]"] = i / 10  # convert to mm
        data["PointModel"]["Pathway"]["FileName"] = pointmodel_file
        data["MaterialDistribution"]["DiffusionTensorActive"] = dti_active
        data["OutputPath"] = output_path
        for electrode in data["Electrodes"]:
            electrode["EncapsulationLayer"]["Thickness[mm]"] = encap_um / 1000
        _activate_contact(data, active_contact_id)

        with open(filename, "w") as file:
            json.dump(data, file, indent=4)

        # cwd=CASE_DIR so the model's relative OutputPath resolves here
        # check=True: abort the sweep on the first failed run instead of silently
        # continuing on to the next radius and then plotting incomplete data
        subprocess.run(["ossdbs", str(filename)], cwd=CASE_DIR, check=True)
        filename.unlink()


if __name__ == "__main__":
    args = parse_args()

    with open(TEMPLATE) as file:
        pointmodel_file = args.pointmodel_file or json.load(file)["PointModel"]["Pathway"]["FileName"]

    output_path = build_output_path(pointmodel_file, args.encap, args.dti)

    run_sweep(1, pointmodel_file, args.encap, args.dti, output_path, args.radius_start, args.radius_stop, args.radius_step)  # E1C1
    run_sweep(2, pointmodel_file, args.encap, args.dti, output_path, args.radius_start, args.radius_stop, args.radius_step)  # E1C2

    plot_rms_bipolar(
        results_path=CASE_DIR / output_path.removeprefix("./"),
        start=args.radius_start,
        stop=args.radius_stop,
        step=args.radius_step,
        fs=1 / dt,
    )
