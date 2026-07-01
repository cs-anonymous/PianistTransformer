#!/usr/bin/env python
import argparse
import json
from pathlib import Path


RAW_KEYS = ("ioi_wass", "duration_wass", "velocity_wass", "pedal_wass")


def parse_args():
    parser = argparse.ArgumentParser(description="Report INR0624 PN/PP raw Wasserstein metrics.")
    parser.add_argument("run_root", type=Path, nargs="?", default=Path("results/inr0624_epr_logscale_4gpu"))
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include older summary.json files under the run root as well.",
    )
    return parser.parse_args()


def get_nested(row, *keys):
    value = row
    for key in keys:
        value = value[key]
    return value


def fmt(value):
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def main():
    args = parse_args()
    summaries = sorted(args.run_root.glob("*/summary.json"))
    if not args.all:
        summaries = [
            path for path in summaries
            if "_seed" in path.parent.name
            and ("bias_correction" in path.parent.name or "calibrated_residual" in path.parent.name)
        ]
    if not summaries:
        raise SystemExit(f"No summary.json found under {args.run_root}")

    header = [
        "run",
        "protocol",
        "PN_ioi",
        "PN_dur",
        "PN_vel",
        "PN_pedal",
        "PP_ioi",
        "PP_dur",
        "PP_vel",
        "PP_pedal",
    ]
    print("\t".join(header))
    for path in summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run = path.parent.name
        for protocol in ("deterministic", "sampling"):
            metrics = payload["metrics"][protocol]["aggregate"]
            row = [run, protocol]
            row.extend(fmt(metrics["pn_wass"].get(key)) for key in RAW_KEYS)
            row.extend(fmt(metrics["pp_wass"].get(key)) for key in RAW_KEYS)
            print("\t".join(row))


if __name__ == "__main__":
    main()
