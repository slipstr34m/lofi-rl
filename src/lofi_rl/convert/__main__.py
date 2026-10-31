"""python -m lofi_rl.convert <bank-id> --big <assets/big> --out <file.sf2>
   python -m lofi_rl.convert all --big <assets/big> --out-dir <folder>       (writes <bank-id>.sf2 for each)

Prints, per bank: the output's sha256 and size, regions kept and dropped (by reason), key and velocity ranges,
how many zones play a loop and where each sample's loop came from, every opcode dropped from a kept region (and
why), the fit errors (each that of the values written), and the presets as palette.sf2_presets reads them back from
the written file. --json writes the same report as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lofi_rl.convert.banks import BANKS, convert_bank
from lofi_rl.palette import sf2_presets


def _print(report: dict, out: Path, presets) -> None:
    p = print
    p(f"== {report['bank']}  ->  {out}")
    p(f"   sha256 {report['sha256']}  ({report['bytes']:,} bytes)")
    p(f"   sfz {report['folder']}/{report['sfz']} ({len(report['files'])} file(s) with includes)")
    p(f"   regions: {report['regions']} read, {report['kept']} kept, {report['dropped']} dropped")
    for reason, n in report["dropped_by_reason"].items():
        p(f"     - {n:4d} {reason}")
    p(f"   keys {report['keys'][0]}..{report['keys'][1]}; velocity layers "
      + ", ".join(f"{a}-{b}" for a, b in report["velocity_layers"]))
    p(f"   samples {report['samples']} source files -> {report['sf2_samples']} SF2 samples, {report['zones']} zones; "
      f"rates {report['sample_rates']}, bits {report['sample_bits']}, channels {report['channels']}")
    p(f"   loops: {report['looped_zones']} of {report['zones']} zones play one (sampleModes 1 or 3); each source "
      f"file's loop comes from {report['sample_loops']}")
    p(f"   velocity curve: max error {report['velocity_curve_max_error_db']} dB against the SFZ curve over each "
      f"zone's velocities (before FluidSynth's whole-centibel rounding)")
    for veltrack, curve, amounts, constant, err in report["velocity_fits"]:
        what = f"amp_veltrack {veltrack:g}" + (f" + amp_velcurve {[tuple(c) for c in curve]}" if curve else "")
        if tuple(amounts) == (960, 0, 0, 0):
            p(f"     - {what}: FluidSynth's default velocity modulator is this curve; no modulators written")
        else:
            p(f"     - {what}: modulator amounts (concave, linear, linear², concave x linear) {list(amounts)}, "
              f"constant {constant} cB, max error {err} dB")
    for k, e in report["time_fit_max_errors"].items():
        unit = "ms" if k.endswith("s)") else "%"
        p(f"   {k}: max error {e * 1000:.2f} {unit}" if unit == "ms" else f"   {k}: max error {e * 100:.2f} %")
    p(f"   attenuation: shift {report['attenuation_shift_cb']} cB, zones {report['attenuation_range_cb'][0]}.."
      f"{report['attenuation_range_cb'][1]} cB")
    if report["dropped_opcodes"]:
        p("   opcodes dropped from kept regions:")
        for k, why in report["dropped_opcodes"].items():
            p(f"     - {k}: {why}")
    for k, why in report["approximations"].items():
        p(f"   approximated {k}: {why}")
    p(f"   presets (palette.sf2_presets): {sorted(presets)}")
    p(f"   why: {report['why']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lofi_rl.convert", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bank", help=f"one of {', '.join(BANKS)}, or all")
    ap.add_argument("--big", required=True, type=Path, help="the assets/big folder holding the fetched banks")
    ap.add_argument("--out", type=Path, help="output .sf2 (one bank)")
    ap.add_argument("--out-dir", type=Path, help="output folder (all: <bank-id>.sf2 each)")
    ap.add_argument("--json", type=Path, help="also write the report(s) as JSON here")
    args = ap.parse_args(argv)
    if args.bank == "all":
        if args.out_dir is None:
            ap.error("all needs --out-dir")
        jobs = [(b, args.out_dir / f"{b}.sf2") for b in BANKS]
    else:
        if args.bank not in BANKS:
            ap.error(f"unknown bank {args.bank!r}; banks: {', '.join(BANKS)}, all")
        if args.out is None:
            ap.error("one bank needs --out")
        jobs = [(args.bank, args.out)]
    reports = {}
    for bank, out in jobs:
        conv = convert_bank(bank, args.big)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        tmp.write_bytes(conv.data)
        tmp.replace(out)
        presets = sf2_presets(out)
        _print(conv.report, out, presets)
        reports[bank] = dict(conv.report, out=str(out), presets=sorted(presets))
    if args.json:
        args.json.write_text(json.dumps(reports, indent=1, default=list) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
