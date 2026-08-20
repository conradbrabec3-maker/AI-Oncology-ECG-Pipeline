#!/usr/bin/env python3
"""
ecg_degrade_mimic.py
────────────────────
Reads WFDB (.hea + .dat) ECG records directly from the MIMIC-IV ECG zip
archive, injects combined baseline wander + EMG/motion artifacts, and
writes one noisy CSV per patient record into an output directory.

Usage
-----
    python ecg_degrade_mimic.py                         # process all records
    python ecg_degrade_mimic.py --start 0 --end 200000  # process slice only

    Optionally override paths via environment variables:
        ECG_ZIP_PATH    path to the MIMIC zip file
        ECG_OUT_DIR     directory to write degraded CSV files

    SLURM array jobs pass SLURM_ARRAY_TASK_ID automatically — the script
    uses it to select the correct slice with no extra arguments needed.

Dependencies
------------
    pip install wfdb numpy pandas scipy
"""

import os
import sys
import zipfile
import tempfile
import shutil
import argparse

import numpy as np
import pandas as pd
import wfdb


# ──────────────────────────────────────────────
# CONFIG  (override with env vars if preferred)
# ──────────────────────────────────────────────
ZIP_PATH = os.environ.get(
    "ECG_ZIP_PATH",
    "/home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip",
)
OUTPUT_DIR = os.environ.get(
    "ECG_OUT_DIR",
    "/home/conrad.brabec/ecg_degraded_output",
)

# ──────────────────────────────────────────────
# NOISE PARAMETERS  (tuned to wearable realism)
# ──────────────────────────────────────────────
WANDER_AMPLITUDE = 0.15   # fraction of signal std  → loose-strap drift
WANDER_FREQ_HZ   = 0.3    # ~respiration rate (0.15 0.4 Hz is physiologic)
EMG_AMPLITUDE    = 0.10   # fraction of signal std  → muscle/motion noise
EMG_BURST_PROB   = 0.02   # per-sample probability of a high-amplitude burst

# Number of parallel slices — must match --array 0-N in the SLURM script
N_SLICES = 5


# ══════════════════════════════════════════════
# ARTIFACT INJECTION FUNCTIONS
# ══════════════════════════════════════════════

def add_baseline_wander(signal: np.ndarray, fs: float) -> np.ndarray:
    if WANDER_AMPLITUDE == 0:
        return signal.copy()
    t = np.linspace(0, len(signal) / fs, len(signal))
    sig_std = np.std(signal) or 1.0
    wander  = WANDER_AMPLITUDE * sig_std * np.sin(2 * np.pi * WANDER_FREQ_HZ * t)
    wander += (WANDER_AMPLITUDE / 2) * sig_std * np.sin(2 * np.pi * WANDER_FREQ_HZ * 2 * t)
    return signal + wander


def add_emg_artifact(signal: np.ndarray, fs: float) -> np.ndarray:
    if EMG_AMPLITUDE == 0:
        return signal.copy()
    sig_std    = np.std(signal) or 1.0
    noise      = EMG_AMPLITUDE * sig_std * np.random.randn(len(signal))
    burst_mask = np.random.rand(len(signal)) < EMG_BURST_PROB
    n_bursts   = burst_mask.sum()
    if n_bursts:
        noise[burst_mask] += EMG_AMPLITUDE * 5 * sig_std * np.random.randn(n_bursts)
    return signal + noise


def degrade_signal(signal: np.ndarray, fs: float) -> np.ndarray:
    noisy = add_baseline_wander(signal, fs)
    noisy = add_emg_artifact(noisy, fs)
    return noisy


# ══════════════════════════════════════════════
# ZIP UTILITIES
# ══════════════════════════════════════════════

def find_hea_files(zf: zipfile.ZipFile) -> list:
    return [
        name for name in zf.namelist()
        if name.endswith(".hea") and "__MACOSX" not in name
    ]


def extract_record_to_tempdir(zf: zipfile.ZipFile, hea_path: str, tmpdir: str) -> str:
    base     = hea_path[:-4]
    dat_path = base + ".dat"
    for member in (hea_path, dat_path):
        if member in zf.namelist():
            zf.extract(member, tmpdir)
        else:
            raise FileNotFoundError(f"Missing zip member: {member}")
    return os.path.join(tmpdir, base)


# ══════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════

def process_zip(start_idx: int, end_idx: int, slice_id: int) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📂  Opening zip: {ZIP_PATH}")
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Zip not found: {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        hea_files = find_hea_files(zf)
        total_all = len(hea_files)

        # ── Slice this job's portion of the work ──────────────────────────
        hea_slice = hea_files[start_idx:end_idx]
        slice_len = len(hea_slice)
        print(f"🔍  Slice {slice_id}: records {start_idx} {end_idx} "
              f"({slice_len} of {total_all} total)\n")

        ok = skipped = errors = 0

        for local_idx, hea_path in enumerate(hea_slice, start=1):
            global_idx = start_idx + local_idx   # position in full dataset

            parts     = hea_path.replace("\\", "/").split("/")
            record_id = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            record_id = record_id.replace(".hea", "")
            out_csv   = os.path.join(OUTPUT_DIR, f"{record_id}_noisy.csv")

            # Resume logic — skip already-written files
            if os.path.exists(out_csv):
                skipped += 1
                continue

            tmpdir = tempfile.mkdtemp(prefix=f"ecg_s{slice_id}_")
            try:
                rec_path = extract_record_to_tempdir(zf, hea_path, tmpdir)
                record   = wfdb.rdrecord(rec_path)

                if record.p_signal is None or record.p_signal.size == 0:
                    print(f"  ⚠️  [{global_idx}/{total_all}] Empty signal, skipping: {record_id}")
                    skipped += 1
                    continue

                fs         = float(record.fs)
                sig_matrix = record.p_signal
                lead_names = record.sig_name or [f"lead_{i}" for i in range(sig_matrix.shape[1])]
                n_samples  = sig_matrix.shape[0]

                df = pd.DataFrame()
                df["timestamp_s"] = np.arange(n_samples) / fs

                for i, lead in enumerate(lead_names):
                    clean     = sig_matrix[:, i].astype(np.float32)
                    noisy     = degrade_signal(clean, fs).astype(np.float32)
                    safe_lead = lead.replace(" ", "_").replace("/", "_")
                    df[f"clean_{safe_lead}"] = clean
                    df[f"noisy_{safe_lead}"] = noisy

                df.to_csv(out_csv, index=False)
                ok += 1

                if local_idx % 500 == 0 or local_idx == slice_len:
                    print(f"  ✅  [{global_idx}/{total_all}] "
                          f"(slice {slice_id}: {local_idx}/{slice_len}) "
                          f"→ {os.path.basename(out_csv)}")

            except Exception as exc:
                errors += 1
                print(f"  ❌  [{global_idx}/{total_all}] ERROR on {record_id}: {exc}")

            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "═" * 55)
    print(f"📊  Slice {slice_id} Summary  (records {start_idx} {end_idx})")
    print(f"  ✅  Processed  : {ok}")
    print(f"  ↩️   Skipped    : {skipped}")
    print(f"  ❌  Errors     : {errors}")
    print(f"\n💾  Output dir  : {OUTPUT_DIR}")
    print("═" * 55)


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=None,
                        help="Start index into the sorted .hea file list")
    parser.add_argument("--end",   type=int, default=None,
                        help="End index (exclusive)")
    parser.add_argument("--slice", type=int, default=None,
                        help="Slice ID label used in log output")
    args = parser.parse_args()

    # ── Resolve slice boundaries ──────────────────────────────────────────
    # Priority: explicit --start/--end  >  SLURM_ARRAY_TASK_ID  >  full run
    total_records = 800035

    if args.start is not None and args.end is not None:
        # Explicit override (useful for manual reruns of a specific range)
        start   = args.start
        end     = args.end
        slc_id  = args.slice if args.slice is not None else 0

    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        # SLURM array job: task IDs 0 4 map to five equal slices
        task_id    = int(os.environ["SLURM_ARRAY_TASK_ID"])
        chunk_size = total_records // N_SLICES
        start      = task_id * chunk_size
        end        = total_records if task_id == N_SLICES - 1 else start + chunk_size
        slc_id     = task_id
        print(f"🔢  SLURM array task {task_id} → records {start} {end}")

    else:
        # Fallback: process everything (single-job mode)
        start  = 0
        end    = total_records
        slc_id = 0
        print("ℹ️   No slice specified — processing all records in single-job mode.")

    np.random.seed(42 + slc_id)   # unique but reproducible seed per slice
    process_zip(start_idx=start, end_idx=end, slice_id=slc_id)
