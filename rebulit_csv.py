#%%
"""
Rebuild summary_all.csv (and each run's summary.csv) from the saved
best_rec_*.pth result files, which contain everything the CSV rows hold.
The existing CSV files are backed up first; the .pth files are only read.
"""
import csv
import glob
import os
import shutil
import time

import torch

ROOT = "Results/results_gridsearch"

# same columns as pnp_utils.CSV_FIELDS
CSV_FIELDS = ["run_id", "data", "flux", "loss", "fidelity", "prior", "algo", "lambda",
              "sigma", "step_scale", "stepsize", "iter", "psnr", "ssim", "microssim"]


def _clean(v):
    """Plain numbers for the CSV: tensors -> floats, 10.0 -> 10, None -> empty."""
    if v is None:
        return ""
    if torch.is_tensor(v):
        v = v.item() if v.numel() == 1 else None
        if v is None:
            return ""
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return int(v)
    return v


def result_row(res, run_id):
    return {
        "run_id": run_id,
        "data": res.get("name", ""),
        "flux": _clean(res.get("flux")),
        "loss": res.get("loss", ""),
        "fidelity": res.get("fidelity", ""),
        "prior": res.get("prior", ""),
        "algo": res.get("algo", ""),
        "lambda": _clean(res.get("lambda")),
        "sigma": _clean(res.get("sigma")),
        "step_scale": _clean(res.get("step_scale")),
        "stepsize": _clean(res.get("stepsize")),
        "iter": _clean(res.get("iter")),
        "psnr": _clean(res.get("psnr")),
        "ssim": _clean(res.get("ssim")),
        "microssim": _clean(res.get("microssim")),
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def rebuild(root=ROOT, per_run=True):
    files = sorted(glob.glob(os.path.join(root, "*", "best_rec_*.pth")))
    print(f"{len(files)} result files found in {root}/*/")

    rows, unreadable = [], []
    for path in files:
        run_id = os.path.basename(os.path.dirname(path))      # the folder is the run
        try:
            res = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            unreadable.append((path, e))
            continue
        rows.append(result_row(res, run_id))

    key = lambda r: (str(r["prior"]), str(r["data"]), str(r["fidelity"]),
                     float(r["flux"]) if r["flux"] != "" else 0.0, str(r["loss"]), r["run_id"])
    rows.sort(key=key)

    # back up and rewrite summary_all.csv
    stamp = time.strftime("%Y%m%d-%H%M%S")
    all_path = os.path.join(root, "summary_all.csv")
    if os.path.exists(all_path):
        shutil.copy(all_path, f"{all_path}.bak-{stamp}")
    write_csv(all_path, rows)
    print(f"summary_all.csv rebuilt: {len(rows)} rows")

    # each run's own summary.csv
    if per_run:
        by_run = {}
        for r in rows:
            by_run.setdefault(r["run_id"], []).append(r)
        for run_id, run_rows in by_run.items():
            p = os.path.join(root, run_id, "summary.csv")
            if os.path.exists(p):
                shutil.copy(p, f"{p}.bak-{stamp}")
            write_csv(p, run_rows)
        print(f"{len(by_run)} run folders: summary.csv rebuilt")

    for path, e in unreadable:
        print(f"could not read {path}: {e}")

    partial = glob.glob(os.path.join(root, "*", "partial_*.pth"))
    if partial:
        print(f"{len(partial)} unfinished searches (partial_*.pth) are not in the table; "
              "rerunning the experiment resumes them")
    flat = glob.glob(os.path.join(root, "best_rec_*.pth"))
    if flat:
        print(f"{len(flat)} result files directly in {root}/ (older layout, no run folder): "
              "register them with register_legacy_results.py")
    return rows


#%%
if __name__ == "__main__":
    rebuild(ROOT)