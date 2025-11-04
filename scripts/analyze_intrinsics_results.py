#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, csv, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##########################
# config
##########################

ROOT = "output/chess_sweep"
EVAL_ROOT = f"{ROOT}/eval"
CSV_FILE = f"{ROOT}/fusion_results_all.csv"
OUT_ROOT = f"{ROOT}/analysis"
PLOTS_DIR = f"{OUT_ROOT}/plots"
TABLE_DIR = f"{OUT_ROOT}/tables"

for d in [OUT_ROOT, PLOTS_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)

# log parse keyword
pattern = re.compile(r"Median Error:\s*([\d\.]+)deg,\s*([\d\.]+)cm")

# baseline values
BASELINE_ROT = 0.1837280332485233
BASELINE_TRANS = 0.5444060079753399
METRICS = ["median_rot_deg", "median_trans_cm"]

# robust normalization settings
CAP_PERCENTILE = 95
EPS = 1e-12


##########################
# utils
##########################

def percent_improve(val, baseline):
    return 100.0 * (val - baseline) / (baseline + EPS)

def percent_reduction(val, baseline):
    return 100.0 * (baseline - val) / (baseline + EPS)

def robust_cap(values, pct=95):
    cap = np.percentile(values, pct)
    return np.minimum(values, cap), cap

def normalize_lower_better(x, min_v, cap_v):
    x = np.clip(x, min_v, cap_v)
    return (x - min_v) / (cap_v - min_v + EPS)

def geometric_mean(a, b):
    return math.sqrt(max(a, EPS) * max(b, EPS))


##########################
# step1: parse logs → csv
##########################

rows = []
for mode in sorted(os.listdir(EVAL_ROOT)):
    logpath = os.path.join(EVAL_ROOT, mode, "eval.log")
    if not os.path.isfile(logpath):
        continue

    with open(logpath) as f:
        txt = f.read()

    m = pattern.search(txt)
    if not m:
        print(f"[WARN] no median in {logpath}")
        continue

    rot, trans = float(m.group(1)), float(m.group(2))
    rows.append(["chess", mode, rot, trans])
    print(f"[OK] {mode}: {rot:.2f} deg, {trans:.2f} cm")

# add baseline
rows.append(["chess", "baseline", BASELINE_ROT, BASELINE_TRANS])
print("[OK] added baseline")

# write csv
with open(CSV_FILE, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "mode", "median_rot_deg", "median_trans_cm"])
    w.writerows(rows)

print(f"✅ saved CSV → {CSV_FILE}")


##########################
# step2: load & visualize
##########################

df = pd.read_csv(CSV_FILE)

def plot_log(sub, scene, metric):
    modes = sub["mode"]
    vals = sub[metric].values
    x = np.arange(len(modes))
    plt.figure(figsize=(10,5))
    plt.bar(x, vals)
    plt.yscale("log")
    plt.xticks(x, modes, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"{scene} | {metric} (log)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{scene}_{metric}_log.png", dpi=200)
    plt.close()

def plot_relative(sub, scene, metric):
    if "baseline" in sub["mode"].values:
        baseline = float(sub[sub["mode"]=="baseline"][metric].iloc[0])
        tag = "baseline"
    else:
        baseline = sub[metric].min()
        tag = "best"

    modes = sub["mode"]
    vals = sub[metric].values
    rel = np.array([percent_improve(v, baseline) for v in vals])
    rel_cap, cap = robust_cap(rel, CAP_PERCENTILE)

    x = np.arange(len(modes))
    plt.figure(figsize=(10,5))
    plt.axhline(0, ls="--")
    plt.bar(x, rel_cap)
    plt.xticks(x, modes, rotation=45, ha="right")
    plt.ylabel(f"Relative to {tag} (%) ↓ better")
    plt.title(f"{scene} | {metric} rel (cap @ {cap:.1f}%)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{scene}_{metric}_relative.png", dpi=200)
    plt.close()

def plot_rank(sub, scene):
    sub = sub.copy()
    for m in METRICS:
        sub[f"rank_{m}"] = sub[m].rank(method="min")
    sub["avg_rank"] = sub[[f"rank_{METRICS[0]}", f"rank_{METRICS[1]}"]].mean(axis=1)
    sub = sub.sort_values("avg_rank")

    modes = sub["mode"]
    vals = sub["avg_rank"].values
    x = np.arange(len(modes))

    plt.figure(figsize=(10,5))
    plt.bar(x, vals)
    plt.xticks(x, modes, rotation=45, ha="right")
    plt.ylabel("Avg Rank (lower better)")
    plt.title(f"{scene} | Fusion Rank")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{scene}_rank.png", dpi=200)
    plt.close()

def plot_radar(sub, scene):
    labels = sub["mode"].tolist()
    data = []
    for m in METRICS:
        v = sub[m].values
        mn = v.min(); cap = np.percentile(v, CAP_PERCENTILE)
        norm = normalize_lower_better(v, mn, cap)
        data.append(1.0 - norm)

    arr = np.stack(data, axis=1)
    angles = np.linspace(0, 2*np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    for i, mode in enumerate(labels):
        vals = arr[i].tolist(); vals += vals[:1]
        ax.plot(angles, vals, marker='o'); ax.fill(angles, vals, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_"," ") for m in METRICS])
    plt.title(f"{scene} | Radar (norm robust)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{scene}_radar.png", dpi=200)
    plt.close()

def composite(sub, scene):
    sub = sub.copy()
    scores = []
    for _, r in sub.iterrows():
        vals = []
        for m in METRICS:
            arr = sub[m].values
            mn = arr.min(); cap = np.percentile(arr, CAP_PERCENTILE)
            x = normalize_lower_better(np.array([r[m]]), mn, cap)[0]
            vals.append(1-x)
        scores.append(geometric_mean(vals[0], vals[1]))

    sub["score"] = scores
    sub = sub.sort_values("score", ascending=False)
    sub.to_csv(f"{TABLE_DIR}/{scene}_composite_rank.csv", index=False)
    print(f"[{scene}] ✅ composite saved")

def latex_escape(text):
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )

def latex_table(sub, scene):
    sub = sub.copy()
    sub = sub.sort_values(METRICS, ascending=[True] * len(METRICS))

    has_baseline = "baseline" in sub["mode"].values
    if has_baseline:
        baseline_row = sub[sub["mode"] == "baseline"].iloc[0]
        baseline_rot = float(baseline_row[METRICS[0]])
        baseline_trans = float(baseline_row[METRICS[1]])
    else:
        baseline_rot = baseline_trans = None

    body = []
    for _, row in sub.iterrows():
        mode_raw = str(row["mode"])
        mode = latex_escape(mode_raw)
        rot = float(row[METRICS[0]])
        trans = float(row[METRICS[1]])
        fields = [mode, f"{rot:.3f}", f"{trans:.3f}"]
        if has_baseline:
            if mode_raw == "baseline":
                rot_delta = trans_delta = 0.0
            else:
                rot_delta = percent_reduction(rot, baseline_rot)
                trans_delta = percent_reduction(trans, baseline_trans)
            fields.extend([f"{rot_delta:.1f}", f"{trans_delta:.1f}"])
        body.append(" & ".join(fields) + " \\")

    header_cols = ["Mode", "$\\mathrm{Rot}\\downarrow$", "$\\mathrm{Trans}\\downarrow$"]
    if has_baseline:
        header_cols.extend(["$\\Delta$ Rot (\%) $\\uparrow$", "$\\Delta$ Trans (\%) $\\uparrow$"])

    col_spec = "l" + "r" * (len(header_cols) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{scene.title()} intrinsics fusion results. Lower is better.}}",
        f"\\label{{tab:{scene}_fusion}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header_cols) + " \\",
        "\\midrule",
        "\n".join(body),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    latex_path = f"{TABLE_DIR}/{scene}_fusion_table.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[{scene}] ✅ LaTeX table saved → {latex_path}")

scene = "chess"
sub = df[df["scene"]=="chess"].copy()

for m in METRICS:
    plot_log(sub, scene, m)
    plot_relative(sub, scene, m)
plot_rank(sub, scene)
plot_radar(sub, scene)
composite(sub, scene)
latex_table(sub, scene)

print("\n✅ ALL DONE")
print(f"📁 results saved in {OUT_ROOT}")
