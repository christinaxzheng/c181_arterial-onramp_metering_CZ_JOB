#!/usr/bin/env python3
"""run_experiments.py -- reproduces every table/figure for Intermediate 1/2.
Usage: python run_experiments.py [--outdir results] [--seeds 3]
Experiments (prediction written in design record BEFORE running):
  E1 ramp-demand sweep, platoon (signal) vs uniform release at identical hourly volume
  E2 time-step sensitivity (dt = 0.5, 0.25, 0.1 s)
  E3 signal timing (cycle/green at equal volume): short vs long cycle
Figures: time-space diagram, detector speed series, sweep plot.
"""
import argparse, csv, os, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ramp_sim as rs

KEYS = ["D3_flow_vph", "D2_min_speed_mps", "D1_min_speed_mps", "D2_frac_intervals_below_15mps",
        "ramp_stopline_residual_veh", "ramp_mean_stopline_wait_s", "ramp_mean_release_to_merge_s",
        "main_mean_entry_wait_s", "min_gap_m"]

def avg(cfg, seeds):
    rows = [rs.run({**cfg, "seed": s})["metrics"] for s in range(1, seeds + 1)]
    assert all(r["conservation_ok"] and r["collisions"] == 0 for r in rows), "conservation/collision failure"
    return {k: st.mean(r[k] for r in rows) for k in KEYS}, rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default="results"); ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    qm = 800.0
    # ---- E1 ----
    sweep = []
    for mode in ("platoon", "uniform"):
        for qr in (0, 100, 200, 300, 400, 500, 600):
            m, _ = avg(dict(q_main=qm, q_ramp=float(qr), ramp_mode=mode), a.seeds)
            sweep.append(dict(mode=mode, q_main=qm, q_ramp=qr, total=qm + qr, **m)); print("E1", mode, qr, round(m["D3_flow_vph"]), round(m["D2_min_speed_mps"], 1))
    with open(os.path.join(a.outdir, "E1_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys())); w.writeheader(); w.writerows(sweep)
    # ---- E2 ----
    rows = []
    for dt in (0.5, 0.25, 0.1):
        m = rs.run(dict(q_main=qm, q_ramp=400.0, ramp_mode="platoon", seed=1, dt=dt))["metrics"]
        rows.append(dict(dt=dt, D3_flow_vph=m["D3_flow_vph"], D2_min_speed=m["D2_min_speed_mps"], ramp_merged=m["ramp_merged"], min_gap_m=m["min_gap_m"])); print("E2", rows[-1])
    with open(os.path.join(a.outdir, "E2_dt_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # ---- E3 ----
    rows = []
    for cyc, g in ((60, 27), (90, 40), (120, 53), (150, 67)):   # same green ratio ~0.44
        m, _ = avg(dict(q_main=qm, q_ramp=400.0, ramp_mode="platoon", cycle=float(cyc), green=float(g)), a.seeds)
        rows.append(dict(cycle_s=cyc, green_s=g, **m)); print("E3", cyc, round(m["D2_min_speed_mps"], 1), round(m["ramp_stopline_residual_veh"]))
    with open(os.path.join(a.outdir, "E3_signal_timing.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # ---- figures ----
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for mode, mk in (("platoon", "o-"), ("uniform", "s--")):
        d = [r for r in sweep if r["mode"] == mode]
        ax[0].plot([r["total"] for r in d], [r["D3_flow_vph"] for r in d], mk, label=mode)
        ax[1].plot([r["total"] for r in d], [r["D2_min_speed_mps"] for r in d], mk, label=mode)
        ax[2].plot([r["total"] for r in d], [r["ramp_mean_release_to_merge_s"] for r in d], mk, label=mode)
    ax[0].set(xlabel="mainline-lane + ramp demand (veh/h)", ylabel="D3 discharge (veh/h)", title="Downstream discharge"); ax[0].plot([800, 1400], [800, 1400], "k:", lw=1, label="demand")
    ax[1].set(xlabel="mainline-lane + ramp demand (veh/h)", ylabel="min 60-s speed at D2 (m/s)", title="Speed dip at merge"); ax[1].axhline(15, color="r", lw=0.8, ls=":")
    ax[2].set(xlabel="mainline-lane + ramp demand (veh/h)", ylabel="mean ramp release-to-merge time (s)", title="Ramp delay")
    for x in ax: x.legend(); x.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "fig_E1_sweep.png"), dpi=150); plt.close()
    for qr, mode in ((400.0, "platoon"), (400.0, "uniform")):
        r = rs.run(dict(q_main=qm, q_ramp=qr, ramp_mode=mode, seed=1, record_traj=True))
        tr = r["traj"]; ids = {}
        for t, vid, x, v, o in tr: ids.setdefault(vid, []).append((t, x, v, o))
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw=dict(width_ratios=[2, 1]))
        for vid, p in ids.items():
            if p[0][0] < 1200 or p[0][0] > 2400 and p[-1][0] > 3000: pass
            ax[0].plot([q[0] for q in p], [q[1] for q in p], lw=0.4, color="tab:red" if p[0][3] == "ramp" else "0.6")
        ax[0].set(xlim=(1500, 2100), ylim=(1000, 2600), xlabel="time (s)", ylabel="position (m)", title=f"Time-space ({mode}, q_ramp={int(qr)}); red = ramp-origin")
        ax[0].axhline(2000, color="b", lw=.5, ls=":"); ax[0].axhline(1800, color="b", lw=.5, ls=":")
        for sid in ("D1", "D2", "D3"):
            d = [x for x in r["records"] if x["sensor_id"] == sid]
            ax[1].plot([(x["interval_start_s"] + x["interval_end_s"]) / 2 for x in d], [x["mean_speed_mps"] for x in d], label=sid)
        ax[1].set(xlabel="time (s)", ylabel="60-s mean speed (m/s)", title="Detector speeds"); ax[1].legend(); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(a.outdir, f"fig_timespace_{mode}.png"), dpi=150); plt.close()
    print("done ->", a.outdir)

if __name__ == "__main__":
    main()
