#!/usr/bin/env python3
"""run_arterial.py -- experiments for the upstream-arterial-signal extension (ramp_sim 0.3).
Usage: python run_arterial.py [--outdir results_art] [--seeds 10]
Set-up (0.3): arterial signal 40 s green / 50 s red (cycle 90 s); the ramp meter is a UNIFORM-RELEASE meter
  (one vehicle per slot of h = 3600/q_meter s, no cycle/green/phase), so there is no signal offset to sweep.
  E4 meter-rate sweep: q_meter = f x ramp demand (f = 1.0 ... 2.0) at q_art = 900 (ramp demand 405 veh/h),
     versus a uniform release at the same volume with no arterial upstream (ramp_mode='uniform')
  E5 demand sweep: arterial feeder + uniform meter at f = 1.0 and f = 1.25 vs the uniform-release ramp
  Figures: time-space of the approach, arrival/release profile (10 s counts), meter-rate sweep, demand sweep.
Prediction (to be checked, not assumed): the 40 s green pulses arrive at the meter as platoons; a meter that releases
at exactly the demand rate (f = 1.0) is critically loaded and builds a queue, so the wait at the meter should fall
steeply as f grows, while the freeway lane (D2 speed, D3 flow) should change little because the long-run ramp
volume is the same.
"""
import argparse, csv, json, os, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ramp_sim as rs

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 19: 2.093}
D = rs.DEFAULT
BASE = dict(q_main=800.0, ramp_mode="arterial", q_art=900.0)
FACTORS = [1.0, 1.1, 1.25, 1.5, 2.0]                        # q_meter / ramp demand
KEYS = ["art_mean_delay_at_art_signal_s", "art_mean_delay_at_meter_s", "ramp_mean_release_to_merge_s",
        "D2_min_speed_mps", "D1_min_speed_mps", "D3_flow_vph", "D2_frac_intervals_below_15mps",
        "meter_max_queue_veh", "art_max_queue_at_art_line_veh", "art_red_runs", "meter_red_crossings",
        "meter_mean_headway_s", "meter_sd_headway_s",
        "ramp_mean_stopline_wait_s", "ramp_mean_total_delay_s", "ramp_unserved_end_veh", "ramp_merged", "min_gap_m"]

def ci(v):
    n = len(v); return T975[n - 1] * st.stdev(v) / n ** 0.5 if n > 1 else 0.0

def batch(cfg, seeds):
    rows = [rs.run({**cfg, "seed": s})["metrics"] for s in range(1, seeds + 1)]
    assert all(r["conservation_ok"] and r["collisions"] == 0 for r in rows), "conservation/collision failure"
    out = {}
    for k in KEYS:
        v = [r[k] for r in rows if k in r]
        if v: out[k] = st.mean(v); out[k + "_ci"] = ci(v)
    return out

def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))      # union of columns, first-seen order
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)

def segs(fn, t0, t1, step=1.0):
    out, t, cur, start = [], t0, None, t0
    while t < t1:
        s = fn(t)
        if s != cur:
            if cur is not None: out.append((start, t - start, cur))
            cur, start = s, t
        t += step
    out.append((start, t1 - start, cur)); return out

def fig_timespace(outdir, ramp_dem):
    """Time-space of the approach for two meter rates (blue: arterial approach, red: ramp lane, black ticks: meter releases)."""
    t0, t1 = 1500.0, 1800.0
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axx, f in zip(ax, (1.0, 1.25)):
        r = rs.run({**BASE, "q_meter": ramp_dem * f, "seed": 1, "record_traj": True, "traj_dt": 1.0}); c = r["cfg"]; tr = {}
        for t, vid, x, v in r["traj_art"] + r["traj_ramp"]:
            if t0 <= t <= t1: tr.setdefault(vid, []).append((t, x))
        for vid, p in tr.items():
            p.sort(); ts = [q[0] for q in p]; xs = [q[1] for q in p]
            k = next((i for i, x in enumerate(xs) if x >= 0.0), len(xs))
            axx.plot(ts[:k + 1], xs[:k + 1], lw=0.6, color="tab:blue")
            if k < len(xs): axx.plot(ts[k:], xs[k:], lw=0.6, color="tab:red")
        col = {"G": "#2fbf71", "Y": "#f2a900", "R": "#d7263d"}
        for s0, w, s in segs(lambda t: rs.art_state_at(t, c), t0, t1): axx.broken_barh([(s0, w)], (c["x_art"] - 10, 20), facecolors=col[s])
        tcs = [t for t in r["events"]["meter_cross"] if t0 <= t <= t1]
        axx.plot(tcs, [0] * len(tcs), "k|", ms=14, label="vehicle released at meter")
        axx.text(t0 + 3, c["x_art"] + 14, "arterial signal (x_art = %d m, 40 s G / 50 s R)" % c["x_art"], fontsize=8)
        axx.text(t0 + 3, 14, "uniform-release meter (x = 0)", fontsize=8)
        axx.set(xlim=(t0, t1), ylim=(-600, 500), xlabel="time (s)", title="Meter rate = %.2f x demand (%.0f veh/h)" % (f, ramp_dem * f)); axx.grid(alpha=.3)
    ax[0].set_ylabel("ramp coordinate (m); blue = arterial approach, red = ramp lane")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_art_timespace.png"), dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default="results_art"); ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True); n = a.seeds
    ramp_dem = BASE["q_art"] * D["p_ramp"]
    # ---------------- E4 meter-rate sweep ----------------
    e4 = []
    for f in FACTORS:
        m = batch({**BASE, "q_meter": ramp_dem * f}, n); e4.append(dict(case="arterial+uniform_meter", rate_factor=f, q_meter=ramp_dem * f, ramp_demand=ramp_dem, **m))
        print("E4 f", f, round(m["art_mean_delay_at_meter_s"], 1), round(m["ramp_mean_release_to_merge_s"], 1), round(m["D2_min_speed_mps"], 1))
    base_u = batch(dict(q_main=800.0, q_ramp=ramp_dem, ramp_mode="uniform", rng_split=1), n)
    e4.append(dict(case="uniform_ramp_no_arterial", rate_factor="", q_meter=ramp_dem, ramp_demand=ramp_dem, **base_u))
    write(os.path.join(a.outdir, "E4_meter_rate_sweep.csv"), e4)
    arts = [r for r in e4 if r["case"] == "arterial+uniform_meter"]
    # ---------------- E5 demand sweep ----------------
    e5 = []
    for qa in (600.0, 900.0, 1200.0, 1500.0):
        qr = qa * D["p_ramp"]
        for case, cfg in (("meter_1.00x_demand", {**BASE, "q_art": qa, "q_meter": qr}),
                          ("meter_1.25x_demand", {**BASE, "q_art": qa, "q_meter": 1.25 * qr}),
                          ("uniform_ramp_no_arterial", dict(q_main=800.0, q_ramp=qr, ramp_mode="uniform", rng_split=1))):
            m = batch(cfg, n); e5.append(dict(case=case, q_art=qa, ramp_demand=qr, **m))
            print("E5", case, qa, round(m.get("art_mean_delay_at_meter_s", m["ramp_mean_stopline_wait_s"]), 1), round(m["D3_flow_vph"]))
    write(os.path.join(a.outdir, "E5_demand_sweep.csv"), e5)
    # ---------------- figures ----------------
    fig_timespace(a.outdir, ramp_dem)
    # (2) arrival profile: stream leaving the arterial signal vs stream released by the uniform meter
    r_art = rs.run({**BASE, "q_meter": ramp_dem * 1.25, "seed": 1})
    w0, w1, bw = 1200.0, 2100.0, 10.0
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for axx, ev, ttl in ((ax[0], r_art["events"]["art_pass"], "Stream leaving the arterial signal (cycle 90 s, green 40 s, red 50 s)"),
                         (ax[1], r_art["events"]["meter_cross"], "Stream released by the uniform meter (1.25 x demand = %d veh/h)" % round(ramp_dem * 1.25))):
        axx.hist([e for e in ev if w0 <= e < w1], bins=[w0 + bw * i for i in range(int((w1 - w0) / bw) + 1)], color="tab:blue", edgecolor="white")
        axx.set(ylabel="vehicles per 10 s", title=ttl); axx.grid(alpha=.3)
    for axx in ax:
        for s0, w, s in segs(lambda t: rs.art_state_at(t, r_art["cfg"]), w0, w1):
            if s == "G": axx.axvspan(s0, s0 + w, color="#2fbf71", alpha=0.18)
    ax[1].set_xlabel("time (s); shaded = arterial green"); plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "fig_art_arrival_profile.png"), dpi=150); plt.close()
    # (3) meter-rate sweep
    x = [r["rate_factor"] for r in arts]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    for axx, k, yl, ttl, base in ((ax[0], "art_mean_delay_at_meter_s", "mean delay at ramp meter (s)", "Wait at the ramp meter", False),
                                  (ax[1], "ramp_mean_total_delay_s", "mean delay, arterial line to merge (s)", "Total delay to merge", True),
                                  (ax[2], "D2_min_speed_mps", "min 60-s speed at D2 (m/s)", "Speed dip at the merge", True)):
        axx.errorbar(x, [r[k] for r in arts], yerr=[r[k + "_ci"] for r in arts], fmt="o-", capsize=3, label="arterial feeder + uniform meter")
        if base: axx.axhline(base_u[k], color="tab:green", ls=":", label="uniform release, no arterial")
        axx.set(xlabel="meter release rate / ramp demand", ylabel=yl, title=ttl); axx.grid(alpha=.3)
    ax[1].legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "fig_art_rate_sweep.png"), dpi=150); plt.close()
    # (4) demand sweep
    fig, ax = plt.subplots(2, 2, figsize=(11, 7)); ax = ax.ravel()
    for case, mk in (("meter_1.00x_demand", "o-"), ("meter_1.25x_demand", "s--"), ("uniform_ramp_no_arterial", "^:")):
        d = [r for r in e5 if r["case"] == case]; xs = [r["ramp_demand"] for r in d]
        for axx, k in zip(ax, ("ramp_mean_total_delay_s", "ramp_unserved_end_veh", "D3_flow_vph", "D2_min_speed_mps")):
            axx.errorbar(xs, [r[k] for r in d], yerr=[r[k + "_ci"] for r in d], fmt=mk, capsize=3, label=case.replace("_", " "))
    for axx, (yl, ttl) in zip(ax, (("mean delay to merge, merged vehicles (s)", "Total delay to merge"), ("vehicles still waiting at end of run", "Unserved ramp vehicles at 3,900 s"),
                                   ("D3 discharge (veh/h)", "Downstream discharge"), ("min 60-s speed at D2 (m/s)", "Speed dip at the merge"))):
        axx.set(xlabel="ramp demand p_ramp x q_art (veh/h)", ylabel=yl, title=ttl); axx.grid(alpha=.3)
    ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "fig_art_demand_sweep.png"), dpi=150); plt.close()
    json.dump(dict(factors=FACTORS, seeds=n, art_green_s=D["g_art"], art_red_s=D["cyc_art"] - D["g_art"] - D["y_art"]),
              open(os.path.join(a.outdir, "summary.json"), "w"))
    print("done ->", a.outdir)

if __name__ == "__main__":
    main()
