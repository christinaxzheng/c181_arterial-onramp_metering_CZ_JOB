#!/usr/bin/env python3
"""run_arterial.py -- experiments for the upstream-arterial-signal extension (ramp_sim 0.2).
Usage: python run_arterial.py [--outdir results_art] [--seeds 10]
  E4 offset sweep: meter green start minus arterial green start (delta), q_art=900 (ramp demand 405 veh/h),
     versus a Poisson-arrival ramp (platoon mode) and a uniform release at the same volume
  E5 demand sweep: coordinated offset (delta = x_art/v0_art) vs worst offset from E4 vs Poisson-arrival ramp
  Figures: time-space of the approach, arrival profile (10 s counts), offset sweep, demand sweep.
Prediction written before running (see design record): coordinating the meter green with the arterial platoon
arrival lowers the wait at the meter; a bad offset raises it. The effect on the freeway lane (D2 speed, D3 flow)
is predicted to be small because total ramp volume (405 veh/h) is unchanged and the freeway lane is below capacity.
"""
import argparse, csv, json, os, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ramp_sim as rs

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 19: 2.093}
D = rs.DEFAULT
ART_GREEN_START = (-D["phi_art"]) % D["cyc_art"]            # arterial green begins at t = 75 s (mod 90)
DT_COORD = abs(D["x_art"]) / D["v0_art"]                    # inter-signal offset = x_meter / v_free = 16.7 s
BASE = dict(q_main=800.0, ramp_mode="arterial", q_art=900.0)
KEYS = ["art_mean_delay_at_art_signal_s", "art_mean_delay_at_meter_s", "ramp_mean_release_to_merge_s",
        "D2_min_speed_mps", "D1_min_speed_mps", "D3_flow_vph", "D2_frac_intervals_below_15mps",
        "meter_max_queue_veh", "art_max_queue_at_art_line_veh", "art_red_runs", "meter_red_crossings",
        "ramp_mean_stopline_wait_s", "ramp_mean_total_delay_s", "ramp_unserved_end_veh", "ramp_merged", "min_gap_m"]

def phi(delta):                       # meter phase so that meter green starts delta s after arterial green starts
    return (ART_GREEN_START + delta) % D["cycle"]

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
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def segs(fn, t0, t1, step=1.0):
    out, t, cur, start = [], t0, None, t0
    while t < t1:
        s = fn(t)
        if s != cur:
            if cur is not None: out.append((start, t - start, cur))
            cur, start = s, t
        t += step
    out.append((start, t1 - start, cur)); return out

def fig_timespace(outdir, worst_delta):
    """Time-space of the approach for coordinated vs worst offset (blue: arterial approach, red: ramp lane)."""
    t0, t1 = 1500.0, 1800.0
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axx, (title, d) in zip(ax, (("Coordinated offset (%.1f s)" % DT_COORD, DT_COORD), ("Worst offset (%d s)" % worst_delta, worst_delta))):
        r = rs.run({**BASE, "meter_phi": phi(d), "seed": 1, "record_traj": True, "traj_dt": 1.0}); c = r["cfg"]; tr = {}
        for t, vid, x, v in r["traj_art"] + r["traj_ramp"]:
            if t0 <= t <= t1: tr.setdefault(vid, []).append((t, x))
        for vid, p in tr.items():
            p.sort(); ts = [q[0] for q in p]; xs = [q[1] for q in p]
            k = next((i for i, x in enumerate(xs) if x >= 0.0), len(xs))
            axx.plot(ts[:k + 1], xs[:k + 1], lw=0.6, color="tab:blue")
            if k < len(xs): axx.plot(ts[k:], xs[k:], lw=0.6, color="tab:red")
        col = {"G": "#2fbf71", "Y": "#f2a900", "R": "#d7263d"}
        for s0, w, s in segs(lambda t: rs.art_state_at(t, c), t0, t1): axx.broken_barh([(s0, w)], (c["x_art"] - 10, 20), facecolors=col[s])
        for s0, w, s in segs(lambda t: "G" if rs.meter_green_at(t, c) else "R", t0, t1): axx.broken_barh([(s0, w)], (-10, 20), facecolors=col[s])
        axx.text(t0 + 3, c["x_art"] + 14, "arterial signal (x_art = %d m)" % c["x_art"], fontsize=8)
        axx.text(t0 + 3, 14, "ramp meter (x = 0)", fontsize=8)
        axx.set(xlim=(t0, t1), ylim=(-600, 500), xlabel="time (s)", title=title); axx.grid(alpha=.3)
    ax[0].set_ylabel("ramp coordinate (m); blue = arterial approach, red = ramp lane")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_art_timespace.png"), dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default="results_art"); ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True); n = a.seeds
    ramp_dem = BASE["q_art"] * D["p_ramp"]
    # ---------------- E4 offset sweep ----------------
    deltas = [0, 10, round(DT_COORD, 1), 25, 35, 45, 55, 65, 75, 85]
    e4 = []
    for d in deltas:
        m = batch({**BASE, "meter_phi": phi(d)}, n); e4.append(dict(case="arterial", delta_s=d, ramp_demand=ramp_dem, **m))
        print("E4 delta", d, round(m["art_mean_delay_at_meter_s"], 1), round(m["ramp_mean_release_to_merge_s"], 1), round(m["D2_min_speed_mps"], 1))
    base_p = batch(dict(q_main=800.0, q_ramp=ramp_dem, ramp_mode="platoon", rng_split=1), n)
    base_u = batch(dict(q_main=800.0, q_ramp=ramp_dem, ramp_mode="uniform", rng_split=1), n)
    e4 += [dict(case="poisson_platoon", delta_s="", ramp_demand=ramp_dem, **base_p), dict(case="uniform", delta_s="", ramp_demand=ramp_dem, **base_u)]
    write(os.path.join(a.outdir, "E4_offset_sweep.csv"), e4)
    arts = [r for r in e4 if r["case"] == "arterial"]
    best = min(arts, key=lambda r: r["art_mean_delay_at_meter_s"]); worst = max(arts, key=lambda r: r["art_mean_delay_at_meter_s"])
    print("best offset", best["delta_s"], "worst offset", worst["delta_s"])
    # ---------------- E5 demand sweep ----------------
    e5 = []
    for qa in (600.0, 900.0, 1200.0, 1500.0):
        qr = qa * D["p_ramp"]
        for case, cfg in (("coordinated", {**BASE, "q_art": qa, "meter_phi": phi(DT_COORD)}),
                          ("worst_offset", {**BASE, "q_art": qa, "meter_phi": phi(worst["delta_s"])}),
                          ("poisson_platoon", dict(q_main=800.0, q_ramp=qr, ramp_mode="platoon", rng_split=1))):
            m = batch(cfg, n); e5.append(dict(case=case, q_art=qa, ramp_demand=qr, **m))
            print("E5", case, qa, round(m.get("art_mean_delay_at_meter_s", m["ramp_mean_stopline_wait_s"]), 1), round(m["D3_flow_vph"]))
    write(os.path.join(a.outdir, "E5_demand_sweep.csv"), e5)
    # ---------------- figures ----------------
    fig_timespace(a.outdir, worst["delta_s"])
    # (2) arrival profile: arterial-released stream vs Poisson ramp arrivals at the same hourly volume
    r_art = rs.run({**BASE, "meter_phi": phi(DT_COORD), "seed": 1}); r_poi = rs.run(dict(q_main=800.0, q_ramp=ramp_dem, ramp_mode="platoon", rng_split=1, seed=1))
    w0, w1, bw = 1200.0, 2100.0, 10.0
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for axx, ev, ttl in ((ax[0], r_art["events"]["art_pass"], "Stream leaving the arterial signal (cycle 90 s, green 35 s)"), (ax[1], r_poi["events"]["ramp_arr"], "Poisson arrivals at the same volume (%d veh/h)" % ramp_dem)):
        axx.hist([e for e in ev if w0 <= e < w1], bins=[w0 + bw * i for i in range(int((w1 - w0) / bw) + 1)], color="tab:blue", edgecolor="white")
        axx.set(ylabel="vehicles per 10 s", title=ttl); axx.grid(alpha=.3)
    for s0, w, s in segs(lambda t: rs.art_state_at(t, r_art["cfg"]), w0, w1):
        if s == "G": ax[0].axvspan(s0, s0 + w, color="#2fbf71", alpha=0.18)
    ax[1].set_xlabel("time (s); shaded = arterial green"); plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "fig_art_arrival_profile.png"), dpi=150); plt.close()
    # (3) offset sweep
    x = [r["delta_s"] for r in arts]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    for axx, k, yl, ttl, base in ((ax[0], "art_mean_delay_at_meter_s", "mean delay at ramp meter (s)", "Wait at the ramp meter", False),
                                  (ax[1], "ramp_mean_total_delay_s", "mean delay, arterial line to merge (s)", "Total delay to merge", True),
                                  (ax[2], "D2_min_speed_mps", "min 60-s speed at D2 (m/s)", "Speed dip at the merge", True)):
        axx.errorbar(x, [r[k] for r in arts], yerr=[r[k + "_ci"] for r in arts], fmt="o-", capsize=3, label="arterial feeder")
        if base:
            axx.axhline(base_p[k], color="tab:orange", ls="--", label="Poisson ramp arrivals"); axx.axhline(base_u[k], color="tab:green", ls=":", label="uniform release")
        axx.axvline(DT_COORD, color="k", lw=0.8, ls="-.", label="x_art / v_free")
        axx.set(xlabel="meter green start minus arterial green start (s)", ylabel=yl, title=ttl); axx.grid(alpha=.3)
    ax[1].legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "fig_art_offset_sweep.png"), dpi=150); plt.close()
    # (4) demand sweep
    fig, ax = plt.subplots(2, 2, figsize=(11, 7)); ax = ax.ravel()
    for case, mk in (("coordinated", "o-"), ("worst_offset", "s--"), ("poisson_platoon", "^:")):
        d = [r for r in e5 if r["case"] == case]; xs = [r["ramp_demand"] for r in d]
        for axx, k in zip(ax, ("ramp_mean_total_delay_s", "ramp_unserved_end_veh", "D3_flow_vph", "D2_min_speed_mps")):
            axx.errorbar(xs, [r[k] for r in d], yerr=[r[k + "_ci"] for r in d], fmt=mk, capsize=3, label=case.replace("_", " "))
    for axx, (yl, ttl) in zip(ax, (("mean delay to merge, merged vehicles (s)", "Total delay to merge"), ("vehicles still waiting at end of run", "Unserved ramp vehicles at 3,900 s"),
                                   ("D3 discharge (veh/h)", "Downstream discharge"), ("min 60-s speed at D2 (m/s)", "Speed dip at the merge"))):
        axx.set(xlabel="ramp demand p_ramp x q_art (veh/h)", ylabel=yl, title=ttl); axx.grid(alpha=.3)
    ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "fig_art_demand_sweep.png"), dpi=150); plt.close()
    json.dump(dict(coord_delta_s=DT_COORD, best=best["delta_s"], worst=worst["delta_s"], seeds=n), open(os.path.join(a.outdir, "summary.json"), "w"))
    print("done ->", a.outdir)

if __name__ == "__main__":
    main()
