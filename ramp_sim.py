#!/usr/bin/env python3
"""
ramp_sim.py  --  minimum freeway + signal-fed on-ramp simulator (CEE 181/281 Fall 2026)

MODEL (units: seconds, meters, m/s internally)
  * Mainline: ONE lane (the right lane, the lane that receives merging traffic) with an
    open upstream boundary (boundary queue is kept, never silently dropped) and a free
    downstream boundary. Vehicle motion: Intelligent Driver Model (Treiber, Hennecke &
    Helbing 2000), ballistic position update.
  * Ramp: stop line (ramp coordinate 0) -> acceleration lane -> end of ramp (ramp_len).
    Ramp vehicles are released by the signal ('platoon' mode: Poisson arrivals queue at
    the stop line and discharge at saturation headway during green) or by a uniform
    release at the same long-run volume ('uniform' mode).
  * Merge: gap-acceptance. The ramp vehicle merges only if (i) its own IDM acceleration
    behind the new leader and (ii) the new follower's IDM acceleration behind it are both
    >= -b_safe, and both bumper-to-bumper gaps >= s0. Otherwise it keeps following the ramp
    leader / a virtual stationary obstacle at the ramp end. The follower's reaction is the
    disturbance we study.
  * NEW in 0.2 (ramp_mode='arterial'): an upstream arterial signal at ramp coordinate x_art (-250 m) feeds
    the ramp meter at ramp coordinate 0. Right-turn vehicles (p_ramp of the arterial approach flow q_art)
    arrive Poisson at x_art_spawn, follow IDM (v0_art), stop behind a virtual leader at x_art when the
    arterial signal is yellow/red (unless they cannot stop at b_go), then travel to the ramp stop line,
    where the same fixed-time meter as before (cycle/green, phase meter_phi) holds them. Vehicles that
    pass the meter line become ordinary ramp vehicles. Through traffic and other approaches are not simulated.
  * Virtual detectors: count crossings and time-mean speed per interval. Intervals with no
    vehicles report mean_speed_mps = NaN (explicitly missing), never 0.

Run:  python ramp_sim.py --selftest
      python ramp_sim.py --q_main 1700 --q_ramp 400 --mode platoon --outdir out
"""
import argparse, csv, json, math, os, random
from collections import deque

MODEL_VERSION = "ramp_sim-0.2"
IDM = dict(v0=30.0, T=1.5, s0=2.0, a=1.2, b=1.8, delta=4.0)   # SI units
CAR_LEN = 5.0

DEFAULT = dict(
    dt=0.5, t_end=3900.0, warm=300.0,
    road_len=3000.0, x_merge_end=2000.0, ramp_len=500.0, accel_start_ramp=200.0,
    q_main=800.0,           # veh/h entering the modeled (right) lane
    q_ramp=400.0,           # veh/h arriving at the ramp signal
    ramp_mode="platoon",    # 'platoon' (signal) or 'uniform'
    cycle=90.0, green=40.0, sat_headway=2.0,
    v_in=27.0, v0_ramp=30.0, b_safe=3.0, b_safe_forced=6.0, forced_zone_m=100.0, seed=1,
    det_x=[1000.0, 1900.0, 2600.0], det_interval=60.0,
    record_traj=False, traj_dt=2.0,
    # --- upstream arterial signal (ramp_mode='arterial') ---
    q_art=900.0,            # total arterial approach flow (veh/h); p_ramp of it turns right onto the ramp
    p_ramp=0.45, cyc_art=90.0, g_art=35.0, y_art=4.0, phi_art=15.0,
    x_art=-250.0,           # arterial stop line, ramp coordinate (m); meter stop line is at 0
    x_art_spawn=-600.0, v0_art=15.0, b_go=4.0,   # b_go: cannot-stop-comfortably threshold (m/s^2)
    meter_phi=0.0,          # meter green starts when (t - meter_phi) mod cycle = 0
    rng_split=0,            # 1: ramp Poisson arrivals use their own random stream
)


def art_state_at(t, c):
    """Arterial signal state 'G','Y','R': S_art(t) = (t + phi_art) mod C_art."""
    s = (t + c["phi_art"]) % c["cyc_art"]
    return "G" if s < c["g_art"] else ("Y" if s < c["g_art"] + c["y_art"] else "R")


def meter_green_at(t, c):
    return ((t - c["meter_phi"]) % c["cycle"]) < c["green"]


def idm_acc(v, gap, dv, v0=None, p=IDM):
    """IDM acceleration. gap: bumper-to-bumper (m) or None (free road); dv = v - v_leader."""
    v0 = p["v0"] if v0 is None else v0
    free = 1.0 - (v / v0) ** p["delta"]
    if gap is None:
        return p["a"] * free
    gap = max(gap, 0.1)
    s_star = p["s0"] + max(0.0, v * p["T"] + v * dv / (2.0 * math.sqrt(p["a"] * p["b"])))
    return p["a"] * (free - (s_star / gap) ** 2)


class Veh:
    __slots__ = ("id", "x", "v", "a", "origin", "t_release", "ga", "gm", "t_arr", "t_a")

    def __init__(self, vid, x, v, origin, t_release):
        self.id, self.x, self.v, self.a, self.origin, self.t_release = vid, x, v, 0.0, origin, t_release
        self.ga = self.gm = self.t_arr = self.t_a = None


def run(cfg):
    c = {**DEFAULT, **cfg}
    rng = random.Random(c["seed"])
    dt = c["dt"]
    nsteps = int(round(c["t_end"] / dt))
    off = c["x_merge_end"] - c["ramp_len"]           # ramp coordinate -> mainline coordinate
    det_x = c["det_x"]
    nint = int(math.ceil(c["t_end"] / c["det_interval"]))
    det_cnt = {f"D{j+1}": [0] * nint for j in range(len(det_x))}
    det_spd = {f"D{j+1}": [[] for _ in range(nint)] for j in range(len(det_x))}
    det_cnt["R1"] = [0] * nint; det_spd["R1"] = [[] for _ in range(nint)]
    R1_X = 100.0                                     # ramp detector (ramp coordinate)

    main, ramp = [], []                              # leader-first (descending x)
    main_q, ramp_q = deque(), deque()
    vid = 0
    n_main = n_released = n_merged = n_exit = 0
    main_wait, stop_wait, merge_wait, tot_delay = [], [], [], []
    max_main_q = max_ramp_q = 0
    min_gap, collisions = 1e9, 0
    traj = []
    # ---- arterial feeder state (ramp_mode == 'arterial') ----
    art_mode = c["ramp_mode"] == "arterial"
    rng_art = random.Random(c["seed"] + 1000)
    rng_ramp = random.Random(c["seed"] + 2000) if c["rng_split"] else rng
    art, art_q = [], deque()
    n_art_arr = n_art_in = art_red_runs = art_yellow_runs = meter_red_cross = 0
    max_art_bq = max_q_meter = max_q_art = 0
    d_art_l, d_met_l = [], []
    ev_art_pass, ev_meter_cross, ev_ramp_arr = [], [], []
    traj_art, traj_ramp = [], []
    lam_art = c["q_art"] * c["p_ramp"] / 3600.0
    t_next_art = rng_art.expovariate(lam_art) if (art_mode and lam_art > 0) else math.inf
    art_min_gap, art_collisions = 1e9, 0

    def mean_hw():
        m = 3600.0 / c["q_main"]; hmin = min(1.0, 0.5 * m)
        return hmin + (rng.expovariate(1.0 / (m - hmin)) if m > hmin else 0.0)

    t_next_main = mean_hw() if c["q_main"] > 0 else math.inf
    lam = c["q_ramp"] / 3600.0
    t_next_ramp = rng_ramp.expovariate(lam) if (lam > 0 and not art_mode) else math.inf
    last_release = -1e9

    for k in range(nsteps):
        t = k * dt
        # ---- arrivals ----
        while t_next_main <= t:
            main_q.append(t_next_main); t_next_main += mean_hw()
        if c["ramp_mode"] == "platoon":
            while t_next_ramp <= t:
                ramp_q.append(t_next_ramp); ev_ramp_arr.append(t_next_ramp)
                t_next_ramp += rng_ramp.expovariate(lam)
        # ---- mainline insertion at x=0 (boundary queue if no safe gap) ----
        if main_q:
            lead = main[-1] if main else None
            v_in = min(c["v_in"], lead.v) if lead else c["v_in"]
            gap = (lead.x - CAR_LEN) if lead else None
            if gap is None or (gap > IDM["s0"] and idm_acc(v_in, gap, v_in - lead.v) >= -0.5 * IDM["b"]):
                arr = main_q.popleft(); vid += 1
                main.append(Veh(vid, 0.0, v_in, "main", t)); n_main += 1
                main_wait.append(t - arr)
        max_main_q = max(max_main_q, len(main_q))
        # ---- ramp release ----
        can_spawn = (not ramp) or (ramp[-1].x >= CAR_LEN + IDM["s0"] + 1.0)
        if c["ramp_mode"] == "platoon":
            green = (t % c["cycle"]) < c["green"]
            if green and ramp_q and can_spawn and t - last_release >= c["sat_headway"] - 1e-9:
                arr = ramp_q.popleft(); vid += 1
                nv = Veh(vid, 0.0, 0.0, "ramp", t); nv.t_arr = arr
                ramp.append(nv); n_released += 1
                last_release = t; stop_wait.append(t - arr)
        elif c["ramp_mode"] == "uniform" and lam > 0:
            if can_spawn and t - last_release >= 3600.0 / c["q_ramp"] - 1e-9:
                vid += 1; nv = Veh(vid, 0.0, 0.0, "ramp", t); nv.t_arr = t
                ramp.append(nv); n_released += 1
                last_release = t; stop_wait.append(0.0)
        max_ramp_q = max(max_ramp_q, len(ramp_q))
        # ---- arterial feeder: arrivals and insertion at x_art_spawn ----
        if art_mode:
            st_art = art_state_at(t, c); mg = meter_green_at(t, c)
            while t_next_art <= t:
                art_q.append(t_next_art); n_art_arr += 1; t_next_art += rng_art.expovariate(lam_art)
            if art_q:
                lead = art[-1] if art else None
                v_in = min(c["v0_art"], lead.v) if lead else c["v0_art"]
                gap = (lead.x - c["x_art_spawn"] - CAR_LEN) if lead else None
                if gap is None or (gap > IDM["s0"] and idm_acc(v_in, gap, v_in - lead.v, v0=c["v0_art"]) >= -0.5 * IDM["b"]):
                    arr = art_q.popleft(); vid += 1
                    nv = Veh(vid, c["x_art_spawn"], v_in, "art", arr); nv.t_arr = arr
                    art.append(nv); n_art_in += 1
            max_art_bq = max(max_art_bq, len(art_q))
        # ---- accelerations (from old state) ----
        for i, v in enumerate(main):
            v.a = idm_acc(v.v, None, 0.0) if i == 0 else idm_acc(v.v, main[i-1].x - v.x - CAR_LEN, v.v - main[i-1].v)
        for i, r in enumerate(ramp):
            # free acceleration toward the ramp speed; the end of the ramp acts as a 'last-chance'
            # stop: brake only when the deceleration needed to stop s0 short of the end exceeds 2.5 m/s^2
            a = idm_acc(r.v, None, 0.0, v0=c["v0_ramp"])
            need = r.v ** 2 / (2.0 * max(c["ramp_len"] - r.x - IDM["s0"], 0.1))
            if need > 2.5:
                a = min(a, -min(need, 6.0))
            if c["ramp_len"] - r.x - IDM["s0"] <= 0.5:      # at the stop line: hold (no creeping)
                a = -6.0
            if i > 0:
                a = min(a, idm_acc(r.v, ramp[i-1].x - r.x - CAR_LEN, r.v - ramp[i-1].v, v0=c["v0_ramp"]))
            r.a = a
        # ---- arterial accelerations: IDM + virtual leaders at the two stop lines ----
        if art_mode:
            for i, v in enumerate(art):
                lead = art[i-1] if i > 0 else (ramp[-1] if ramp else None)
                a = (idm_acc(v.v, None, 0.0, v0=c["v0_art"]) if lead is None
                     else idm_acc(v.v, lead.x - v.x - CAR_LEN, v.v - lead.v, v0=c["v0_art"]))
                if v.x < c["x_art"]:
                    if st_art == "G":
                        v.ga = None
                    else:
                        if v.ga is None:       # decide once at onset: stop unless it cannot stop at b_go
                            v.ga = v.v ** 2 / (2.0 * max(c["x_art"] - v.x, 0.1)) > c["b_go"]
                        if not v.ga:
                            a = min(a, idm_acc(v.v, c["x_art"] - v.x, v.v, v0=c["v0_art"]))
                if v.x < 0.0:
                    if mg:
                        v.gm = None
                    else:
                        if v.gm is None:
                            v.gm = v.v ** 2 / (2.0 * max(0.0 - v.x, 0.1)) > c["b_go"]
                        if not v.gm:
                            a = min(a, idm_acc(v.v, 0.0 - v.x, v.v, v0=c["v0_art"]))
                v.a = a
        # ---- position update + detector crossings ----
        tn = t + dt
        b = min(int(tn / c["det_interval"]), nint - 1)
        for v in main:
            vn = max(0.0, v.v + v.a * dt); xo = v.x
            v.x += 0.5 * (v.v + vn) * dt; v.v = vn
            for j, xd in enumerate(det_x):
                if xo < xd <= v.x:
                    det_cnt[f"D{j+1}"][b] += 1; det_spd[f"D{j+1}"][b].append(vn)
        for r in ramp:
            vn = max(0.0, r.v + r.a * dt); xo = r.x
            r.x += 0.5 * (r.v + vn) * dt; r.v = vn
            if r.x > c["ramp_len"] - IDM["s0"]:             # never pass the end of the ramp
                r.x = c["ramp_len"] - IDM["s0"]; r.v = 0.0
            if xo < R1_X <= r.x:
                det_cnt["R1"][b] += 1; det_spd["R1"][b].append(vn)
        # ---- arterial update; vehicles crossing the meter line become ramp vehicles ----
        if art_mode:
            for v in art:
                vn = max(0.0, v.v + v.a * dt); xo = v.x
                v.x += 0.5 * (v.v + vn) * dt; v.v = vn
                if xo < c["x_art"] <= v.x:
                    v.t_a = tn; ev_art_pass.append(tn)
                    if st_art == "R": art_red_runs += 1
                    elif st_art == "Y": art_yellow_runs += 1
            while art and art[0].x >= 0.0:
                v = art.pop(0)
                if not mg: meter_red_cross += 1
                n_released += 1; ev_meter_cross.append(tn)
                d_art_l.append((v.t_a - v.t_arr) - (c["x_art"] - c["x_art_spawn"]) / c["v0_art"])
                d_met = (tn - v.t_a) - (0.0 - c["x_art"]) / c["v0_art"]
                d_met_l.append(d_met); stop_wait.append(d_met)
                v.origin = "ramp"; v.t_release = tn; ramp.append(v)
            max_q_meter = max(max_q_meter, sum(1 for v in art if v.x >= c["x_art"] and v.v < 1.0))
            max_q_art = max(max_q_art, sum(1 for v in art if v.x < c["x_art"] and v.v < 1.0))
        # ---- exits ----
        while main and main[0].x >= c["road_len"]:
            main.pop(0); n_exit += 1
        # ---- merges (front ramp vehicle first) ----
        while ramp and ramp[0].x >= c["accel_start_ramp"]:
            r = ramp[0]; xm = r.x + off
            idx = 0
            while idx < len(main) and main[idx].x >= xm:
                idx += 1
            lead = main[idx-1] if idx > 0 else None
            foll = main[idx] if idx < len(main) else None
            ok = True
            bs = c["b_safe_forced"] if (c["ramp_len"] - r.x) <= c["forced_zone_m"] else c["b_safe"]
            if lead:
                g = lead.x - xm - CAR_LEN
                ok = ok and g >= IDM["s0"] and idm_acc(r.v, g, r.v - lead.v) >= -bs
            if foll and ok:
                g = xm - foll.x - CAR_LEN
                ok = ok and g >= IDM["s0"] and idm_acc(foll.v, g, foll.v - r.v) >= -bs
            if not ok:
                break
            ramp.pop(0); r.x = xm; r.origin = "ramp"; main.insert(idx, r)
            n_merged += 1; merge_wait.append(tn - r.t_release)
            # total delay to merge, measured from the stop-line arrival (baseline) or from leaving the arterial
            # stop line minus its free-flow travel time to the meter (arterial mode)
            ref = r.t_a if r.t_a is not None else r.t_arr
            free = (-c["x_art"] / c["v0_art"]) if r.t_a is not None else 0.0
            tot_delay.append(tn - ref - free)
        # ---- gap audit ----
        for i in range(1, len(main)):
            g = main[i-1].x - main[i].x - CAR_LEN
            min_gap = min(min_gap, g)
            if g < 0: collisions += 1
        if art_mode:
            for i in range(1, len(art)):
                g = art[i-1].x - art[i].x - CAR_LEN
                art_min_gap = min(art_min_gap, g)
                if g < 0: art_collisions += 1
            if art and ramp:
                g = ramp[-1].x - art[0].x - CAR_LEN
                art_min_gap = min(art_min_gap, g)
                if g < 0: art_collisions += 1
        if c["record_traj"] and art_mode and k % int(round(c["traj_dt"] / dt)) == 0:
            for v in art: traj_art.append((tn, v.id, v.x, v.v))
            for r in ramp: traj_ramp.append((tn, r.id, r.x, r.v))
        if c["record_traj"] and k % int(round(c["traj_dt"] / dt)) == 0:
            for v in main:
                traj.append((tn, v.id, v.x, v.v, v.origin))

    # ---- detector records (interface format) ----
    records = []
    for sid in det_cnt:
        for i in range(nint):
            sp = det_spd[sid][i]
            records.append(dict(sensor_id=sid, interval_start_s=i * c["det_interval"],
                                interval_end_s=(i + 1) * c["det_interval"], count=det_cnt[sid][i],
                                mean_speed_mps=(sum(sp) / len(sp) if sp else float("nan")),
                                quality_flag="ok" if sp else "no_vehicles"))
    w0 = int(c["warm"] / c["det_interval"]); span = (nint - w0) * c["det_interval"]
    metrics = {}
    for sid in det_cnt:
        cnts = det_cnt[sid][w0:]
        spd = [r["mean_speed_mps"] for r in records if r["sensor_id"] == sid
               and r["interval_start_s"] >= c["warm"] and not math.isnan(r["mean_speed_mps"])]
        metrics[f"{sid}_flow_vph"] = sum(cnts) * 3600.0 / span
        metrics[f"{sid}_min_speed_mps"] = min(spd) if spd else float("nan")
        metrics[f"{sid}_mean_speed_mps"] = sum(spd) / len(spd) if spd else float("nan")
        metrics[f"{sid}_frac_intervals_below_15mps"] = (sum(s < 15.0 for s in spd) / len(spd)) if spd else float("nan")
    metrics.update(
        main_boundary_max_queue_veh=max_main_q, main_boundary_residual_veh=len(main_q),
        main_mean_entry_wait_s=(sum(main_wait) / len(main_wait) if main_wait else 0.0),
        ramp_stopline_max_queue_veh=max_ramp_q, ramp_stopline_residual_veh=len(ramp_q),
        ramp_vehicles_on_ramp_end=len(ramp),
        ramp_mean_stopline_wait_s=(sum(stop_wait) / len(stop_wait) if stop_wait else 0.0),
        ramp_mean_release_to_merge_s=(sum(merge_wait) / len(merge_wait) if merge_wait else 0.0),
        ramp_mean_total_delay_s=(sum(tot_delay) / len(tot_delay) if tot_delay else 0.0),
        ramp_unserved_end_veh=len(ramp_q) + len(ramp) + len(art) + len(art_q),
        ramp_released=n_released, ramp_merged=n_merged, main_spawned=n_main, main_exited=n_exit,
        min_gap_m=min_gap, collisions=collisions,
        conservation_ok=bool(n_main + n_merged == n_exit + len(main) and n_released == n_merged + len(ramp)),
    )
    if art_mode:
        mean = lambda l: (sum(l) / len(l) if l else 0.0)
        metrics.update(
            art_arrived=n_art_arr, art_inserted=n_art_in, art_released_to_ramp=n_released,
            art_on_approach_end=len(art), art_boundary_residual_veh=len(art_q), art_boundary_max_queue_veh=max_art_bq,
            art_mean_delay_at_art_signal_s=mean(d_art_l), art_mean_delay_at_meter_s=mean(d_met_l),
            art_max_queue_at_art_line_veh=max_q_art, meter_max_queue_veh=max_q_meter,
            art_red_runs=art_red_runs, art_yellow_runs=art_yellow_runs, meter_red_crossings=meter_red_cross,
            art_min_gap_m=art_min_gap, art_collisions=art_collisions,
            art_conservation_ok=bool(n_art_in == n_released + len(art)))
        metrics["conservation_ok"] = bool(metrics["conservation_ok"] and metrics["art_conservation_ok"] and art_collisions == 0)
    qr_eff = c["q_art"] * c["p_ramp"] if art_mode else c["q_ramp"]
    meta = dict(scenario_id=f"{c['ramp_mode']}_qm{int(c['q_main'])}_qr{int(round(qr_eff))}",
                demand_id=f"qm{int(c['q_main'])}_qr{int(round(qr_eff))}", seed=c["seed"],
                model_version=MODEL_VERSION, controller_version="none(fixed signals)", dt=dt)
    return dict(meta=meta, metrics=metrics, records=records, traj=traj, cfg=c,
                events=dict(art_pass=ev_art_pass, meter_cross=ev_meter_cross, ramp_arr=ev_ramp_arr),
                traj_art=traj_art, traj_ramp=traj_ramp)


def selftest():
    print("[1] free motion: single car from rest must approach v0 monotonically, never exceed it")
    v, x, dt, ok = 0.0, 0.0, 0.1, True
    for _ in range(int(120 / dt)):
        a = idm_acc(v, None, 0.0); vn = v + a * dt
        ok &= vn >= v - 1e-9 and vn <= IDM["v0"] + 1e-9; v = vn
    print("    final v = %.3f m/s (v0 = %.1f)  -> %s" % (v, IDM["v0"], "PASS" if ok and v > 29.5 else "FAIL"))
    print("[2] equilibrium: at s_e(v) = (s0+vT)/sqrt(1-(v/v0)^4) acceleration must be 0")
    for v in (10.0, 20.0, 25.0):
        se = (IDM["s0"] + v * IDM["T"]) / math.sqrt(1 - (v / IDM["v0"]) ** IDM["delta"])
        print("    v=%4.1f  s_e=%6.2f m  a=%.2e  -> %s" % (v, se, idm_acc(v, se, 0.0),
              "PASS" if abs(idm_acc(v, se, 0.0)) < 1e-9 else "FAIL"))
    print("[3] standstill: stopped car with gap s0 must not move")
    print("    a = %.3f -> %s" % (idm_acc(0.0, IDM["s0"], 0.0), "PASS" if abs(idm_acc(0.0, IDM["s0"], 0.0)) < 1e-9 else "FAIL"))
    print("[4] full run, zero ramp demand: conservation, no collisions, D3 flow ~ demand")
    m = run(dict(q_ramp=0.0, seed=1))["metrics"]
    print("    conservation_ok=%s collisions=%d min_gap=%.1f D3_flow=%.0f veh/h (demand 800)"
          % (m["conservation_ok"], m["collisions"], m["min_gap_m"], m["D3_flow_vph"]))
    print("[5] full run with ramp demand: conservation and collisions")
    m = run(dict(q_ramp=400.0, seed=1))["metrics"]
    print("    conservation_ok=%s collisions=%d min_gap=%.1f released=%d merged=%d"
          % (m["conservation_ok"], m["collisions"], m["min_gap_m"], m["ramp_released"], m["ramp_merged"]))


def selftest_arterial():
    print("[6] arterial feeder (q_art=900, p_ramp=0.45): conservation, gaps, signal obedience")
    r = run(dict(ramp_mode="arterial", seed=1)); m = r["metrics"]
    print("    conservation_ok=%s art_collisions=%d art_min_gap=%.1f main_min_gap=%.1f"
          % (m["conservation_ok"], m["art_collisions"], m["art_min_gap_m"], m["min_gap_m"]))
    print("    art arrived=%d inserted=%d released_to_ramp=%d  red_runs=%d yellow_runs=%d meter_red_crossings=%d"
          % (m["art_arrived"], m["art_inserted"], m["art_released_to_ramp"], m["art_red_runs"], m["art_yellow_runs"], m["meter_red_crossings"]))
    c = r["cfg"]; bad = sum(1 for t in r["events"]["art_pass"] if art_state_at(t - c["dt"], c) == "R")
    print("    crossings of x_art during red: %d (must be 0 or rare)  -> %s" % (bad, "PASS" if bad <= 2 else "CHECK"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--outdir", default="out")
    for k, v in DEFAULT.items():
        if isinstance(v, (int, float, str)) and not isinstance(v, bool):
            ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args()
    if a.selftest:
        selftest(); return selftest_arterial()
    cfg = {k: getattr(a, k) for k in DEFAULT if hasattr(a, k)}
    res = run(cfg)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "detectors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res["records"][0].keys())); w.writeheader(); w.writerows(res["records"])
    with open(os.path.join(a.outdir, "evaluation.json"), "w") as f:
        json.dump(dict(**res["meta"], **res["metrics"]), f, indent=2)
    print(json.dumps(res["metrics"], indent=2))


if __name__ == "__main__":
    main()