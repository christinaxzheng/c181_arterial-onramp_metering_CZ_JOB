# Ramp-merge simulator (CEE 181/281 Fall 2026) -- version 0.2 (adds upstream arterial signal)

Dependencies: Python 3.9+, matplotlib (plots only). No other packages.

    python ramp_sim.py --selftest                       # verification checks (free motion, equilibrium, standstill, conservation, arterial feeder)
    python ramp_sim.py --q_main 800 --q_ramp 400 --ramp_mode platoon --outdir out   # one run -> out/detectors.csv, out/evaluation.json
    python ramp_sim.py --q_main 800 --q_art 900 --ramp_mode arterial --outdir out_art  # arterial feeder (ramp demand = p_ramp x q_art)
    python run_experiments.py --outdir results --seeds 3          # earlier experiments E1-E3 (about 1-2 min)
    python run_arterial.py --outdir results_art --seeds 10        # arterial experiments E4-E5 + 4 figures (about 4-5 min)

Inputs: all in `DEFAULT` in ramp_sim.py (also CLI flags). Units: s, m, m/s internally; demands in veh/h.
Outputs: detectors.csv (sensor_id, interval_start_s, interval_end_s, count, mean_speed_mps [NaN = missing], quality_flag),
evaluation.json (scenario_id, demand_id, seed, model_version, controller_version, metrics incl. boundary queues).

## What changed in 0.2
* New `ramp_mode="arterial"`: right-turn vehicles (p_ramp x q_art) arrive Poisson at x_art_spawn (-600 m), follow IDM (v0_art = 15 m/s)
  and stop behind a virtual leader at the arterial stop line x_art (-250 m) when the arterial signal is yellow or red
  (S_art = (t + phi_art) mod cyc_art; green < g_art, yellow < g_art + y_art). A vehicle that would need more than b_go = 4 m/s^2 to stop
  proceeds. They then queue at the ramp meter (stop line at ramp coordinate 0, fixed time cycle/green, phase meter_phi) and become ordinary
  ramp vehicles after crossing it. Offset delta = meter green start minus arterial green start; meter_phi = (75 + delta) mod 90.
* New metrics: art_mean_delay_at_art_signal_s, art_mean_delay_at_meter_s, meter_max_queue_veh, art_max_queue_at_art_line_veh,
  art_red_runs, meter_red_crossings, art_min_gap_m, art_conservation_ok, ramp_mean_total_delay_s, ramp_unserved_end_veh.
* New option `rng_split=1`: ramp Poisson arrivals use their own random stream, so the mainline arrivals are identical across modes
  for the same seed (arterial arrivals always use their own stream). Default 0 keeps version 0.1 results.
* platoon and uniform modes are unchanged: results are identical to version 0.1 (checked for seeds 1 and 2).

Known limitations (state these in the report): one modeled lane (right lane) -- other freeway lanes and mainline yielding are not simulated;
gap-acceptance merge with a 'forced' zone (follower may brake to 6 m/s^2 in the last 100 m), merge is an instantaneous transfer;
open downstream boundary (no spillback); passenger cars only; the arterial feeder simulates only the right-turn stream in its own lane
(no through or left traffic, no other approaches); the meter has no yellow or all-red, so vehicles that cannot stop at 4 m/s^2 cross it on red;
ramp mean delay is averaged over merged vehicles only (see ramp_unserved_end_veh when the ramp is saturated); the baseline ramp starts
vehicles from rest at the meter while arterial-fed vehicles may arrive with a rolling start.
