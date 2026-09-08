"""Altitude oscillation analysis: period, and what the command is actually doing.

The distilled student holds the right MEAN height but oscillates into the floor.
Summary stats (alt_std) cannot distinguish the possible causes, which need
different fixes:

  * cmd ~ -vz, in phase        -> correct damping, oscillation is elsewhere
  * cmd ~ -(z - z0), in phase  -> POSITION feedback with no damping term: an
                                  undamped oscillator by construction
  * cmd ~ -vz but LAGGED       -> right law, wrong phase; delay turns damping
                                  into anti-damping near the resonant period

Reads the traces written by `eval_flight.py --trace` (student) and
`gate_rollout.py --trace-output` (teacher).

    python3 scripts/alt_phase.py logs/trace_student.json logs/trace_teacher.json
"""
import json
import sys

import numpy as np


def dominant_period(z, dt):
    """Dominant oscillation period of the detrended signal, in seconds."""
    z = np.asarray(z, dtype=float)
    if z.size < 16:
        return None
    z = z - np.polyval(np.polyfit(np.arange(z.size), z, 1), np.arange(z.size))
    spec = np.abs(np.fft.rfft(z * np.hanning(z.size)))
    freqs = np.fft.rfftfreq(z.size, d=dt)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    return (1.0 / freqs[k]) if freqs[k] > 0 else None


def best_lag(a, b, max_lag):
    """Lag (in samples) maximizing |corr(a[t], b[t-lag])|, plus that corr.

    Positive lag = `a` responds to what `b` did earlier, i.e. `a` LAGS `b`.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a - a.mean(); b = b - b.mean()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0, 0.0
    best = (0, 0.0)
    for lag in range(0, max_lag + 1):
        x = a[lag:] if lag else a
        y = b[:len(b) - lag] if lag else b
        n = min(len(x), len(y))
        if n < 8:
            break
        c = float(np.corrcoef(x[:n], y[:n])[0, 1])
        if abs(c) > abs(best[1]):
            best = (lag, c)
    return best


def analyze(tag, episodes, dt):
    print(f"\n=== {tag} ===")
    print(f"{'ep':>3} {'n':>5} {'period':>8} {'z_std':>7} "
          f"{'corr(cmd,-vz)':>14} {'lag':>5} {'corr(cmd,-z)':>13} {'lag':>5}")
    rows = []
    for i, ep in enumerate(episodes):
        tr = ep.get("trace", ep)
        z = np.asarray(tr.get("z", []), dtype=float)
        vz = np.asarray(tr.get("vz", []), dtype=float)
        cmd = np.asarray(tr.get("alt_cmd", []), dtype=float)
        n = min(len(z), len(vz), len(cmd))
        if n < 32:
            continue
        z, vz, cmd = z[:n], vz[:n], cmd[:n]
        period = dominant_period(z, dt)
        # correlate the command against -vz (damping) and -z (position feedback)
        lag_v, c_v = best_lag(cmd, -vz, max_lag=int(round(1.5 / dt)))
        lag_z, c_z = best_lag(cmd, -(z - z.mean()), max_lag=int(round(1.5 / dt)))
        rows.append((period, z.std(), c_v, lag_v * dt, c_z, lag_z * dt))
        print(f"{i:>3} {n:>5} {('%.2fs' % period) if period else '   n/a':>8} "
              f"{z.std():>7.3f} {c_v:>14.2f} {lag_v * dt:>4.2f}s {c_z:>13.2f} {lag_z * dt:>4.2f}s")
    if rows:
        arr = np.array([[r[0] if r[0] else np.nan, r[1], r[2], r[3], r[4], r[5]] for r in rows])
        print(f"{'MEAN':>3} {'':>5} {np.nanmean(arr[:,0]):>7.2f}s {np.nanmean(arr[:,1]):>7.3f} "
              f"{np.nanmean(arr[:,2]):>14.2f} {np.nanmean(arr[:,3]):>4.2f}s "
              f"{np.nanmean(arr[:,4]):>13.2f} {np.nanmean(arr[:,5]):>4.2f}s")
    return rows


def main():
    dt = 1.0 / 20.0
    for path in sys.argv[1:]:
        with open(path) as fh:
            d = json.load(fh)
        eps = d.get("episodes", [])
        analyze(path, eps, dt)
    print("""
Reading the table:
  corr(cmd,-vz) near +1 at lag 0  -> command IS a velocity damper, in phase.
  corr(cmd,-z)  near +1 at lag 0  -> command is POSITION feedback (undamped).
  high corr at a LAG of a quarter period -> right law, destabilizing phase.""")


if __name__ == "__main__":
    main()
