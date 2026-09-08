"""Keep <out_dir>/model.zip abreast of the newest file in <out_dir>/checkpoints/.

rl_train.py writes model.zip only when model.learn() returns, so a run killed
by a signal resumes from whatever model.zip last held -- possibly far behind
the checkpoints. This promotes each new checkpoint pair as it appears, capping
the loss from any crash at one checkpoint interval.

Safe to run against a live trainer: it only reads model.zip at startup, and
each promotion is an atomic os.replace of a fully-copied temp file.

    python3 scripts/promote_checkpoint.py --daemon        # detach and watch
    python3 scripts/promote_checkpoint.py --once          # single pass
"""
import argparse
import os
import re
import shutil
import sys
import time

STEP_RE = re.compile(r"_(\d+)_steps\.zip$")
SETTLE_SEC = 15.0          # ignore files still being written


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [promote] {msg}", flush=True)


def newest_pair(ckpt_dir, settle=SETTLE_SEC):
    """Newest (steps, model_zip, vecnorm_pkl) that has settled, or None."""
    best = None
    for name in os.listdir(ckpt_dir):
        m = STEP_RE.search(name)
        if not m:
            continue
        steps = int(m.group(1))
        zip_path = os.path.join(ckpt_dir, name)
        pkl_path = os.path.join(
            ckpt_dir, STEP_RE.sub(f"_vecnormalize_{steps}_steps.pkl", name)
        )
        if not os.path.exists(pkl_path):
            continue
        now = time.time()
        if now - os.path.getmtime(zip_path) < settle:
            continue
        if now - os.path.getmtime(pkl_path) < settle:
            continue
        if best is None or steps > best[0]:
            best = (steps, zip_path, pkl_path)
    return best


def atomic_copy(src, dst):
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def promote(out_dir, pair):
    steps, zip_path, pkl_path = pair
    model_zip = os.path.join(out_dir, "model.zip")
    # Never clobber something the trainer itself wrote more recently -- that is
    # the clean-finish case, where model.zip already holds the final state.
    if os.path.exists(model_zip) and os.path.getmtime(model_zip) > os.path.getmtime(zip_path):
        log(f"skip {steps}: model.zip is newer than the checkpoint")
        return False
    atomic_copy(pkl_path, os.path.join(out_dir, "vecnorm.pkl"))
    atomic_copy(zip_path, model_zip)
    log(f"promoted {steps} steps -> model.zip + vecnorm.pkl")
    return True


def alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError) as exc:
        return isinstance(exc, PermissionError)
    return True


def watch(out_dir, supervisor_pid, interval):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    last = 0
    log(f"watching {ckpt_dir} (supervisor pid {supervisor_pid}, every {interval}s)")
    while True:
        if supervisor_pid and not alive(supervisor_pid):
            # The trainer writes the final model.zip on a clean exit; stop
            # before we could overwrite it with an older checkpoint.
            log(f"supervisor pid {supervisor_pid} is gone; exiting")
            return
        try:
            pair = newest_pair(ckpt_dir)
            if pair and pair[0] > last and promote(out_dir, pair):
                last = pair[0]
        except OSError as exc:
            log(f"error: {exc}")
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="models/rl/ppo_gates_v4")
    ap.add_argument("--supervisor-pid", type=int, default=0,
                    help="exit when this pid dies; 0 to run until killed")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--log", default="logs/promote_checkpoint.log")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    if args.once:
        pair = newest_pair(os.path.join(out_dir, "checkpoints"))
        if not pair:
            log("no settled checkpoint pair found")
            return 1
        promote(out_dir, pair)
        return 0

    if args.daemon:
        log_path = os.path.abspath(args.log)
        pid = os.fork()
        if pid:
            print(f"detached pid {pid}, logging to {log_path}")
            return 0
        os.setsid()
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
        os.dup2(fd, 1)
        os.dup2(fd, 2)

    watch(out_dir, args.supervisor_pid, args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
