"""Launch a command in its own session so process-group kills don't reach it.

macOS has no setsid(1); this forks, calls os.setsid(), redirects stdio to the
log, and execs. The parent returns immediately with the child's pid.
"""
import os
import sys

LOG = "/Users/Dan/dev/nanopilot-ai/logs/rl_train_ppo_gates_v5.log"
CWD = "/Users/Dan/dev/nanopilot-ai"
CMD = [
    "./scripts/rl_train.sh",
    "--total-steps=400000",
    "--out=models/rl/ppo_gates_v5",
    "--control-altitude",
    "--altitude-scale=0.09",
]

pid = os.fork()
if pid == 0:
    os.setsid()                      # new session: immune to the parent's group kill
    os.chdir(CWD)
    fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.execvp(CMD[0], CMD)
    os._exit(127)
print(f"detached pid {pid}")
