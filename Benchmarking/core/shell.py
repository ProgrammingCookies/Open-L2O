"""Helpers for invoking a library's own script as a subprocess. (Making it easier to use already created scripts from other repositories.)

Has help functions to make sure the scripts for the different methods are run in the correct environment. 
It also has helpers to make sure the results are saved in the correct format.
It also makes it so when multiple ones are run in the same script (which is another way to do it) 
multiple imports don't collide with each other in sys.modules.

Sys.executable makes sures every process uses the same interpreter so we don't get problems with that.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root = two levels up from this file (Benchmarking/core/shell.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


def flags_to_argv(flags: Dict[str, Any]) -> List[str]:
    """
    Turn a dict of CLI flags into an argv list.
    """
    argv: List[str] = []
    for key, value in flags.items():
        flag = "--{}".format(key)
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, (list, tuple)):
            for v in value:
                argv.extend([flag, str(v)])
        else:
            argv.extend([flag, str(value)])
    return argv


def run_script(lib_dir: str, script: str, flags: Dict[str, Any],
               log_path: Optional[str] = None,
               env: Optional[Dict[str, str]] = None) -> str:
    """Run python <script> <flags> with cwd=lib_di.

    Streams combined stdout/stderr to log_path (if given) and returns the
    exact command string for provenance. 
    Raises CalledProcessError on a non-zero exit so a failed run never masquerades as a successful benchmark.
    """
    argv = [sys.executable, script] + flags_to_argv(flags)
    cmd_str = " ".join(argv)

    run_env = dict(os.environ)
    if env:
        run_env.update(env)

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w") as log:
            log.write("# cwd: {}\n# cmd: {}\n\n".format(lib_dir, cmd_str))
            log.flush()
            subprocess.run(argv, cwd=lib_dir, env=run_env, check=True,
                           stdout=log, stderr=subprocess.STDOUT, text=True)
    else:
        subprocess.run(argv, cwd=lib_dir, env=run_env, check=True, text=True)
    return cmd_str
