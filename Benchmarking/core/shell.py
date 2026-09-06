"""Helpers for invoking a library's own script as a subprocess. (Making it easier to use already created scripts from other repositories.)

Has help functions to make sure the scripts for the different methods are run in the correct environment. 
It also has helpers to make sure the results are saved in the correct format.
It also makes it so when multiple ones are run in the same script (which is another way to do it) 
multiple imports don't collide with each other in sys.modules.

Sys.executable makes sures every process uses the same interpreter so we don't get problems with that.
"""

from __future__ import annotations

import difflib
import functools
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

# Repo root = two levels up from this file (Benchmarking/core/shell.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Matches argparse's, the two
# flag-definition styles used by every train/evaluate script currently used.
# NOTE: ONLY WORKS WITH FLAGS defined like that, so other styles won't be recognized.
_ARGPARSE_FLAG_RE = re.compile(r"""add_argument\(\s*["']--([A-Za-z0-9_]+)["']""")
_ABSL_FLAG_RE = re.compile(r"""flags\.DEFINE_\w+\(\s*["']([A-Za-z0-9_]+)["']""")


@functools.lru_cache(maxsize=None)
def discover_script_flags(script_path: str) -> FrozenSet[str]:
    """Scans the script at `script_path`' as a plain text. searching for flags that match the regex.

    Returns an empty set if the file can't be read, or neither pattern
    matches anything -- callers should treat that as "validation unavailable"
    for this script, not "every flag is invalid".
    """
    try:
        text = Path(script_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return frozenset()
    found = set(_ARGPARSE_FLAG_RE.findall(text))
    found.update(_ABSL_FLAG_RE.findall(text))
    return frozenset(found)


def check_known_flags(script_path: str, flags: Dict[str, Any]) -> None:
    """Raise ValueError if `flags` has a key `script_path` doesn't define.

    """
    known = discover_script_flags(script_path)
    if not known:
        return
    unknown = sorted(k for k in flags if k not in known)
    if not unknown:
        return
    detail = []
    for key in unknown:
        match = difflib.get_close_matches(key, known, n=1)
        hint = " -- did you mean '{}'?".format(match[0]) if match else ""
        detail.append("  '{}'{}".format(key, hint))
    raise ValueError(
        "{} does not define these flags: \n{}\n"
        "Valid flags for this script: {}".format(
            script_path, "\n".join(detail), ", ".join(sorted(known))))


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
               env: Optional[Dict[str, str]] = None,
               validate_flags: bool = True) -> str:
    """Run python <script> <flags> with cwd=lib_di.

    Streams combined stdout/stderr to log_path (if given) and returns the
    exact command string for provenance.
    Raises CalledProcessError on a non-zero exit so a failed config never runs.
    """
    if validate_flags:
        check_known_flags(os.path.join(lib_dir, script), flags)
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
