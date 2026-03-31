#!/usr/bin/env python3
"""Common auto-postprocess hook for ISCE2 applications."""

import os
import shlex
import subprocess
import sys
import importlib.util


_TRUE_VALUES = frozenset(("1", "true", "yes", "on", "y"))
_FALSE_VALUES = frozenset(("0", "false", "no", "off", "n"))


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default

    token = str(value).strip().lower()
    if token in _TRUE_VALUES:
        return True
    if token in _FALSE_VALUES:
        return False
    return default


def _build_command():
    command = os.environ.get("ISCE_AUTO_POSTPROCESS_CMD")
    args = os.environ.get("ISCE_AUTO_POSTPROCESS_ARGS", "")
    args = args.strip()

    if command is None:
        mod_name = "isce2.docker.isce2_tops_postprocess"
        try:
            has_module = importlib.util.find_spec(mod_name) is not None
        except Exception:
            has_module = False
        parts = [sys.executable, "-m", mod_name] if has_module else ["isce2-tops-postprocess"]
        if args:
            parts.extend(shlex.split(args))
        return parts

    command = command.strip()
    if not command:
        return []

    parts = shlex.split(command)
    if args:
        parts.extend(shlex.split(args))
    return parts


def run_auto_postprocess(logger, workflow_name):
    """Run postprocess command before endup.

    Environment variables:
      ISCE_AUTO_POSTPROCESS: enable/disable auto run (default: true)
      ISCE_AUTO_POSTPROCESS_STRICT: raise on failure (default: false)
      ISCE_AUTO_POSTPROCESS_CMD: command to execute
      ISCE_AUTO_POSTPROCESS_ARGS: extra args appended to command
    """

    enabled = _env_bool("ISCE_AUTO_POSTPROCESS", True)
    strict = _env_bool("ISCE_AUTO_POSTPROCESS_STRICT", False)
    if not enabled:
        logger.info(
            "Auto postprocess disabled (workflow=%s, ISCE_AUTO_POSTPROCESS=%s).",
            workflow_name,
            os.environ.get("ISCE_AUTO_POSTPROCESS"),
        )
        return 0

    cmd = _build_command()
    if not cmd:
        message = (
            "Auto postprocess command is empty. "
            "Set ISCE_AUTO_POSTPROCESS_CMD or disable ISCE_AUTO_POSTPROCESS."
        )
        if strict:
            raise RuntimeError(message)
        logger.warning(message)
        return 2

    logger.info(
        "Running auto postprocess (workflow=%s): %s",
        workflow_name,
        " ".join(shlex.quote(token) for token in cmd),
    )
    run_env = os.environ.copy()
    run_env["ISCE_AUTO_POSTPROCESS_WORKFLOW"] = str(workflow_name)
    try:
        result = subprocess.run(cmd, check=False, env=run_env)
    except OSError as err:
        message = "Failed to launch auto postprocess command: {0}".format(err)
        if strict:
            raise RuntimeError(message)
        logger.warning(message)
        return 127

    if result.returncode != 0:
        message = "Auto postprocess exited with code {0}".format(result.returncode)
        if strict:
            raise RuntimeError(message)
        logger.warning(message)
    else:
        logger.info("Auto postprocess completed (workflow=%s).", workflow_name)

    return result.returncode
