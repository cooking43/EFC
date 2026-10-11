"""Analytic roll/pitch deck pose for the corrected H1-2 experiment."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _quat_roll_pitch(roll: float, pitch: float) -> np.ndarray:
    """Return MuJoCo's wxyz quaternion for yaw=0, Rz(yaw)Ry(pitch)Rx(roll)."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp], dtype=float)


def _startup_ramp(t: float, duration: float) -> tuple[float, float]:
    """Raised-cosine startup ramp and its derivative.

    The ramp belongs to the simulator clock and is therefore evaluated only
    from continuous deck time.  Robot episode resets must not restart it.
    """
    if duration <= 0.0 or t >= duration:
        return 1.0, 0.0
    if t <= 0.0:
        return 0.0, 0.0
    phase = math.pi * t / duration
    return 0.5 - 0.5 * math.cos(phase), 0.5 * math.pi * math.sin(phase) / duration


def get_deck_pose(t: float, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate deck position, orientation and rates at time ``t``."""
    cfg = config.get("deck", config)
    ar = math.radians(float(cfg["roll_amp_deg"]))
    ap = math.radians(float(cfg["pitch_amp_deg"]))
    wr = 2.0 * math.pi * float(cfg["roll_freq_hz"])
    wp = 2.0 * math.pi * float(cfg["pitch_freq_hz"])
    roll_phase = float(cfg.get("roll_phase_rad", 0.0))
    pitch_phase = float(cfg.get("pitch_phase_rad", 0.0))
    z0 = float(cfg.get("surface_height_m", 0.0)) - float(cfg["thickness"]) / 2.0
    # Counterfactual episodes share the exact same level pose at t=0, while
    # their scheduled future motion has an independently signed direction.
    # The peak magnitude remains ar/ap; this does not increase deck amplitude.
    counterfactual = bool(cfg.get("counterfactual_enabled", False))
    roll_sign = float(cfg.get("counterfactual_roll_sign", 1.0)) if counterfactual else 1.0
    pitch_sign = float(cfg.get("counterfactual_pitch_sign", 1.0)) if counterfactual else 1.0
    branch_start = float(cfg.get("counterfactual_start_time_s", 0.0)) if counterfactual else 0.0
    branch_active = (not counterfactual) or t >= branch_start
    motion_t = t - branch_start if counterfactual else t
    ramp, ramp_rate = _startup_ramp(t, float(cfg.get("startup_ramp_s", 1.0)))
    roll_wave = roll_sign * ar * math.sin(wr * motion_t + roll_phase) if branch_active else 0.0
    pitch_wave = pitch_sign * ap * math.sin(wp * motion_t + pitch_phase) if branch_active else 0.0
    roll_wave_rate = roll_sign * ar * wr * math.cos(wr * motion_t + roll_phase) if branch_active else 0.0
    pitch_wave_rate = pitch_sign * ap * wp * math.cos(wp * motion_t + pitch_phase) if branch_active else 0.0
    roll = ramp * roll_wave
    pitch = ramp * pitch_wave
    return {
        "position": np.array([0.0, 0.0, z0], dtype=float),
        "quaternion": _quat_roll_pitch(roll, pitch),
        "roll": roll,
        "pitch": pitch,
        "heave": 0.0,
        "roll_rate": ramp_rate * roll_wave + ramp * roll_wave_rate,
        "pitch_rate": ramp_rate * pitch_wave + ramp * pitch_wave_rate,
        "heave_rate": 0.0,
        "startup_ramp": ramp,
    }
