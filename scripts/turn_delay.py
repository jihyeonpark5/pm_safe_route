"""Compatibility entry point for historical experiments."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmroute.turn_delay import TurnDelayProfile, turn_angle, valid_walk
