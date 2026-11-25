"""Training-script style reference: must NOT mark MainNet as a component."""

from __future__ import annotations

from .main_model import MainNet


def build_model():
    return MainNet()


def train():
    model = build_model()
    return model
