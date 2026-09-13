"""Config loading and the CRN key discipline for AETHER v3.

One committed-defaults YAML (configs/config.yaml), attribute
access via a recursive namespace, CLI overrides as --set key.sub=value.

CRN: master key = sha256(world|regime|seed) (builtin hash() is process-salted and NOT
reproducible). Seven named subkeys; every stochastic stream evolves per slot via
fold_in independent of the schedule, so all methods see identical realizations.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import yaml

DEFAULT_YAML = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


class Cfg(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


def _to_ns(obj):
    if isinstance(obj, dict):
        return Cfg(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def _parse_value(s):
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def load_config(path=None, overrides=()):
    """overrides: iterable of 'sect.key=value' strings."""
    with open(path or DEFAULT_YAML) as f:
        raw = yaml.safe_load(f)
    for ov in overrides:
        key, _, val = ov.partition("=")
        node = raw
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = _parse_value(val)
    return _to_ns(raw)


class CRNKeys(NamedTuple):
    world_gen: jax.Array
    target_gen: jax.Array
    fading: jax.Array
    blockage: jax.Array
    clutter: jax.Array
    det_noise: jax.Array
    outliers: jax.Array


def master_key(world: str, regime: str, seed: int):
    h = hashlib.sha256(f"{world}|{regime}|{seed}".encode()).digest()
    return jax.random.PRNGKey(int.from_bytes(h[:4], "little"))


def crn_keys(world: str, regime: str, seed: int) -> CRNKeys:
    return CRNKeys(*jax.random.split(master_key(world, regime, seed), 7))


def config_hash(cfg) -> str:
    def flatten(ns, prefix=""):
        items = []
        for k, v in sorted(vars(ns).items()):
            if isinstance(v, Cfg):
                items += flatten(v, f"{prefix}{k}.")
            else:
                items.append(f"{prefix}{k}={v}")
        return items
    return hashlib.sha256("|".join(flatten(cfg)).encode()).hexdigest()[:12]


def config_dict(cfg):
    """JSON-serializable effective settings for experiment provenance."""
    if isinstance(cfg, SimpleNamespace):
        return {key: config_dict(value) for key, value in vars(cfg).items()}
    if isinstance(cfg, dict):
        return {key: config_dict(value) for key, value in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [config_dict(value) for value in cfg]
    return cfg
