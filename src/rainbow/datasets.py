"""Dataset definitions for rainbow."""

from typing import Dict, List

import t5

from . import utils


# N.B. The names and sizes for the datasets must correspond to the names
# and sizes used during dataset preparation (see
# $REPO/bin/prepare.py and $REPO/src/rainbow/preparation/).


# core classes


@utils.data
class Split:
    """A single split of a dataset."""

    name: str
    size: int


@utils.data
class Dataset:
    """A dataset."""

    name: str
    splits: Dict[str, Split]


# constants

RAINBOW_DATASETS = {
    # AlphaNLI
    "anli": Dataset(
        name="anli",
        splits={
            "train": Split(name="train", size=169654),
            "validation": Split(name="validation", size=1532),
            "test": Split(name="test", size=3040),
        },
    ),
    # CosmosQA
    "cosmosqa": Dataset(
        name="cosmosqa",
        splits={
            "train": Split(name="train", size=25262),
            "validation": Split(name="validation", size=2985),
            "test": Split(name="test", size=6963),
        },
    ),
    # HellaSWAG
    "hellaswag": Dataset(
        name="hellaswag",
        splits={
            "train": Split(name="train", size=39905),
            "validation": Split(name="validation", size=10042),
            "test": Split(name="test", size=10050),
        },
    ),
    # PhysicalIQA
    "physicaliqa": Dataset(
        name="physicaliqa",
        splits={
            "train": Split(name="train", size=16113),
            "validation": Split(name="validation", size=1838),
            "test": Split(name="test", size=3446),
        },
    ),
    # SocialIQA
    "socialiqa": Dataset(
        name="socialiqa",
        splits={
            "train": Split(name="train", size=33410),
            "validation": Split(name="validation", size=1954),
            "test": Split(name="test", size=2059),
        },
    ),
    # WinoGrande
    "winogrande": Dataset(
        name="winogrande",
        splits={
            "train": Split(name="train", size=40398),
            "train_xs": Split(name="train_xs", size=160),
            "train_s": Split(name="train_s", size=640),
            "train_m": Split(name="train_m", size=2558),
            "train_l": Split(name="train_l", size=10234),
            "train_xl": Split(name="train_xl", size=40398),
            "validation": Split(name="validation", size=1267),
            "test": Split(name="test", size=1767),
        },
    ),
}
"""Rainbow datasets."""


KNOWLEDGE_GRAPH_DATASETS = {
    # ATOMIC
    "atomic": Dataset(
        name="atomic",
        splits={
            "train": Split(name="train", size=2 * 709996),
            "validation": Split(name="validation", size=2 * 79600),
        },
    ),
    # ConceptNet
    "conceptnet": Dataset(
        name="conceptnet",
        splits={
            "train": Split(name="train", size=2 * 100000),
            "validation": Split(name="validation", size=2 * 1200),
        },
    ),
}
"""Commonsense knowledge graph datasets."""
COMMONSENSE_DATASETS = {
    # CommonsenseQA
    "commonsenseqa": Dataset(
        name="commonsenseqa",
        splits={
            "train": Split(name="train", size=9741),
            "validation": Split(name="validation", size=1221),
            "test": Split(name="test", size=1140),
        },
    ),
    # JHU Ordinal Commonsense Inference
    "joci": Dataset(
        name="joci",
        splits={
            "train": Split(name="train", size=34092),
            "validation": Split(name="validation", size=2500),
        },
    ),
    # CycIC
    "cyc": Dataset(
        name="cyc",
        splits={
            "train": Split(name="train", size=10678),
            "validation": Split(name="validation", size=1525),
            "test": Split(name="test", size=3051),
        },
    ),
}
"""Commonsense datasets besides rainbow."""
