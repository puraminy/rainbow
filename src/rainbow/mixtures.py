"""Mixture definitions for rainbow."""

import t5

from . import datasets, rates, settings

import rainbow.tasks

# N.B. tasks must be imported before mixtures, so that the mixtures can use
# the tasks in their definitions.


# Create the individual rainbow task mixtures.
for dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        base_name = (
            f"{dataset.name}" if size is None else f"{dataset.name}_{size:05}"
        )
        t5.data.MixtureRegistry.add(
            f"{base_name}_mixture", [f"{base_name}_task"], default_rate=1.0
        )

# Create paired rainbow task mixtures.
for dataset1 in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for dataset2 in datasets.RAINBOW_DATASETS.values():
            for rate_name, rate_func in rates.MIXING_RATES.items():
                dataset1_base_name = (
                    f"{dataset1.name}"
                    if size is None
                    else f"{dataset1.name}_{size:05}"
                )
                t5.data.MixtureRegistry.add(
                    f"{dataset1_base_name}_{dataset2.name}_{rate_name}_mixture",
                    [f"{dataset1_base_name}_task", f"{dataset2.name}_task"],
                    default_rate=rate_func,
                )

# bbb
# My tasks
# Create paired directions for single knowlege graph
for dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for mt in ["t5", "mt5"]:
            for dir1 in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
                for dir2 in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
                    if dir1 == dir2:
                        continue
                    base_name = (
                        f"{mt}_{dataset.name}_{dir1}"
                        if size is None
                        else f"{mt}_{dataset.name}_{dir1}_{size:05}"
                    )
                    t5.data.MixtureRegistry.add(
                        f"{base_name}_{dir2}_mixture",
                        [
                            f"{base_name}_task",
                            f"{mt}_{dataset.name}_{dir2}_task",
                        ],
                        default_rate=1.0,
                    )

for dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for direction in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
            for mt in ["t5", "mt5"]:
                base_name = (
                    f"{mt}_{dataset.name}_{direction}"
                    if size is None
                    else f"{mt}_{dataset.name}_{direction}_{size:05}"
                )
                t5.data.MixtureRegistry.add(
                    f"{base_name}_mixture",
                    [f"{base_name}_task"],
                    default_rate=1.0,
                )
# Create the individual external commonsense datasets mixtures.
for dataset in datasets.COMMONSENSE_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        base_name = (
            f"{dataset.name}" if size is None else f"{dataset.name}_{size:05}"
        )
        t5.data.MixtureRegistry.add(
            f"{base_name}_mixture", [f"{base_name}_task"], default_rate=1.0
        )

## My task
for lang in ["e2e", "e2p", "p2e", "p2p"]:
    t5.data.MixtureRegistry.add(
        f"{lang}_rel_mixture",
        [
            f"{lang}_{rel}_task"
            for rel in ["xAttr", "xEffect", "oEffect", "xIntent", "xWant", "oWant", "xNeed", "xReact" , "oReact"]
        ],
        default_rate=rates.proportional_rate,
    )

t5.data.MixtureRegistry.add(
    f"l2l_rel_mixture",
    [
        f"e2e_{rel}_task"
        for rel in ["xAttr", "xEffect", "oEffect", "xIntent", "xWant", "oWant", "xNeed", "xReact" , "oReact"]
    ] + 
    [
        f"e2p_{rel}_task"
        for rel in ["xAttr", "xEffect", "oEffect", "xIntent", "xWant", "oWant", "xNeed", "xReact" , "oReact"]
    ] +
    [
        f"p2e_{rel}_task"
        for rel in ["xAttr", "xEffect", "oEffect", "xIntent", "xWant", "oWant", "xNeed", "xReact" , "oReact"]
    ] +
    [
        f"p2p_{rel}_task"
        for rel in ["xAttr", "xEffect", "oEffect", "xIntent", "xWant", "oWant", "xNeed", "xReact" , "oReact"]
    ],
    default_rate=rates.proportional_rate,
)
# Create the Rainbow mixtures.
for rate_name, rate_func in rates.MIXING_RATES.items():
    t5.data.MixtureRegistry.add(
        f"rainbow_{rate_name}_mixture",
        [
            f"{dataset.name}_task"
            for dataset in datasets.RAINBOW_DATASETS.values()
        ],
        default_rate=rate_func,
    )

# Create leave-one-out Rainbow mixtures.
for dataset in datasets.RAINBOW_DATASETS.values():
    for rate_name, rate_func in rates.MIXING_RATES.items():
        t5.data.MixtureRegistry.add(
            f"{dataset.name}_00000_rainbow_{rate_name}_mixture",
            [
                f"{other_dataset.name}_task"
                for other_dataset in datasets.RAINBOW_DATASETS.values()
                if other_dataset != dataset
            ],
            default_rate=rate_func,
        )

# Create knowledge graph-only mixtures with a single knowledge graph.
# Create mixtures with rainbow and all knowledge graphs.
for direction in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
    for rate_name, rate_func in rates.MIXING_RATES.items():
        for mt in ["t5", "mt5"]:
            t5.data.MixtureRegistry.add(
                f"{mt}_rainbow_comet_{direction}_{rate_name}_mixture",
                [
                    # include the knowledge graphs
                    f"{mt}_{dataset.name}_{direction}_task"
                    for dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values()
                ]
                + [
                    # include the rainbow datasets
                    f"{dataset.name}_task"
                    for dataset in datasets.RAINBOW_DATASETS.values()
                ],
                default_rate=rate_func,
            )

# Create the Rainbow multi-tasking learning curve mixtures.
for dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        if size is None:
            continue

        for rate_name, rate_func in rates.MIXING_RATES.items():
            t5.data.MixtureRegistry.add(
                f"{dataset.name}_{size:05}_rainbow_{rate_name}_mixture",
                [f"{dataset.name}_{size:05}_task"]
                + [
                    f"{other_dataset.name}_task"
                    for other_dataset in datasets.RAINBOW_DATASETS.values()
                    if other_dataset != dataset
                ],
                default_rate=rate_func,
            )

# Create rainbow external evaluation mixtures.
for dataset in datasets.COMMONSENSE_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for rate_name, rate_func in rates.MIXING_RATES.items():
            base_name = (
                dataset.name if size is None else f"{dataset.name}_{size:05}"
            )
            t5.data.MixtureRegistry.add(
                f"{base_name}_rainbow_{rate_name}_mixture",
                [f"{base_name}_task"]
                + [
                    f"{rainbow_dataset.name}_task"
                    for rainbow_dataset in datasets.RAINBOW_DATASETS.values()
                ],
                default_rate=rate_func,
            )
