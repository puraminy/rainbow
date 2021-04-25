"""Task definitions for rainbow."""

import os

import t5
import tensorflow as tf
import seqio

from . import core, datasets, preprocessors, settings

T5_DEFAULT_SPM_PATH = os.path.join(
    settings.BASE_DIR, "pretrained/t5/sentencepiece.model"
)
MT5_DEFAULT_SPM_PATH = os.path.join(
    settings.BASE_DIR, "pretrained/mt5/sentencepiece.model"
)


T5_DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(T5_DEFAULT_SPM_PATH)
MT5_DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(MT5_DEFAULT_SPM_PATH)
MT5_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=MT5_DEFAULT_VOCAB, add_eos=True, required=False
    ),
    "targets": seqio.Feature(vocabulary=MT5_DEFAULT_VOCAB, add_eos=True),
}

T5_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True
    ),
    "targets": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True
    ),
}

# Create tasks for the datasets.
t5.data.TaskRegistry.add(
    "t5_tvs_atomic",
    # Specify the task type.
    core.TvsTask,
    # Supply a function which returns a tf.data.Dataset.
    split_to_filepattern={
        split: os.path.join(
            settings.PREPROCESSED_DATASETS_DIR, "atomic_" + split + ".tsv"
        )
        for split in ["train", "validation"]
    },
    num_input_examples={"train": 600, "validation": 700},
    text_preprocessor=[preprocessors.tvs_preprocessor],
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text,
    # output_features=DEFAULT_OUTPUT_FEATURES
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    # Not required, but helps for mixing and auto-caching.
    # num_input_examples=num_atomic_examples
)

# seqio.TaskRegistry.add(
#    "nq_context_free",
#    source=seqio.TextLineDataSource(
#        split_to_filepattern=nq_tsv_path,
#        num_input_examples=num_nq_examples),
#    preprocessors=[
#      functools.partial(
#          t5.data.preprocessors.parse_tsv,
#          field_names=["question", "answer"]),
#      trivia_preprocessor,
#      seqio.preprocessors.tokenize_and_append_eos,
#    ],
#    postprocess_fn=t5.data.postprocessors.lower_text,
#    metric_fns=[t5.evaluation.metrics.accuracy],
#    output_features=DEFAULT_OUTPUT_FEATURES,
# )

# Create the rainbow tasks.
for dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        task_name = (
            f"{dataset.name}_task"
            if size is None
            else f"{dataset.name}_{size:05}_task"
        )
        t5.data.TaskRegistry.add(
            name=task_name,
            task_cls=core.CsvTask,
            # args for CsvTask
            #   dataset configuration and location
            split_to_filepattern={
                split.name: os.path.join(
                    settings.PREPROCESSED_DATASETS_DIR,
                    dataset.name,
                    settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                        split=split.name, dataset=dataset.name
                    ),
                )
                for split in dataset.splits.values()
            },
            num_input_examples={
                split.name: split.size for split in dataset.splits.values()
            },
            text_preprocessor=[
                preprocessors.make_add_field_names_preprocessor(
                    field_indices=[1, 2], field_names=["inputs", "targets"]
                )
            ],
            metric_fns=[t5.evaluation.metrics.accuracy],
            #   CSV parsing
            record_defaults=[tf.int32, tf.string, tf.string],
            compression_type=None,
            buffer_size=None,
            header=True,
            field_delim=",",
            use_quote_delim=True,
            na_value="",
            select_cols=None,
            #   dataset truncation
            truncate_to=size,
            # args for the task class
            postprocess_fn=t5.data.postprocessors.lower_text,
        )

# bbbb
# Create knowledge graph tasks.
for dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for direction in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
            for output_features in [T5_OUTPUT_FEATURES, MT5_OUTPUT_FEATURES]:
                mt = "t5" if output_features is T5_OUTPUT_FEATURES else "mt5"
                task_name = (
                    f"{mt}_{dataset.name}_{direction}_task"
                    if size is None
                    else f"{mt}_{dataset.name}_{direction}_{size:05}_task"
                )

                if direction == "forward":
                    predicate = lambda x: tf.strings.regex_full_match(
                        x["targets"], r"^<object>.*"
                    )
                elif direction == "backward":
                    predicate = lambda x: tf.strings.regex_full_match(
                        x["targets"], r"^<subject>.*"
                    )
                elif direction == "bidirectional":
                    predicate = lambda x: True
                else:
                    raise ValueError(f"Unrecognized direction: {direction}.")

                t5.data.TaskRegistry.add(
                    name=task_name,
                    task_cls=core.CsvTask,
                    # args for CsvTask
                    #   dataset configuration and location
                    split_to_filepattern={
                        split.name: os.path.join(
                            settings.PREPROCESSED_DATASETS_DIR,
                            dataset.name,
                            settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                                split=split.name, dataset=dataset.name
                            ),
                        )
                        for split in dataset.splits.values()
                    },
                    num_input_examples={
                        split.name: split.size / 2
                        for split in dataset.splits.values()
                    },
                    text_preprocessor=[
                        preprocessors.make_add_field_names_preprocessor(
                            field_indices=[1, 2],
                            field_names=["inputs", "targets"],
                            direction=direction,
                        ),
                        preprocessors.make_filter_preprocessor(predicate),
                    ],
                    output_features=output_features,
                    metric_fns=[t5.evaluation.metrics.accuracy],
                    #   CSV parsing
                    record_defaults=[tf.int32, tf.string, tf.string],
                    compression_type=None,
                    buffer_size=None,
                    header=True,
                    field_delim=",",
                    use_quote_delim=True,
                    na_value="",
                    select_cols=None,
                    #   dataset truncation
                    truncate_to=size,
                    # args for the task class
                    postprocess_fn=t5.data.postprocessors.lower_text,
                )


# Create the commonsense tasks.
for dataset in datasets.COMMONSENSE_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        task_name = (
            f"{dataset.name}_task"
            if size is None
            else f"{dataset.name}_{size:05}_task"
        )
        t5.data.TaskRegistry.add(
            name=task_name,
            task_cls=core.CsvTask,
            # args for CsvTask
            #   dataset configuration and location
            split_to_filepattern={
                split.name: os.path.join(
                    settings.PREPROCESSED_DATASETS_DIR,
                    dataset.name,
                    settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                        split=split.name, dataset=dataset.name
                    ),
                )
                for split in dataset.splits.values()
            },
            num_input_examples={
                split.name: split.size for split in dataset.splits.values()
            },
            text_preprocessor=[
                preprocessors.make_add_field_names_preprocessor(
                    field_indices=[1, 2], field_names=["inputs", "targets"]
                )
            ],
            metric_fns=[t5.evaluation.metrics.accuracy],
            #   CSV parsing
            record_defaults=[tf.int32, tf.string, tf.string],
            compression_type=None,
            buffer_size=None,
            header=True,
            field_delim=",",
            use_quote_delim=True,
            na_value="",
            select_cols=None,
            #   dataset truncation
            truncate_to=size,
            # args for the task class
            postprocess_fn=t5.data.postprocessors.lower_text,
        )
