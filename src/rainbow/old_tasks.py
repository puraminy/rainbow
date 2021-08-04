"""Task definitions for rainbow."""
from pathlib import Path
import os
import pandas as pd
import t5
import tensorflow as tf
import functools
from . import core, datasets, preprocessors, settings
BASE_DIR = "/drive2/"
T5_DEFAULT_SPM_PATH = os.path.join(
    BASE_DIR, "pretrained/t5/sentencepiece.model"
)
MT5_DEFAULT_SPM_PATH = os.path.join(
    BASE_DIR, "pretrained/mt5/sentencepiece.model"
)


T5_DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(T5_DEFAULT_SPM_PATH)
MT5_DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(MT5_DEFAULT_SPM_PATH)
MT5_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=MT5_DEFAULT_VOCAB, add_eos=True, required=False
    ),
    "targets": t5.data.Feature(vocabulary=MT5_DEFAULT_VOCAB, add_eos=True),
}

T5_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=T5_DEFAULT_VOCAB, add_eos=True
    ),
    "targets": t5.data.Feature(
        vocabulary=T5_DEFAULT_VOCAB, add_eos=True
    ),
}
for nn in ["","_nn"]: # include or exclude none values
    for rel in ["xAttr", "xEffect", "oEffect", "xIntent", "xWant", "oWant", "xNeed", "xReact" , "oReact"]:
        paths={
            split: os.path.join(
                "/drive3/pouramini/data/atomic/en_fa/", f"{rel}_{split}_no_dups{nn}.tsv"
            )
            for split in ["en_train","fa_train", "en_fa_train", "en_valid", "fa_valid", "en_fa_validation"]
        }
        num_lines =  {split: sum(1 for line in open(path)) for split,path in paths.items()}
        t5.data.TaskRegistry.add(
            f"enfa_{rel}{nn}_task",
            # Specify the task type.
            core.MyTsvTask,
            sel_cols = ["prefix", "input_text", "target_text"],
            # Supply a function which returns a tf.data.Dataset.
            split_to_filepattern=paths,
            num_input_examples=num_lines,
            text_preprocessor=[preprocessors.tsv_rel_preprocessor('en2fa')],
            # Lowercase targets before computing metrics.
            # postprocess_fn=t5.data.postprocessors.lower_text,
            # output_features=DEFAULT_OUTPUT_FEATURES
            # We'll use accuracy as our evaluation metric.
            metric_fns=[t5.evaluation.metrics.accuracy],
            # Not required, but helps for mixing and auto-caching.
            # num_input_examples=num_atomic_examples
            output_features=MT5_OUTPUT_FEATURES #if lang == "per" else None
        )

# Create tasks for the datasets.
t5.data.TaskRegistry.add(
    "eng_task",
    # Specify the task type.
    core.MyTsvTask,
    sel_cols = ["prefix", "input_text", "target_text"],
    # Supply a function which returns a tf.data.Dataset.
    split_to_filepattern={
        split: os.path.join(
            "/drive3/pouramini/data/atomic/", "natural_all_atomic_" + split + ".tsv"
        )
        for split in ["train", "validation"]
    },
    num_input_examples={"train": 708000, "validation": 79000},
    text_preprocessor=[preprocessors.tvs_preprocessor],
    # Lowercase targets before computing metrics.
    # postprocess_fn=t5.data.postprocessors.lower_text,
    # output_features=DEFAULT_OUTPUT_FEATURES
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    # Not required, but helps for mixing and auto-caching.
    # num_input_examples=num_atomic_examples
    output_features=MT5_OUTPUT_FEATURES
)

print("!!!!!!!!!!!!!!!!!!! Tasks was called !!!!!!!!!!!!!")
rels = ["xAttr", "xEffect", "oEffect", "xIntent", "xWant", "oWant", "xNeed", "xReact" , "oReact"]
rels = ["xIntent"]
sel_langs = ["en", "fa"]
for lang in sel_langs: # include or exclude none values
    for natural in ["natural_"]: # include or exclude none values
        for rel in rels:
            paths={}
            paths["src"] = os.path.join( "/drive3/pouramini/data/atomic/en_fa/", f"{rel}_en_fa_de_train_no_dups.tsv")

#                paths["en_fa_train"] = os.path.join(
#                        "/drive3/pouramini/data/atomic/en_fa/", f"{rel}_en_fa_train_no_dups.tsv")
#                paths["en_train"] = os.path.join(
#                        "/drive3/pouramini/data/atomic/en_fa/", f"{rel}_en_train_no_dups{natural}.tsv")
#                paths["fa_train"] = os.path.join(
#                        "/drive3/pouramini/data/atomic/en_fa/", f"{rel}_fa_train_no_dups{natural}.tsv")
#                paths["en_valid"] = os.path.join(
#                        "/drive3/pouramini/data/atomic/en_fa/", f"{rel}_en_valid_no_dups.tsv")
#                paths["fa_valid"] = os.path.join(
#                        "/drive3/pouramini/data/atomic/en_fa/", f"{rel}_fa_valid_no_dups.tsv")
#                paths["en_fa_validation"] = os.path.join(
#                        "/drive3/pouramini/data/atomic/en_fa/", f"{rel}_en_fa_validation_no_dups.tsv")
#
            for vlang in sel_langs:
                input_col = f"{natural}input_text_{vlang}"
                target_col = "target_text"
                sel_cols = ["prefix", input_col, target_col]
                df = pd.read_table(paths["src"], index_col =0)
                if not "prefix" in df:
                    raise Exception(f"prefix is not in dataframe")
                if not input_col in df:
                    raise Exception(f"{input_col} is not in dataframe")
                if not target_col in df:
                    raise Exception(f"{target_col} is not in dataframe")

                new_df = df[sel_cols]
                new_df.columns = ["prefix", "input_text", "target_text"]
                data_folder = str(Path(paths["src"]).parent)
                p = data_folder + f"/{rel}_{vlang}_train_no_dups_{natural}newexp.tsv"
#
                new_df.to_csv(p, sep="\t", index = False)
                paths[vlang + "_train"] = p
                print(p)
            p = data_folder + f"/{rel}_{lang}_train_no_dups_{natural}newexp.tsv"
            paths["train"]= p
            num_lines =  {split: sum(1 for line in open(path)) for split,path in paths.items()}
            t5.data.TaskRegistry.add(
                f"newexp_{lang}2en_{rel}_{natural}task",
                # Specify the task type.
                core.MyTsvTask,
                #record_defaults = ["", "", ""],
                # Supply a function which returns a tf.data.Dataset.
                sel_cols=sel_cols,
                split_to_filepattern=paths,
                num_input_examples=num_lines,
                text_preprocessor=[preprocessors.tsv_rel_preprocessor(f'{lang}2en')],
                # Lowercase targets before computing metrics.
                # postprocess_fn=t5.data.postprocessors.lower_text,
                # output_features=DEFAULT_OUTPUT_FEATURES
                # We'll use accuracy as our evaluation metric.
                metric_fns=[t5.evaluation.metrics.accuracy],
                # Not required, but helps for mixing and auto-caching.
                # num_input_examples=num_atomic_examples
                output_features=MT5_OUTPUT_FEATURES #if model == "" else T5_OUTPUT_FEATURES
            )

# Create tasks for the datasets.
t5.data.TaskRegistry.add(
    "en_fa_task",
    # Specify the task type.
    core.MyTsvTask,
    sel_cols = ["prefix", "input_text", "target_text"],
    # Supply a function which returns a tf.data.Dataset.
    split_to_filepattern={
        split: os.path.join(
            "/drive3/pouramini/data/atomic/en_fa/", "en_fa_train.tsv"
        )
        for split in ["train", "validation"]
    },
    num_input_examples={"train": 74206, "validation": 74206},
    text_preprocessor=[preprocessors.tvs_preprocessor],
    # Lowercase targets before computing metrics.
    # postprocess_fn=t5.data.postprocessors.lower_text,
    # output_features=DEFAULT_OUTPUT_FEATURES
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    # Not required, but helps for mixing and auto-caching.
    # num_input_examples=num_atomic_examples
    output_features=MT5_OUTPUT_FEATURES
)
# Create tasks for the datasets.
t5.data.TaskRegistry.add(
    "per_task",
    # Specify the task type.
    core.MyTsvTask,
    sel_cols = ["prefix", "input_text", "target_text"],
    # Supply a function which returns a tf.data.Dataset.
    split_to_filepattern={
        split: os.path.join(
            "/drive3/pouramini/data/atomic/", "translate_" + split + ".tsv"
        )
        for split in ["train", "validation"]
    },
    num_input_examples={"train": 708000, "validation": 79000},
    text_preprocessor=[preprocessors.tvs_preprocessor],
    # Lowercase targets before computing metrics.
    # postprocess_fn=t5.data.postprocessors.lower_text,
    # output_features=DEFAULT_OUTPUT_FEATURES
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    # Not required, but helps for mixing and auto-caching.
    # num_input_examples=num_atomic_examples
    output_features=MT5_OUTPUT_FEATURES
)
split_list = {}
split_list["e2e"] = ["e2e", "p2e"]
split_list["e2p"] = ["e2p", "p2p"]
split_list["p2e"] = ["e2e", "p2e"]
split_list["p2p"] = ["p2p", "e2p"]
for lang in ["e2e", "e2p", "p2e", "p2p"]:
    for rel in ["xAttr", "xEffect", "oEffect", "xIntent", "xWant", "oWant", "xNeed", "xReact" , "oReact"]:
        paths={
            split: os.path.join(
                "/drive3/pouramini/data/atomic/mix/", f"{split}_{rel}_validation.tsv"
            )
            for split in split_list[lang]
        }
        paths["train"] = os.path.join("/drive3/pouramini/data/atomic/mix/", f"{lang}_{rel}_train.tsv")
        num_lines =  {split: sum(1 for line in open(path)) for split,path in paths.items()}
        sel_cols = ["prefix", "input_text", "target_text"]
        df = pd.read_table(paths["train"])
        sel_cols = [df.columns.get_loc(c) for c in sel_cols if c in df]

        t5.data.TaskRegistry.add(
            f"{lang}_{rel}_task",
            # Specify the task type.
            core.MyTsvTask,
            sel_cols = sel_cols,
            # Supply a function which returns a tf.data.Dataset.
            split_to_filepattern=paths,
            num_input_examples=num_lines,
            text_preprocessor=[preprocessors.tsv_rel_preprocessor(lang)],
            # Lowercase targets before computing metrics.
            # postprocess_fn=t5.data.postprocessors.lower_text,
            # output_features=DEFAULT_OUTPUT_FEATURES
            # We'll use accuracy as our evaluation metric.
            metric_fns=[t5.evaluation.metrics.accuracy],
            # Not required, but helps for mixing and auto-caching.
            # num_input_examples=num_atomic_examples
            output_features=MT5_OUTPUT_FEATURES #if lang == "per" else None
        )

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
