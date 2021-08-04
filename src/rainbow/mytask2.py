"""Task definitions for rainbow."""
from pathlib import Path
import os
import pandas as pd
import t5
import tensorflow as tf
import functools
from . import core, datasets, preprocessors, settings, rates
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

rels = ["xAttr", "xEffect", "oEffect", "xIntent", "xWant", "oWant", "xNeed", "xReact" , "oReact"]
rels = ["xIntent"]
sel_langs = ["en", "fa"]
print("==================== Registering my tasks ====================")
for lang in sel_langs: # include or exclude none values
    for natural in ["natural_"]: # include or exclude none values
        for rel in rels:
            task_name = f"newexp_{lang}2en_{rel}_{natural}task"
            print("Task:", task_name)
            paths={}
            paths["src"] = os.path.join( "/drive3/pouramini/data/atomic/en_fa/", f"{rel}_en_fa_de_train_no_dups.tsv")

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
                task_name,
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

sel_langs = ["fa"]
slangs = "newexp_" + "".join(sel_langs)
for lang in sel_langs:
    for natural in ["_natural"]:
        for rel in rels: 
            mixture_name = f"{slangs}_{rel}{natural}_mixture"
            if  mixture_name in t5.data.MixtureRegistry.names():
                print(mixture_name, " already registered")
            else:
                t5.data.MixtureRegistry.add(
                    mixture_name,
                    [
                        f"newexp_{lang}2en_{rel}{natural}_task", f"newexp_en2en_{rel}{natural}_task"
                    ],
                    default_rate=1.0 #rates.proportional_rate,
                )
