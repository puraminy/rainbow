#! /usr/bin/env python

"""Fine-tune the model on the rainbow datasets."""

import logging
import os
import shutil
import click
import t5
import glob
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from rainbow import utils, settings, core, rates, preprocessors
import pandas as pd
from pathlib import Path

tf.get_logger().setLevel("ERROR")
# N.B. We must import rainbow.mixtures here so that the mixtures are registered
# and available for training.

BASE_DIR = "/drive2/"
DATA_DIR = "."
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

logger = logging.getLogger(__name__)
PRETRAINED_MODELS = {
    "t5_small": os.path.join(BASE_DIR, "pretrained/t5/small"),
    "t5_large": os.path.join(BASE_DIR, "pretrained/t5/large"),
    "mt5_small": os.path.join(BASE_DIR, "pretrained/mt5/small"),
    "mt5_base": os.path.join(BASE_DIR, "pretrained/mt5/base"),
    "mt5_sa2": os.path.join(BASE_DIR, "pretrained/mt5/small_atomic_2"),
    "mt5_ba2": os.path.join(BASE_DIR, "pretrained/mt5/base_atomic_2"),
    "mt5_snaa": os.path.join(BASE_DIR, "pretrained/mt5/small_natural_all_atomic"),
    "mt5_large": os.path.join(BASE_DIR, "pretrained/mt5/large"),
}

@click.command()
@click.argument("input_cols", type=str)
@click.option(
    "--target_col",
    default="target_text",
    type=str,
    help=""
)
@click.option(
    "--rel",
    default="xIntent",
    type=str,
    help="The relation between input and target"
)
@click.option(
    "--model-dir",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
)
@click.option(
    "--do_train",
    "-t",
    is_flag=True,
    help=""
)
@click.option(
    "--do_eval",
    "-v",
    is_flag=True,
    help=""
)
@click.option(
    "--pm",
    type=str,
    default="t5_small",
    help="The path to or name of the pretrained model. ",
)
@click.option(
    "--n-steps",
    type=int,
    default=110000,
    help="The number of gradient updates. Defaults to 25,000.",
)
@click.option(
    "--eval-step",
    default=-1,
    type=int,
    help="Step for evaluaiton, defaults to last checkpoint"
)
@click.option(
    "--learning-rate",
    type=float,
    default=3e-3,
    help="The learning rate to use for training. Defaults to 3e-3.",
)
@click.option(
    "--extra", type=str, default="",
)
@click.option(
    "--bs",
    type=int,
    default=8,
    help="Batch size",
)
@click.option(
    "--save-checkpoints-steps",
    "-s",
    type=int,
    default=5000,
    help=(
        "The number of steps to take before saving a checkpoint. Defaults to"
        " 5000."
    ),
)
@click.option(
    "--n-checkpoints-to-keep",
    "-n",
    type=int,
    default=2,
    help=(
        "The number of checkpoints to keep during fine-tuning. Defaults"
        " to 4."
    ),
)
@click.option(
    "--info",
    "-i",
    is_flag=True,
    help="Get info from training dataset"
)
@click.option(
    "--ds_fname",
    default="train",
    type=str,
    help=""
)
def fine_tune(
    input_cols: str,
    target_col: str,
    rel:str ,
    model_dir: str,
    do_train,
    do_eval,
    pm: str,
    n_steps: int,
    eval_step: int,
    learning_rate: float,
    extra: str,
    bs: int,
    save_checkpoints_steps: int,
    n_checkpoints_to_keep: int,
    info,
    ds_fname
) -> None:
    if not do_train and not do_eval and not info:
        print("Specify train or evaluation flag. --do_train or --do_eval")
        return
    if not Path(model_dir).exists():
        Path(model_dir).mkdir(parents=True, exist_ok=True)
    if do_train and any(os.scandir(model_dir)):
        temp = glob.glob("model.ckpt*")
        if len(temp) == 0:
            print("The folder must contain a checkpoint or be empty!")
            return

    task_names = []
    input_prefix = input_postfix = target_prefix = target_postfix = ""
    for input_col in input_cols.split(","):
        input_col = input_col.strip()
        if ":" in input_col:
            input_prefix, input_col, input_postfix = input_col.split(":")
            if input_prefix and not input_prefix.endswith(" "):
                input_prefix += " "
            if input_postfix and not input_postfix.startswith(" "):
                input_postfix = " " + input_postfix
        if ":" in target_col:
            target_prefix, target_col, target_postfix = target_col.split(":")
            if target_prefix and not target_prefix.endswith(" "):
                target_prefix += " "
            if target_postfix and not target_postfix.startswith(" "):
                target_postfix = " " + target_postfix
        task_name  = input_col + "_" + target_col
        task_names.append(task_name)
        print("Task:", task_name)
        paths={}
        paths["src"] = os.path.join(DATA_DIR, f"atomic_train.tsv")

        sel_cols = ["prefix", input_col, target_col]
        df = pd.read_table(paths["src"])
        if not "prefix" in df:
            raise Exception(f"prefix is not in dataframe")
        if not input_col in df:
            raise Exception(f"{input_col} is not in dataframe")
        if not target_col in df:
            raise Exception(f"{target_col} is not in dataframe")
        if info:
            for col in df.columns:
                print(col)
            print(df[[input_col, target_col]].head())
            input_file = open(ds_fname + "." + input_col, "w")
            target_file = open(ds_fname + "." + target_col, "w")
            for idx, row in df.iterrows():
                print(input_prefix + str(row[input_col]) + input_postfix, file=input_file)
                print(target_prefix + str(row[target_col]) + target_postfix, file=target_file)
            input_file.close()
            target_file.close()
            return

        new_df = df[sel_cols]
        new_df.columns = ["prefix", "input_text", "target_text"]
        data_folder = os.path.join(model_dir, "data")
        Path(data_folder).mkdir(parents=True, exist_ok=True)
        p = data_folder + f"/{rel}_{input_col}_train.tsv"
    #
        new_df.to_csv(p, sep="\t", index = False)
        if do_train:
            split = "train"
        else:
            split = input_col
        paths[split] = p
        print(p)
        exp = Path(model_dir).stem
        print("Exp lable:", exp)
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
            text_preprocessor=[preprocessors.tsv_rel_preprocessor(input_prefix, input_postfix, target_prefix, target_postfix)],
            # Lowercase targets before computing metrics.
            # postprocess_fn=t5.data.postprocessors.lower_text,
            # output_features=DEFAULT_OUTPUT_FEATURES
            # We'll use accuracy as our evaluation metric.
            metric_fns=[t5.evaluation.metrics.accuracy],
            # Not required, but helps for mixing and auto-caching.
            # num_input_examples=num_atomic_examples
            output_features=MT5_OUTPUT_FEATURES  if pm.startswith("mt5") else None
        )


    mixture = "_".join(task_names)
    print("Mixture:", mixture, ":", task_names)
    t5.data.MixtureRegistry.add(
        mixture,
        task_names,
        default_rate=1.0 #rates.proportional_rate,
    )


    """Fine-tune the model on MIXTURE, writing results to RESULTS_DIR."""
    if not pm in PRETRAINED_MODELS:
        raise ValueError(pm + " path isn't introduced in PRETRAINED_MODELS")

    if pm.startswith("t5_") and mixture.startswith("mt5_"):
        raise ValueError(pm + " isn't matched with the mixture")

    if "large" in pm or ("mt5" in pm):
        bs = 4
    if "large" in pm and ("mt5" in pm):
        bs = 1
    if "base" in pm and ("mt5" in pm):
        bs = 1
    batch_size = bs
    print("Using ", pm, " Batch Size:", bs)
    print("==============================================")
    utils.configure_logging(clear=True)
    if extra:
        results_dir = os.path.join(
            settings.BASE_DIR, "models/rb", pm, extra + "/" + mixture
        )
    else:
        results_dir = os.path.join(settings.BASE_DIR, "models/rb", pm, task_name)

    MODEL_TYPE = pm.split("_")[0]

    # Validate arguments.

#    task = t5.data.MixtureRegistry.get(mixture)
#    ds = task.get_dataset(
#        split="train", sequence_length={"inputs": 128, "targets": 128} 
#    )
#    # bbb
#    print("A few preprocessed validation examples...")
#    for ex in tfds.as_numpy(ds.take(5)):
#        tf.print(ex)
#    # Process arguments.
    pm = PRETRAINED_MODELS[pm]

    # Run fine-tuning.

    model = t5.models.MtfModel(
        tpu=None,
        model_dir=model_dir,
        model_parallelism=8,
        batch_size=batch_size,
        sequence_length={"inputs": 128, "targets": 128},
        mesh_shape="model:1,batch:1",
        mesh_devices=["gpu:0"],
        learning_rate_schedule=learning_rate,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=n_checkpoints_to_keep,
        iterations_per_loop=100,
    )
    if do_train:
        print("============================ Training =========================")
        try:
            model.finetune(
                mixture_or_task_name=mixture,
                pretrained_model_dir=pm,
                finetune_steps=n_steps,
                split=split,
            )
        except KeyboardInterrupt:
            print("Saving model")

        export_dir = os.path.join(results_dir, "export")
        task_vocab = t5.models.utils.get_vocabulary(mixture)

        model.batch_size = 1  # make one prediction per call
        saved_model_path = model.export(
            export_dir,
            checkpoint_step=-1,  # use most recent
            beam_size=1,  # no beam search
            vocabulary=task_vocab,
            temperature=1.0,  # sample according to predicted distribution
        )
        print("cd ", results_dir, "")

    if do_eval:
        print("============================ Validating =========================")
        summary_dir = "validation"
        split = input_col
        split_dir = os.path.join(model_dir, summary_dir, split)
        #split_dir = summary_dir
        if True: #not os.path.isdir(split_dir):
            tf.io.gfile.makedirs(split_dir)
            print("=====================", split_dir, "=========================")
            ref_file = paths["src"]
            shutil.copy(ref_file, split_dir + "/src_df.tsv")
            model.eval(
                mixture_or_task_name=mixture,
                summary_dir=split_dir,
                checkpoint_steps=eval_step,
                #eval_with_score=True,
                split=split,
            )

if __name__ == "__main__":
    fine_tune()
