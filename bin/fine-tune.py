#! /usr/bin/env python

"""Fine-tune the model on the rainbow datasets."""
import seqio
import logging
import glob
import os
import shutil
import click
import t5.models
import glob
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from comet.train.common import *
from comet.train.eval import *
from rainbow import utils, settings, core, rates, preprocessors
import pandas as pd
from pathlib import Path
from datasets import Dataset

tf.get_logger().setLevel("ERROR")
# N.B. We must import rainbow.mixtures here so that the mixtures are registered
# and available for training.

BASE_DIR = "/home/pouramini/"
DATA_DIR = "/home/pouramini/atomic"
T5_DEFAULT_SPM_PATH = os.path.join(
    BASE_DIR, "pret/t5/sentencepiece.model"
)
MT5_DEFAULT_SPM_PATH = os.path.join(
    BASE_DIR, "pret/mt5/sentencepiece.model"
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
    "t5_small": os.path.join(BASE_DIR, "pret/t5/small"),
    "t5_base": os.path.join(BASE_DIR, "pret/t5/base"),
    "t5_large": os.path.join(BASE_DIR, "pret/t5/large"),
    "mt5_small": os.path.join(BASE_DIR, "pret/mt5/small"),
    "mt5_base": os.path.join(BASE_DIR, "pret/mt5/base"),
    "mt5_sa2": os.path.join(BASE_DIR, "pret/mt5/small_atomic_2"),
    "mt5_ba2": os.path.join(BASE_DIR, "pret/mt5/base_atomic_2"),
    "mt5_snaa": os.path.join(BASE_DIR, "pret/mt5/small_natural_all_atomic"),
    "mt5_large": os.path.join(BASE_DIR, "pret/mt5/large"),
}

@click.command()
@click.option(
    "--methods",
    "-mt",
    default="sup",
    type=str,
    help=""
)
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
    "--model_dir",
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
    default="t5_base",
    help="The path to or name of the pret model. ",
)
@click.option(
    "--n-steps",
    "-ns",
    type=int,
    default=0,
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
@click.option(
    "--split",
    "-sp",
    default="",
    type=str,
    help=""
)
@click.option(
    "--train_samples",
    "-n",
    default=0,
    type=int,
    help=""
)
@click.option(
    "--test_set",
    "-ts",
    default="test",
    type=str,
    help=""
)
@click.option(
    "--val_samples",
    "-vn",
    default=150,
    type=int,
    help=""
)
@click.option(
    "--test_samples",
    "-tn",
    default=0,
    type=int,
    help=""
)
@click.option(
    "--pred_pat",
    "-pp",
    default="",
    type=str,
    help=""
)
@click.option(
    "--do_score",
    "-ds",
    is_flag=True,
    help=""
)
def fine_tune(
    methods: str,
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
    ds_fname,
    split, train_samples, test_set, val_samples, test_samples,
    pred_pat, do_score
) -> None:
    if not do_train and not do_eval and not info and not do_score:
        print("Specify train or evaluation flag. --do_train or --do_eval --do_score")
        return
    if not split:
        if do_train: split = "train"
        if do_eval or do_score: split = "test"
    print("split:", split)
    if n_steps == 0 and do_train:
        n_steps = train_samples
    if not Path(model_dir).exists():
        Path(model_dir).mkdir(parents=True, exist_ok=True)
    if do_train and any(os.scandir(model_dir)):
        temp = glob.glob("model.ckpt*")
        if len(temp) == 0:
            print("The folder must contain a checkpoint or be empty!")
            return

    task_names = []
    input_prefix = input_postfix = target_prefix = target_postfix = ""
    input_col = "input_text"
    for method in methods.split(","):
        method = method.strip()
        task_name  = method + "_" + target_col
        task_names.append(task_name)
        print("Task:", task_name)
        paths={}
        paths["train"] = os.path.join(DATA_DIR, f"train.tsv")
        paths["val"] = os.path.join(DATA_DIR, f"val_all_rels.tsv")
        paths["test"] = os.path.join(DATA_DIR, f"test.tsv")
        num_samples = {"train": int(train_samples), "val":int(val_samples), "sample":0, "test":int(test_samples)}
        myds = {}
        for split_name, df_path in paths.items():
            if do_train and split_name != "train":
                continue
            if do_eval or do_score and split_name != split:
                continue
            
            split_df = pd.read_table(df_path)
            ds = MyDataset(split_df, split_name, method, 
                    num_samples = num_samples[split_name])
            myds[split_name]=ds

            _iter = iter(ds)
            pbar = tqdm(total=ds.num_records, position=0, leave=True) #,dynamic_ncols=True)
            ds_rows = []
            for batch_list in batched(list(_iter), 10):
                pbar.update(bs)
                for (query, tail, rel, qid, reps) in batch_list:
                    _data = {"event":query, "resp":tail, "rel":rel, "index":qid, "rep":reps}
                    ds_rows.append(_data)

            sel_df = pd.DataFrame(data = ds_rows,
                                  columns = ["event","rel", "resp"])
            temp_path = os.path.join(model_dir, "data") 
            mkdir(temp_path)
            temp_path += "/" + Path(df_path).name.replace(".tsv",f".{method}.tsv")
            print(temp_path)
            paths[split_name] = temp_path
            sel_df.to_csv(temp_path, sep="\t", index=False) 

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

        exp = Path(model_dir).stem
        print("Exp lable:", exp)
        num_lines =  {split: sum(1 for line in open(path)) for split,path in paths.items()}
        print(paths)
        sel_cols = ["event", "rel", "resp"]

        t5.data.TaskRegistry.add(
            task_name,
            # Specify the task type.
            core.MyTsvTask,
            #record_defaults = ["", "", ""],
            # Supply a function which returns a tf.data.Dataset.
            sel_cols=sel_cols,
            split_to_filepattern=paths,
            num_input_examples=num_lines,
            text_preprocessor=[preprocessors.atomic_pp()], 
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
    # vvvvvvvvvvvvvvvvvvvvv
    split_dir = model_dir
    if do_eval:
        print("============================ Validating =========================")
        summary_dir = "validation"
        split_dir = os.path.join(model_dir, summary_dir, test_set)
        #split_dir = summary_dir
        if True: #not os.path.isdir(split_dir):
            tf.io.gfile.makedirs(split_dir)
            print("=====================", split_dir, "=========================")
            ref_file = paths[test_set]
            shutil.copy(ref_file, split_dir + "/src_df.tsv")
            model.eval(
                mixture_or_task_name=mixture,
                summary_dir=split_dir,
                checkpoint_steps=eval_step,
                #eval_with_score=True,
                split=test_set,
            )
    if do_score:
        inps = glob.glob(f"{split_dir}/*{pred_pat}*predictions")
        if len(inps) == 0:
            print(f"A file with this pattern '{pred_pat}*predictions' wasn't found")
            return
        preds_file = inps[0]
        ds = myds[split]
        print("DS:", ds, "split:", split)
        extra = "_" + now
        model_id = Path(pm).stem
        m_name = model_id + "-" + method
        lang = "en2en"
        w_str = "unwrapped"
        f_str = "unfrozen"
        epochs_num = 1
        trial = 1
        experiment = Path(split_dir).stem
        exp_info = {"exp":experiment, "model":model_id, "lang": lang, 
                        "method":method, 
                        "wrap": w_str, 
                        "frozen":f_str, 
                        "steps":train_samples,
                        "epochs":epochs_num,
                        "trial":trial,
                        "date":extra}
        evaluate(ds, split_dir, exp_info, 
                test_samples, preds_file = preds_file)

if __name__ == "__main__":
    fine_tune()
