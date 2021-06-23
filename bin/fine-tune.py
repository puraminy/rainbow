#! /usr/bin/env python

"""Fine-tune the model on the rainbow datasets."""

import logging
import os
import click
import t5
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from rainbow import utils, settings

import rainbow.mixtures

tf.get_logger().setLevel("ERROR")
# N.B. We must import rainbow.mixtures here so that the mixtures are registered
# and available for training.

logger = logging.getLogger(__name__)
BASE_DIR = "/drive2/"
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
@click.argument("mixture", type=str)
# @click.argument("results_dir", type=str)
@click.option(
    "--split",
    type=str,
    default="train",
    help="The split on which to train. Defaults to 'train'.",
)
@click.option(
    "--pm",
    type=str,
    default="t5_small",
    help="The path to or name of the pretrained model. Defaults to 3B.",
)
@click.option(
    "--n-steps",
    type=int,
    default=25000,
    help="The number of gradient updates. Defaults to 25,000.",
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
    type=int,
    default=15000,
    help=(
        "The number of steps to take before saving a checkpoint. Defaults to"
        " 5000."
    ),
)
@click.option(
    "--n-checkpoints-to-keep",
    type=int,
    default=3,
    help=(
        "The number of checkpoints to keep during fine-tuning. Defaults"
        " to 4."
    ),
)
# def cli()
#   pass
def fine_tune(
    mixture: str,
    #    results_dir: str,
    split: str,
    pm: str,
    n_steps: int,
    learning_rate: float,
    extra: str,
    bs: int,
    save_checkpoints_steps: int,
    n_checkpoints_to_keep: int,
) -> None:
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
        results_dir = os.path.join(settings.BASE_DIR, "models/rb", pm, mixture)

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
    print("=====================================================")
    pm = PRETRAINED_MODELS[pm]

    # Run fine-tuning.

    model = t5.models.MtfModel(
        tpu=None,
        model_dir=results_dir,
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

    model.finetune(
        mixture_or_task_name=mixture,
        pretrained_model_dir=pm,
        finetune_steps=n_steps,
        split=split,
    )
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


mixture = "t5_atomic_backward_mixture"  # @param {type:"string"}
if __name__ == "__main__":
    fine_tune()
    # fine_tune(mixture, "train", "t5_small", 25000, 0.003, 8, 1, 5000, 2)
