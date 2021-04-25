#! /usr/bin/env python

"""Evaluate the model on the rainbow datasets."""
import os
import logging

import click
import t5
import tensorflow as tf

from rainbow import utils
from pathlib import Path
import rainbow.mixtures

# N.B. We must import rainbow.mixtures here so that the mixtures are registered
# and available for evaluation.

tf.get_logger().setLevel("ERROR")

logger = logging.getLogger(__name__)


@click.command()
# @click.argument("mixture", type=str)
@click.option(
    "--model-dir",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
)
@click.option(
    "--mixture", type=str, default="", help="The mixture name",
)
@click.option(
    "--summary-dir",
    type=str,
    default="",
    help="The path to or name of validation dir.",
)
@click.option(
    "--split",
    type=str,
    default="validation",
    help="The split on which to evaluate. Defaults to 'validation'.",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help=(
        "The batch size to use for prediction. For efficient prediction on the"
        " TPU, choose a multiple of either 8 or 128. Defaults to 64."
    ),
)
@click.option(
    "--checkpoint-steps",
    type=str,
    default="-1",
    help=(
        "The check points to use for prediction. For efficient prediction on the"
        " TPU, choose a multiple of either 8 or 128. Defaults to 64."
    ),
)
@click.option(
    "--model-parallelism",
    type=int,
    default=8,
    help="The degree of model parallelism to use. Defaults to 8.",
)
def evaluate(
    mixture: str,
    model_dir: str,
    summary_dir: str,
    split: str,
    batch_size: int,
    checkpoint_steps,
    model_parallelism: int,
) -> None:
    """Evaluate the model located at RESULTS_DIR on MIXTURE."""
    utils.configure_logging(clear=True)
    if not summary_dir:
        summary_dir = os.path.join(model_dir, "validation")
    if not mixture:
        mixture = Path(model_dir).stem

    tf.io.gfile.makedirs(summary_dir)
    if checkpoint_steps == "-1":
        checkpoint_steps = -1
    elif checkpoint_steps.isdigit():
        checkpoint_steps = int(checkpoint_steps)
    # Validate arguments.

    #    if not results_dir.startswith("gs://"):
    #        raise ValueError(f"RESULTS_DIR ({results_dir}) must be a GCS path.")
    #    elif not tf.io.gfile.exists(results_dir):
    #        raise IOError(f"RESULTS_DIR ({results_dir}) doesn't exist.")

    # Run evaluation.

    model = t5.models.MtfModel(
        tpu=None,
        model_dir=model_dir,
        model_parallelism=model_parallelism,
        batch_size=batch_size,
        sequence_length={"inputs": 128, "targets": 128},
        mesh_shape="model:1,batch:1",
        mesh_devices=["gpu:0"],
        learning_rate_schedule=None,
        save_checkpoints_steps=5000,
        keep_checkpoint_max=None,
        iterations_per_loop=100,
    )
    model.eval(
        mixture_or_task_name=mixture,
        summary_dir=summary_dir,
        checkpoint_steps=checkpoint_steps,
        split=split,
    )


if __name__ == "__main__":
    evaluate()  # pylint: disable=no-value-for-parameter
