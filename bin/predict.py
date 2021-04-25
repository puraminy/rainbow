import rainbow.mixtures
import t5
import tensorflow as tf
import click
import os
from pathlib import Path

import tensorflow_text  # Required to run exported model.


@click.command()
# @click.argument("mixture", type=str)
@click.option(
    "--model-dir",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
)
def predict(model_dir):
    def load_predict_fn(model_dir):
        if tf.executing_eagerly():
            print("Loading SavedModel in eager mode.")
            imported = tf.saved_model.load(model_dir, ["serve"])
            return lambda x: imported.signatures["serving_default"](
                tf.constant(x)
            )["outputs"].numpy()
        else:
            print("Loading SavedModel in tf 1.x graph mode.")
            tf.compat.v1.reset_default_graph()
            sess = tf.compat.v1.Session()
            meta_graph_def = tf.compat.v1.saved_model.load(
                sess, ["serve"], model_dir
            )
            signature_def = meta_graph_def.signature_def["serving_default"]
            return lambda x: sess.run(
                fetches=signature_def.outputs["outputs"].name,
                feed_dict={signature_def.inputs["input"].name: x},
            )

    predict_fn = load_predict_fn(model_dir)

    def answer(question):
        return predict_fn([question])[0].decode("utf-8")

        question_1 = "علی کتاب خرید علی این کار را برای"
        question_2 = "علی به رضا کمک کرد. رضا "
        question_3 = "علی همه را قانع کرد. در نتیجه دیگران"
        question_4 = (
            "PersonX convinces every ___. As a result, others feel"  # persuaded
        )

        questions = [question_1, question_2, question_3, question_4]

    for question in questions:
        print(answer(question))


if __name__ == "__main__":
    predict()  # pylint: disable=no-value-for-parameter
