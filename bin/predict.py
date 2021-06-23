import rainbow.mixtures
import t5
import csv
import tensorflow as tf
import click
import os, sys
import codecs
from pathlib import Path
import pandas as pd
import tensorflow_text  # Required to run exported model.

from rainbow.preparation.prepare_natural import *

@click.command()
# @click.argument("mixture", type=str)
@click.option(
    "--model-dir",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
)
@click.option(
    "--samples",
    default="/drive3/pouramini/data/atomic/natural_all_atomic_dev.tsv",
    type=str,
    help=""
)
@click.option(
    "--natural",
    default=True,
    type=bool,
    help=""
)
def predict(model_dir, samples, natural):
    df_list = [
    "/drive3/pouramini/data/atomic/atomic_dev.tsv",
    "/drive3/pouramini/data/atomic/natural_all_atomic_validation.tsv",
    "/drive3/pouramini/data/atomic/translate_validation.tsv",
    "/drive3/pouramini/data/atomic/translate_train.tsv",
    ]
    for i, item in enumerate(df_list):
        print(i, ":", item)
    sel = input("Select:") or 0
    samples = df_list[int(sel)]
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
        pred = predict_fn([question])[0].decode("utf-8")
        return pred

    data = pd.read_table(samples)
    if sel == "0":
        data = data[data["prefix"] == "xIntent"]

    fname = "/drive3/pouramini/data/atomic/sel_atomic_train.tsv"
    tsvfile = open(fname, 'a') 
    writer = csv.writer(tsvfile, delimiter='\t')
    writer.writerow(["prefix", "input_text", "target_text"])    
    q = "begin"
    debug = False
    use_sample = True
    natural = True
    while q != "end":
        print("==============================================================")
        sample = data.sample(n = 1)
        #print(sample)
        fact = sample.to_dict('records')[0]
        prefix = ""
        if sel == "0":
            prefix = fact["prefix"]
        head = fact["input_text"]
        target = fact["target_text"]
        if sel == "0" and natural:
            prompt = fact_to_prompt("atomic", fact)
            addition = prompt.replace(".","").replace(head,"")
        else:
            prompt = head + " " + prefix
            addition = prefix
        if debug: print("addition:", addition)
        show_prompt = prompt.replace("PersonX", "علی").replace("PersonY", "رضا")
        if use_sample: print(show_prompt)
        try:
            q = input("Question:")
            if q == "end":
                break
            if q == "s":
                continue
            if q == "sample":
                use_sample = not use_sample
                print("Sample set to " + str(use_sample))
            if sel == "0":
                q2 = q.replace("علی", "PersonX").replace("رضا", "PersonY")
            else:
                q2 = q.replace("PersonX", "علی").replace("PersonY", "رضا")
            if use_sample: 
                q2 += " " + addition
            else:
                q2 = q2.replace("می خواهد", "xIntent").replace("واکنش", "xReact")
            print(q2)
            try:
                if use_sample: print("Answer:", answer(prompt))
                #print("Answer (my):", answer(q))
                if sel == "0":
                    print("Answer q2:", answer(q2))
                if use_sample: print("Target:", target)
            except:
                print("Error!!!!!!!")
        except UnicodeDecodeError:
            print("Decoding input Error!")
        if use_sample:
            n = input("Sel:")
            if n and n != "end":
                writer.writerow([prefix, head, target])                
            print("was written!")

    tsvfile.close()

if __name__ == "__main__":
    predict()  # pylint: disable=no-value-for-parameter
