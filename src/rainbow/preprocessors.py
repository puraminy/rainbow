"""Preprocessors for modeling rainbow."""

from typing import Callable, Optional, Sequence

import tensorflow as tf
from comet.train.common import *
def make_add_field_names_preprocessor(
    field_names: Sequence[str],
    field_indices: Optional[Sequence[int]] = None,
    direction=None,
) -> Callable:
    """Make a preprocessor to add field names to a dataset.

    Create a preprocessor that converts a dataset of lists of tensors
    into a dataset of dictionaries mapping strings to tensors.

    Parameters
    ----------
    field_names : Sequence[str], required
        A sequence of strings representing the field names for the new
        dictionaries.
    field_indices : Optional[Sequence[int]], optional (default=None)
        The indices corresponding to each field name in
        ``field_names``. If ``field_indices`` is ``None``, then each
        field name's corresponding index is assumed to be its index in
        the sequence.

    Returns
    -------
    Callable
        A function taking a ``tf.data.Dataset`` and returning a
        ``tf.data.Dataset``, that converts each sequence of tensors into an
        dictionary mapping the field names to the tensors at their
        corresponding indices.
    """
    if field_indices is None:
        field_indices = range(len(field_names))

    def add_field_names_preprocessor(
        dataset: tf.data.Dataset,
    ) -> tf.data.Dataset:
        return dataset.map(
            lambda *row: {
                field_name: row[field_index]
                for field_name, field_index in zip(field_names, field_indices)
            }
        )

    def my_preprocessor(ds):
        def to_inputs_and_targets(*row):
            inp = row[1]
            if direction:
                inp = direction + ":" + inp
                # inp = tf.strings.join([direction, row[1]])
            return {"inputs": inp, "targets": row[2]}

        return ds.map(
            to_inputs_and_targets,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    return my_preprocessor


def make_filter_preprocessor(predicate: Callable) -> Callable:
    """Make a preprocessor to filter examples from the dataset.

    Create a preprocessor that filters out any examples from the dataset
    for which the predicate returns ``False``.

    Parameters
    ----------
    predicate : Callable, required
        A function that takes an example and returns a boolean, ``True``
        if the example should remain in the dataset, ``False`` if it
        should not.

    Returns
    -------
    Callable
        A function taking a ``tf.data.Dataset`` and returning a
        ``tf.data.Dataset`` with all examples for which ``predicate``
        evaluates to ``False`` removed.
    """

    def filter_preprocessor(dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.filter(predicate)

    return filter_preprocessor


def remove_tags(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    # text = tf.strings.lower(text)
    # text = tf.strings.regex_replace(text,"<", "[")
    return text

def normalize_text(text):
    """Lowercase and remove quotes and tags from a TensorFlow string."""
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "'(.*)'", r"\1")
    text = tf.strings.regex_replace(text, "<[^>]+>", " ")
    return text

def tsv_atomic(mt):
    def rel_preprocessor(ds):
        def to_inputs_and_targets(ex):
            rel = ex["prefix"]
            event = ex["input_text"]
            resp = ex["target_text"]
            qtemp, anstemp, ex_qtemp, ex_anstemp, context = create_templates(mt)
            qtemp = tf.convert_to_tensor(qtemp, dtype=tf.string) 
            anstemp = tf.convert_to_tensor(anstemp, dtype=tf.string) 
            query = tf.strings.regex_replace(event, "\{event\}", event)
            query = tf.strings.regex_replace(resp, "\{ph\}", "<extra_id_0>")
            response = tf.strings.regex_replace(anstemp, "\{resp\}", resp)
            response = tf.strings.regex_replace(response, "\{ph\}", "<extra_id_0>")
            return {
                "inputs": query,
                "targets":response 
                }
        return ds.map(
            to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    return rel_preprocessor

def atomic_pp(input_col="event", target_col="resp"):
    def rel_preprocessor(ds):
        def to_inputs_and_targets(ex):
            return {
                "inputs": ex[input_col],
                "targets": ex[target_col]
            }
        return ds.map(
            to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    return rel_preprocessor

def tvs_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            "inputs": tf.strings.join(
                ["atomic: ", normalize_text(ex["input_text"])]
            ),
            "targets": normalize_text(ex["target_text"]),
        }

    return ds.map(
        to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

def unicode_encode(text):
    text = tf.strings.unicode_encode(text,"UTF-8")
    return text

def tvs_unicode_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            "inputs": tf.strings.join(
                ["atomic: ", unicode_encode(ex["input_text"])]
            ),
            "targets": unicode_encode(ex["target_text"]),
        }

    return ds.map(
        to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )


def rt_preprocessor(ds):
    def remove_tags(*row):
        r1 = row[1]
        r2 = normalize_text(row[2])  # normalize_text(row[2])
        tf.print(r2)
        return {"inputs": r1, "targets": r2}

    return ds.map(remove_tags, num_parallel_calls=tf.data.experimental.AUTOTUNE)
