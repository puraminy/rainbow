"""Dataset preparation for ConceptNet."""

import codecs
import csv
import gzip
import hashlib
import logging
import os

import tensorflow as tf

from .. import settings
from . import preparer, utils


logger = logging.getLogger(__name__)


# main class


class ConceptNetPreparer(preparer.Preparer):
    """Prepare ConceptNet for text-to-text modeling."""

    CONCEPTNET = {
        "name": "conceptnet",
        "splits": {
            "train": {
                "name": "train",
                "size": 100000,
                "files": [
                    {
                        "url": "https://ttic.uchicago.edu/~kgimpel/comsense_resources/train100k.txt.gz",
                        "checksum": "a44a27fc0c6f5d4a426a935cdb6c02c101efba69f34346b0b627ee7c7d17a64e",
                        "file_name": "train100k.txt.gz",
                    },
                ],
            },
            "validation": {
                "name": "validation",
                "size": 1200,
                "files": [
                    {
                        "url": "https://ttic.uchicago.edu/~kgimpel/comsense_resources/dev1.txt.gz",
                        "checksum": "b101d1b27d1862190674470911326297d0de5df58a36a2acd6c0f17b289a90fc",
                        "file_name": "dev1.txt.gz",
                    },
                    {
                        "url": "https://ttic.uchicago.edu/~kgimpel/comsense_resources/dev2.txt.gz",
                        "checksum": "fd3c7a41dd280581de03c1b62197fd1af5c1f810846dfed048d71f2cebd13ded",
                        "file_name": "dev2.txt.gz",
                    },
                ],
            },
        },
    }
    """Configuration data for ConceptNet."""

    def prepare(self, src: str, dst: str, force_download: bool = False) -> None:
        """See ``rainbow.preparation.preparer.Preparer``."""
        # Create the directory for saving the source files.
        tf.io.gfile.makedirs(os.path.join(src, self.CONCEPTNET["name"]))

        # Create the directory for ConceptNet's prepared files.
        tf.io.gfile.makedirs(os.path.join(dst, self.CONCEPTNET["name"]))

        # Prepare the splits.
        for split in self.CONCEPTNET["splits"].values():
            rows_written = 0
            for file_ in split["files"]:
                src_path = os.path.join(
                    src, self.CONCEPTNET["name"], file_["file_name"]
                )

                # Copy the file to src_path from the URL.
                if not tf.io.gfile.exists(src_path) or force_download:
                    logger.info(
                        f"Downloading {self.CONCEPTNET['name']}'s"
                        f" {file_['file_name']} file from {file_['url']} to"
                        f" {src_path}."
                    )
                    utils.copy_url_to_gfile(file_["url"], src_path)

                with tf.io.gfile.GFile(src_path, "rb") as src_file:
                    # Verify the dataset file against its checksum.
                    sha256 = hashlib.sha256()
                    chunk = None
                    while chunk != b"":
                        # Read in 64KB at a time.
                        chunk = src_file.read(64 * 1024)
                        sha256.update(chunk)
                    checksum = sha256.hexdigest()
                    if checksum != file_["checksum"]:
                        raise IOError(
                            f"The {self.CONCEPTNET['name']} file,"
                            f" {file_['file_name']}, did not have the expected"
                            f" checksum. Try running with force_download=True"
                            f" to redownload all files, or consider updating"
                            f" the datasets' checksums."
                        )
                    # Return to the beginning of the file.
                    src_file.seek(0)

                    with gzip.open(src_file, "r") as src_gunzipped:
                        # Decode src_gunzipped.
                        src_gunzipped = codecs.getreader("utf-8")(src_gunzipped)

                        # Prepare and write out the split.
                        dst_path = os.path.join(
                            dst,
                            self.CONCEPTNET["name"],
                            settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                                split=split["name"],
                                dataset=self.CONCEPTNET["name"],
                            ),
                        )
                        with tf.io.gfile.GFile(dst_path, "a") as dst_file:
                            writer = csv.DictWriter(
                                dst_file,
                                fieldnames=["index", "inputs", "targets"],
                                dialect="unix",
                            )
                            writer.writeheader()

                            reader = csv.reader(src_gunzipped, delimiter="\t")

                            for i, row_in in enumerate(reader):
                                if float(row_in[3]) <= 0:
                                    # The last row being equal to zero
                                    # signifies that the triple is a negative
                                    # sample, so skip it.
                                    continue

                                inputs = (
                                    f"[{self.CONCEPTNET['name']}]:\n"
                                    f"<subject>{row_in[1]}</subject>\n"
                                    f"<relation>{row_in[0]}</relation>"
                                )
                                targets = row_in[2]

                                row_out = {
                                    "index": rows_written,
                                    "inputs": inputs,
                                    "targets": targets,
                                }
                                if i == 0:
                                    logger.info(
                                        f"\n\n"
                                        f"Example {row_out['index']} from"
                                        f" {self.CONCEPTNET['name']}'s"
                                        f" {split['name']} split:\n"
                                        f"inputs:\n"
                                        f"{row_out['inputs']}\n"
                                        f"targets:\n"
                                        f"{row_out['targets']}\n"
                                        f"\n"
                                    )

                                # Write to the CSV.
                                writer.writerow(row_out)
                                rows_written += 1

            if rows_written != split["size"]:
                logger.error(
                    f"Expected to write {split['size']} rows for the"
                    f" {split['name']} split of {self.CONCEPTNET['name']},"
                    f" instead {rows_written} were written."
                )

        logger.info("Finished preparing ConceptNet.")
