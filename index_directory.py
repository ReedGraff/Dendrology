import multiprocessing
import os
import random
import time
import warnings

import numpy as np
import tensorflow.compat.v2 as tf

# isort: off
from tensorflow.python.util.tf_export import keras_export

# Additional
from keras.utils.dataset_utils import index_subdirectory

def index_directory(
    directory,
    labels,
    formats,
    class_names=None,
    shuffle=True,
    seed=None,
    follow_links=False,
    as_list=True,
):
    """Make list of all files in the subdirs of `directory`, with their labels.
    Args:
      directory: The target directory (string).
      labels: Either "inferred"
          (labels are generated from the directory structure),
          None (no labels),
          or a list/tuple of integer labels of the same size as the number of
          valid files found in the directory. Labels should be sorted according
          to the alphanumeric order of the image file paths
          (obtained via `os.walk(directory)` in Python).
      formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").
      class_names: Only valid if "labels" is "inferred". This is the explicit
          list of class names (must match names of subdirectories). Used
          to control the order of the classes
          (otherwise alphanumerical order is used).
      shuffle: Whether to shuffle the data. Default: True.
          If set to False, sorts the data in alphanumeric order.
      seed: Optional random seed for shuffling.
      follow_links: Whether to visits subdirectories pointed to by symlinks.
    Returns:
      tuple (file_paths, labels, class_names).
        file_paths: list of file paths (strings).
        labels: list of matching integer labels (same length as file_paths)
        class_names: names of the classes corresponding to these labels, in
          order.
    """
    if labels is None:
        # in the no-label case, index from the parent directory down.
        subdirs = [""]
        class_names = subdirs
    else:
        subdirs = []
        for subdir in sorted(tf.io.gfile.listdir(directory)):
            if tf.io.gfile.isdir(tf.io.gfile.join(directory, subdir)):
                if subdir.endswith("/"):
                    subdir = subdir[:-1]
                subdirs.append(subdir)
        if not class_names:
            class_names = subdirs
        else:
            if set(class_names) != set(subdirs):
                raise ValueError(
                    "The `class_names` passed did not match the "
                    "names of the subdirectories of the target directory. "
                    f"Expected: {subdirs}, but received: {class_names}"
                )
    class_indices = dict(zip(class_names, range(len(class_names))))

    if as_list:

        # Build an index of the files
        # in the different class subfolders.
        pool = multiprocessing.pool.ThreadPool()
        results = []
        filenames = []

        for dirpath in (tf.io.gfile.join(directory, subdir) for subdir in subdirs):
            results.append(
                pool.apply_async(
                    index_subdirectory,
                    (dirpath, class_indices, follow_links, formats),
                )
            )
        labels_list = []
        for res in results:
            partial_filenames, partial_labels = res.get()
            labels_list.append(partial_labels)
            filenames += partial_filenames
        if labels not in ("inferred", None):
            if len(labels) != len(filenames):
                raise ValueError(
                    "Expected the lengths of `labels` to match the number "
                    "of files in the target directory. len(labels) is "
                    f"{len(labels)} while we found {len(filenames)} files "
                    f"in directory {directory}."
                )
        else:
            i = 0
            labels = np.zeros((len(filenames),), dtype="int32")
            for partial_labels in labels_list:
                labels[i : i + len(partial_labels)] = partial_labels
                i += len(partial_labels)

        if labels is None:
            print(f"Found {len(filenames)} files.")
        else:
            print(
                f"Found {len(filenames)} files belonging "
                f"to {len(class_names)} classes."
            )
        pool.close()
        pool.join()
        file_paths = [tf.io.gfile.join(directory, fname) for fname in filenames]

        if shuffle:
            # Shuffle globally to erase macro-structure
            if seed is None:
                seed = np.random.randint(1e6)
            rng = np.random.RandomState(seed)
            rng.shuffle(file_paths)
            rng = np.random.RandomState(seed)
            rng.shuffle(labels)
    else:
        
    return file_paths, labels, class_names

if __name__ == "__main__":
    image_paths, labels, class_names = index_directory(
        "downloads/",
        "inferred",
        formats=(".bmp", ".gif", ".jpeg", ".jpg", ".png"),
        follow_links=False,
    )

    print(image_paths)
    #     class_structure = dict(zip(class_names, [[] for _ in range(len(class_names))]))