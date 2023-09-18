import numpy as np
from keras.utils import dataset_utils


image_paths, labels, class_names = dataset_utils.index_directory(
    "downloads/",
    "inferred",
    formats=(".bmp", ".gif", ".jpeg", ".jpg", ".png"),
    shuffle=True,
    seed=123,
    follow_links=False,
)
print(image_paths)
print(labels)
print(class_names)

def find_indices(list_to_check, item_to_find):
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]

vals = np.random.sample(range(len(class_names)), 2)
element = []
element.append(image_paths.index(vals[0]))
element.append(image_paths.index(vals[1]))