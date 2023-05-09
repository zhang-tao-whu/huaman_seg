# Copyright (c) Facebook, Inc. and its affiliates.
import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

HUMAN_SEM_SEG_FULL_CATEGORIES = [
    {"name": "human", "id": 1, "trainId": 1},
]

def _get_human_seg_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in HUMAN_SEM_SEG_FULL_CATEGORIES]

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in HUMAN_SEM_SEG_FULL_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret

def load_human_sem_seg(root, json_file):
    json_path = os.path.join(root, json_file)
    with open(json_path, "r") as f:
        data = json.load(f)
    dataset = []
    for idx, item_ori in enumerate(data):
        item = dict()
        item["file_name"] = os.path.join(root, item_ori["image_path"])
        item["image_id"] = idx
        item["sem_seg_file_name"] = os.path.join(root, item_ori["mask_path"])
        item["width"] = item_ori["width"]
        item["height"] = item_ori["height"]
        dataset.append(item)
    return dataset

def register_human_seg(root):
    root = os.path.join(root, "human_seg")
    meta = _get_human_seg_meta()
    for name, json_file in [("train", "train_94w_raw_with_info.json"), ("val", "all_test_with_info.json")]:
        name = "human_sem_seg_{}".format(name)
        DatasetCatalog.register(
            name, lambda x=root, y=json_file: load_human_sem_seg(x, y)
        )
        MetadataCatalog.get(name).set(
            evaluator_type="sem_seg",
            mask_to_label=True, # 将mask映射为label
            ignore_label=None,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_human_seg(_root)


import numpy as np
from detectron2.data.transforms import Augmentation, Transform

class ValueTrans(Transform):
    """
    Transform mask value from 0-255 to 0-1.
    """

    def __init__(self):
        """
        Args:
        """
        super().__init__()

    def apply_image(self, img):
        return img
    
    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        if len(segmentation.shape) == 3:
            segmentation = segmentation[:, :, 0]
        segmentation = np.where(segmentation>220, 1, 0)
        return segmentation
