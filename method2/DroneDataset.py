from collections import defaultdict
import json

import torch
import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import torchvision.transforms as T
import torch.nn.functional as F

class DroneDataset(Dataset):

    def __init__(self, image_names, image_annotations, image_dir):

        self.image_names = image_names
        self.image_annotations = image_annotations
        self.image_dir = image_dir

        # ImageNet standart normalization
        self.normalize_fn = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )


    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):

        image_name =  self.image_names[idx]
        image_annotation =  self.image_annotations[idx]

        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path)
        # cv2.imread reads turn image into BGR 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_height, image_width = image.shape[:2]

        # compress pixel values between 0-1
        image = image.astype(np.float32) / 255.0
        # [Height Weight, Channel] ---> [Channel, Height, Weight] 
        image = image.transpose(2, 0, 1)
        # Transform image into tensor
        image_tensor = torch.from_numpy(image) 
        # Normalize
        image_tensor = self.normalize_fn(image_tensor) 

        # calculate grid size, 
        # ResNet's CNN layers reduce the input image's dimension by factor 32
        grid_height = (image_height + 31) // 32
        grid_width = (image_width + 31) // 32

        # create tensor for grid_labels
        grid_labels = torch.zeros((5, grid_height, grid_width), dtype=torch.float32)

        # cell size
        cell_width = 32
        cell_height = 32

        for bbox in image_annotation:
            x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

            # labels show left upper corner, update them to show center
            x = x  + (width / 2)
            y = y + (height / 2)

            # find which cell corresponds to object center
            cell_col = int(x // cell_width)
            cell_row = int(y // cell_height)

            # normalize labels between 0 and 1
            normalized_x = (x % cell_width) / cell_width
            normalized_y = (y % cell_height) / cell_height
            normalized_height = height / cell_height
            normalized_width = width / cell_width

            grid_labels[0, cell_row, cell_col] = 1.0
            grid_labels[1, cell_row, cell_col] = normalized_x
            grid_labels[2, cell_row, cell_col] = normalized_y
            grid_labels[3, cell_row, cell_col] = normalized_width
            grid_labels[4, cell_row, cell_col] = normalized_height

        return image_tensor, grid_labels, image_name




def read_annotations(src_folder):
    """reads annotations and returns the annotations in the format, which DroneDataset takes input"""

    image_names = []
    image_annotations = []

    coco_path = os.path.join(src_folder, "_annotations.coco.json")

    with open(coco_path, "r", encoding="utf-8") as coco_json_file:
        coco_json = json.load(coco_json_file)

    images = coco_json.get("images", [])

    annotations = coco_json.get("annotations", [])

    # for O(1) lookup, if key not exists return empty list
    bbox_dict = defaultdict(list)
    for annotation in annotations:
        bbox_dict[annotation["image_id"]].append(annotation["bbox"])

    for image in images:
        image_names.append(image["file_name"])
        image_annotations.append(bbox_dict[image["id"]])

    return image_names, image_annotations, src_folder


def add_padding_fn(batch):
    """
        Since ResNet-18's CNN layers reduce the input image's dimension by factor 32,
        there may be some error or unexpected behaviour if input image dimensions (A x B)  A % 32 != 0 or B % 32 != 0.
        In addition if every image in a batch does not share the same dimensions it leads to an error.
        To avoid this problem, this method adds padding image for to make 
        every image's dimesions equal to each other and multiple of 32.

        Paddings will not affect to loss function hence affect the backpropogation.
    """

    images = [element[0] for element in batch]
    labels = [element[1] for element in batch]
    image_names = [element[2] for element in batch]

    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)

    # make sure dimensions are multiple of 32
    max_height = ((max_height + 31) // 32) * 32
    max_width = ((max_width + 31) // 32) * 32

    padded_images = []
    padded_labels = []

    for image, label in zip(images, labels):

        pad_h = max_height - image.shape[1]
        pad_w = max_width - image.shape[2]

        # add padding to edge below and right edge
        padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0)
        padded_images.append(padded_image)

        # update the labels with respect to new dimensions
        original_h, original_w = label.shape[1], label.shape[2]

        new_grid_results = torch.zeros(6, (max_height // 32), (max_width // 32), dtype=torch.float32)
        new_grid_results[:5, :original_h, :original_w] = label

        # 6. Channel: 
        # real image cell ---> 1.0 
        # padding cell ---> 0.0
        new_grid_results[5, :original_h, :original_w] = 1.0
        padded_labels.append(new_grid_results)

    return torch.stack(padded_images), torch.stack(padded_labels), image_names




