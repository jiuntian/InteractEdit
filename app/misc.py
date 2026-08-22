import base64
from collections import defaultdict
from datetime import datetime
import json
import logging
import os
import random
import re

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageChops
from diffusers.utils.torch_utils import is_compiled_module
from torch.utils.data import Dataset
from torchvision import transforms


def get_output_dir(save_name: str) -> str:
    # Use the current working directory as the root (no trailing slash).
    root = os.getcwd()

    # Generate a timestamp, e.g., "20230131-150210"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Combine everything into a single path
    output_dir = os.path.join(root, "outputs", f"{save_name}-{timestamp}")

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_details(image: Image.Image,
                 sbj, obj, action, detections, base_model, save_name,
                 initial_learning_rate, learning_rate, text_encoder_lr,
                 train_batch_size, rank, text_encoder_rank,
                 lambda_attention, phase1_train_steps, phase2_train_steps,
                 resolution,
                 adam_beta1, adam_beta2,
                 adam_weight_decay, adam_weight_decay_text_encoder,
                 optimizer, use_8bit_adam,
                 enable_xformers, seed, use_dora, detected_image, output_dir, clip_skip):

    train_info = {
        'subject': sbj,
        'object': obj,
        'action': action,
        'base_model': base_model,
        'save_name': save_name,
        'initial_learning_rate': initial_learning_rate,
        'learning_rate': learning_rate,
        'text_encoder_lr': text_encoder_lr,
        'train_batch_size': train_batch_size,
        'rank': rank,
        'text_encoder_rank': text_encoder_rank,
        'lambda_attention': lambda_attention,
        'phase1_train_steps': phase1_train_steps,
        'phase2_train_steps': phase2_train_steps,
        'resolution': resolution,
        'adam_beta1': adam_beta1,
        'adam_beta2': adam_beta2,
        'adam_weight_decay': adam_weight_decay,
        'adam_weight_decay_text_encoder': adam_weight_decay_text_encoder,
        'optimizer': optimizer,
        'use_8bit_adam': use_8bit_adam,
        'enable_xformers': enable_xformers,
        'seed': seed,
        'use_dora': use_dora,
        'clip_skip': clip_skip,
    }

    image.save(os.path.join(output_dir, "src.jpg"))
    json.dump(train_info, open(os.path.join(output_dir, "inversion.json"), "w"))
    
    base64_str = re.sub(r"^data:image/[^;]+;base64,", "", detected_image.plot)
    with open(os.path.join(output_dir, "detection.png"), "wb") as output_file:
        output_file.write(base64.b64decode(base64_str))


class SegmentationValueError(ValueError):
    def __init__(self, *args):
        super().__init__(*args)


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        image: Image.Image,
        detections,
        placeholder_tokens,
        subject_object_label,
        action,
        size=(512, 512),
        # repeats=1,
        center_crop=False,
        num_of_assets=3,
        flip_p=0.5,
    ):
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.custom_instance_prompts = None
        self.action = action
        logging.info(f"training with action: {self.action}")
        assert len(subject_object_label) == 2, 'must be only subject and object'

        # we load the training data
        self.placeholder_tokens = placeholder_tokens
        image = image.copy()
        image = image.resize(self.size)

        self.instance_masks = []

        self.instance_image = self.image_transforms(image)

        obj_to_detection = defaultdict(list)
        for detection in detections:
            obj_to_detection[detection.label.rstrip('.')].append(detection)
        obj_to_detection = {
            key: sorted(value, key=lambda x: x.score, reverse=True)
            for key, value in obj_to_detection.items()
        }
        if subject_object_label[0] == subject_object_label[1]:
            subject_mask = obj_to_detection[subject_object_label[0]][0].mask
            object_mask = obj_to_detection[subject_object_label[1]][1].mask
        else:
            subject_mask = obj_to_detection[subject_object_label[0]][0].mask
            object_mask = obj_to_detection[subject_object_label[1]][0].mask

        mask_subject = Image.fromarray(subject_mask)
        mask_object = Image.fromarray(object_mask)
        or_result = ImageChops.logical_or(mask_subject.convert(
            "1"), mask_object.convert("1"))  # interaction mask
        nor_result = ImageChops.invert(or_result)  # bg_mask

        for curr_mask in [mask_subject, mask_object, nor_result]:
            curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
            self.instance_masks.append(curr_mask)
        self.instance_masks = torch.cat(self.instance_masks)

        self.num_instance_images = 1  # len(self.instance_images)
        self._length = self.num_instance_images

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        num_of_tokens = random.randrange(1, len(self.placeholder_tokens) + 1)
        tokens_ids_to_use = random.sample(
            range(len(self.placeholder_tokens)), k=num_of_tokens
        )

        # ori
        tokens_to_use = [self.placeholder_tokens[tkn_i]
                         for tkn_i in tokens_ids_to_use]
        prompt = "a photo of " + " and ".join(tokens_to_use)

        example["instance_images"] = self.instance_image
        example["instance_masks"] = self.instance_masks[tokens_ids_to_use]
        example["token_ids"] = torch.tensor(tokens_ids_to_use)

        if random.random() > self.flip_p:
            example["instance_images"] = TF.hflip(example["instance_images"])
            example["instance_masks"] = TF.hflip(example["instance_masks"])

        example["instance_prompt"] = prompt

        return example
