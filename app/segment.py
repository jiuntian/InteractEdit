from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

from PIL import Image
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import requests
import torch
from transformers import (
    AutoModelForMaskGeneration,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    pipeline,
)

from misc import SegmentationValueError


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))


def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(
        np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(
            image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> plt.Figure:
    # Define a color palette with two distinct colors
    # Here, we choose red and green for maximum distinctness
    # You can change this to red and blue if preferred
    color_palette = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 1.0)   # Cyan
    ]
    
    # Optionally define hatch patterns if you want different patterns per mask
    # We'll alternate between // and \\ for clarity
    hatch_patterns = ["//", "\\\\"]
    
    image_np = np.array(image) if isinstance(image, Image.Image) else image
    
    # Create a figure and axis to draw on
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)
    ax.axis("off")  # Hide the axis for a cleaner look
    
    # We'll collect legend entries in a list
    legend_patches = []
    
    # Iterate over detection results
    for i, detection in enumerate(detection_results):
        label = detection.label
        score = detection.score
        mask = detection.mask

        # If a mask is provided, overlay it
        if mask is not None:
            # Select color from the palette
            color = color_palette[i % len(color_palette)]
            hatch_style = hatch_patterns[i % len(hatch_patterns)]
            
            # # Create an RGB overlay for the mask
            # mask_colored = np.zeros_like(image_np, dtype=np.float32)
            # mask_bool = mask > 0.5  # Adjust threshold to your mask's format
            # mask_colored[mask_bool, 0] = color[0]
            # mask_colored[mask_bool, 1] = color[1]
            # mask_colored[mask_bool, 2] = color[2]
            
            # # Overlay the mask on the image with alpha transparency
            # ax.imshow(mask_colored, alpha=0.3)
            
            # Create a legend patch (so we don't clutter the image text)
            legend_label = f"{label.rstrip('.')} ({score:.2f})"
            patch = mpatches.Patch(color=color, label=legend_label)
            legend_patches.append(patch)
            
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_reshaped = cnt.reshape(-1, 2)
                ax.plot(cnt_reshaped[:,0], cnt_reshaped[:,1], color=color, linewidth=3)
                
                # Create a polygon for each contour
                # facecolor includes alpha so underlying image still shows a bit
                facecolor = (color[0], color[1], color[2], 0.3)
                
                polygon = mpatches.Polygon(
                    cnt_reshaped,
                    closed=True,
                    facecolor=facecolor,
                    edgecolor=color,
                    hatch=hatch_style,
                    linewidth=2
                )
                ax.add_patch(polygon)
                
            
    # Add the legend to the figure
    # Increase 'fontsize' here if you want bigger text
    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower right", fontsize=14, fancybox=True,)
    plt.tight_layout(pad=0)
    
    return fig
    

class GroundedSAM():
    def __init__(self,
                 detector="IDEA-Research/grounding-dino-tiny",
                 segmenter="facebook/sam-vit-base",
                 device: Optional[torch.device] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        #     detector).to(device)
        self.detector = pipeline(model=detector, task="zero-shot-object-detection", device=device)
        self.segmenter = AutoModelForMaskGeneration.from_pretrained(
            segmenter).to(device)
        self.seg_processor = AutoProcessor.from_pretrained(segmenter)
        self.device = device

    def detect(self,
               image: Image.Image,
               labels: List[str],
               threshold: float = 0.3,):
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """
        labels = [label if label.endswith(
            ".") else label+"." for label in labels]

        results = self.detector(
            image,  candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]

        return results

    def segment(self,
                image: Image.Image,
                detection_results: List[Dict[str, Any]],
                polygon_refinement: bool = False,):
        boxes = get_boxes(detection_results)
        inputs = self.seg_processor(
            images=image, input_boxes=boxes, return_tensors="pt").to(self.device)

        outputs = self.segmenter(**inputs)
        masks = self.seg_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        return detection_results

    def grounded_segmentation(
        self,
        image: Union[Image.Image, str],
        labels: List[str],
        threshold: float = 0.3,
        polygon_refinement: bool = True,
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        if isinstance(image, str):
            image = load_image(image)

        detections = self.detect(image, labels, threshold)
        self.verify_detections(detections, labels)
        detections = self.segment(image, detections, polygon_refinement)

        return image, detections
    
    def verify_detections(self, detections, labels):
        obj_to_detection = defaultdict(list)
        for detection in detections:
            obj_to_detection[detection.label.rstrip('.')].append(detection)
        obj_to_detection = {
            key: sorted(value, key=lambda x: x.score, reverse=True)
            for key, value in obj_to_detection.items()
        }
        
        if labels[0] == labels[1]:
            if labels[0].rstrip('.') not in obj_to_detection:
                raise SegmentationValueError(f"Subject/Object not detected. Found 0, but expect at least 2.")
            same_obj = obj_to_detection[labels[0].rstrip('.')]
            if len(same_obj) < 2:
                raise SegmentationValueError(f"Subject/Object not detected. Found {len(same_obj)}, but expect at least 2.")
        else:
            if labels[0].rstrip('.') not in obj_to_detection:
                raise SegmentationValueError(f"Subject not detected. Found 0, but expect at least 1.")
            if labels[1].rstrip('.') not in obj_to_detection:
                raise SegmentationValueError(f"Object not detected. Found 0, but expect at least 1.")
            sbj = obj_to_detection[labels[0].rstrip('.')]
            obj = obj_to_detection[labels[1].rstrip('.')]
            
            if len(sbj) < 1 or len(obj) < 1:
                raise SegmentationValueError(f"Subject/Object not detected. Found {len(sbj)} subject and {len(obj)} object, but expect at least 1 each.")
            
