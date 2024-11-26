import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
from pathlib import Path
import os
from tqdm import tqdm

# Add the class mapping
CLASS_MAPPING = {
    "bacterial_spot": 1,
    "early_blight": 2,
    "gray_spot": 3,
    "healthy": 4,
    "late_blight": 5,
    "leaf_miner": 6,
    "leaf_mold": 7,
    "magnesium_deficiency": 8,
    "mosaic_virus": 9,
    "nitrogen_deficiency": 10,
    "potassium_deficiency": 11,
    "powdery_mildew": 12,
    "septoria_leaf_spot": 13,
    "spider_mites_two-spotted_spider_mite": 14,
    "spotted_wilt_virus": 15,
    "target_spot": 16,
    "yellow_leaf_curl_virus": 17,
}

# Generate random and distinct visualization colors for each class
np.random.seed(42)
COLORS = {
    class_id: tuple(map(int, np.random.randint(0, 255, 3)))
    for class_id in CLASS_MAPPING.values()
}


def setup_sam():
    """
    Initialize the SAM model with the default checkpoint.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_TYPE = "vit_h"
    # You'll need to download this
    CHECKPOINT_PATH = "/data/ftisu/minhduc/autoseg/ckpts/sam_vit_h_4b8939.pth"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Increase this value if you're getting too many small masks
    )
    print(f"Init SAM model with device: {DEVICE}")
    return mask_generator


def get_best_mask(masks, image_shape):
    height, width = image_shape[:2]

    def edge_intersection_ratio(mask):
        top = mask[0, :].sum()
        bottom = mask[-1, :].sum()
        left = mask[:, 0].sum()
        right = mask[:, -1].sum()
        perimeter = 2 * (height + width)
        return (top + bottom + left + right) / perimeter

    def get_mask_coverage(mask, image_shape):
        mask_array = mask["segmentation"]
        return np.sum(mask_array) / (height * width)

    def is_same_mask(mask1, mask2):
        if mask1 is None or mask2 is None:
            return False
        return np.array_equal(mask1["segmentation"], mask2["segmentation"])

    max_edge_intersection_mask = None
    max_edge_intersection = float("-inf")
    for mask in masks:
        mask_array = mask["segmentation"]
        edge_intersection = edge_intersection_ratio(mask_array)
        if edge_intersection > max_edge_intersection:
            max_edge_intersection = edge_intersection
            max_edge_intersection_mask = mask

    sorted_masks = sorted(
        masks, key=lambda x: get_mask_coverage(x, image_shape), reverse=True
    )

    i = 0
    while i < len(sorted_masks):
        current_mask = sorted_masks[i]
        if not is_same_mask(current_mask, max_edge_intersection_mask):
            return current_mask
        i += 1

    return None


def process_image(
    image_path: Path,
    class_id: int,
    class_label: str,
    mask_generator: SamAutomaticMaskGenerator,
    output_dir: Path,
) -> None:
    """
    Process a single image and save the segmentation masks.
    """
    # Define mask and visualization output paths
    mask_filename = (
        output_dir
        / "masks"
        / f"class_{class_id}_{class_label}_{image_path.stem}_mask.png"
    )
    vis_filename = (
        output_dir
        / "visualizations"
        / f"class_{class_id}_{class_label}_{image_path.stem}_visualization.png"
    )

    # Skip processing if image has already been processed
    if os.path.exists(str(mask_filename)) or os.path.exists(str(vis_filename)):
        print(f"Skipping image {image_path} (already processed)")
        return

    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate masks
    masks = mask_generator.generate(image)
    if len(masks) == 0:
        raise Exception("No masks generated")

    best_mask = get_best_mask(masks, image.shape)
    if best_mask is None:
        raise Exception("No suitable mask found")

    # Save masks for each detected object
    if best_mask:
        multi_class_mask = (
            best_mask["segmentation"].astype(np.uint8) * (255 - class_id)
        ).astype(np.uint8)

        if np.sum(multi_class_mask) == 0:
            raise Exception("Generated mask is empty")

        # Save mask
        cv2.imwrite(str(mask_filename), multi_class_mask)

        # Save visualization
        visualization = image.copy()
        visualization[multi_class_mask > 0] = (
            visualization[multi_class_mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
        )
        cv2.imwrite(str(vis_filename), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    else:
        raise Exception("No masks detected")


def main():
    # Setup paths
    input_dir = Path("/data/ftisu/minhduc/datasets/unlabeled")
    output_dir = Path("/data/ftisu/minhduc/autoseg/output")

    # Create output dirs if not already exist
    os.makedirs(output_dir / "masks", exist_ok=True)
    os.makedirs(output_dir / "visualizations", exist_ok=True)

    # Initialize SAM
    mask_generator = setup_sam()

    # Filter types of image file extensions to procefss
    image_extensions = [".jpg", ".jpeg", ".png"]

    # Create error log file
    error_log = Path("errored_images.txt")

    class_dirs = [dir for dir in input_dir.iterdir() if dir.is_dir()]
    total_classes = len(class_dirs)

    for class_idx, class_dir in enumerate(class_dirs, 1):
        class_label = class_dir.stem
        class_id = CLASS_MAPPING[class_label]

        # Get all images for current class
        image_files = [
            f
            for f in (class_dir / "leaf").iterdir()
            if f.suffix.lower() in image_extensions
        ]
        total_images = len(image_files)

        # Initialize counters
        processed = 0
        errors = 0

        # Create progress bar
        pbar = tqdm(
            image_files,
            desc=f"[{class_idx}/{total_classes}] Processing {class_label}",
            total=total_images,
            unit="img",
        )

        for image_path in pbar:
            try:
                process_image(
                    image_path, class_id, class_label, mask_generator, output_dir
                )
                processed += 1
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                errors += 1
                # Log error
                with open(error_log, "a") as f:
                    f.write(f"{image_path}\t{str(e)}\n")
                    f.close()

            # Update progress bar
            pbar.set_postfix(
                {"processed": f"{processed}/{total_images}", "errors": errors}
            )

        # Class summary
        print(f"\nCompleted {class_label}:")
        print(
            f"Successfully processed: {
              max(0, processed-errors)}/{total_images}"
        )
        print(f"Errors: {errors}/{total_images}\n")


if __name__ == "__main__":
    main()
