# src/main.py
import argparse
from pathlib import Path
import random
import cv2
import numpy as np
from utils.util import get_detections


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instance segmentation with Mask R-CNN (COCO) using OpenCV DNN."
    )
    ROOT = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "models" / "frozen_inference_graph.pb",
        help="Path to Mask R-CNN weights (.pb).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "models" / "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt",
        help="Path to Mask R-CNN config (.pbtxt).",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=ROOT / "models" / "class.names",
        help="Path to COCO class names (one per line).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "data" / "samples" / "cat_and_dog.png",
        help="Path to input image.",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.5,
        help="Detection confidence threshold (0-1). Default: 0.5",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open OpenCV windows to visualize results.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "build" / "outputs",
        help="Directory to save outputs.",
    )
    return parser.parse_args()


def load_classes(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {path}")
    names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return names


def main():
    args = parse_args()

    # Ensure output dir
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Validate files
    for p in [args.weights, args.config, args.classes, args.image]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # Load image
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")
    H, W, C = image.shape

    # Load model
    net = cv2.dnn.readNetFromTensorflow(model=str(args.weights), config=str(args.config))

    # Prepare blob (keep default scaling; Mask R-CNN expects raw BGR)
    blob = cv2.dnn.blobFromImage(image=image)

    # Forward pass
    boxes, masks = get_detections(net=net, blob=blob)

    # Safety checks
    if boxes is None or masks is None or len(masks) == 0:
        print("No object detected (empty outputs).")
        return

    # Load class names
    class_names = load_classes(args.classes)

    # Colors (index up to len(class_names)); add one extra for safety
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
              for _ in range(len(class_names) + 2)]

    # Accumulators
    mask_img = np.zeros((H, W, C), dtype=np.uint8)
    class_name_list = []

    # Iterate detections
    # boxes shape: [1,1,N,7]; for each i, bbox = boxes[0,0,i]
    N = boxes.shape[2]
    for i in range(N):
        bbox = boxes[0, 0, i]
        class_id = int(bbox[1])          # class index as float -> int
        score = float(bbox[2])           # confidence

        if score < args.thr:
            continue

        # Mask R-CNN returns masks[i] with shape [num_classes, 15, 15]
        mask_all = masks[i]
        if class_id < 0 or class_id >= mask_all.shape[0]:
            # unexpected class index; skip
            continue

        # Get bbox coordinates (normalized)
        x1 = int(bbox[3] * W)
        y1 = int(bbox[4] * H)
        x2 = int(bbox[5] * W)
        y2 = int(bbox[6] * H)

        # Clamp coordinates to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # Select mask for this class
        mask = mask_all[class_id].astype(np.float32)

        # Resize mask to bbox size
        mask = cv2.resize(mask, (x2 - x1, y2 - y1))
        _, mask_bin = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

        # Paint mask region
        color = colors[class_id]
        for c in range(3):
            mask_img[y1:y2, x1:x2, c] = np.maximum(
                mask_img[y1:y2, x1:x2, c],
                (mask_bin * color[c]).astype(np.uint8)
            )

        # Class name (COCO files are typically 1-based; keep safe guard)
        name_index = max(0, min(len(class_names) - 1, class_id - 1))
        class_name_str = class_names[name_index]
        class_name_list.append(f"{class_name_str} ({score:.2f})")

    # Blend overlay
    overlay = ((0.6 * mask_img) + (0.4 * image)).astype("uint8")

    # Put class labels
    for i, txt in enumerate(class_name_list):
        cv2.putText(overlay, txt, (10, 30 + 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save outputs
    out_mask = args.outdir / "mask.png"
    out_overlay = args.outdir / "overlay.png"
    cv2.imwrite(str(out_mask), mask_img)
    cv2.imwrite(str(out_overlay), overlay)
    print(f"Saved: {out_mask}")
    print(f"Saved: {out_overlay}")

    # Optional display
    if args.show:
        cv2.imshow("mask", mask_img)
        cv2.imshow("overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
