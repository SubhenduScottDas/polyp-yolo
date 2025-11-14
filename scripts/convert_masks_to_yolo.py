# scripts/convert_masks_to_yolo.py
# flake8: noqa: E401 - Multiple imports acceptable for utility scripts
import os, glob, cv2  # noqa: E401
import numpy as np
from pathlib import Path

def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return x_min, y_min, x_max, y_max


def mask_to_bboxes_connected_components(mask):
    # find contours on binary mask and return list of bboxes (x_min,y_min,x_max,y_max)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        bboxes.append((x, y, x + w, y + h))
    return bboxes

def write_yolo_label(label_path, bboxes, img_w, img_h, class_id=0):
    # bboxes: list of (x_min,y_min,x_max,y_max)
    lines = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min
        # normalize
        x_center /= img_w
        y_center /= img_h
        w /= img_w
        h /= img_h
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    with open(label_path, 'w') as f:
        f.writelines(lines)


def convert_dataset(images_dir, masks_dir, labels_out_dir, image_exts=('.jpg','.png','.jpeg'), multi=False):
    os.makedirs(labels_out_dir, exist_ok=True)
    img_paths = []
    for ext in image_exts:
        img_paths.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    img_paths = sorted(img_paths)
    print(f"Found {len(img_paths)} images")

    missing_masks = 0
    no_mask_count = 0
    for img_path in img_paths:
        img_name = Path(img_path).stem
        mask_paths = [os.path.join(masks_dir, img_name + ext) for ext in ['.png', '.jpg', '_mask.png', '_mask.jpg']]
        mask_file = None
        for p in mask_paths:
            if os.path.exists(p):
                mask_file = p
                break
        if mask_file is None:
            missing_masks += 1
            open(os.path.join(labels_out_dir, img_name + '.txt'), 'w').close()
            continue

        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            missing_masks += 1
            open(os.path.join(labels_out_dir, img_name + '.txt'), 'w').close()
            continue

        _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if multi:
            bboxes = mask_to_bboxes_connected_components(bin_mask)
            if len(bboxes) == 0:
                no_mask_count += 1
                open(os.path.join(labels_out_dir, img_name + '.txt'), 'w').close()
                continue
        else:
            bbox = mask_to_bbox(bin_mask)
            if bbox is None:
                no_mask_count += 1
                open(os.path.join(labels_out_dir, img_name + '.txt'), 'w').close()
                continue
            bboxes = [bbox]

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        write_yolo_label(os.path.join(labels_out_dir, img_name + '.txt'), bboxes, w, h, class_id=0)

    print(f"Missing masks: {missing_masks}, No mask pixels: {no_mask_count}")
    print("Done.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True)
    p.add_argument('--masks', required=True)
    p.add_argument('--labels_out', required=True)
    p.add_argument('--multi', action='store_true', help='Export multiple bboxes per image using connected components')
    args = p.parse_args()
    convert_dataset(args.images, args.masks, args.labels_out, multi=args.multi)