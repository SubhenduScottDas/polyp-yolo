"""Evaluate a trained YOLO model on the validation set using Ultralytics' val method.

Usage:
    python scripts/eval_val.py --weights runs/detect/train/weights/best.pt --data yolo_data.yaml --imgsz 640
"""
from ultralytics import YOLO
import argparse


def evaluate(weights, data, imgsz=640, batch=16):
    model = YOLO(weights)
    print(f"Running validation: model={weights}, data={data}")
    results = model.val(data=data, imgsz=imgsz, batch=batch)
    # results may be a mapping/dict or printed by ultralytics; print returned object
    print("Validation results:")
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Trained model weights (.pt)')
    parser.add_argument('--data', default='yolo_data.yaml', help='Path to data yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    evaluate(args.weights, args.data, args.imgsz, args.batch)
