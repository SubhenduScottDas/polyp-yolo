from ultralytics import YOLO
import cv2, os, argparse
from pathlib import Path

def draw_boxes(img, boxes, scores, classes, names):
    for (x1,y1,x2,y2), s, c in zip(boxes, scores, classes):
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        label = f"{names[int(c)]} {s:.2f}"
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return img

def main(weights, imgs_dir, out_dir, conf=0.25):
    model = YOLO(weights)
    names = model.names if hasattr(model, 'names') else {0:'polyp'}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img_paths = list(Path(imgs_dir).glob('*'))
    for p in img_paths:
        res = model.predict(source=str(p), conf=conf, save=False, imgsz=640, verbose=False)
        r = res[0]
        img = cv2.imread(str(p))
        if r.boxes is None or len(r.boxes) == 0:
            out = img
        else:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            out = draw_boxes(img, boxes, scores, classes, names)
        cv2.imwrite(os.path.join(out_dir, p.name), out)
    print("Done. Results saved to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--imgs', required=True)
    parser.add_argument('--out', default='results/predictions')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    main(args.weights, args.imgs, args.out, args.conf)