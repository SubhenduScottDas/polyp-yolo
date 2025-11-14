import os, glob, shutil, random
from pathlib import Path

def train_val_split(images_dir, labels_dir, out_dir, val_ratio=0.2, seed=42):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    for split in ['images/train','images/val','labels/train','labels/val']:
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(images_dir, '*')))
    random.shuffle(img_files)
    n_val = int(len(img_files) * val_ratio)
    val_imgs = set([Path(p).name for p in img_files[:n_val]])
    for p in img_files:
        fname = Path(p).name
        src_label = os.path.join(labels_dir, Path(p).stem + '.txt')
        if fname in val_imgs:
            shutil.copy2(p, os.path.join(out_dir, 'images/val', fname))
            if os.path.exists(src_label):
                shutil.copy2(src_label, os.path.join(out_dir, 'labels/val', Path(p).stem + '.txt'))
        else:
            shutil.copy2(p, os.path.join(out_dir, 'images/train', fname))
            if os.path.exists(src_label):
                shutil.copy2(src_label, os.path.join(out_dir, 'labels/train', Path(p).stem + '.txt'))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--images')
    p.add_argument('--labels')
    p.add_argument('--out')
    p.add_argument('--val_ratio', type=float, default=0.2)
    args = p.parse_args()
    train_val_split(args.images, args.labels, args.out, args.val_ratio)