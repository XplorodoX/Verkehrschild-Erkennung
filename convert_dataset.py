"""
Konvertiert GTSDB-Annotationen in das YOLO-Format.

Erwartet die vorhandene Verzeichnisstruktur:
    archive(2)/
    ├── TrainIJCNN2013/TrainIJCNN2013/
    │   ├── 00000.ppm ... 00599.ppm   (600 vollständige Straßenbilder)
    │   ├── gt.txt                    (851 Annotationen)
    │   └── 00/ ... 42/               (zugeschnittene Zeichen – wird ignoriert)
    └── TestIJCNN2013/TestIJCNN2013Download/
        └── 00000.ppm ... (301 Bilder ohne Annotationen – nur für Demo)

Aufruf:
    python convert_dataset.py
"""

import random
from pathlib import Path
from PIL import Image
import pandas as pd


ARCHIVE      = Path("archive(2)")
TRAIN_SRC    = ARCHIVE / "TrainIJCNN2013" / "TrainIJCNN2013"
TEST_SRC     = ARCHIVE / "TestIJCNN2013"  / "TestIJCNN2013Download"
GT_FILE      = TRAIN_SRC / "gt.txt"

IMG_TRAIN    = Path("data/images/train")
IMG_VAL      = Path("data/images/val")
LBL_TRAIN    = Path("data/labels/train")
LBL_VAL      = Path("data/labels/val")

VAL_SPLIT    = 0.2
SEED         = 42


def ppm_to_jpg(src: Path, dst: Path) -> tuple[int, int]:
    img = Image.open(src).convert("RGB")
    img.save(dst, "JPEG", quality=95)
    return img.size  # (width, height)


def yolo_bbox(x1: int, y1: int, x2: int, y2: int,
              w: int, h: int) -> tuple[float, float, float, float]:
    x_center = (x1 + x2) / 2 / w
    y_center = (y1 + y2) / 2 / h
    width    = (x2 - x1) / w
    height   = (y2 - y1) / h
    return x_center, y_center, width, height


def main() -> None:
    if not TRAIN_SRC.exists():
        print(f"[FEHLER] Ordner nicht gefunden: {TRAIN_SRC}")
        return
    if not GT_FILE.exists():
        print(f"[FEHLER] Annotationsdatei nicht gefunden: {GT_FILE}")
        return

    df = pd.read_csv(GT_FILE, sep=";",
                     names=["filename", "x1", "y1", "x2", "y2", "class_id"])
    print(f"Annotationen geladen: {len(df)} Einträge über {df['filename'].nunique()} Bilder")

    # Nur PPM-Dateien direkt im TRAIN_SRC-Ordner (keine Unterordner mit Crops)
    all_images = sorted(TRAIN_SRC.glob("*.ppm"))
    if not all_images:
        print(f"[FEHLER] Keine PPM-Bilder in {TRAIN_SRC}")
        return
    print(f"Trainingsbilder gefunden: {len(all_images)}")

    random.seed(SEED)
    shuffled = all_images[:]
    random.shuffle(shuffled)
    split_idx  = int(len(shuffled) * (1 - VAL_SPLIT))
    train_set  = set(img.name for img in shuffled[:split_idx])

    annotations = df.groupby("filename")
    converted   = 0

    for img_path in all_images:
        is_train    = img_path.name in train_set
        img_dst_dir = IMG_TRAIN if is_train else IMG_VAL
        lbl_dst_dir = LBL_TRAIN if is_train else LBL_VAL

        dst_img = img_dst_dir / (img_path.stem + ".jpg")
        dst_lbl = lbl_dst_dir / (img_path.stem + ".txt")

        w, h = ppm_to_jpg(img_path, dst_img)

        if img_path.name in annotations.groups:
            rows  = annotations.get_group(img_path.name)
            lines = []
            for _, row in rows.iterrows():
                xc, yc, bw, bh = yolo_bbox(
                    int(row["x1"]), int(row["y1"]),
                    int(row["x2"]), int(row["y2"]),
                    w, h
                )
                lines.append(f"{int(row['class_id'])} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            dst_lbl.write_text("\n".join(lines))
        else:
            dst_lbl.write_text("")  # Hintergrundbild ohne Zeichen

        converted += 1
        if converted % 100 == 0:
            print(f"  {converted}/{len(all_images)} konvertiert...")

    n_train = len(train_set)
    n_val   = len(all_images) - n_train
    print(f"\nKonvertierung abgeschlossen!")
    print(f"  Train: {n_train} Bilder  →  data/images/train/")
    print(f"  Val:   {n_val}   Bilder  →  data/images/val/")
    print(f"\nNächster Schritt: python train.py")


if __name__ == "__main__":
    main()
