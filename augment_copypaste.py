"""
Copy-Paste-Augmentierung: klebt GTSRB-Crops aus dem Archiv auf Szenenbilder.

Ziel: Seltene Klassen auf mindestens TARGET_PER_CLASS Annotationen hochskalieren.
Die erzeugten Bilder landen in data/images/train/ und data/labels/train/
und werden beim nächsten Training automatisch mitgenutzt.

Aufruf:
    python augment_copypaste.py
"""

import random
from typing import Optional
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ARCHIVE      = Path("archive(2)/TrainIJCNN2013/TrainIJCNN2013")
IMG_TRAIN    = Path("data/images/train")
LBL_TRAIN    = Path("data/labels/train")
TARGET_PER_CLASS = 150  # Mindestanzahl Annotationen pro Klasse
MAX_SIGNS_PER_IMG = 4   # Zeichen pro synthetischem Bild
SEED         = 0
NUM_CLASSES  = 43


def load_crops(archive: Path) -> dict[int, list[Path]]:
    """Lädt alle GTSRB-Crop-Pfade geordnet nach Klasse."""
    crops: dict[int, list[Path]] = {}
    for cls in range(NUM_CLASSES):
        cls_dir = archive / f"{cls:02d}"
        if cls_dir.exists():
            imgs = list(cls_dir.glob("*.ppm"))
            if imgs:
                crops[cls] = imgs
    return crops


def count_existing(lbl_dir: Path) -> dict[int, int]:
    """Zählt wie viele Annotationen pro Klasse bereits vorhanden sind."""
    counts: dict[int, int] = {i: 0 for i in range(NUM_CLASSES)}
    for lbl_file in lbl_dir.glob("*.txt"):
        for line in lbl_file.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                counts[cls] = counts.get(cls, 0) + 1
    return counts


def load_backgrounds(img_dir: Path) -> list[Path]:
    return list(img_dir.glob("*.jpg"))


def paste_crop(background: Image.Image, crop: Image.Image,
               scale_range=(0.04, 0.12)) -> Optional[tuple[Image.Image, tuple[float, float, float, float]]]:
    """
    Klebt einen Crop zufällig skaliert auf den Hintergrund.
    Gibt (Bild, (x_center, y_center, w, h)) zurück (normalisiert).
    """
    bg_w, bg_h = background.size

    # Zufällige Größe relativ zur Bildbreite
    scale = random.uniform(*scale_range)
    sign_w = max(20, int(bg_w * scale))
    sign_h = max(20, int(sign_w * crop.height / max(crop.width, 1)))

    if sign_w >= bg_w or sign_h >= bg_h:
        return None

    crop_resized = crop.resize((sign_w, sign_h), Image.LANCZOS)

    # Zufällige Position (nicht zu nah am Rand)
    margin = 10
    x = random.randint(margin, bg_w - sign_w - margin)
    y = random.randint(margin, bg_h - sign_h - margin)

    # Leichte Helligkeitsvariation für Realismus
    brightness = random.uniform(0.75, 1.25)
    crop_resized = ImageEnhance.Brightness(crop_resized).enhance(brightness)

    result = background.copy()
    if crop_resized.mode == "RGBA":
        result.paste(crop_resized, (x, y), crop_resized)
    else:
        result.paste(crop_resized, (x, y))

    # YOLO-Koordinaten (normalisiert)
    xc = (x + sign_w / 2) / bg_w
    yc = (y + sign_h / 2) / bg_h
    bw = sign_w / bg_w
    bh = sign_h / bg_h

    return result, (xc, yc, bw, bh)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    crops     = load_crops(ARCHIVE)
    existing  = count_existing(LBL_TRAIN)
    bgs       = load_backgrounds(IMG_TRAIN)

    if not bgs:
        print("[FEHLER] Keine Hintergrundbilder in data/images/train/ — erst convert_dataset.py ausführen.")
        return

    print("Aktuelle Annotationen pro Klasse (nur Klassen < Target):")
    short = {cls: cnt for cls, cnt in existing.items() if cnt < TARGET_PER_CLASS and cls in crops}
    for cls, cnt in sorted(short.items(), key=lambda x: x[1]):
        print(f"  Klasse {cls:2d}: {cnt:3d} → Ziel {TARGET_PER_CLASS}")

    if not short:
        print(f"Alle Klassen haben bereits ≥ {TARGET_PER_CLASS} Annotationen.")
        return

    # Klassen gewichten: seltene Klassen öfter ziehen
    needs        = {cls: max(0, TARGET_PER_CLASS - cnt) for cls, cnt in short.items()}
    total_needed = sum(needs.values())
    print(f"\n{total_needed} zusätzliche Annotationen werden erzeugt ...")

    generated  = 0
    img_index  = len(list(IMG_TRAIN.glob("aug_*.jpg")))

    # Solange noch Klassen unter Target sind
    remaining = dict(needs)
    while any(v > 0 for v in remaining.values()):
        bg_path = random.choice(bgs)
        bg      = Image.open(bg_path).convert("RGB")

        labels = []
        n_signs = random.randint(1, MAX_SIGNS_PER_IMG)

        # Klassen nach Bedarf gewichten
        pool = [cls for cls, v in remaining.items() if v > 0 and cls in crops]
        if not pool:
            break

        weights = [remaining[cls] for cls in pool]
        total_w = sum(weights)
        probs   = [w / total_w for w in weights]

        chosen_classes = np.random.choice(pool, size=min(n_signs, len(pool)),
                                          replace=False, p=probs[:len(pool)])

        for cls in chosen_classes:
            crop_path = random.choice(crops[cls])
            crop_img  = Image.open(crop_path).convert("RGB")
            result = paste_crop(bg, crop_img)
            if result is None:
                continue
            bg, (xc, yc, bw, bh) = result
            labels.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            remaining[cls] = max(0, remaining[cls] - 1)
            generated += 1

        if not labels:
            continue

        # Speichern
        out_name = f"aug_{img_index:05d}"
        bg.save(IMG_TRAIN / f"{out_name}.jpg", "JPEG", quality=92)
        (LBL_TRAIN / f"{out_name}.txt").write_text("\n".join(labels))
        img_index += 1

        if generated % 50 == 0:
            still_short = sum(1 for v in remaining.values() if v > 0)
            print(f"  {generated} Annotationen erzeugt, {still_short} Klassen noch unter Target ...")

    print(f"\nFertig! {generated} zusätzliche Annotationen in {img_index} neuen Bildern.")
    print("Nächster Schritt: python train_improved.py")


if __name__ == "__main__":
    main()
