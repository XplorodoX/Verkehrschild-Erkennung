"""
Verbessertes Training: yolov8s, 100 Epochen, optimierte Augmentierung für Verkehrszeichen.

Wichtigste Unterschiede zu train.py:
  - Größeres Modell (yolov8s: ~11M Parameter statt 3M)
  - copy_paste=0.3  (YOLOs eingebaute Copy-Paste-Aug.)
  - Kein horizontales/vertikales Spiegeln (Schilder sind nicht symmetrisch)
  - Mehr Farb- und Beleuchtungsvariation (Wetter, Tageszeit)
  - Höherer patience-Wert damit Early Stopping nicht zu früh greift
  - Kleinere Lernrate für stabileres Fine-Tuning

Aufruf:
    python train_improved.py
    python train_improved.py --model yolov8m   # noch größer
    python train_improved.py --resume          # nach Unterbrechung fortsetzen
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="yolov8s",  help="yolov8s | yolov8m | yolov8l")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--batch",   type=int, default=8,
                   help="Kleiner Batch wegen großem Modell auf MPS")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--name",    default="verkehrszeichen_v2")
    p.add_argument("--patience",type=int, default=20)
    p.add_argument("--resume",  action="store_true",
                   help="Letztes Checkpoint fortsetzen")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_yaml = Path("dataset.yaml")
    if not dataset_yaml.exists():
        raise FileNotFoundError("dataset.yaml nicht gefunden.")

    if args.resume:
        last = Path(f"runs/detect/{args.name}/weights/last.pt")
        if not last.exists():
            raise FileNotFoundError(f"Kein Checkpoint gefunden: {last}")
        print(f"Setze Training fort von: {last}")
        model = YOLO(str(last))
    else:
        model_file = f"{args.model}.pt"
        print(f"Lade vortrainiertes Modell: {model_file}")
        model = YOLO(model_file)

    print(f"\nVerbessertes Training:")
    print(f"  Modell        : {args.model}")
    print(f"  Epochen       : {args.epochs}  (patience={args.patience})")
    print(f"  Batch         : {args.batch}")
    print(f"  Bildgröße     : {args.imgsz}")
    print(f"  Name          : {args.name}\n")

    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        resume=args.resume,

        # --- Lernrate ---
        lr0=0.001,          # niedrig für stabiles Fine-Tuning
        lrf=0.01,           # Endlernrate = lr0 * lrf
        warmup_epochs=3,

        # --- Augmentierung speziell für Schilder ---
        flipud=0.0,         # Schilder nicht auf den Kopf stellen
        fliplr=0.0,         # Richtungsschilder nicht spiegeln

        # Farbe & Beleuchtung (Wetter, Tageszeit, Verschmutzung)
        hsv_h=0.02,         # Farbton-Variation
        hsv_s=0.8,          # Sättigung
        hsv_v=0.5,          # Helligkeit

        # Geometrie
        degrees=12.0,       # leichte Rotation (Wind, schräge Kameras)
        translate=0.1,
        scale=0.6,          # Zoom-Variation (Schilder in verschiedenen Entfernungen)
        shear=3.0,          # Perspektive
        perspective=0.0003,

        # Mosaic & Copy-Paste
        mosaic=1.0,
        copy_paste=0.3,     # kopiert Objekte aus anderen Bildern → hilft bei seltenen Klassen

        # Sonstiges
        mixup=0.1,          # mischt zwei Bilder leicht
        close_mosaic=10,    # letzten 10 Epochen ohne Mosaic für saubere Konvergenz
    )

    best = Path(f"runs/detect/{args.name}/weights/best.pt")
    print(f"\nTraining abgeschlossen! Beste Gewichte: {best}")
    print(f"\nEvaluierung: python predict.py --weights {best} --val")
    print(f"Webcam-Demo: python webcam_demo.py --weights {best}")


if __name__ == "__main__":
    main()
