"""
YOLOv8 Fine-Tuning auf dem GTSDB-Datensatz.

Aufruf:
    python train.py                  # Standard (nano, 50 Epochen)
    python train.py --model yolov8s  # Größeres Modell
    python train.py --epochs 100     # Mehr Epochen
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="yolov8n",  help="yolov8n | yolov8s | yolov8m")
    p.add_argument("--epochs",  type=int, default=50)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--name",    default="verkehrszeichen_v1")
    p.add_argument("--patience",type=int, default=10, help="Early-Stopping-Geduld")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_yaml = Path("dataset.yaml")
    if not dataset_yaml.exists():
        raise FileNotFoundError("dataset.yaml nicht gefunden – bitte im Projektverzeichnis ausführen.")

    model_file = f"{args.model}.pt"
    print(f"Lade Modell: {model_file}")
    model = YOLO(model_file)

    print(f"\nStarte Training:")
    print(f"  Modell   : {args.model}")
    print(f"  Epochen  : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Bildgröße: {args.imgsz}")
    print(f"  Name     : {args.name}\n")

    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        # Augmentierungen
        mosaic=1.0,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,   # Verkehrszeichen nicht vertikal spiegeln
        fliplr=0.0,   # Verkehrszeichen nicht horizontal spiegeln
    )

    best_weights = Path(f"runs/detect/{args.name}/weights/best.pt")
    print(f"\nTraining abgeschlossen!")
    print(f"Beste Gewichte: {best_weights}")
    print(f"\nNächster Schritt: python predict.py --weights {best_weights}")


if __name__ == "__main__":
    main()
