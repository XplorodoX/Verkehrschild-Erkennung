"""
Evaluierung & Einzelbild-Vorhersage des trainierten Modells.

Aufruf:
    python predict.py --weights best.pt --val
    python predict.py --weights best.pt --per-class
    python predict.py --weights best.pt --export
    python predict.py --weights best.pt --image testbild.jpg
    python predict.py --weights best.pt --video testvideo.mp4
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights",    required=True, help="Pfad zur best.pt")
    p.add_argument("--image",      default=None,  help="Einzelnes Testbild")
    p.add_argument("--video",      default=None,  help="Videodatei")
    p.add_argument("--conf",       type=float, default=0.5, help="Konfidenz-Schwelle")
    p.add_argument("--val",        action="store_true", help="Validierung auf Val-Set")
    p.add_argument("--per-class",  action="store_true", help="Per-Klassen-AP-Tabelle")
    p.add_argument("--export",     action="store_true", help="Modell nach ONNX exportieren")
    return p.parse_args()


def run_validation(model: YOLO) -> None:
    print("\n=== Validierung auf dem Val-Set ===")
    metrics = model.val(data="dataset.yaml")
    print(f"\n  mAP50      : {metrics.box.map50:.3f}")
    print(f"  mAP50-95   : {metrics.box.map:.3f}")
    print(f"  Precision  : {metrics.box.mp:.3f}")
    print(f"  Recall     : {metrics.box.mr:.3f}")
    print("\nKonfusionsmatrix und Kurven unter runs/detect/.../")


def run_per_class(model: YOLO) -> None:
    print("\n=== Per-Klassen-Analyse ===")
    metrics = model.val(data="dataset.yaml", verbose=False)
    names = model.names

    rows = []
    for i, ap in enumerate(metrics.box.ap50):
        rows.append((names[i], float(ap)))

    rows.sort(key=lambda x: x[1])

    print(f"\n{'Klasse':<35} {'AP50':>6}")
    print("-" * 43)
    for name, ap in rows:
        bar = "█" * int(ap * 20)
        flag = "  ⚠" if ap < 0.3 else ""
        print(f"{name:<35} {ap:>5.3f}  {bar}{flag}")

    weak = [name for name, ap in rows if ap < 0.3]
    if weak:
        print(f"\n{len(weak)} Klassen unter AP50=0.3 – mehr Augmentierung empfohlen:")
        for name in weak:
            print(f"  - {name}")


def run_export(weights: Path) -> None:
    print(f"\n=== ONNX-Export ===")
    model = YOLO(str(weights))
    out = model.export(format="onnx", imgsz=640, simplify=True)
    print(f"\nONNX-Modell gespeichert: {out}")
    print("Nutzung in webcam_demo.py:")
    print(f"  python webcam_demo.py --weights {Path(out).name}")


def run_image(model: YOLO, image_path: str, conf: float) -> None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"[FEHLER] Bild nicht gefunden: {image_path}")
        return

    results = model(img, conf=conf)
    annotated = results[0].plot()

    out_path = Path(image_path).stem + "_detected.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Ergebnis gespeichert: {out_path}")

    cv2.imshow("Erkennung", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(model: YOLO, video_path: str, conf: float) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[FEHLER] Video nicht gefunden: {video_path}")
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_path = Path(video_path).stem + "_detected.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf)
        annotated = results[0].plot()
        writer.write(annotated)
        cv2.imshow("Verkehrszeichen-Erkennung", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"{frame_count} Frames verarbeitet → {out_path}")


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[FEHLER] Gewichte nicht gefunden: {weights}")
        print("Erst trainieren: python train_improved.py")
        return

    if args.export:
        run_export(weights)
        return

    model = YOLO(str(weights))

    if args.per_class:
        run_per_class(model)
    elif args.val:
        run_validation(model)
    elif args.image:
        run_image(model, args.image, args.conf)
    elif args.video:
        run_video(model, args.video, args.conf)
    else:
        run_validation(model)


if __name__ == "__main__":
    main()
