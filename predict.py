"""
Evaluierung & Einzelbild-Vorhersage des trainierten Modells.

Aufruf:
    python predict.py --weights runs/detect/verkehrszeichen_v1/weights/best.pt
    python predict.py --weights best.pt --image testbild.jpg
    python predict.py --weights best.pt --video testvideo.mp4
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Pfad zur best.pt")
    p.add_argument("--image",   default=None,  help="Einzelnes Testbild")
    p.add_argument("--video",   default=None,  help="Videodatei")
    p.add_argument("--conf",    type=float, default=0.5, help="Konfidenz-Schwelle")
    p.add_argument("--val",     action="store_true",    help="Validierung auf Val-Set")
    return p.parse_args()


def run_validation(model: YOLO) -> None:
    print("\n=== Validierung auf dem Val-Set ===")
    metrics = model.val(data="dataset.yaml")
    print(f"\n  mAP50      : {metrics.box.map50:.3f}")
    print(f"  mAP50-95   : {metrics.box.map:.3f}")
    print(f"  Precision  : {metrics.box.mp:.3f}")
    print(f"  Recall     : {metrics.box.mr:.3f}")
    print("\nKonfusionsmatrix und Kurven unter runs/detect/.../")


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

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        print("Erst trainieren: python train.py")
        return

    model = YOLO(str(weights))

    if args.val:
        run_validation(model)
    elif args.image:
        run_image(model, args.image, args.conf)
    elif args.video:
        run_video(model, args.video, args.conf)
    else:
        run_validation(model)


if __name__ == "__main__":
    main()
