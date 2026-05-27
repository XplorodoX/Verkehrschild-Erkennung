import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="verkehrszeichen_v1",
                   help="Name des Trainings-Runs (Unterordner in runs/detect/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ROOT = Path(__file__).resolve().parent
    run_dir = ROOT / "runs" / "detect" / args.run
    RESULTS_CSV = run_dir / "results.csv"
    OUTPUT_HTML = run_dir / "results_plot.html"

    if not RESULTS_CSV.exists():
        print(f"[FEHLER] Keine results.csv gefunden: {RESULTS_CSV}")
        print("Erst trainieren: python train.py  oder  python train_improved.py")
        return

    df = pd.read_csv(RESULTS_CSV)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Losses",
            "Precision / Recall",
            "mAP",
            "Learning Rate",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["train/box_loss"], name="train/box_loss", mode="lines+markers"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["val/box_loss"], name="val/box_loss", mode="lines+markers"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["train/cls_loss"], name="train/cls_loss", mode="lines+markers"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["val/cls_loss"], name="val/cls_loss", mode="lines+markers"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["train/dfl_loss"], name="train/dfl_loss", mode="lines+markers"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["val/dfl_loss"], name="val/dfl_loss", mode="lines+markers"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["metrics/precision(B)"], name="precision", mode="lines+markers"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["metrics/recall(B)"], name="recall", mode="lines+markers"),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["metrics/mAP50(B)"], name="mAP50", mode="lines+markers"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["metrics/mAP50-95(B)"], name="mAP50-95", mode="lines+markers"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["lr/pg0"], name="lr/pg0", mode="lines+markers"),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["lr/pg1"], name="lr/pg1", mode="lines+markers"),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["lr/pg2"], name="lr/pg2", mode="lines+markers"),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Training Verlauf - Verkehrszeichen Erkennung",
        template="plotly_white",
        height=900,
        width=1400,
        legend_title_text="Metrics",
    )
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="Learning rate", row=2, col=2)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
    print(f"Gespeichert: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()