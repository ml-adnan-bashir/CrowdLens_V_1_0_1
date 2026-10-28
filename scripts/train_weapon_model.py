"""Example command for user-supplied and licensed weapon bounding-box data."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local YOLO weapon model on labeled data")
    parser.add_argument("--data", required=True, help="Dataset YAML, not included in this repository")
    parser.add_argument("--base", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    from ultralytics import YOLO

    YOLO(args.base).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()

