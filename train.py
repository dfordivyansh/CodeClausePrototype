import argparse
import os
from ultralytics import YOLO
import torch

# Default training parameters
EPOCHS = 40                      # Train for longer
MOSAIC = 0.5                     # Reduced aggressive augmentation
MIXUP = 0.1                      # Less mixup improves object clarity
HSV_H = 0.0138                   # Subtle color jitter
HSV_S = 0.664
HSV_V = 0.464
OPTIMIZER = 'SGD'               # More stable convergence
MOMENTUM = 0.937
LR0 = 0.0032                    # Smaller learning rate
LRF = 0.12                      # Final LR factor
WEIGHT_DECAY = 0.00036          # Slightly reduced to prevent over-penalizing
WARMUP_EPOCHS = 3
PATIENCE = 20                   # More patience helps generalize
SINGLE_CLS = False
MODEL_NAME = "yolov8s.pt"
DATA_YAML = "yolo_params.yaml"
DEFAULT_DEVICE = "0"  # Use GPU (device 0) by default

if __name__ == '__main__':
    # Check for GPU
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Make sure PyTorch is installed with CUDA and your drivers are up to date.")
        print("👉 Install instructions: https://pytorch.org/get-started/locally/")
        exit(1)

    # Argument parser
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--mosaic', type=float, default=MOSAIC)
    parser.add_argument('--mixup', type=float, default=MIXUP)
    parser.add_argument('--hsv_h', type=float, default=HSV_H)
    parser.add_argument('--hsv_s', type=float, default=HSV_S)
    parser.add_argument('--hsv_v', type=float, default=HSV_V)
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER)
    parser.add_argument('--momentum', type=float, default=MOMENTUM)
    parser.add_argument('--lr0', type=float, default=LR0)
    parser.add_argument('--lrf', type=float, default=LRF)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--warmup_epochs', type=int, default=WARMUP_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS)
    parser.add_argument('--model', type=str, default=MODEL_NAME)
    parser.add_argument('--data', type=str, default=DATA_YAML)
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help="Device to train on, e.g. 'cpu', '0', or '0,1'")

    args = parser.parse_args()

    # Set working directory
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    # Load model
    model_path = os.path.join(this_dir, args.model)
    model = YOLO(model_path)

    # Train
    results = model.train(
        data=os.path.join(this_dir, args.data),
        epochs=args.epochs,
        device=args.device,  # Defaults to GPU: '0'
        single_cls=args.single_cls,
        mosaic=args.mosaic,
        mixup=args.mixup,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        optimizer=args.optimizer,
        momentum=args.momentum,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience
    )

    print("✅ Training complete.")
