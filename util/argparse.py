import argparse
from util.data import Checkpoints


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="prod")
    parser.add_argument("--yolo_checkpoint", default="checkpoints/yolo_checkpoint.pt")
    parser.add_argument('--mae_checkpoint', default="checkpoints/mae_checkpoint.pt")
    parser.add_argument('--swinir_checkpoint', default="checkpoints/swinir_checkpoint.pt")
    parser.add_argument("--img_path", default="data/openlogo/test/images/logos32plus_001262.jpg")
    parser.add_argument("--device", default="cuda:0")
    return parser


def get_checkpoints(args: argparse.Namespace) -> Checkpoints:
    pths = Checkpoints(yolo=args.yolo_checkpoint, mae=args.mae_checkpoint, swinir=args.swinir_checkpoint)
    return pths
