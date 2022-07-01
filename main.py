import argparse

import cv2 as cv
import torch

from detection import Detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--yolo_checkpoint", default="checkpoints/yolo_checkpoint.pth")
    parser.add_argument("--test_img_path", default="data/openlogo/test/images/zara8.jpg")
    args = parser.parse_args()

    if args.mode == "train":
        pass
    elif args.mode == "test_mae":
        pass
    elif args.mode == "test_yolo":
        test_yolo(args)
    elif args.mode == "prod":
        pass
    else:
        assert False, "Unknown mode"


def test_yolo(args: argparse.Namespace):
    DEVICE = torch.device("cuda:0")
    det: Detector = Detector(args.yolo_checkpoint, device=DEVICE)
    img = cv.imread(args.test_img_path)
    img = cv.resize(img, (640, 640))
    input_imgs: torch.Tensor = (
        torch.tensor(img).permute(2, 0, 1).unsqueeze(0).expand(3, -1, -1, -1).to(DEVICE)
        / 255.0
    )
    print(input_imgs.shape)
    print(det.get_bboxes(input_imgs))


if __name__ == "__main__":
    main()
