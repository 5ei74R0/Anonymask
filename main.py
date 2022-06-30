import argparse

import cv2 as cv
import torch

from detection import Detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    args = parser.parse_args()

    if args.mode == "train":
        pass
    elif args.mode == "test_mae":
        pass
    elif args.mode == "test_yolo":
        DEVICE = torch.device("cuda:0")
        det: Detector = Detector(device=DEVICE)
        img_path = "/path/to/img"
        input_imgs: torch.Tensor = (
            torch.tensor(cv.imread(img_path)).permute(2, 0, 1).unsqueeze(0).expand(3, -1, -1, -1).to(DEVICE)
            / 255.0
        )
        print(det.get_bboxes(input_imgs))

    elif args.mode == "prod":
        pass
    else:
        assert False, "Unknown mode"


if __name__ == "__main__":
    main()
