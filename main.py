import argparse

import cv2 as cv
import numpy as np
from cv2 import resize

from mask.inpaint import Inpainter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    parser.add_argument('--mae_checkpoint', default="checkpoints/mae_checkpoint.pth")
    args = parser.parse_args()

    if args.mode == "train":
        pass
    elif args.mode == "test_mae":
        test_mae(args)
    elif args.mode == "test_yolo":
        pass
    elif args.mode == "prod":
        pass
    else:
        assert False, "Unknown mode"


def test_mae(args):
    inpainter = Inpainter(args.mae_checkpoint)
    x = cv.imread("/home/initial/Downloads/download.png")
    x = resize(x, (224, 224))
    x = x.astype(np.float32) / 255.
    y = inpainter(cv.cvtColor(x, cv.COLOR_BGR2RGB), np.array([[0, 0], [40, 40]]))
    y = cv.cvtColor(y, cv.COLOR_RGB2BGR)
    cv.imshow("img", np.hstack([x, y]))
    cv.waitKey(0)


if __name__ == "__main__":
    main()
