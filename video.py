import cv2 as cv
from tqdm import tqdm
from main import Anonymask
from util.argparse import get_argparser, get_checkpoints


def main():
    parser = get_argparser()
    parser.add_argument("--video_path", default="data/video.mp4")
    args = parser.parse_args()
    checkpoints = get_checkpoints(args)
    anonymask = Anonymask(checkpoints, device=args.device)

    cap = cv.VideoCapture(args.video_path)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    frame_rate = int(cap.get(cv.CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    fmt = cv.VideoWriter_fourcc(*"MJPG")
    writer = cv.VideoWriter('./output/video.avi', fmt, frame_rate, size)

    ret = True
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = anonymask(frame)
        frame[frame < 0] = 0
        frame[frame > 255] = 255
        writer.write(frame)

    writer.release()
    cap.release()


if __name__ == "__main__":
    main()
