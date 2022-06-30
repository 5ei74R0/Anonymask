import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    args = parser.parse_args()

    if args.mode == "train":
        pass
    elif args.mode == "test_mae":
        pass
    elif args.mode == "test_yolo":
        pass
    elif args.mode == "prod":
        pass
    else:
        assert False, "Unknown mode"


if __name__ == "__main__":
    main()