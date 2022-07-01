export PYTHONPATH=`pwd`/lib/mae:`pwd`/lib/yolo:$PYTHONPATH
python main.py --mode=test_mae "$@"