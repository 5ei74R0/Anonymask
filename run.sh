export PYTHONPATH=`pwd`/lib/mae:`pwd`/lib/yolo:$PYTHONPATH
python main.py --mode=train "$@"