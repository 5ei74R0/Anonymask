export PYTHONPATH=`pwd`/lib/mae:`pwd`/lib/yolo:`pwd`/lib/swinir:$PYTHONPATH
python main.py "$@"