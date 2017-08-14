LOG=../../Log/train-'rap-vespa'.log
/home/ywang/lib/caffe-git/build/tools/caffe train \
--solver=./solver.prototxt \
--weights=./googlenet_after_surgery-vespa.caffemodel \
--gpu=0 2>&1 |tee $LOG
