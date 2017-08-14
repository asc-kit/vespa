LOG=../../Log/train-'peta-vespa'.log
/home/ywang/lib/caffe-git/build/tools/caffe train \
--solver=./solver.prototxt \
--weights=./after_surgery.caffemodel \
--gpu=0 2>&1 |tee $LOG
