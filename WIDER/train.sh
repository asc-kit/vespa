#!/usr/bin/env sh
#set -e
#TOOLS= ../lib/caffe-git/build/tools
#GLOG_logtostderr=0 GLOG_log_dir=/home/ywang/Multi-attributeLearning/Log/ \
#LOG=./Log/train-'try_adam_titan5-con7200'.log
LOG=../Log/train-'wider-20-4000'.log
/home/ywang/lib/caffe-git/build/tools/caffe train \
--solver=/home/ywang/Multi-attributeLearning/WIDER/solver.prototxt \
--weights=/home/ywang/Multi-attributeLearning/snapshot/8/rap-anglenprob-1inc-7000_iter_16000.caffemodel \
--gpu=0 2>&1 |tee >$LOG
#--weights=/home/ywang/Multi-attributeLearning/googlenet_bn_stepsize_6400_iter_1200000.caffemodel \
#--snapshot=/home/ywang/Multi-attributeLearning/snapshot/7/_iter_60000.solverstate \
#--snapshot=/home/ywang/Multi-attributeLearning/snapshot/13/3loss_iter_3000.solverstate \
#--snapshot=/home/ywang/Multi-attributeLearning/snapshot/2/peta32-recover_iter_8000.solverstate 
#--weights=/home/ywang/Multi-attributeLearning/snapshot/8/best/rap-angleprob_iter_10000.caffemodel \
#--snapshot=/home/ywang/Multi-attributeLearning/snapshot/2/peta-angleprob_iter_1000.solverstate \
#--weights=/home/ywang/Multi-attributeLearning/snapshot/8/best/rap-angleprob_iter_10000.caffemodel \
#--weights=/home/ywang/Multi-attributeLearning/PETA/angle-googlenet/rap_googlenet_after_surgery-angleprob.caffemodel \
#--snapshot=/home/ywang/Multi-attributeLearning/snapshot/2/peta-angleprobi-2test-ls-m3_iter_8000.solverstate \
#--weights=/home/ywang/Multi-attributeLearning/snapshot/8/rap-anglenprob-1inc-7000_iter_16000.caffemodel \
#--snapshot=/home/ywang/Multi-attributeLearning/snapshot/2/wider-20_iter_7000.solverstate \


