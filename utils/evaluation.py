#!/usr/bin/env python
#give the path of caffe
import sys
import os
import os.path as osp
sys.path.insert(0,'..')
caffe_root="/home/ywang/lib/caffe-ssd-36"
import time

import numpy as np
from PIL import Image
import scipy.misc
sys.path.insert(0, os.path.join(caffe_root,'/python'))
#sys.path.append("./layers") # the datalayers we will use are in this directory.
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

def safe_divide(x,y):
    if y == 0.0: return 0.0
    else:        return x/y


def load_mean(fname_bp):
    """
    Load mean.binaryproto file.
    """
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(fname_bp , 'rb' ).read()
    blob.ParseFromString(data)
    return np.array(caffe.io.blobproto_to_array(blob))


def compute_metrics(fname_testdata, fname_predictions, thr=0.5):
    """
    """
    # read stuff
    with open(fname_testdata, 'r') as f:
        attributes = np.asarray([[int(v) for v in l.strip().split()[1].split(',')[:-1]]
                                 for l in f.readlines()])
    predictions = np.load(fname_predictions)
    predictions = (predictions > thr).astype(np.int32)
    n_examples, n_attributes = attributes.shape

    TP_att = np.zeros(n_attributes)
    P_att  = np.zeros(n_attributes)
    TN_att = np.zeros(n_attributes)
    N_att  = np.zeros(n_attributes)
    precision = 0.0
    accuracy = 0.0
    recall = 0.0
    for gt, pred in zip(attributes, predictions):
        # example based
        TP = float(len([1 for a,p in zip(gt,pred) if a==1 and p==1]))
        either_pos = len([1 for a,p in zip(gt,pred) if a==1 or p==1])
        accuracy  += safe_divide(TP, either_pos)
        precision += safe_divide(TP, pred.sum())
        recall    += safe_divide(TP, gt.sum())

        # label based
        P_att  += [a==1          for a,p in zip(gt,pred)]
        TP_att += [a==1 and p==1 for a,p in zip(gt,pred)]
        N_att  += [a==0          for a,p in zip(gt,pred)]
        TN_att += [a==0 and p==0 for a,p in zip(gt,pred)]

    # aggregate metrics
    mA        = (1.0 / (2*n_attributes)) * (TP_att/P_att + TN_att/N_att).sum()
    accuracy  /= n_examples
    precision /= n_examples
    recall    /= n_examples
    F1        = (2*precision*recall) / (precision+recall)

    return (mA, accuracy, precision, recall, F1)



def run_evaluation(fname_model, fname_weights, fname_testdata, path_out, fname_mean, path_dataset, layer_pred="prob-attr", batchsize=16):
    tstart = time.time()

    # read testdata
    with open(fname_testdata, 'r') as f:
        testdata = [(l.strip().split()[0], l.strip().split()[1].split(',')) for l in f.readlines()]

    # generate attribute prediction file
    if not osp.exists(path_out): os.makedirs(path_out)
    fname_features = osp.join(path_out, "predictions.npy")
    #fname_angles = osp.join(path_out, "angles.npy") 
    if not osp.exists(fname_features):
        # load mean
        mean = load_mean(fname_mean)[0,:]
        mean = mean[:,:,::-1]           # wrong, but consistant for training and testing (channel is first dimension)
        mean = mean.transpose((1,2,0))

        # create net
        net = caffe.Net(fname_model, fname_weights, caffe.TEST)
        bs = batchsize
        net.blobs['data'].reshape(bs, 3, 227, 227)
        net.reshape()

        # predict
        features = []
        #angles = []
        for i in xrange(0,len(testdata),bs):
            batch = []
            for j in range(bs):
                idx = min(i+j, len(testdata)-1)
                fname_image = testdata[idx][0]

                img = np.asarray(Image.open(osp.join(path_dataset, fname_image)))
                img = scipy.misc.imresize(img, (256,256), interp='bicubic')
                img = np.subtract(img, mean)
                img = img[15:15+227, 15:15+227, :]
                img = img[:,:,::-1]
                img = img.transpose((2,0,1))
                batch.append(img)

            batch = np.asarray(batch)
            net.blobs['data'].data[...] = batch
            o = net.forward()

            for j in range(bs):
                if i+j >= len(testdata): break
                features.append(np.copy(net.blobs[layer_pred].data.squeeze()[j,:]))
                #angles.append(np.copy(net.blobs['prob-angle'].data.squeeze()[j,:]))
            if i > 0 and i % 100 == 0:
                print("computing features: {}/{}...".format(i, len(testdata)))

        # store
        np.save(fname_features, np.asarray(features))
        #print np.asarray(angles).shape
        #np.save(fname_angles, np.asarray(angles))
        with open(osp.join(path_out, "time_prediction.txt"), 'w') as f:
            f.write("{:.2f}\n".format(time.time()-tstart))

    # evaluate
    mA, accuracy, precision, recall, F1 = compute_metrics(fname_testdata, fname_features)
    with open(osp.join(path_out, "metrics.txt"), 'w') as f:
        f.write("mA:        {:.4f}\n".format(mA*100))
        f.write("Accuracy:  {:.4f}\n".format(accuracy*100))
        f.write("Precision: {:.4f}\n".format(precision*100))
        f.write("Recall:    {:.4f}\n".format(recall*100))
        f.write("F1:        {:.4f}\n".format(F1*100))





sw = 1

if sw == 1:
    fname_model    = "../PETA/deploy_peta.prototxt"
    fname_weights  = "../snapshots/vespa-peta_iter_12000.caffemodel"
    fname_testdata = "../generated/PETA_test_list.txt"
    path_dataset   = "/home/ywang/Downloads/"
    fname_mean     = "../generated/peta_mean.binaryproto"
    path_out       = "../eval_peta/"
    run_evaluation(fname_model, fname_weights, fname_testdata, path_out, fname_mean, path_dataset)

elif sw == 2:
    fname_model    = "../RAP/deploy_rap.prototxt"
    fname_weights  = "../snapshots/vespa-rap_iter_16000.caffemodel"
    fname_testdata = "../generated/RAP_test_list.txt"
    path_dataset   = "/cvhci/users/aschuman/datasets/RAP/"
    fname_mean     = "../generated/rap_mean.binaryproto"
    path_out       = "../eval_rap/"
    run_evaluation(fname_model, fname_weights, fname_testdata, path_out, fname_mean, path_dataset)
