import numpy as np
import scipy.misc
import caffe
def num2attributes(ests, labels, num_attr, dic):
    k=1
    for est,lab in zip(ests,labels):
        e_attr=[]
        l_attr=[]

        for i in range(num_attr):
            if est[i]==1:
                e_attr.append(dic[i])
            if lab[i]==1:
                l_attr.append(dic[i])
        print 'image ',k
        print 'estimated attributes: ', e_attr,'\n'
        print "label attributes: ", l_attr
   	print '\n\n'
        k+=1
def load_mean_binaryproto( binaryproto_name):
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open( binaryproto_name , 'rb' ).read()
        blob.ParseFromString(data)

        arr = np.array( caffe.io.blobproto_to_array(blob) )
        #print arr
        #print arr.shape
        return arr
def get_list_from_file(filepath):
    result_list = []

    with open(filepath, 'r') as reader:
        for line in reader.readlines():
            k = []
            k.append(line.split(' ')[0])
            k.append(map(int, list(line.split(' ')[1].split(','))[:-1]))
            result_list.append(k)

    return result_list



