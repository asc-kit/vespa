import sys
sys.pathh.append("..")
import caffe

import numpy as np
from PIL import Image
from random import shuffle
from utils.blob import load_mean_binaryproto, get_list_from_file
import scipy.misc

from utils.tools import SimpleTransformer

class MultiLabelDataLayerSync(caffe.Layer):
    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        params = eval(self.param_str)

        self.batch_size = params['batch_size']
	self.label_size = params['label_size']
        self.batch_loader=BatchLoader(params)
        
        # === reshape tops ===
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        
	top[1].reshape(self.batch_size, self.label_size)
	
	
    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, multilabel = self.batch_loader.load_next_imlabelpair()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = multilabel
            
        
    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass
    
    
class BatchLoader(object):
    def __init__(self,params):
        self.batch_size = params['batch_size']
        self.dataset_path = params['dataset_path']
        self.im_shape = params['im_shape']
        self.split = params['split']
	self.mean_file = params['mean_file']

        list_path = params['img_label_list_file']
        #print list_path
        self.img_veclabel_list = get_list_from_file(list_path)
        # print self.img_veclabel_list
        
        
        self.cur = 0  # current image
  
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()
        #self.transformer.set_mean([104.0, 117.0, 123.0])
        self.transformer.set_mean([0.0, 0.0, 0.0])
        #self.transformer.set_scale = 1.0/128.0
	self.mean = load_mean_binaryproto(self.mean_file)[0,:]

	# wrong but consistant with training and testing
	self.mean = self.mean[:,:,::-1]#change to RGB
        self.mean = self.mean.transpose((1,2,0))#c w h to w h c

	
    def load_next_imlabelpair(self):
        
        if self.cur == len(self.img_veclabel_list):
            self.cur = 0
            shuffle(self.img_veclabel_list)
        imName = self.img_veclabel_list[self.cur][0]
        imName = self.dataset_path + imName
        
        im = np.asarray(Image.open(imName))
        if self.split == 'val':
            im = scipy.misc.imresize(im, (256,256), interp='bicubic')
	    im = np.subtract(im, self.mean)
            #randomly cropped
            crop_x = 15#np.random.randint(0, pad + 1)
            crop_y = 15#np.random.randint(0, pad + 1)
            im = im[crop_y:crop_y + self.im_shape[0], crop_x:crop_x + self.im_shape[1], :]

        else:
            im = scipy.misc.imresize(im, (256,256), interp='bicubic')
            im = np.subtract(im, self.mean)  
            #randomly cropped
            pad = 29
            crop_x = np.random.randint(0, pad + 1)
            crop_y = np.random.randint(0, pad + 1)
            im = im[crop_y:crop_y + self.im_shape[0], crop_x:crop_x + self.im_shape[1], :]
            #randomly mirrored
            if np.random.choice([True, False]):
                im = np.fliplr(im)
        
        multilabel = self.img_veclabel_list[self.cur][1]
        self.cur += 1
        
        
                        
        return self.transformer.preprocess(im), multilabel
  
    
    



        


 
