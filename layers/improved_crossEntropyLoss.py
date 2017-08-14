
import caffe
import numpy as np
import os
import warnings

class SigmoidCrossEntropyLayer(caffe.Layer):

    def setup(self, bottom, top):
        
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        params = eval(self.param_str)
        self.count_file_path = params['count_file_path']
        self.positive_ratio = np.zeros((bottom[1].data.shape[1],), dtype=np.float32)
        if os.path.exists(self.count_file_path) == True:
	    print 'exist', self.count_file_path
            k=0
            with open(self.count_file_path, 'r') as f:
                for line in f:
                    self.positive_ratio[k] =np.float32(line)
                    k += 1
	else:
	    print 'no such ratio file'
	    print self.count_file_path
    def reshape(self, bottom, top):
    # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
    
    # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
    # loss output is scalar
        top[0].reshape(1)
    
    def forward(self, bottom, top):
         
        if any(self.positive_ratio)==False:
	    self.weights = np.ones_like(bottom[0].data, dtype=np.float32)
        else:
            self.weights = np.exp(bottom[1].data + (1 - bottom[1].data*2) * self.positive_ratio)
        warnings.filterwarnings('error')
	try:
	    probs = 1.0/(1.0+self.safe_exp(-bottom[0].data)) #expit(bottom[0].data)
        except Warning:
	    print bottom[0].data
	    print probs
	try:
	    
	    loss = - np.sum(self.weights*(bottom[1].data * self.safe_log(probs) + (1.0 - bottom[1].data) * self.safe_log(1.0 - probs))) / bottom[0].num
        except Warning:
            print probs
	    print bottom[1].data
           
 	
        # loss = - np.sum(self.weights*(bottom[1].data * np.log(probs) + (1.0 - bottom[1].data) * np.log(1.0 - probs))) / bottom[0].num
        
        self.diff[...] = probs - bottom[1].data
        top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1] == 1:
            print 'crash because propagate_down[1] is set.'
        if propagate_down[0] == 1:
            
            bottom[0].diff[...] = self.weights*self.diff/bottom[0].num
        
    def safe_log(self, x, minval=0.0000000001):
	return np.log(x.clip(min=minval))   
    def safe_exp(self, x, minval=-88.722835540, maxval=88.72283554):
	return np.exp(x.clip(min=minval, max=maxval)) 
