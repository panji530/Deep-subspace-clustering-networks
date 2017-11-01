import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
import matlab.engine
import scipy.io as sio

def next_batch(data, _index_in_epoch ,batch_size , _epochs_completed):
    _num_examples = data.shape[0]
    start = _index_in_epoch
    _index_in_epoch += batch_size
    if _index_in_epoch > _num_examples:
        # Finished epoch
        _epochs_completed += 1
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        data = data[perm]
        #label = label[perm]
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples
    end = _index_in_epoch
    return data[start:end], _index_in_epoch, _epochs_completed

class ConvAE(object):
    def __init__(self, n_input, kernel_size,n_hidden, learning_rate = 1e-3, batch_size = 256,\
        reg = None, denoise = False ,model_path = None,restore_path = None, logs_path = '/home/pan/workspace-eclipse/deep-subspace-clustering/models_face'):    
    #n_hidden is a arrary contains the number of neurals on every layer
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.iter = 0
        weights = self._initialize_weights()
        
        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])        

        if denoise == False:
            x_input = self.x
            latent, shape = self.encoder(x_input, weights)

        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                               mean = 0,
                                               stddev = 0.2,
                                               dtype=tf.float32))

            latent,shape = self.encoder(x_input, weights)
        self.z = tf.reshape(latent,[batch_size, -1])
        self.x_r = self.decoder(latent, weights, shape)
        self.saver = tf.train.Saver()
        # cost for reconstruction
        # l_2 loss 
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r, self.x), 2.0))   # choose crossentropy or l2 loss
        tf.summary.scalar("l2_loss", self.cost)          
        
        self.merged_summary_op = tf.summary.merge_all()        
        
        self.loss = self.cost

        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss) #GradientDescentOptimizer #AdamOptimizer
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))

        all_weights['enc_w1'] = tf.get_variable("enc_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0],self.n_hidden[1]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))

        all_weights['enc_w2'] = tf.get_variable("enc_w2", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1],self.n_hidden[2]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['enc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype = tf.float32))        

        all_weights['dec_w0'] = tf.get_variable("dec_w0", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1],self.n_hidden[2]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['dec_b0'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))

        all_weights['dec_w1'] = tf.get_variable("dec_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0],self.n_hidden[1]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['dec_b1'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))

        all_weights['dec_w2'] = tf.get_variable("dec_w2", shape=[self.kernel_size[0], self.kernel_size[0],1, self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
        all_weights['dec_b2'] = tf.Variable(tf.zeros([1], dtype = tf.float32))
        return all_weights


    # Building the encoder
    def encoder(self,x, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(x.get_shape().as_list())
        layer1 = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
        layer1 = tf.nn.relu(layer1)
        shapes.append(layer1.get_shape().as_list())
        layer2 = tf.nn.bias_add(tf.nn.conv2d(layer1, weights['enc_w1'], strides=[1,2,2,1],padding='SAME'),weights['enc_b1'])
        layer2 = tf.nn.relu(layer2)
        shapes.append(layer2.get_shape().as_list())
        layer3 = tf.nn.bias_add(tf.nn.conv2d(layer2, weights['enc_w2'], strides=[1,2,2,1],padding='SAME'),weights['enc_b2'])
        layer3 = tf.nn.relu(layer3)
        return  layer3, shapes
    # Building the decoder
    def decoder(self,z, weights, shapes):
        # Encoder Hidden layer with relu activation #1
        shape_de1 = shapes[2]
        layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack([tf.shape(self.x)[0],shape_de1[1],shape_de1[2],shape_de1[3]]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b0'])
        layer1 = tf.nn.relu(layer1)
        shape_de2 = shapes[1]
        layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights['dec_w1'], tf.stack([tf.shape(self.x)[0],shape_de2[1],shape_de2[2],shape_de2[3]]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b1'])
        layer2 = tf.nn.relu(layer2)
        shape_de3= shapes[0]
        layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights['dec_w2'], tf.stack([tf.shape(self.x)[0],shape_de3[1],shape_de3[2],shape_de3[3]]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b2'])
        layer3 = tf.nn.relu(layer3)
        return layer3

    def partial_fit(self, X): 
        cost, summary, _ = self.sess.run((self.cost, self.merged_summary_op, self.optimizer), feed_dict = {self.x: X})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost 

    def reconstruct(self,X):
        return self.sess.run(self.x_r, feed_dict = {self.x:X})

    def transform(self, X):
        return self.sess.run(self.z, feed_dict = {self.x:X})

    def save_model(self):
        save_path = self.saver.save(self.sess,self.model_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")

def ae_feature_clustering(CAE, X):
    CAE.restore()
    
    #eng = matlab.engine.start_matlab()
    #eng.addpath(r'/home/pan/workspace-eclipse/deep-subspace-clustering/SSC_ADMM_v1.1',nargout=0)
    #eng.addpath(r'/home/pan/workspace-eclipse/deep-subspace-clustering/EDSC_release',nargout=0)
    
    Z = CAE.transform(X)
    
    sio.savemat('AE_YaleB.mat', dict(Z = Z) )
    
    return

def train_face(Img, CAE, n_input, batch_size):    
    it = 0
    display_step = 300
    save_step = 900
    _index_in_epoch = 0
    _epochs= 0

    # CAE.restore()
    # train the network
    while True:
        batch_x,  _index_in_epoch, _epochs =  next_batch(Img, _index_in_epoch , batch_size , _epochs)
        batch_x = np.reshape(batch_x,[batch_size,n_input[0],n_input[1],1])
        cost = CAE.partial_fit(batch_x)
        it = it +1
        avg_cost = cost/(batch_size)
        if it % display_step == 0:
            print ("epoch: %.1d" % _epochs)
            print  ("cost: %.8f" % avg_cost)
        if it % save_step == 0:
            CAE.save_model()
    return

def test_face(Img, CAE, n_input):
    
    batch_x_test = Img[200:300,:]
    batch_x_test= np.reshape(batch_x_test,[100,n_input[0],n_input[1],1])
    CAE.restore()
    x_re = CAE.reconstruct(batch_x_test)

    plt.figure(figsize=(8,12))
    for i in range(5):
        plt.subplot(5,2,2*i+1)
        plt.imshow(batch_x_test[i,:,:,0], vmin=0, vmax=255, cmap="gray") #
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(x_re[i,:,:,0], vmin=0, vmax=255, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
        plt.tight_layout()
    plt.show()
    return

if __name__ == '__main__':
    
    data = sio.loadmat('/home/pan/workspace-eclipse/deep-subspace-clustering/face_datasets/YaleBCrop025.mat')
    img = data['Y']
    I = []
    Label = []
    for i in range(img.shape[2]):
        for j in range(img.shape[1]):
            temp = np.reshape(img[:,j,i],[42,48])
            Label.append(i)
            I.append(temp)
    I = np.array(I)
    Label = np.array(Label)
    Img = np.transpose(I,[0,2,1])
    Img = np.expand_dims(Img,3)   
    
    
    n_input = [48,42]
    kernel_size = [5,3,3]
    n_hidden = [10,20,30]
    batch_size = Img.shape[0]    
    lr = 1.0e-3 # learning rate
    model_path = '/home/pan/workspace-eclipse/deep-subspace-clustering/models_face/model-102030-48x42-yaleb.ckpt'
    CAE = ConvAE(n_input = n_input, n_hidden = n_hidden, learning_rate = lr, kernel_size = kernel_size, 
                 batch_size = batch_size, model_path = model_path, restore_path = model_path)
    #test_face(Img, CAE, n_input)
    train_face(Img, CAE, n_input, batch_size)
    #X = np.reshape(Img, [Img.shape[0],n_input[0],n_input[1],1])
    #ae_feature_clustering(CAE, X)
    
    
    
    
    
    
    
    
    
    
    
    
