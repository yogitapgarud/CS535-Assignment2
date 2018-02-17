"""
Yogita Garud
"""


from __future__ import division
from __future__ import print_function

import sys

import _pickle as cPickle
import numpy as np

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
        self.W = W

    def forward(self, x):
        w_transpose = np.transpose(self.W)
        l = np.dot(x, w_transpose())

        return l

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
        self.W = self.W - np.dot(learning_rate, grad_output)
	# DEFINE backward function
# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
	# DEFINE forward function
        x[x<0] = 0
        return x 

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
        self.w = w
    # DEFINE backward function
# ADD other operations in ReLU if needed

    def gradRelu(x):
        x[x>0] = 1
        return x

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):

	def forward(self, x):
            
            return sigmoid(x)

	def backward(
            self, 
            grad_output, 
            learning_rate=0.0,
            momentum=0.0,
            l2_penalty=0.0
        ):

            self.w = u * mlp.momentumw1 - learning_rate * dw1                                             
            #mlp.momentumw2 = u * mlp.momentumw2 - learning_rate * dw2

            mlp.W1 = mlp.W1 + mlp.momentumw1
            mlp.W2 = mlp.W2 + mlp.momentumw2

# ADD other operations and data entries in SigmoidCrossEntropy if needed

	def sigmoid(z):
            return 1/(1 + np.exp(-z)) 

# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.W1 = np.random.rand(hidden_units, input_dims)
        self.W2 = np.random.rand(1, hidden_units)
        self.momentumw1 = np.zeros(hidden_units, input_dims)
        self.momentumw2 = np.zeros(hidden_units, 1)

    def train(
        self, 
        x_batch, 
        y_batch, 
        learning_rate, 
        momentum,
        l2_penalty,
    ):
	# INSERT CODE for training the network
        lt1 = LinearTransform(self.W1)
        lt2 = LinearTransform(self.W2)
        relu = ReLU()
        op = SigmoidCrossEntropy(self.W2)

        z1 = lt1.forward(x_batch)
        a1 = relu.forward(z1)
        z2 = lt2.forward(a1)
        a2 = op.forward(z2)
        
        da2dz2 = np.subtract(y_batch, a2)

        da1 = self.W2
        dz1 = relu.gradrelu(z1)
        
        dw1 = np.dot(da2dz2, np.dot(da1, np.dot(dz1, x_batch)))
        
        dw2 = np.dot(np.transpose(da2dz2), a1)

        loss = np.dot(y_batch, np.log(z2)) + np.dot((1-y), np.log(1-z2))
        print("loss: ", loss.shape())

        self.W2 = sig.backward(dw2)
        self.W1 = relu.backward(dw1)

        return dw1, dw2, loss

    def evaluate(self, x, y):
        z = x.y
	# INSERT CODE for testing the network
# ADD other operations and data entries in MLP if needed

if __name__ == '__main__':

    data = cPickle.load(open('cifar_2class_py2.p', 'rb'))

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']
	
    num_examples, input_dims = train_x.shape
    hidden_units = 10
	# INSERT YOUR CODE HERE
	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 10
    num_batches = 10
    mlp = MLP(input_dims, hidden_units)

    train_loss = 0
    total_loss = 0
    batch_size = num_examples/num_batches

    for epoch in xrange(num_epochs):

	# INSERT YOUR CODE FOR EACH EPOCH HERE
        np.random.shuffle(train_x)

        for b in xrange(num_batches):
      	    total_loss = 0.0
# INSERT YOUR CODE FOR EACH MINI_BATCH HERE
# MAKE SURE TO UPDATE total_loss
      	    i = b * batch_size

      	    dw1 , dw2 = mlp.train(x[i:i+batch_size,:],y[i:i+batch_size], learning_rate, momentumw1, momentumw2, penalty)
                        
            #mlp.momentumw1 = u * mlp.momentumw1 - learning_rate * dw1                                             
            #mlp.momentumw2 = u * mlp.momentumw2 - learning_rate * dw2

            #mlp.W1 = mlp.W1 + mlp.momentumw1
            #mlp.W2 = mlp.W2 + mlp.momentumw2

      	    print(
               '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    total_loss,
                ),
                end='',
            )
      	    sys.stdout.flush()
# INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
# MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        print()
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            train_loss,
            100. * train_accuracy,
        ))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            test_loss,
            100. * test_accuracy,
        ))
