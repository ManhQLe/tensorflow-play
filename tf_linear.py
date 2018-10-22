import os
import tensorflow as tf

# Disable verbose log of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2 inputs 1 operation like [0,0,0,1]
# Last entry is operator, we want to teach the machine to do logic operation
# 00 - AND, 01 - OR , 10 - XOR 
# 
#
#
#
#


n_of_feature = 4
n_of_out = 1

x = tf.placeholder(tf.float32,[n_of_feature])
y_true = tf.placeholder(tf.float32,[n_of_out])

weights = tf.Variable(tf.zeros([n_of_feature,n_of_out]))

biases = tf.Variable(tf.zeros([n_of_out]))

logits = tf.matmul(x,weights) + biases

out = tf.nn.sigmoid(logits)




