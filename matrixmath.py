import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([1,2,3,4,5,6],shape=[2,3])
b = tf.constant([2,2,2,2,2,2],shape=[3,2])
mult = tf.matmul(a,b)

with tf.Session() as sess:
    out = sess.run(mult)
    print(type(out))
    print(out)
    
