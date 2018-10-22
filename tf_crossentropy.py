import os
import tensorflow as tf

# Disable verbose log of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

yhat = tf.Variable([0.05,0.9,0.05])
y = tf.Variable([0.0,1.0,0.0])

crossentropy = -tf.reduce_sum(tf.multiply(tf.log(yhat),y))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out = sess.run(crossentropy)
    print(out)