import os
import tensorflow as tf

# Disable verbose log of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

const1 = tf.placeholder(tf.float32)

softmax = tf.nn.softmax(const1)

var1 = tf.Variable([1.0,2.0,3.0])
softmax2 = tf.nn.softmax(var1)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    out = sess.run(softmax, {const1:[1,2,3]})
    out2= sess.run(softmax2)
    print(out)
    print(out2)