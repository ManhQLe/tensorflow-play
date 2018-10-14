import os
import tensorflow as tf

# Disable verbose log of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.constant(1.0)
hold = tf.placeholder(tf.float32)
var = tf.Variable(2.0)

compute = tf.add(x,tf.multiply(hold,var))

init = tf.initializers.global_variables()
with tf.Session() as sess:    
    sess.run(init)
    out = sess.run(compute,feed_dict={hold:15})
    print(out)
