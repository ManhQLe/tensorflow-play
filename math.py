import os
import tensorflow as tf

# Disable verbose log of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.constant(10)
y = tf.constant(13)
z = tf.add(x,y)
w = tf.subtract(z,1)

with tf.Session() as sess:
    out = sess.run(z)
    out1 = sess.run(w)
    print(out)
    print(out1)