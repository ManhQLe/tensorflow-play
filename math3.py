import os
import tensorflow as tf

# Disable verbose log of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.constant([3,2])
y = tf.constant([3])
p = tf.pow(x,y)
with tf.Session() as sess:
    out = sess.run(p)
    print(out)


