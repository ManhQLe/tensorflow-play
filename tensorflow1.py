import os
import tensorflow as tf

# Disable verbose log of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.string)

with tf.Session() as sess:
    out1 = sess.run(x,feed_dict = {x:123})
    out2 = sess.run(y,feed_dict = {y:"A string"})
    print(out1)
    print(out2)
