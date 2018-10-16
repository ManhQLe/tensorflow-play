import os
import tensorflow as tf

# Disable verbose log of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

weights = tf.Variable(tf.truncated_normal((1,6)))
zeros = tf.Variable(tf.zeros(5))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    output =  sess.run(weights)
    out2 = sess.run(zeros)
    print(output)
    print(out2)
    print(zeros)