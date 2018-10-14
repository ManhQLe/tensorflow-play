import tensorflow as tf

hello_constant = tf.constant("Say hello to Tensorflow with GPU")

with tf.Session() as sess1:
    output = sess1.run(hello_constant)
    print(output)
