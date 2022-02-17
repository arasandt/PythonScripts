import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

x = tf.placeholder(tf.float32)

linear_model = W * x + b
y = tf.placeholder(tf.float32)

loss = tf.reduce_mean(tf.square(linear_model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(optimizer,feed_dict={x:x_train, y:y_train})
    print(sess.run([W,b]))
    print(sess.run([loss],feed_dict={x:x_train, y:y_train}))
    

