#from Foundatios of TF
import tensorflow as tf
tf.reset_default_graph() 
# =============================================================================
# a = tf.constant(6.5, name='constant_a')
# b = tf.constant(3.4, name='constant_b')
# c = tf.constant(3.0, name='constant_c')
# d = tf.constant(100.2, name='constant_d')
# =============================================================================


# =============================================================================
# mul = tf.multiply(a,b,name='mul')
# div = tf.div(c, d, name='div')
# addn = tf.add_n([mul,div], name='addn')
# 
# =============================================================================

# =============================================================================
# square = tf.square(a,name='square_a')
# power = tf.pow(b,c,name='pow_b_c')
# sqrt = tf.sqrt(d,name='sqrt_d')
# 
# final_sum = tf.add_n([square, power, sqrt],name='final_sum')
# 
# with tf.Session() as sess:
#     print(sess.run(square))
#     print(sess.run(power))
#     print(sess.run(sqrt))
#     print(sess.run(final_sum))
# 
# writer = tf.summary.FileWriter('./m2_example1', sess.graph)
# writer.close()
# =============================================================================


# =============================================================================
# x = tf.constant([100,200,300],name='x')
# y = tf.constant([1,2,3],name='y')
# 
# sum_x = tf.reduce_sum(x,name='sum_x')
# prod_y = tf.reduce_prod(y,name='prod_y')
# final_mean = tf.reduce_mean([sum_x,prod_y],name='final_mean')
# 
# with tf.Session() as sess:
#     print(sess.run(sum_x))
#     print(sess.run(prod_y))
#     print(sess.run(final_mean))
# 
# =============================================================================

# y = Wx + b

#W = tf.constant([10,100],name='const_W')
W = tf.Variable([2.5,4.0],tf.float32,name='var_W')

x = tf.placeholder(tf.float32,name='x')
b = tf.Variable([5.0,10.0],tf.float32,name='var_b')


y = W * x + b

init = tf.global_variables_initializer()


#Wx = tf.multiply(W,x,name='Wx')
#y = tf.add(Wx,b,name='y')
#y_ = tf.subtract(x,b,name='y_')

with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(Wx,feed_dict={x:[3,33]}))
    #print(sess.run(fetches=[y,y_],feed_dict={x:[5,50], b:[7,9]}))
    ## we can also do below so Wx value is taken from feed dict
    #print(sess.run(fetches=y,feed_dict={Wx:[100,1000], b:[7,9]}))
    print(sess.run(fetches=y,feed_dict={x:[10,100]}))
    























