import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/",one_hot=True)

training_digits, training_labels = mnist.train.next_batch(5000)
test_digits, test_labels = mnist.train.next_batch(200)

training_digits_pl = tf.placeholder("float",[None,784]) #None is list of images which is unknown
test_digits_pl = tf.placeholder("float",[784]) # we are doing to give only one as input, so there is no None
#training_labels_pl = tf.placeholder("float",[None,10]) 
#test_labels_pl = tf.placeholder("float",[10])

l1_distance = tf.abs(tf.add(training_digits_pl,tf.negative(test_digits_pl)))

distance = tf.reduce_sum(l1_distance,axis=1)

#pred = tf.arg_min(distance,0)

accuracy = 0.

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(len(test_digits)):
        pred = sess.run(distance,feed_dict={training_digits_pl: training_digits,test_digits_pl: test_digits[i,:] })
        #nn_index = sess.run(pred,feed_dict={training_digits_pl: training_digits,test_digits_pl: test_digits[i,:] })
        nn_index = sess.run(tf.arg_min(pred,0))
        print("Test",i,"Prediction:",np.argmax(training_labels[nn_index]),"True Label:",np.argmax(test_labels[i]))
# =============================================================================
#         if np.argmax(training_labels[nn_index]) != np.argmax(test_labels[i]):
#             testImage = (np.array(training_digits[nn_index], dtype='float')).reshape(28,28)
#             img = Image.fromarray(np.uint8(testImage * 255) , 'L')
#             img.show()
#             testImage = (np.array(test_digits[i], dtype='float')).reshape(28,28)
#             img = Image.fromarray(np.uint8(testImage * 255) , 'L')
#             img.show()
#             break
#             #Image.fromarray(training_digits[nn_index].astype('uint8'),'RGB').show()
# =============================================================================
    
        if np.argmax(training_labels[nn_index]) == np.argmax(test_labels[i]):
            accuracy += 1./len(test_digits) #simple way to calculate accuracy
        
    print("Accuracy:", accuracy)
        
        