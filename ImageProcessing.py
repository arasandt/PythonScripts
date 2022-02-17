# =============================================================================
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.image as mp_img
# import os
# 
# filename = "D:\\Arasan\\Misc\\GitHub\\ML\\tf_hub\\examples\\image_retraining\\test_images\\3.jpg"
# 
# image = mp_img.imread(filename)
# 
# print(image.shape,image)
# 
# plt.imshow(image)
# plt.show()
# 
# x = tf.Variable(image, name='x')
# 
# init = tf.global_variables_initializer()
# 
# with tf.Session() as sess:
#     sess.run(init)
#     #transpose = tf.transpose(x,perm=[1,0,2]) #original was 0,1,2
#     transpose = tf.image.transpose_image(x)
#     result = sess.run(transpose)
#     print(result.shape)
#     plt.imshow(result)
#     plt.show()
# 
#     
#     
# 
# 
# =============================================================================

import tensorflow as tf
from PIL import Image

original_image_list = ['D:\\Arasan\\Misc\\GitHub\\ML\\tf_hub\\examples\\image_retraining\\test_images\\1.jpg',
                       'D:\\Arasan\\Misc\\GitHub\\ML\\tf_hub\\examples\\image_retraining\\test_images\\2.jpg',
                       'D:\\Arasan\\Misc\\GitHub\\ML\\tf_hub\\examples\\image_retraining\\test_images\\3.jpg',]


filename_queue = tf.train.string_input_producer(original_image_list)

image_reader = tf.WholeFileReader()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    
    image_list = []
    for i in range(len(original_image_list)):
        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_jpeg(image_file)
        image = tf.image.resize_images(image,[224,224])
        image.set_shape((224,224,3))
        
        image_array = sess.run(image)
        print(image_array.shape)
        
        #Image.fromarray(image_array.astype('uint8'),'RGB').show()
        image_tensor = tf.stack(image_array)
        print(image_tensor)
        #image_list.append(tf.expand_dims(image_array,0))
        image_list.append(image_tensor)
        
    coord.request_stop()
    coord.join(threads)
        
    images_tensor = tf.stack(image_list) # convert to 4-D
    print(images_tensor)
        
        
        
        
        
        
        
        
        
        
    
    