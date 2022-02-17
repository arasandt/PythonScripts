import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

cifar_folder = ".\\cifar-10\\cifar-10-batches-py\\"

def cifar_load(batch_id):
    with open(cifar_folder + 'data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='bytes')

    keys = [(i.decode('ascii'), i) for i in batch.keys()]
    for x,y in keys:
        batch[x] = batch.pop(y)
    
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0,2,3,1)
    labels = batch['labels']
    
    return features, labels

features, labels = cifar_load(1)
#print(len(features),features[0].shape,labels[:5])

lookup = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def display_fl(features,labels,index):
    if index >= len(features):
        raise Exception('Index out of range')
    
    print('Label :', lookup[labels[index]])
    plt.imshow(features[index])
        
        
#display_fl(features,labels,7)

train_size = int(len(features) * 0.8)
training_images = features[:train_size,:,:]
training_labels = labels[:train_size]
print(len(training_images), len(training_labels))

test_images = features[train_size:,:,:]
test_labels = labels[train_size:]
print(len(test_images), len(test_labels))

height = 32
width = 32
channels = 3
n_inputs = height * width

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[None, height,width,channels], name='X')
dropout_rate = 0.3

training = tf.placeholder_with_default(False,shape=(),name='training')
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

y = tf.placeholder(tf.int32,shape=[None],name='y')

conv1 = tf.layers.conv2d(X_drop, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, name='conv1')
print(conv1.shape)
conv2 = tf.layers.conv2d(conv1,  filters=64, kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu, name='conv2')
print(conv2.shape)
pool3 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
print(pool3.shape)
conv4 = tf.layers.conv2d(pool3,  filters=128,kernel_size=4, strides=3, padding='SAME', activation=tf.nn.relu, name='conv4')
print(conv4.shape)
pool5 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,1,1,1], padding='VALID')
print(pool5.shape)
pool5_flat = tf.reshape(pool5, shape=[-1, 128 * 2 * 2])
print(pool5_flat.shape)
fullyconn1 = tf.layers.dense(pool5_flat,128,activation=tf.nn.relu, name='fc1' )
print(fullyconn1.shape)
fullyconn2 = tf.layers.dense(fullyconn1,64,activation=tf.nn.relu, name='fc2' )
print(fullyconn2.shape)

logits = tf.layers.dense(fullyconn2, 10, name='output')

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def get_next_batch(features, labels, train_size, batch_index, batch_size):
    training_images = features[:train_size,:,:]
    training_labels = labels[:train_size]

    test_images = features[train_size:,:,:]
    test_labels = labels[train_size:]
    
    start_index = batch_index * batch_size
    end_index = start_index + batch_size
    
    return features[start_index: end_index, :,:], labels[start_index:end_index], test_images, test_labels


n_epochs = 1
batch_size = 128

with tf.Session() as sess:
    init.run()
    #tf.summary.FileWriterCache.clear()
    #writer = tf.summary.FileWriter('board_beginner')    
    #writer.add_graph(sess.graph)
    train_writer = tf.summary.FileWriter("logs", sess.graph)
    
    for epoch  in range(n_epochs):
        batch_index = 0
        
        for iteration in range(train_size // batch_size):
            X_batch, y_batch, test_images, test_labels = get_next_batch(features, labels, train_size, batch_index, batch_size)
            batch_index += 1
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training:True})
        

        
            
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: test_images, y: test_labels})
        #tf.summary.scalar('accuracy', accuracy)
        print(epoch, "Train Accuracy: ", acc_train, "test Accuracy: ", acc_test)
        
        #writer.add_summary(acc_test)
        save_path = saver.save(sess,"./my_mnist_model")
        
    
    #merged = tf.summary.merge_all()

    #tf.train.SummarySaverHook
    #train_writer.close()



































