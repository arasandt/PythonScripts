import tensorflow as tf

def printTensors(pb_file):

    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name='')
        with tf.Session() as sess:
            #writer = tf.summary.FileWriter('logs', sess.graph)
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            print(embeddings.get_shape()[1])
            #writer.close()
    # print operations
    #for op in graph.get_operations():
    #    print(op.name)
    
    
	

printTensors("C:\\Users\\128537\\Downloads\\20180408-102900\\20180408-102900\\20180408-102900.pb")

#tensorboard --logdir="logs"