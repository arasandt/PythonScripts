import tensorflow as tf
import cv2
import classpkg.facenet as facenet
import classpkg.detect_face as detect_face
#import logging

class BoundingBox(object):

    def __init__(self, frame, img_path=None):
        self.IMG_PATH = img_path
        self.frame = frame
        self.MODEL_DIR = './model/FaceNet_20180408-102900.pb'
        self.NUMPY_FILE='./model/'
        self.MINSIZE = 20  # minimum size of face
        self.THRESHOLD = [0.6, 0.7, 0.7]  # three steps's threshold
        self.FACTOR = 0.709  # scale factor
        self.MARGIN = 44
        self.FRAME_INTERVAL = 3
        self.BATCH_SIZE = 1000
        self.IMAGE_SIZE = 182
        self.INPUT_IMAGE_SIZE = 160
        self.PER_PROCESS_GPU_MEMORY_FRACTION=0.6


    def get_mtcnn_data(self):
         return detect_face.create_mtcnn(self.sess, self.NUMPY_FILE)

     
    def detect_face_cordinates(self):
        facenet.load_model(self.MODEL_DIR)
        #self.frame = cv2.imread(self.IMG_PATH,0)
            
        if self.frame.ndim == 2:
            self.frame = facenet.to_rgb(self.frame)
    
        self.bounding_boxes,_ = detect_face.detect_face(self.frame, 
                                        self.MINSIZE, self.pnet, self.rnet,
                                        self.onet, self.THRESHOLD, self.FACTOR)
        
        return self.bounding_boxes
        

    def get_bounding_boxes(self):
        with tf.Graph().as_default():
            self.person = {}
            #self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
            #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))
            self.pnet, self.rnet, self.onet = self.get_mtcnn_data()
            self.bb = self.detect_face_cordinates()
            #print('Total no of faces found is {}'.format(self.bb.shape[0]))
            
            #for i in range(int(self.bb.shape[0])):
            #    self.person['Person_' + str(i)] = list(self.bb[i])[0:4]
                
        return self.person
        
if (__name__ == '__main__'):
    img_path='download.jpg'
    try:
        bb = BoundingBox(img_path)
        boundingBox = bb.get_bounding_boxes()
        
        if boundingBox:
            for i in boundingBox.keys():
                print(i,boundingBox[i])
        else:
            print('No persons available in the frame')
            
    except Exception as e:
        print('An Error has occured while fetching the cordinates ' + str(e))