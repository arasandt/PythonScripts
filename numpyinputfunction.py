import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.inputs import numpy_io

age = np.arange(4) * 1.0
height = np.arange(32, 36)
x = {'age': age, 'height': height}
y = np.arange(-32, -28)

with tf.Session() as session:
  input_fn = numpy_io.numpy_input_fn(
      x, y, batch_size=2, shuffle=False, num_epochs=1)
  print(type(input_fn))
  print(input_fn())