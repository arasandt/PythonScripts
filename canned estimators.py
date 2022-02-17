import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
train_df = pd.read_csv('./train.csv')

crim = tf.feature_column.numeric_column('crim', dtype=tf.float64, shape=())
zn = tf.feature_column.numeric_column('zn', dtype=tf.float64, shape=())
indus = tf.feature_column.numeric_column('indus', dtype=tf.float64, shape=())
chas = tf.feature_column.numeric_column('chas', dtype=tf.int64, shape=())
nox = tf.feature_column.numeric_column('nox', dtype=tf.float64, shape=())
rm = tf.feature_column.numeric_column('rm', dtype=tf.float64, shape=())
age = tf.feature_column.numeric_column('age', dtype=tf.float64, shape=())
dis = tf.feature_column.numeric_column('dis', dtype=tf.float64, shape=())
rad = tf.feature_column.numeric_column('rad', dtype=tf.int64, shape=())
tax = tf.feature_column.numeric_column('tax', dtype=tf.int64, shape=())
ptratio = tf.feature_column.numeric_column('ptratio', dtype=tf.float64, shape=())
black = tf.feature_column.numeric_column('black', dtype=tf.float64, shape=())
lstat = tf.feature_column.numeric_column('lstat', dtype=tf.float64, shape=())

feature_cols = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat]
feature_names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
label_name = 'medv'
features_ndarray = train_df[feature_names]
label_ndarray = train_df[label_name]
X_train, X_test, y_train, y_test = train_test_split(features_ndarray, label_ndarray, random_state=0, test_size=0.3) #random_state is the seed


def train_input():
    _dataset = tf.data.Dataset.from_tensor_slices(({'crim': X_train['crim'], 
                                                   'zn': X_train['zn'], 
                                                   'indus': X_train['indus'],
                                                   'chas': X_train['chas'],
                                                   'nox': X_train['nox'],
                                                   'rm': X_train['rm'],
                                                   'age': X_train['age'],
                                                   'dis': X_train['dis'],
                                                   'rad': X_train['rad'],
                                                   'tax': X_train['tax'],
                                                   'ptratio': X_train['ptratio'],
                                                   'black': X_train['black'],
                                                   'lstat': X_train['lstat']
                                                  }, y_train))
    dataset = _dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def val_input():
    _dataset = tf.data.Dataset.from_tensor_slices(({'crim': X_test['crim'], 
                                                   'zn': X_test['zn'], 
                                                   'indus': X_test['indus'],
                                                   'chas': X_test['chas'],
                                                   'nox': X_test['nox'],
                                                   'rm': X_test['rm'],
                                                   'age': X_test['age'],
                                                   'dis': X_test['dis'],
                                                   'rad': X_test['rad'],
                                                   'tax': X_test['tax'],
                                                   'ptratio': X_test['ptratio'],
                                                   'black': X_test['black'],
                                                   'lstat': X_test['lstat']
                                                  }, y_test))
    dataset = _dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


estimator = tf.estimator.LinearRegressor(feature_columns=feature_cols)
estimator.train(input_fn=train_input, steps=None) #steps=none will train until data is fetched

train_e = estimator.evaluate(input_fn=train_input)
print(train_e)
test_e = estimator.evaluate(input_fn=val_input)
print(test_e)
preds = estimator.predict(input_fn=val_input)
print(preds[0])
predictions = np.array([item['predictions'][0] for item in preds])
print(list(zip(predictions,y_test)))






















