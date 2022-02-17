
from sklearn.externals import joblib
import os
import sys
import pandas as pd

if (len(sys.argv) !=4):
    print ("Error: Not enough arguments. Format: python predictionEngineBatch.py <modelname> <modelversion> <inputdata> ")
    quit()

ModelName=sys.argv[1]
ModelVersion=sys.argv[2]
fileName=sys.argv[3]

#print('Loding model ... ')
model = joblib.load('modelstore/'+ModelName+'.joblib') 
#print('...load complete! ')

#print ("Processing file :", fileName)
input_data = pd.read_csv(fileName)

# Predict
prediction = model.predict(input_data)
print(max(prediction))