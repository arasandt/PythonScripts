
from sklearn.externals import joblib
from math import exp
import os
import pandas as pd
from io import StringIO

from flask import Flask, request
from flask_restful import Resource, Api
from webargs import fields, validate
from webargs.flaskparser import use_args, use_kwargs, parser, abort
from flask_cors import CORS

print('Loding model ... ')

#Model Dictionary
Models = {}
Confidence = {}

# Diabetes Model 
model = joblib.load('modelstore/diabetes.joblib') 
diabetesmodel = model
f= open('modelstore/diabetes.confidence','r')
Confidence["Diabetes-1.0"] = f.readline()
f.close()
Models["Diabetes-1.0"] = model

# Thyroid Model
model = joblib.load('modelstore/thyroid.joblib') 
f= open('modelstore/thyroid.confidence','r')
Confidence["Thyroid-1.0"] = f.readline()
f.close()
Models["Thyroid-1.0"] = model

# Fallrisk Model
model = joblib.load('modelstore/fallrisk.joblib') 
f= open('modelstore/fallrisk.confidence','r')
Confidence["Fallrisk-1.0"] = f.readline()
f.close()
Models["Fallrisk-1.0"] = model

# Heart Model
model = joblib.load('modelstore/heart.joblib') 
f= open('modelstore/heart.confidence','r')
Confidence["Heart-1.0"] = f.readline()
f.close()
Models["Heart-1.0"] = model

# Neurology Model
model = joblib.load('modelstore/neurology.joblib') 
f= open('modelstore/neurology.confidence','r')
Confidence["Neurology-1.0"] = f.readline()
f.close()
Models["Neurology-1.0"] = model

data = "mac-opt-dsc-dist,opt-dsc-dia\n2,1"
input_data = pd.read_csv(StringIO(data))

# Retinopathy Model
model = joblib.load('modelstore/retinopathy.joblib') 

prediction = model.predict(input_data)
print(prediction)


f= open('modelstore/retinopathy.confidence','r')
Confidence["Retinopathy-1.0"] = f.readline()
f.close()
Models["Retinopathy-1.0"] = model

print('...load complete! ')





# WebApp 
app = Flask(__name__)
CORS(app)
api = Api(app)

class Diabetes(Resource):
    def get(self):
            features= ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
            featurethreshold = 5
            return runModel("Diabetes","1.0",features,featurethreshold)

class Thyroid(Resource):
    def get(self):
            features = ['T3_resin','Serum_thyrpxin','Serum_triiodothyronine','Basal_TSH','Abs_diff_TSH']
            featurethreshold = 3
            return runModel("Thyroid","1.0",features,featurethreshold)

class Fallrisk(Resource):
    def get(self):
            features = getSensorFeatures()
            featurethreshold = 400
            return runModel("Fallrisk","1.0",features,featurethreshold)

class Heart(Resource):
    def get(self):
            features = ['age','sex','cp','trestbps','chol','restecg','thalach','slope','ca','thal']
            featurethreshold = 6
            return runModel("Heart","1.0",features,featurethreshold)

class Neurology(Resource):
    def get(self):
            features = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR']
            featurethreshold = 8
            return runModel("Neurology","1.0",features,featurethreshold)

class Retinopathy(Resource):
    def get(self):
            features = ['mac-opt-dsc-dist','opt-dsc-dia']
            featurethreshold = 2
            return runModel("Retinopathy","1.0",features,featurethreshold)

class Cohort(Resource):
    def get(self):
                sex=0
                type_of_claim=0
                accident_in_office_or_not=0
                disability_indicator=0
                any_treatment_in_past_18_months=0
                diagnosis_code_back_injury=1
                elbow_injury=0
                over_exertion=0
                pregnancy=0
                
                sex = int(request.args.get("sex"))
                type_of_claim = int(request.args.get("type_of_claim"))
                accident_in_office_or_not = int(request.args.get("accident_in_office_or_not"))
                disability_indicator = int(request.args.get("disability_indicator"))
                any_treatment_in_past_18_months = int(request.args.get("any_treatment_in_past_18_months"))
                elbow_injury = int(request.args.get("elbow_injury"))
                over_exertion = int(request.args.get("over_exertion"))
                pregnancy = int(request.args.get("pregnancy"))
                diagnosis_code_back_injury = int(request.args.get("diagnosis_code_back_injury"))

                x = (-74.8273+(16.805*sex)+\
                     (-74.8273*type_of_claim)+\
                     (80.4952*accident_in_office_or_not)+\
                     (93.2384*disability_indicator)+\
                     (-25.7049*any_treatment_in_past_18_months)+\
                     (-160.7691*diagnosis_code_back_injury)+\
                     (83.5794*elbow_injury) +\
                     (float(110.4445)*float(over_exertion)) +\
                     71.2002*pregnancy)
                     
                proba = (exp(x))/(1+exp(x))
                return {
                'Result' : 'Success',
                'ModelName' : "LTDCohorts",
                'ModelVersion' : "1.0",
                'Confidence' : 0.472365,
                'Prediction': float(proba*100)}


api.add_resource(Diabetes, '/diabetes') 
api.add_resource(Thyroid, '/thyroid') 
api.add_resource(Fallrisk, '/fallrisk') 
api.add_resource(Heart, '/heart') 
api.add_resource(Neurology, '/neurology') 
api.add_resource(Retinopathy, '/retinopathy') 
api.add_resource(Cohort, '/cohort') 

def runModel(ModelName,ModelVersion,features,featurethreshold):
    data = []
    filledupFeatures = 0
    for feature in features:
        inputField = request.args.get(feature)
        if inputField == None:
            data.append(0)
        else:
            data.append(inputField)
            filledupFeatures+=1

    print ("No of inputs:" + str(filledupFeatures))
    if (filledupFeatures < featurethreshold):
        return {
            'Result' : 'Not_enough_data',
            'ModelName' : ModelName,
            'ModelVersion' : ModelVersion
        }
    else:
        for i in range(len(features)):
            print(features[i], ":",data[i])

        # Input data
        nameString = ",".join(features)
        dataString = ",".join(map(str,data))

        data = nameString + "\n" + dataString
        input_data = pd.read_csv(StringIO(data))

        # Predict
        prediction = Models[ModelName+"-"+ModelVersion].predict(input_data)
        print(prediction)

        return {
            'Result' : 'Success',
            'ModelName' : ModelName,
            'ModelVersion' : ModelVersion,
            'Confidence' : Confidence[ModelName+"-"+ModelVersion],
            'Prediction': int(prediction[0])}

def getSensorFeatures():
    return ['tBodyAcc-Mean-1',
        'tBodyAcc-Mean-2',
        'tBodyAcc-Mean-3',
        'tBodyAcc-STD-1',
        'tBodyAcc-STD-2',
        'tBodyAcc-STD-3',
        'tBodyAcc-Mad-1',
        'tBodyAcc-Mad-2',
        'tBodyAcc-Mad-3',
        'tBodyAcc-Max-1',
        'tBodyAcc-Max-2',
        'tBodyAcc-Max-3',
        'tBodyAcc-Min-1',
        'tBodyAcc-Min-2',
        'tBodyAcc-Min-3',
        'tBodyAcc-SMA-1',
        'tBodyAcc-Energy-1',
        'tBodyAcc-Energy-2',
        'tBodyAcc-Energy-3',
        'tBodyAcc-IQR-1',
        'tBodyAcc-IQR-2',
        'tBodyAcc-IQR-3',
        'tBodyAcc-ropy-1',
        'tBodyAcc-ropy-2',
        'tBodyAcc-ropy-3',
        'tBodyAcc-ARCoeff-1',
        'tBodyAcc-ARCoeff-2',
        'tBodyAcc-ARCoeff-3',
        'tBodyAcc-ARCoeff-4',
        'tBodyAcc-ARCoeff-5',
        'tBodyAcc-ARCoeff-6',
        'tBodyAcc-ARCoeff-7',
        'tBodyAcc-ARCoeff-8',
        'tBodyAcc-ARCoeff-9',
        'tBodyAcc-ARCoeff-10',
        'tBodyAcc-ARCoeff-11',
        'tBodyAcc-ARCoeff-12',
        'tBodyAcc-Correlation-1',
        'tBodyAcc-Correlation-2',
        'tBodyAcc-Correlation-3',
        'tGravityAcc-Mean-1',
        'tGravityAcc-Mean-2',
        'tGravityAcc-Mean-3',
        'tGravityAcc-STD-1',
        'tGravityAcc-STD-2',
        'tGravityAcc-STD-3',
        'tGravityAcc-Mad-1',
        'tGravityAcc-Mad-2',
        'tGravityAcc-Mad-3',
        'tGravityAcc-Max-1',
        'tGravityAcc-Max-2',
        'tGravityAcc-Max-3',
        'tGravityAcc-Min-1',
        'tGravityAcc-Min-2',
        'tGravityAcc-Min-3',
        'tGravityAcc-SMA-1',
        'tGravityAcc-Energy-1',
        'tGravityAcc-Energy-2',
        'tGravityAcc-Energy-3',
        'tGravityAcc-IQR-1',
        'tGravityAcc-IQR-2',
        'tGravityAcc-IQR-3',
        'tGravityAcc-ropy-1',
        'tGravityAcc-ropy-2',
        'tGravityAcc-ropy-3',
        'tGravityAcc-ARCoeff-1',
        'tGravityAcc-ARCoeff-2',
        'tGravityAcc-ARCoeff-3',
        'tGravityAcc-ARCoeff-4',
        'tGravityAcc-ARCoeff-5',
        'tGravityAcc-ARCoeff-6',
        'tGravityAcc-ARCoeff-7',
        'tGravityAcc-ARCoeff-8',
        'tGravityAcc-ARCoeff-9',
        'tGravityAcc-ARCoeff-10',
        'tGravityAcc-ARCoeff-11',
        'tGravityAcc-ARCoeff-12',
        'tGravityAcc-Correlation-1',
        'tGravityAcc-Correlation-2',
        'tGravityAcc-Correlation-3',
        'tBodyAccJerk-Mean-1',
        'tBodyAccJerk-Mean-2',
        'tBodyAccJerk-Mean-3',
        'tBodyAccJerk-STD-1',
        'tBodyAccJerk-STD-2',
        'tBodyAccJerk-STD-3',
        'tBodyAccJerk-Mad-1',
        'tBodyAccJerk-Mad-2',
        'tBodyAccJerk-Mad-3',
        'tBodyAccJerk-Max-1',
        'tBodyAccJerk-Max-2',
        'tBodyAccJerk-Max-3',
        'tBodyAccJerk-Min-1',
        'tBodyAccJerk-Min-2',
        'tBodyAccJerk-Min-3',
        'tBodyAccJerk-SMA-1',
        'tBodyAccJerk-Energy-1',
        'tBodyAccJerk-Energy-2',
        'tBodyAccJerk-Energy-3',
        'tBodyAccJerk-IQR-1',
        'tBodyAccJerk-IQR-2',
        'tBodyAccJerk-IQR-3',
        'tBodyAccJerk-ropy-1',
        'tBodyAccJerk-ropy-2',
        'tBodyAccJerk-ropy-3',
        'tBodyAccJerk-ARCoeff-1',
        'tBodyAccJerk-ARCoeff-2',
        'tBodyAccJerk-ARCoeff-3',
        'tBodyAccJerk-ARCoeff-4',
        'tBodyAccJerk-ARCoeff-5',
        'tBodyAccJerk-ARCoeff-6',
        'tBodyAccJerk-ARCoeff-7',
        'tBodyAccJerk-ARCoeff-8',
        'tBodyAccJerk-ARCoeff-9',
        'tBodyAccJerk-ARCoeff-10',
        'tBodyAccJerk-ARCoeff-11',
        'tBodyAccJerk-ARCoeff-12',
        'tBodyAccJerk-Correlation-1',
        'tBodyAccJerk-Correlation-2',
        'tBodyAccJerk-Correlation-3',
        'tBodyGyro-Mean-1',
        'tBodyGyro-Mean-2',
        'tBodyGyro-Mean-3',
        'tBodyGyro-STD-1',
        'tBodyGyro-STD-2',
        'tBodyGyro-STD-3',
        'tBodyGyro-Mad-1',
        'tBodyGyro-Mad-2',
        'tBodyGyro-Mad-3',
        'tBodyGyro-Max-1',
        'tBodyGyro-Max-2',
        'tBodyGyro-Max-3',
        'tBodyGyro-Min-1',
        'tBodyGyro-Min-2',
        'tBodyGyro-Min-3',
        'tBodyGyro-SMA-1',
        'tBodyGyro-Energy-1',
        'tBodyGyro-Energy-2',
        'tBodyGyro-Energy-3',
        'tBodyGyro-IQR-1',
        'tBodyGyro-IQR-2',
        'tBodyGyro-IQR-3',
        'tBodyGyro-ropy-1',
        'tBodyGyro-ropy-2',
        'tBodyGyro-ropy-3',
        'tBodyGyro-ARCoeff-1',
        'tBodyGyro-ARCoeff-2',
        'tBodyGyro-ARCoeff-3',
        'tBodyGyro-ARCoeff-4',
        'tBodyGyro-ARCoeff-5',
        'tBodyGyro-ARCoeff-6',
        'tBodyGyro-ARCoeff-7',
        'tBodyGyro-ARCoeff-8',
        'tBodyGyro-ARCoeff-9',
        'tBodyGyro-ARCoeff-10',
        'tBodyGyro-ARCoeff-11',
        'tBodyGyro-ARCoeff-12',
        'tBodyGyro-Correlation-1',
        'tBodyGyro-Correlation-2',
        'tBodyGyro-Correlation-3',
        'tBodyGyroJerk-Mean-1',
        'tBodyGyroJerk-Mean-2',
        'tBodyGyroJerk-Mean-3',
        'tBodyGyroJerk-STD-1',
        'tBodyGyroJerk-STD-2',
        'tBodyGyroJerk-STD-3',
        'tBodyGyroJerk-Mad-1',
        'tBodyGyroJerk-Mad-2',
        'tBodyGyroJerk-Mad-3',
        'tBodyGyroJerk-Max-1',
        'tBodyGyroJerk-Max-2',
        'tBodyGyroJerk-Max-3',
        'tBodyGyroJerk-Min-1',
        'tBodyGyroJerk-Min-2',
        'tBodyGyroJerk-Min-3',
        'tBodyGyroJerk-SMA-1',
        'tBodyGyroJerk-Energy-1',
        'tBodyGyroJerk-Energy-2',
        'tBodyGyroJerk-Energy-3',
        'tBodyGyroJerk-IQR-1',
        'tBodyGyroJerk-IQR-2',
        'tBodyGyroJerk-IQR-3',
        'tBodyGyroJerk-ropy-1',
        'tBodyGyroJerk-ropy-2',
        'tBodyGyroJerk-ropy-3',
        'tBodyGyroJerk-ARCoeff-1',
        'tBodyGyroJerk-ARCoeff-2',
        'tBodyGyroJerk-ARCoeff-3',
        'tBodyGyroJerk-ARCoeff-4',
        'tBodyGyroJerk-ARCoeff-5',
        'tBodyGyroJerk-ARCoeff-6',
        'tBodyGyroJerk-ARCoeff-7',
        'tBodyGyroJerk-ARCoeff-8',
        'tBodyGyroJerk-ARCoeff-9',
        'tBodyGyroJerk-ARCoeff-10',
        'tBodyGyroJerk-ARCoeff-11',
        'tBodyGyroJerk-ARCoeff-12',
        'tBodyGyroJerk-Correlation-1',
        'tBodyGyroJerk-Correlation-2',
        'tBodyGyroJerk-Correlation-3',
        'tBodyAccMag-Mean-1',
        'tBodyAccMag-STD-1',
        'tBodyAccMag-Mad-1',
        'tBodyAccMag-Max-1',
        'tBodyAccMag-Min-1',
        'tBodyAccMag-SMA-1',
        'tBodyAccMag-Energy-1',
        'tBodyAccMag-IQR-1',
        'tBodyAccMag-ropy-1',
        'tBodyAccMag-ARCoeff-1',
        'tBodyAccMag-ARCoeff-2',
        'tBodyAccMag-ARCoeff-3',
        'tBodyAccMag-ARCoeff-4',
        'tGravityAccMag-Mean-1',
        'tGravityAccMag-STD-1',
        'tGravityAccMag-Mad-1',
        'tGravityAccMag-Max-1',
        'tGravityAccMag-Min-1',
        'tGravityAccMag-SMA-1',
        'tGravityAccMag-Energy-1',
        'tGravityAccMag-IQR-1',
        'tGravityAccMag-ropy-1',
        'tGravityAccMag-ARCoeff-1',
        'tGravityAccMag-ARCoeff-2',
        'tGravityAccMag-ARCoeff-3',
        'tGravityAccMag-ARCoeff-4',
        'tBodyAccJerkMag-Mean-1',
        'tBodyAccJerkMag-STD-1',
        'tBodyAccJerkMag-Mad-1',
        'tBodyAccJerkMag-Max-1',
        'tBodyAccJerkMag-Min-1',
        'tBodyAccJerkMag-SMA-1',
        'tBodyAccJerkMag-Energy-1',
        'tBodyAccJerkMag-IQR-1',
        'tBodyAccJerkMag-ropy-1',
        'tBodyAccJerkMag-ARCoeff-1',
        'tBodyAccJerkMag-ARCoeff-2',
        'tBodyAccJerkMag-ARCoeff-3',
        'tBodyAccJerkMag-ARCoeff-4',
        'tBodyGyroMag-Mean-1',
        'tBodyGyroMag-STD-1',
        'tBodyGyroMag-Mad-1',
        'tBodyGyroMag-Max-1',
        'tBodyGyroMag-Min-1',
        'tBodyGyroMag-SMA-1',
        'tBodyGyroMag-Energy-1',
        'tBodyGyroMag-IQR-1',
        'tBodyGyroMag-ropy-1',
        'tBodyGyroMag-ARCoeff-1',
        'tBodyGyroMag-ARCoeff-2',
        'tBodyGyroMag-ARCoeff-3',
        'tBodyGyroMag-ARCoeff-4',
        'tBodyGyroJerkMag-Mean-1',
        'tBodyGyroJerkMag-STD-1',
        'tBodyGyroJerkMag-Mad-1',
        'tBodyGyroJerkMag-Max-1',
        'tBodyGyroJerkMag-Min-1',
        'tBodyGyroJerkMag-SMA-1',
        'tBodyGyroJerkMag-Energy-1',
        'tBodyGyroJerkMag-IQR-1',
        'tBodyGyroJerkMag-ropy-1',
        'tBodyGyroJerkMag-ARCoeff-1',
        'tBodyGyroJerkMag-ARCoeff-2',
        'tBodyGyroJerkMag-ARCoeff-3',
        'tBodyGyroJerkMag-ARCoeff-4',
        'fBodyAcc-Mean-1',
        'fBodyAcc-Mean-2',
        'fBodyAcc-Mean-3',
        'fBodyAcc-STD-1',
        'fBodyAcc-STD-2',
        'fBodyAcc-STD-3',
        'fBodyAcc-Mad-1',
        'fBodyAcc-Mad-2',
        'fBodyAcc-Mad-3',
        'fBodyAcc-Max-1',
        'fBodyAcc-Max-2',
        'fBodyAcc-Max-3',
        'fBodyAcc-Min-1',
        'fBodyAcc-Min-2',
        'fBodyAcc-Min-3',
        'fBodyAcc-SMA-1',
        'fBodyAcc-Energy-1',
        'fBodyAcc-Energy-2',
        'fBodyAcc-Energy-3',
        'fBodyAcc-IQR-1',
        'fBodyAcc-IQR-2',
        'fBodyAcc-IQR-3',
        'fBodyAcc-ropy-1',
        'fBodyAcc-ropy-2',
        'fBodyAcc-ropy-3',
        'fBodyAcc-MaxInds-1',
        'fBodyAcc-MaxInds-2',
        'fBodyAcc-MaxInds-3',
        'fBodyAcc-MeanFreq-1',
        'fBodyAcc-MeanFreq-2',
        'fBodyAcc-MeanFreq-3',
        'fBodyAcc-Skewness-1',
        'fBodyAcc-Kurtosis-1',
        'fBodyAcc-Skewness-2',
        'fBodyAcc-Kurtosis-2',
        'fBodyAcc-Skewness-3',
        'fBodyAcc-Kurtosis-3',
        'fBodyAcc-BandsEnergyOld-1',
        'fBodyAcc-BandsEnergyOld-2',
        'fBodyAcc-BandsEnergyOld-3',
        'fBodyAcc-BandsEnergyOld-4',
        'fBodyAcc-BandsEnergyOld-5',
        'fBodyAcc-BandsEnergyOld-6',
        'fBodyAcc-BandsEnergyOld-7',
        'fBodyAcc-BandsEnergyOld-8',
        'fBodyAcc-BandsEnergyOld-9',
        'fBodyAcc-BandsEnergyOld-10',
        'fBodyAcc-BandsEnergyOld-11',
        'fBodyAcc-BandsEnergyOld-12',
        'fBodyAcc-BandsEnergyOld-13',
        'fBodyAcc-BandsEnergyOld-14',
        'fBodyAcc-BandsEnergyOld-15',
        'fBodyAcc-BandsEnergyOld-16',
        'fBodyAcc-BandsEnergyOld-17',
        'fBodyAcc-BandsEnergyOld-18',
        'fBodyAcc-BandsEnergyOld-19',
        'fBodyAcc-BandsEnergyOld-20',
        'fBodyAcc-BandsEnergyOld-21',
        'fBodyAcc-BandsEnergyOld-22',
        'fBodyAcc-BandsEnergyOld-23',
        'fBodyAcc-BandsEnergyOld-24',
        'fBodyAcc-BandsEnergyOld-25',
        'fBodyAcc-BandsEnergyOld-26',
        'fBodyAcc-BandsEnergyOld-27',
        'fBodyAcc-BandsEnergyOld-28',
        'fBodyAcc-BandsEnergyOld-29',
        'fBodyAcc-BandsEnergyOld-30',
        'fBodyAcc-BandsEnergyOld-31',
        'fBodyAcc-BandsEnergyOld-32',
        'fBodyAcc-BandsEnergyOld-33',
        'fBodyAcc-BandsEnergyOld-34',
        'fBodyAcc-BandsEnergyOld-35',
        'fBodyAcc-BandsEnergyOld-36',
        'fBodyAcc-BandsEnergyOld-37',
        'fBodyAcc-BandsEnergyOld-38',
        'fBodyAcc-BandsEnergyOld-39',
        'fBodyAcc-BandsEnergyOld-40',
        'fBodyAcc-BandsEnergyOld-41',
        'fBodyAcc-BandsEnergyOld-42',
        'fBodyAccJerk-Mean-1',
        'fBodyAccJerk-Mean-2',
        'fBodyAccJerk-Mean-3',
        'fBodyAccJerk-STD-1',
        'fBodyAccJerk-STD-2',
        'fBodyAccJerk-STD-3',
        'fBodyAccJerk-Mad-1',
        'fBodyAccJerk-Mad-2',
        'fBodyAccJerk-Mad-3',
        'fBodyAccJerk-Max-1',
        'fBodyAccJerk-Max-2',
        'fBodyAccJerk-Max-3',
        'fBodyAccJerk-Min-1',
        'fBodyAccJerk-Min-2',
        'fBodyAccJerk-Min-3',
        'fBodyAccJerk-SMA-1',
        'fBodyAccJerk-Energy-1',
        'fBodyAccJerk-Energy-2',
        'fBodyAccJerk-Energy-3',
        'fBodyAccJerk-IQR-1',
        'fBodyAccJerk-IQR-2',
        'fBodyAccJerk-IQR-3',
        'fBodyAccJerk-ropy-1',
        'fBodyAccJerk-ropy-2',
        'fBodyAccJerk-ropy-3',
        'fBodyAccJerk-MaxInds-1',
        'fBodyAccJerk-MaxInds-2',
        'fBodyAccJerk-MaxInds-3',
        'fBodyAccJerk-MeanFreq-1',
        'fBodyAccJerk-MeanFreq-2',
        'fBodyAccJerk-MeanFreq-3',
        'fBodyAccJerk-Skewness-1',
        'fBodyAccJerk-Kurtosis-1',
        'fBodyAccJerk-Skewness-2',
        'fBodyAccJerk-Kurtosis-2',
        'fBodyAccJerk-Skewness-3',
        'fBodyAccJerk-Kurtosis-3',
        'fBodyAccJerk-BandsEnergyOld-1',
        'fBodyAccJerk-BandsEnergyOld-2',
        'fBodyAccJerk-BandsEnergyOld-3',
        'fBodyAccJerk-BandsEnergyOld-4',
        'fBodyAccJerk-BandsEnergyOld-5',
        'fBodyAccJerk-BandsEnergyOld-6',
        'fBodyAccJerk-BandsEnergyOld-7',
        'fBodyAccJerk-BandsEnergyOld-8',
        'fBodyAccJerk-BandsEnergyOld-9',
        'fBodyAccJerk-BandsEnergyOld-10',
        'fBodyAccJerk-BandsEnergyOld-11',
        'fBodyAccJerk-BandsEnergyOld-12',
        'fBodyAccJerk-BandsEnergyOld-13',
        'fBodyAccJerk-BandsEnergyOld-14',
        'fBodyAccJerk-BandsEnergyOld-15',
        'fBodyAccJerk-BandsEnergyOld-16',
        'fBodyAccJerk-BandsEnergyOld-17',
        'fBodyAccJerk-BandsEnergyOld-18',
        'fBodyAccJerk-BandsEnergyOld-19',
        'fBodyAccJerk-BandsEnergyOld-20',
        'fBodyAccJerk-BandsEnergyOld-21',
        'fBodyAccJerk-BandsEnergyOld-22',
        'fBodyAccJerk-BandsEnergyOld-23',
        'fBodyAccJerk-BandsEnergyOld-24',
        'fBodyAccJerk-BandsEnergyOld-25',
        'fBodyAccJerk-BandsEnergyOld-26',
        'fBodyAccJerk-BandsEnergyOld-27',
        'fBodyAccJerk-BandsEnergyOld-28',
        'fBodyAccJerk-BandsEnergyOld-29',
        'fBodyAccJerk-BandsEnergyOld-30',
        'fBodyAccJerk-BandsEnergyOld-31',
        'fBodyAccJerk-BandsEnergyOld-32',
        'fBodyAccJerk-BandsEnergyOld-33',
        'fBodyAccJerk-BandsEnergyOld-34',
        'fBodyAccJerk-BandsEnergyOld-35',
        'fBodyAccJerk-BandsEnergyOld-36',
        'fBodyAccJerk-BandsEnergyOld-37',
        'fBodyAccJerk-BandsEnergyOld-38',
        'fBodyAccJerk-BandsEnergyOld-39',
        'fBodyAccJerk-BandsEnergyOld-40',
        'fBodyAccJerk-BandsEnergyOld-41',
        'fBodyAccJerk-BandsEnergyOld-42',
        'fBodyGyro-Mean-1',
        'fBodyGyro-Mean-2',
        'fBodyGyro-Mean-3',
        'fBodyGyro-STD-1',
        'fBodyGyro-STD-2',
        'fBodyGyro-STD-3',
        'fBodyGyro-Mad-1',
        'fBodyGyro-Mad-2',
        'fBodyGyro-Mad-3',
        'fBodyGyro-Max-1',
        'fBodyGyro-Max-2',
        'fBodyGyro-Max-3',
        'fBodyGyro-Min-1',
        'fBodyGyro-Min-2',
        'fBodyGyro-Min-3',
        'fBodyGyro-SMA-1',
        'fBodyGyro-Energy-1',
        'fBodyGyro-Energy-2',
        'fBodyGyro-Energy-3',
        'fBodyGyro-IQR-1',
        'fBodyGyro-IQR-2',
        'fBodyGyro-IQR-3',
        'fBodyGyro-ropy-1',
        'fBodyGyro-ropy-2',
        'fBodyGyro-ropy-3',
        'fBodyGyro-MaxInds-1',
        'fBodyGyro-MaxInds-2',
        'fBodyGyro-MaxInds-3',
        'fBodyGyro-MeanFreq-1',
        'fBodyGyro-MeanFreq-2',
        'fBodyGyro-MeanFreq-3',
        'fBodyGyro-Skewness-1',
        'fBodyGyro-Kurtosis-1',
        'fBodyGyro-Skewness-2',
        'fBodyGyro-Kurtosis-2',
        'fBodyGyro-Skewness-3',
        'fBodyGyro-Kurtosis-3',
        'fBodyGyro-BandsEnergyOld-1',
        'fBodyGyro-BandsEnergyOld-2',
        'fBodyGyro-BandsEnergyOld-3',
        'fBodyGyro-BandsEnergyOld-4',
        'fBodyGyro-BandsEnergyOld-5',
        'fBodyGyro-BandsEnergyOld-6',
        'fBodyGyro-BandsEnergyOld-7',
        'fBodyGyro-BandsEnergyOld-8',
        'fBodyGyro-BandsEnergyOld-9',
        'fBodyGyro-BandsEnergyOld-10',
        'fBodyGyro-BandsEnergyOld-11',
        'fBodyGyro-BandsEnergyOld-12',
        'fBodyGyro-BandsEnergyOld-13',
        'fBodyGyro-BandsEnergyOld-14',
        'fBodyGyro-BandsEnergyOld-15',
        'fBodyGyro-BandsEnergyOld-16',
        'fBodyGyro-BandsEnergyOld-17',
        'fBodyGyro-BandsEnergyOld-18',
        'fBodyGyro-BandsEnergyOld-19',
        'fBodyGyro-BandsEnergyOld-20',
        'fBodyGyro-BandsEnergyOld-21',
        'fBodyGyro-BandsEnergyOld-22',
        'fBodyGyro-BandsEnergyOld-23',
        'fBodyGyro-BandsEnergyOld-24',
        'fBodyGyro-BandsEnergyOld-25',
        'fBodyGyro-BandsEnergyOld-26',
        'fBodyGyro-BandsEnergyOld-27',
        'fBodyGyro-BandsEnergyOld-28',
        'fBodyGyro-BandsEnergyOld-29',
        'fBodyGyro-BandsEnergyOld-30',
        'fBodyGyro-BandsEnergyOld-31',
        'fBodyGyro-BandsEnergyOld-32',
        'fBodyGyro-BandsEnergyOld-33',
        'fBodyGyro-BandsEnergyOld-34',
        'fBodyGyro-BandsEnergyOld-35',
        'fBodyGyro-BandsEnergyOld-36',
        'fBodyGyro-BandsEnergyOld-37',
        'fBodyGyro-BandsEnergyOld-38',
        'fBodyGyro-BandsEnergyOld-39',
        'fBodyGyro-BandsEnergyOld-40',
        'fBodyGyro-BandsEnergyOld-41',
        'fBodyGyro-BandsEnergyOld-42',
        'fBodyAccMag-Mean-1',
        'fBodyAccMag-STD-1',
        'fBodyAccMag-Mad-1',
        'fBodyAccMag-Max-1',
        'fBodyAccMag-Min-1',
        'fBodyAccMag-SMA-1',
        'fBodyAccMag-Energy-1',
        'fBodyAccMag-IQR-1',
        'fBodyAccMag-ropy-1',
        'fBodyAccMag-MaxInds-1',
        'fBodyAccMag-MeanFreq-1',
        'fBodyAccMag-Skewness-1',
        'fBodyAccMag-Kurtosis-1',
        'fBodyAccJerkMag-Mean-1',
        'fBodyAccJerkMag-STD-1',
        'fBodyAccJerkMag-Mad-1',
        'fBodyAccJerkMag-Max-1',
        'fBodyAccJerkMag-Min-1',
        'fBodyAccJerkMag-SMA-1',
        'fBodyAccJerkMag-Energy-1',
        'fBodyAccJerkMag-IQR-1',
        'fBodyAccJerkMag-ropy-1',
        'fBodyAccJerkMag-MaxInds-1',
        'fBodyAccJerkMag-MeanFreq-1',
        'fBodyAccJerkMag-Skewness-1',
        'fBodyAccJerkMag-Kurtosis-1',
        'fBodyGyroMag-Mean-1',
        'fBodyGyroMag-STD-1',
        'fBodyGyroMag-Mad-1',
        'fBodyGyroMag-Max-1',
        'fBodyGyroMag-Min-1',
        'fBodyGyroMag-SMA-1',
        'fBodyGyroMag-Energy-1',
        'fBodyGyroMag-IQR-1',
        'fBodyGyroMag-ropy-1',
        'fBodyGyroMag-MaxInds-1',
        'fBodyGyroMag-MeanFreq-1',
        'fBodyGyroMag-Skewness-1',
        'fBodyGyroMag-Kurtosis-1',
        'fBodyGyroJerkMag-Mean-1',
        'fBodyGyroJerkMag-STD-1',
        'fBodyGyroJerkMag-Mad-1',
        'fBodyGyroJerkMag-Max-1',
        'fBodyGyroJerkMag-Min-1',
        'fBodyGyroJerkMag-SMA-1',
        'fBodyGyroJerkMag-Energy-1',
        'fBodyGyroJerkMag-IQR-1',
        'fBodyGyroJerkMag-ropy-1',
        'fBodyGyroJerkMag-MaxInds-1',
        'fBodyGyroJerkMag-MeanFreq-1',
        'fBodyGyroJerkMag-Skewness-1',
        'fBodyGyroJerkMag-Kurtosis-1',
        'tBodyAcc-AngleWRTGravity-1',
        'tBodyAccJerk-AngleWRTGravity-1',
        'tBodyGyro-AngleWRTGravity-1',
        'tBodyGyroJerk-AngleWRTGravity-1',
        'tXAxisAcc-AngleWRTGravity-1',
        'tYAxisAcc-AngleWRTGravity-1',
        'tZAxisAcc-AngleWRTGravity-1'
        ]
        
# This error handler is necessary for usage with Flask-RESTful
@parser.error_handler
def handle_request_parsing_error(err, req, schema):
    """webargs error handler that uses Flask-RESTful's abort function to return
    a JSON error response to the client.
    """
    abort(422, errors=err.messages)

if __name__ == '__main__':
    app.env='development'
    app.run(ssl_context=('/home/ubuntu/openssl/client-cert.pem', '/home/ubuntu/openssl/client-key.pem'),port='8080',host='0.0.0.0',)
#    app.run(ssl_context=('adhoc'),port='8080',host='0.0.0.0',)
