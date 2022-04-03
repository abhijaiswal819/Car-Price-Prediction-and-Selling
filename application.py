from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

import os 
#from pml import app 


app=Flask(__name__)
app._static_folder = "static"
cors=CORS(app)
model=pickle.load(open('RandomForestRegressorModel.pkl','rb'))
car=pd.read_csv('Cleaned_CarDataset.csv')

@app.route('/',methods=['GET','POST'])
def index():
    companies=sorted(car['Company'].unique())
    car_models=sorted(car['Model'].unique())
    locations=sorted(car['Location'].unique())
    year=sorted(car['Year'].unique(),reverse=True)
    fuel_type=sorted(car['Fuel_Type'].unique())
    transmission=sorted(car['Transmission'].unique())
    owner_type=car['Owner_Type'].unique()
    seat=sorted(car['Seats'].unique()) 

    companies.insert(0,'Select Company')
    locations.insert(0,'Select Location')
    year.insert(0,'Select Year')
    fuel_type.insert(0,'Select Fuel Type')
    transmission.insert(0,'Select Transmission')
    #owner_type.insert(0,'Select Owner')
    seat.insert(0,'Select Seat')
    return render_template('index.html',companies=companies, car_models=car_models, locations=locations, years=year,fuel_types=fuel_type, transmissions=transmission, owner_types=owner_type, seats=seat)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    company=request.form.get('company')

    car_model=request.form.get('car_models')
    location=request.form.get('location')
    
    years=request.form.get('year')
    year = int(years)

    fuel=request.form.get('fuel')
    transmission=request.form.get('transmission')
    owner=request.form.get('owner')
    seats = request.form.get('seat')
    seat = int(seats)
    drivens = request.form.get('kilo_driven')
    driven = float(drivens)
    mileages = request.form.get('mileage')
    mileage = float(mileages)
    engines = request.form.get('engine')
    engine = float(engines)
    powers = request.form.get('power')
    power = float(powers)

    prediction=model.predict(pd.DataFrame([[car_model,company,location,year,driven,fuel,transmission,owner,mileage,engine,power,seat]], columns=['Model', 'Company', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats']))
    
    #prediction=model.predict(pd.DataFrame(columns=['Model', 'Company', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats'], data=np.array([car_model,company,location,year,driven,fuel,transmission,owner,mileage,engine,power,seat]).reshape(1, 12)))
    print(prediction)

    return str(np.round(prediction[0],2))


if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000)) 
    app.run(host='0.0.0.0', port=port)
    #app.run()
