import pickle
import uvicorn
import numpy as np
import pandas as pd
from BankNotes import BankNote
from fastapi import FastAPI

app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello World'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome to my FastAPI application': f'{name}'}

@app.post('/predict')
def predict_banknote(data: BankNote):
    data = data.dict()
    variance = data["variance"]
    skewness = data["skewness"]
    curtosis = data["curtosis"]
    entropy = data["entropy"]
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if prediction[0] > 0.5:
        prediction = 'Fake note'
    else:
        prediction = 'It\'s a Bank Note'
    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
