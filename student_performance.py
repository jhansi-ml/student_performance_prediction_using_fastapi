from fastapi import FastAPI
from pydantic import BaseModel
import joblib

#load pkl file
model=joblib.load("student_model.pkl")
scaler=joblib.load("scaler.pkl")
app=FastAPI()

#define input schema
class StudentData(BaseModel):
    hours_studied: float
    sleep_hours: float
    previous_score: float

def get_grade(score):
    if score>=90:
        return "A"
    elif score>=75:
        return "B"
    elif score>=60:
        return "C"
    elif score>=50:
        return "D"
    else:
        return "Fail"
#Home route
@app.get("/")
def home():
    return {"Message":"ML API is running"}

#predict
@app.post("/predict")
def predict(data: StudentData):
    features=[[data.hours_studied, data.sleep_hours, data.previous_score]]
    features=scaler.transform(features)
    prediction=model.predict(features)
    score=prediction[0]
    grade=get_grade(score)
    return {"prediction_score":score,
            "Grade":grade}
