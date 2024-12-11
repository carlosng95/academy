from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import warnings
warnings.filterwarnings('ignore')

class Depto(BaseModel):
    city: float
    day: float
    shared_room: float
    private_room: float
    person_capacity: float
    superhost: float
    multiple_rooms: float
    business: float
    cleanliness_rating: float
    guest_satisfaction: float
    bedrooms: float
    city_center_km: float
    metro_distance_km: float
    attraction_index: float
    normalised_attraction_index: float
    restraunt_index: float
    normalised_restraunt_index: float
    
app = FastAPI()

@app.get('/')
async def root():
    return {"message": "Bienvenidos a nuestro modelo"}

@app.post('/prediction/')
async def get_predictions(payload: Depto):
    values_list = [
        payload.city,
        payload.day,
        payload.shared_room,
        payload.private_room,
        payload.person_capacity,
        payload.superhost,
        payload.multiple_rooms,
        payload.business,
        payload.cleanliness_rating,
        payload.guest_satisfaction,
        payload.bedrooms,
        payload.city_center_km,
        payload.metro_distance_km,
        payload.attraction_index,
        payload.normalised_attraction_index,
        payload.restraunt_index,
        payload.normalised_restraunt_index
    ]
    with open('modelo_lineal.pkl', 'rb') as file:
        model = pickle.load(file)
        
    prediction = model.predict([values_list])[0][0]
    return {"prediction": prediction}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    