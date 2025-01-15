import time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import joblib
from dotenv import load_dotenv
from io import BytesIO
import collections
import json
import firebase_admin
from firebase_admin import credentials, db
import asyncio
import tensorflow as tf
from PIL import Image
from torchvision import transforms
import google.generativeai as genai
import os
import urllib.parse 

load_dotenv()

# Firebase Admin SDK setup
cred = credentials.Certificate(os.getenv('FILE_PATH'))  # Path to your Firebase service account JSON file
firebase_admin.initialize_app(cred, {
    'databaseURL': os.getenv('DATABASE_URL')
})

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load models
freshness = tf.keras.models.load_model(r'D:\FinFinder - Copy\api\models\freshness.h5')
identification = tf.keras.models.load_model(r'D:\FinFinder - Copy\api\models\Identify.h5')

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post("/identify_fish")
async def identify_fish(
    image: UploadFile = File(...)
):
#     try:
        # Read the uploaded files
    
    transform = transforms.Compose([
        transforms.Resize(256)
    ])    
    
    image_data = await image.read()
    
    img = Image.open(BytesIO(image_data))

    resized_image = transform(img)
    
    # Convert the image to a NumPy array
    img_array = np.array(resized_image)
    img_array = np.expand_dims(img_array,axis=0)        

    prediction = identification.predict(img_array)
    
    fish = ['Common Carp', 'Mori', 'Rohu', 'Silver Carp', 'catla']
    
    predicted_class = fish[np.argmax(prediction)]

    return JSONResponse(content={
        'class':predicted_class
    })

    # except Exception as e:
    #     return JSONResponse(content={'error': str(e)}, status_code=500)
@app.post("/freshness_assesment")
async def predict_behavior(
    image: UploadFile = File(...)
):
#     try:
        # Read the uploaded files
    transform = transforms.Compose([
        transforms.Resize(256)
    ])    
    
    image_data = await image.read()
    
    img = Image.open(BytesIO(image_data))

    resized_image = transform(img)
    
    # Convert the image to a NumPy array
    img_array = np.array(resized_image)
    img_array = np.expand_dims(img_array,axis=0)        

    prediction = freshness.predict(img_array)
    
    fresh = ['Fresh Eyes', 'Nonfresh Eyes']
    
    predicted_class = fresh[np.argmax(prediction)]

    return JSONResponse(content={
        'class':predicted_class
    })

@app.post("/get_recipes")
async def get_recipes(
    fish: str = Form(...)
):

    genai.configure(api_key=os.getenv("GENAI_API_KEY"))
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content("three recipes for "+fish+" fish.recipes name, description and steps and instruction and give output in with tags in html for formatting i'm directly going to show as it is in website. Use bold tages instead of header tags and assign number rahter that li ul tags ")
    print(response.text.replace('```html','').replace('```',''))
    print(fish)
    return JSONResponse(content={
        'recipes': response.text.replace('```html','').replace('```','')
    })
    

@app.post("/get_price")
async def get_price(
    fish: str = Form(...)
):

    genai.configure(api_key=os.getenv("GENAI_API_KEY"))
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content("Price for "+fish+" fish in indian market general estimate and do not suggest any website to check realtime price. demand of fish general estimate and give output in with tags in html for formatting i'm directly going to show as it is in website. Use bold tages instead of header tags and assign number rahter that li ul tags ")
    print(response.text.replace('```html','').replace('```',''))
    print(fish)
    return JSONResponse(content={
        'price': response.text.replace('```html','').replace('```','')
    })
  
@app.post("/signup")
async def signup(
    name: str = Form(...),
    username: str = Form(...),
    password: str = Form(...)
):

    ref = db.reference(f'finfinder/{username}')
    
    await asyncio.to_thread(ref.set, {
        'name': name,
        'username': username,
        'password': password  # Ensure that password is stored correctly
    })
    
    return JSONResponse(content={'message': 'User created successfully'}, status_code=201)

@app.post("/login")
async def login(
    username: str = Form(...),
    password: str = Form(...)
):

    ref = db.reference(f'finfinder/{username}')
    
    user = await asyncio.to_thread(ref.get)
    
    if user is None:
        return JSONResponse(content={'valid': False, 'message': 'User not found'}, status_code=400)
    elif user['password'] == password:
        return JSONResponse(content={'valid': True, 'message': 'Login successful'}, status_code=200)
    else:
        return JSONResponse(content={'valid': False, 'message': 'Invalid credentials'}, status_code=400)
    
def get_user_inventory_ref(username: str):
    return db.reference(f'finfinder/{username}/inventory')

@app.post("/add_fish")
async def add_fish(
    fish_name: str = Form(...),
    count: str = Form(...),
    username: str = Form(...)
):
    """Add a new fish to the inventory for the specified username."""
    
    count = int(count)

    user_ref = get_user_inventory_ref(username)

    user_inventory = await asyncio.to_thread(user_ref.get)

    if user_inventory is None:
        user_inventory = {}

    user_inventory[fish_name] = user_inventory.get(fish_name, 0) + count

    await asyncio.to_thread(user_ref.set, user_inventory)

    return JSONResponse(status_code=200, content={"message": "Fish added successfully."})

@app.post("/update_fish_count")
async def update_fish_count(
    fish_name: str = Form(...),
    operation: str = Form(...),  # This will either be 'increase' or 'decrease'
    username: str = Form(...),
):
    """Update the count of an existing fish in the inventory based on the operation."""
    
    # Get the Firebase reference for the user's inventory
    user_ref = get_user_inventory_ref(username)

    # Retrieve the current inventory asynchronously
    user_inventory = await asyncio.to_thread(user_ref.get)

    # If no inventory or the fish is not found, return an error message
    if user_inventory is None or fish_name not in user_inventory:
        return JSONResponse(status_code=400, content={"message": "Fish not found in inventory."})

    # Get the current count of the fish
    current_count = user_inventory[fish_name]

    # Based on the operation, either increase or decrease the count
    if operation == "increase":
        # Increase the count by 1 (or you can use a different value if needed)
        user_inventory[fish_name] = current_count + 1
    elif operation == "decrease":
        # Decrease the count by 1 (ensure count does not go below 0)
        if current_count > 0:
            user_inventory[fish_name] = current_count - 1
        else:
            return JSONResponse(status_code=400, content={"message": "Count cannot be negative."})
    else:
        return JSONResponse(status_code=400, content={"message": "Invalid operation. Use 'increase' or 'decrease'."})

    # Save the updated inventory back to Firebase
    await asyncio.to_thread(user_ref.set, user_inventory)

    return JSONResponse(status_code=200, content={"message": "Fish count updated successfully."})

# Retrieve the inventory for the specified username
@app.post("/get_inventory")
async def get_inventory(
    username: str = Form(...)
):
    """Retrieve the inventory for the specified username."""
    user_ref = get_user_inventory_ref(username)

    # Retrieve the current inventory asynchronously
    user_inventory = await asyncio.to_thread(user_ref.get)

    if user_inventory is None:
        return JSONResponse(status_code=404, content={"message": "User inventory not found."})

    # Return inventory data directly as a JSON response
    return {"inventory": user_inventory}

@app.post("/delete_fish")
async def delete_fish(
    fish_name: str = Form(...),
    username: str = Form(...),
):
    """Delete a fish from the inventory."""
    
    user_ref = get_user_inventory_ref(username)

    # Retrieve the current inventory asynchronously
    user_inventory = await asyncio.to_thread(user_ref.get)

    # If no inventory or the fish is not found, return an error message
    if user_inventory is None or fish_name not in user_inventory:
        return JSONResponse(status_code=400, content={"message": "Fish not found in inventory."})

    # Delete the fish from the inventory
    del user_inventory[fish_name]

    # Save the updated inventory back to Firebase
    await asyncio.to_thread(user_ref.set, user_inventory)

    # Return success response
    return JSONResponse(status_code=200, content={"message": "Fish deleted successfully."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


