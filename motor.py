# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:28:37 2023

@author: Ankit
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st

MODEL_FILE_PATH = "trained_model.sav"

def load_model():
    with open(MODEL_FILE_PATH, 'rb') as f:
        return pickle.load(f)

model = load_model()

def predict_motor_speed(input_data):
    input_data_asarray = np.asarray(input_data)
    input_data_reshaped = input_data_asarray.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    return prediction

def main():
    st.title("Motor Speed Prediction")
    
    # Input fields
    ambient_temp = st.text_input("Ambient Temperature (C)", value="25.0")
    coolant_temp = st.text_input("Coolant Temperature (C)", value="25.0")
    u_d = st.text_input("Voltage D-Component (V)", value="0.0")
    u_q = st.text_input("Voltage Q-Component (V)", value="0.0")
    i_d = st.text_input("Current D-Component (A)", value="0.0")
    i_q = st.text_input("Current Q-Component (A)", value="0.0")
    pm = st.text_input("Permanent Magnet (T)", value="0.0")
    stator_winding = st.text_input("Stator Winding Temperature (C)", value="25.0")
    
    # Prediction button
    if st.button("Predict Motor Speed"):
        # Validate input
        try:
            input_data = [float(ambient_temp), float(coolant_temp), float(u_d), float(u_q), float(i_d), float(i_q), float(pm), float(stator_winding)]
        except ValueError:
            st.error("Invalid input data. Please enter numerical values.")
            return
        
        # Make prediction
        prediction = predict_motor_speed(input_data)
        
        # Display prediction
        st.success(f"Predicted motor speed: {prediction[0]:.2f} RPM")
    
if __name__ == "__main__":
    main()
