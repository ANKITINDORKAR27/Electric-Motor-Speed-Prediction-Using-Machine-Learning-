# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:56:03 2023

@author: Ankit
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/Ankit/Desktop/Documents/Data Science Projects/Excelr/trained_model.sav', 'rb'))

def Random_Forest(input_data):
    input_data_asarray = np.asarray(input_data)
    input_data_reshaped = input_data_asarray.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def main():
    st.title("Moter Speed Prediction")
    
    
    Ambient = st.text_input("Ambient Temperature")
    Coolant = st.text_input("Coolant Temperature")
    U_D = st.text_input("Voltage_D_Component")
    U_Q = st.text_input("Voltage_Q_Component")
    I_D = st.text_input("Current_D_Component")
    I_Q = st.text_input("Current_Q_Component")
    PM = st.text_input("Permanent Magnet")
    Stater_Winding = st.text_input("Stater_Winding")
    
    empty = " "
    
    if st.button("Predict Motor Speed"):
        empty = Random_Forest([Ambient,Coolant,U_D,U_Q,I_D,I_Q,PM,Stater_Winding])
    st.success(empty)
    
if __name__==' __main__':
    main()