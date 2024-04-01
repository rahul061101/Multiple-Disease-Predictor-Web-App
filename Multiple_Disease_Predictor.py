# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:20:46 2024

@author: HP
"""

import streamlit as st
import pickle
import joblib
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu


#loading the saved model

diabetes_model = pickle.load(open('diabetes.sav', 'rb'))

heart_model = joblib.load(open('heart_disease.sav', 'rb'))

parkinsons_model = pickle.load(open('parkinsons.sav', 'rb'))

breast_model = joblib.load(open('breastcancer.sav', 'rb'))

doctors_df = pd.read_csv('Doctors list.csv')



# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Breast Cancer Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'asterisk'],
                           default_index=0)


# Function to fetch the assigned doctor based on the predicted disease
def display_top_five_doctors():
    st.header("Top Five Doctors")
    if selected == 'Diabetes Prediction':
        st.dataframe(doctors_df[doctors_df['DISEASE'] == 'Diabetes'].head(5))
    elif selected == 'Heart Disease Prediction':
        st.dataframe(doctors_df[doctors_df['DISEASE'] == 'Heart'].head(5))
    elif selected == "Parkinsons Prediction":
        st.dataframe(doctors_df[doctors_df['DISEASE'] == 'Parkinson'].head(5))
    elif selected == "Breast Cancer Prediction":
        st.dataframe(doctors_df[doctors_df['DISEASE'] == 'Breast Cancer'].head(5))


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')
    image = Image.open('diabetes image.jpg')
    st.image(image)
    
    # getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
            Pregnancies = st.number_input('Number of Pregnancies (0-17)',0,17)
            
    with col2:
            Glucose = st.number_input('Glucose Level(0-199)',0,199)
        
    with col1:
            SkinThickness = st.number_input('Skin Thickness value(0-110)',0,110)
        
    with col2:
            Insulin = st.number_input('Insulin Level(0-744)',0,744)
        
    with col1:
            BMI = st.number_input('BMI value(0-81)',0,81)
        
    with col2:
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value(0.078-2.43)',0.078,2.43)
        
    with col1:
            Age = st.number_input('Age of the Person(21-81)',21,81)



    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, SkinThickness, Insulin,
                  BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        
            # Display the test result
            st.success(diab_diagnosis)

            # Display top five doctors for diabetes only if the person has the disease
            if diab_diagnosis == 'The person is diabetic':
                display_top_five_doctors()
        else:
            diab_diagnosis = 'The person is not diabetic'
            st.success(diab_diagnosis)



# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')
    image = Image.open('heart disease image.jpg')
    st.image(image)


    col1, col2 = st.columns(2)
    
    with col1:
        age = st.text_input('Age') 
        
    with col2:
        sex = st.number_input('Sex (1-male,0-female)',0,1)
        
    with col1:
        cp = st.number_input('Chest Pain types (1-typical angina,2-atypical angina,3-non-aginal pain,4-asympotic)',1,4)
        
    with col2:
        trestbps = st.text_input('Resting Blood Pressure (mmHg(unit))')
        
    with col1:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col2:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl then 1 else 0',0,1)
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results (0-normal,1-having ST-T abnormality,2-left ventricular hyperthorophy',0,2)
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col1:
        exang = st.number_input('Exercise Induced Angina (1-yes,0-no)',0,1)
        
    with col2:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col1:
        slope = st.number_input('Slope of the peak exercise ST segment (1-upsloping,2-flat,3-downsloping)',1,3)
        
    with col2:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect',0,2)

    

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):
        user_input_heart = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, thal]

        user_input_heart = [float(x) for x in user_input_heart]

        heart_prediction = heart_model.predict([user_input_heart])
        
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        
            # Display the test result
            st.success(heart_diagnosis)

            # Display top five doctors for heart disease only if the person has the disease
            if heart_diagnosis == 'The person is having heart disease':
                display_top_five_doctors()
        else:
            heart_diagnosis = 'The person does not have any heart disease'
            st.success(heart_diagnosis)
        


    

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")
    image = Image.open('parkinsons image.jpg')
    st.image(image)


    col1, col2 = st.columns(2)

    with col1:
        fo = st.number_input('MDVP-Fo(Hz) (88-260)',88,260)          
    
    with col2:
        Shimmer = st.number_input('MDVP-Shimmer  (0.00954-0.11908)',0.00954,0.11908)
        
    with col1:
        Shimmer_dB = st.number_input('MDVP-Shimmer(dB) (0.085-1.302)',0.085,1.302)
        
    with col2:
        APQ3 = st.number_input('Shimmer-APQ3 (0.00455-0.05647) ',0.00455,0.05647)
        
    with col1:
        APQ5 = st.number_input('Shimmer-APQ5 (0.0057-0.0794)',0.0057,0.0794)
        
    with col2:
        APQ = st.number_input('MDVP-APQ (0.00719-0.13778)',0.00719,0.13778)
        
    with col1:
        DDA = st.number_input('Shimmer-DDA (0.01364-0.16942)',0.01364,0.16942)
        
    with col2:
        NHR = st.number_input('NHR (0.00065-0.31482)',0.00065,0.31482)
        
    with col1:
        HNR = st.number_input('HNR (8.441-33.047)',8.441,33.047)
        
    with col2:
        RPDE = st.number_input('RPDE (0.25657-0.685151)',0.25657,0.685151)
        
    with col1:
        DFA = st.number_input('DFA (0.574-0.825)',0.574,0.825)
        
    with col2:
        spread1 = st.number_input('spread1 (-7 (to) -2)',-7,-2)
        
    with col1:
        spread2 = st.number_input('spread2 (0.006247-0.450493)',0.006247,0.450493)
        
    with col2:
        D2 = st.number_input('D2 (1.423-3.67115)',1.423,3.67115)    
   
    with col1:
        PPE = st.number_input('PPE (.044539-0.5273)',0.044539,0.5273)  

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        user_input_parkinsons = [fo, Shimmer, Shimmer_dB, APQ3, APQ5,
                             APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input_parkinsons = [float(x) for x in user_input_parkinsons]

        parkinsons_prediction = parkinsons_model.predict([user_input_parkinsons])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        
            # Display the test result
            st.success(parkinsons_diagnosis)

            # Display top five doctors for Parkinson's only if the person has the disease
            if parkinsons_diagnosis == "The person has Parkinson's disease":
                display_top_five_doctors()
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
            st.success(parkinsons_diagnosis)
    
    
    
# Breast Cancer Prediction Page
if selected == "Breast Cancer Prediction":
    # page title
    st.title("Breast Cancer Disease Prediction using ML")
    image = Image.open('breast cancer image.jpg')
    # Resize the image to the desired height while preserving aspect ratio
    desired_height = 200
    width_percent = (desired_height / float(image.size[1]))
    new_width = int((float(image.size[0]) * float(width_percent)))
    resized_image = image.resize((new_width, desired_height))
    
   
    st.image(image)
    
    col1, col2, col3, col4 = st.columns(4)


    with col1:
        mn_radius = st.number_input('Enter the mean Radius (6.5-28.5)',6.5,28.5)

    with col2:
        mn_texture = st.number_input('mean texture (9.5-39.5)',9.5,39.5)

    with col3:
        mn_perimeter = st.number_input('mean perimeter (43-189)',43,189)

    with col4:
        mn_area = st.number_input('mean area (143-2501)',143,2501)

   

    with col1:
        mn_smoothness = st.number_input('mean smoothness (0.05-0.170)',0.05,0.170)
            
        
    with col2:
        mn_compactness = st.number_input('mean compactness (0.019 - 0.345)',0.019,0.345)

    with col3:
        mn_concavity = st.number_input('mean concavity (0.0 - 0.426)',0.0,0.426)

    with col4:
        
        mn_concavepts = st.number_input('mean concave points (0.0 - 0.201)',0.0,0.201)
        

    with col1:
        mn_symmetry = st.number_input('mean symmetry (0.106 - 0.304)',0.106,0.304)

    with col2:
        mn_fractaldm = st.number_input('mean fractal dimension (0.05 - 0.097)',0.05,0.097)

   
                
            
    # code for Prediction

    breast_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Breast Cancer Test Result"):
        user_input_breast = [mn_radius, mn_texture, mn_perimeter, mn_area, mn_smoothness, mn_compactness, 
                             mn_concavity,mn_concavepts, mn_symmetry, mn_fractaldm]

        user_input_breast = [float(x) for x in user_input_breast]

        breast_prediction = breast_model.predict([user_input_breast])

        if breast_prediction[0] == 1:
            breast_diagnosis = "The person has Breast Cancer Disease"
        
            # Display the test result
            st.success(breast_diagnosis)

            # Display top five doctors for breast cancer only if the person has the disease
            if breast_diagnosis == "The person has Breast Cancer Disease":
                display_top_five_doctors()
        else:
            breast_diagnosis = "The person does not have Breast Cancer Disease"
            st.success(breast_diagnosis)