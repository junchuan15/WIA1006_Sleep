import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler

# Load the saved model
model1 = joblib.load(r"best_sleepefficiency_model.pkl")
model2 = joblib.load(r"best_model_sleep_apnea.pkl")
model3 = joblib.load(r"best_model_insomnia.pkl")

with st.sidebar:
    selected = option_menu(
        "Sweet Dream 💤 : Main Menu",
        ["Introduction", "Sleep Efficiency Predictor", "Sleep Disorder Predictor"],
        default_index=0,
    )
if selected == "Introduction":
    st.title("Sweet Dream 💤")
    st.header("Introduction")
    st.header("__________________________________________")
    text = """
    Welcome to our Sweet Dream - your ultimate sleep companion! We understand the importance of quality sleep for your well-being. With our powerful tool, you'll receive valuable insights into your sleep health. We provide you with the sleep efficiency, which measures how well you're sleeping, along with the percentage risk of sleep apnea and insomnia.

    Our innovative technology analyzes your sleep patterns and health data to calculate your sleep efficiency. This helps you understand how efficiently you're sleeping and identify areas for improvement.

    Additionally, we estimate your percentage risk of developing sleep apnea and insomnia. Sleep apnea, a common disorder involving breathing interruptions during sleep, and insomnia, characterized by difficulty falling or staying asleep, can significantly impact your well-being. By identifying your risks of sleep apnea and insomnia, you can take proactive steps to optimize your sleep routine and seek appropriate support if needed. 

    Discover the power of our Sweet Dream! Gain a better understanding of your sleep quality with the sleep efficiency coefficient and stay ahead of potential sleep apnea and insomnia risks with the percentage risk estimation. Start your journey to better sleep and improved well-being now!
        """
    st.markdown(text)
    st.header("__________________________________________")

elif selected == "Sleep Efficiency Predictor":
    st.title("Sweet Dream 💤")
    st.header("Sleep Efficiency Predictor")

    # Prompt the user for input
    age = st.number_input("Enter your age:", min_value=0, max_value=100)
    gender = st.radio("Choose your gender:", ("Male", "Female"))
    sleep_duration = st.number_input("Enter your sleep duration (in hours):",min_value=0.0,max_value=24.0,value=8.0,)
    valid_input = False
    while not valid_input:
        rem_percentage = st.slider(
            "Percentage of REM sleep:", min_value=0, max_value=100)
        max_deep_percentage = 100 - rem_percentage
        deep_percentage = st.slider(
            "Percentage of deep sleep:",
            min_value=0,
            max_value=max_deep_percentage,
            value=0,
        )
        light_percentage = 100 - rem_percentage - deep_percentage

        # Check if the sum of rem_percentage, deep_percentage, and light_percentage is 100
        sleep_percentage_sum = rem_percentage + deep_percentage + light_percentage
        if sleep_percentage_sum != 100:
            st.warning(
                "The sum of REM, deep, and light sleep percentages should be equal to 100. Please re-enter the values."
            )
        else:
            valid_input = True

    # Display the light sleep percentage as a disabled slider
    light_placeholder = st.empty()

    # Update the displayed value of light sleep percentage
    light_placeholder.text("Percentage of light sleep: {}".format(light_percentage))

    awakenings = st.number_input("Enter the number of awakenings:", min_value=0)
    caffeine_consumption = st.number_input(
        "Enter your caffeine consumption (in mg):", min_value=0.0, value=0.0
    )
    alcohol_consumption = st.number_input(
        "Enter your alcohol consumption (unit):", min_value=0, value=0
    )
    smoking_status = st.selectbox("Choose your smoking status:", ("No", "Yes"))
    exercise_frequency = st.selectbox(
        "Choose your exercise frequency (days per week):", (0, 1, 2, 3, 4, 5, 6, 7)
    )
    
    gender_mapping = {'Male': 1, 'Female': 0}
    smoking_mapping = {'Yes': 1, 'No': 0}
    gender_val = gender_mapping[gender]
    smoking_val = smoking_mapping[smoking_status]
    awakenings_float = float(awakenings) if awakenings else 0.0
    alcohol_float = float(alcohol_consumption) if alcohol_consumption else 0.0
    exercise_frequency_float = float(exercise_frequency) if exercise_frequency else 0.0

   
      # Create a dictionary with the user input
    user_input = {
    "Age": [age],
    "Gender": [gender_val],
    "Sleep duration": [sleep_duration],
    "REM sleep percentage": [rem_percentage],
    "Deep sleep percentage": [deep_percentage],
    "Light sleep percentage": [light_percentage],
    "Awakenings": [awakenings_float],
    "Caffeine consumption": [caffeine_consumption],
    "Alcohol consumption": [alcohol_float],
    "Smoking status": [smoking_val],
    "Exercise frequency": [exercise_frequency_float],
    }

    user_input_df = pd.DataFrame(user_input)
    
    # Display a button to trigger the prediction
    if st.button("Predict Sleep Efficiency"):

       
       # Check the number of features in scaled_user_input_df
        num_features_expected = 11
        num_features_actual = user_input_df.shape[1]

       # Compare the number of features
        if num_features_actual != num_features_expected:
           st.error("The number of features in the input data does not match the expected number of features.")
        else:
           input_array = user_input_df.values
           if len(input_array.shape) == 1:
            input_array = input_array.reshape(1, -1)
           predicted_sleep_efficiency = model1.predict(input_array)

        # Classify the predicted sleep efficiency into categories
           if predicted_sleep_efficiency < 0.3:
                sleep_efficiency_class = "Poor Sleep Efficiency"
                message = "Your predicted sleep efficiency indicates poor sleep quality. We recommend consulting a healthcare professional to assess your sleep health and provide guidance on improving your sleep quality."
           elif predicted_sleep_efficiency <= 0.7:
                sleep_efficiency_class = "Average Sleep Efficiency"
                message = "Your predicted sleep efficiency suggests average sleep quality. We recommend you to maintain a regular sleep schedule and take note of your lifestyle habits such as avoiding alcoholic drinks and exercise regularly."
           else:
                sleep_efficiency_class = "Good Sleep Efficiency"
                message = "Congratulations! Your predicted sleep efficiency indicates good sleep quality. Keep it up!"

            # Display the prediction and sleep efficiency class
           st.markdown(
            "<h1 style='text-align: center; font-size: 24px;'>Sleep Efficiency Percentage: {}%</h1>".format(
                round(predicted_sleep_efficiency[0] * 100, 2)
            ),
            unsafe_allow_html=True,
            )
           st.markdown(
            "<h2 style='text-align: center; font-size: 24px;'>{}</h2>".format(
                sleep_efficiency_class
            ),
            unsafe_allow_html=True,
           )
           st.write(message)
        
            # Print the preprocessed input for debugging
           st.write("User Input:")
           st.write(user_input_df)
           st.write("Number of Features:")
           st.write(num_features_actual)
           st.write("Predicted Sleep Efficiency:")
           st.write(predicted_sleep_efficiency)

elif selected == "Sleep Disorder Predictor":
    st.title("Sweet Dream 💤")
    st.header("Sleep Disorder Predictor")
    st.header("[Imsomnia / Sleep Apnea]")

    # Gender input
    gender = st.radio("Enter your gender:", ("Male", "Female"))

    # Age input
    age = st.number_input("Enter your age:", min_value=0, max_value=100)

    # Occupation input
    occupation_categories = [
        "Accountant",
        "Doctor",
        "Engineer",
        "Lawyer",
        "Manager",
        "Nurse",
        "Sales Representative",
        "Salesperson",
        "Scientist",
        "Software Engineer",
        "Teacher",
    ]
    occupation = st.selectbox("Choose your occupation", occupation_categories)

    # Sleep duration input
    sleep_duration = st.number_input(
        "Enter your sleep duration (in hours):",
        min_value=0.0,
        max_value=24.0,
        value=8.0,
    )

    # Sleep quality input
    sleep_quality = st.slider(
        "Set your quality of sleep (1-10):", min_value=0, max_value=10
    )

    # Physical activity level input
    physical_activity_level = st.slider(
        "Set your physical activity level (1-100):", min_value=0, max_value=100
    )

    # Stress level input
    stress_level = st.number_input(
        "Enter your stress level (1-10):", min_value=0, max_value=10
    )

    # Height and weight inputs
    height = st.number_input(
        "Enter your height (in cm):", min_value=0.0, max_value=300.0
    )
    weight = st.number_input(
        "Enter your weight (in kg):", min_value=0.0, max_value=500.0
    )
    
    
    # BMI calculation function
    def calculate_bmi(height, weight):
      bmi = weight / ((height/100) ** 2)
      return bmi

    # Validate inputs and calculate BMI
    if height > 0 and weight > 0:
      bmi = calculate_bmi(height, weight)
      if bmi < 25.0:
         BMI = "Normal"
      elif 25.0 <= bmi < 30.0:
         BMI = "Overweight"
      else:
         BMI = "Obese"
      st.write("Your BMI is", bmi,"("+BMI+")")
    else:
      st.warning("Please provide valid height and weight values.")


    # Heart rate input
    heartrate = st.number_input(
        "Enter your heart rate (bpm):", min_value=0, max_value=150
    )

    # Daily steps input
    dailySteps = st.number_input("Set your daily steps:", min_value=0, max_value=30000)

    # Systolic and diastolic pressure inputs
    systolic_pressure = st.number_input(
        "Set your systolic pressure:", min_value=0, max_value=200
    )
    diastolic_pressure = st.number_input(
        "Set your diastolic pressure:", min_value=0, max_value=200
    )

    if st.button("Predict Sleep Disease"):
        user_input = {
            "Age": age,
            "Gender": gender,
            "Occupation": "Occupation_" + occupation,
            "Sleep duration": sleep_duration,
            "Quality of Sleep": sleep_quality,
            "Physical Activity Level": physical_activity_level,
            "Stress Level": stress_level,
            "BMI Category": BMI,
            "Heart Rate": heartrate,
            "Daily Steps": dailySteps,
            "Systolic Pressure": systolic_pressure,
            "Diastolic Pressure": diastolic_pressure,
        }

        # Convert 'Male' to 1 and 'Female' to 0 in 'Gender'
        user_input["Gender"] = 1 if user_input["Gender"] == "Male" else 0

        # Define the BMI category mapping
        bmi_mapping = {"Normal": 0, "Overweight": 1, "Obese": 2}

        # Convert 'BMI Category' values using the mapping
        user_input["BMI Category"] = bmi_mapping.get(user_input["BMI Category"], 0)
        occupation_categories = [
            "Occupation_Accountant",
            "Occupation_Doctor",
            "Occupation_Engineer",
            "Occupation_Lawyer",
            "Occupation_Manager",
            "Occupation_Nurse",
            "Occupation_Sales Representative",
            "Occupation_Salesperson",
            "Occupation_Scientist",
            "Occupation_Software Engineer",
            "Occupation_Teacher",
        ]

        # Get the index of the selected occupation
        selected_occupation_index = occupation_categories.index(
            user_input["Occupation"]
        )

        # Create binary features for occupation
        occupation_features = ["Occupation_" + occ for occ in occupation_categories]
        user_input.update(
            {
                occ: 1 if i == selected_occupation_index else 0
                for i, occ in enumerate(occupation_features)
            }
        )

        # Remove the 'Occupation' key from user_input
        user_input.pop("Occupation")

        # Convert user input to a 1D array
        input_array = np.array(list(user_input.values()))

        # Reshape the input array to 2D
        input_array = input_array.reshape(1, -1)

        # Perform Min-Max scaling on numeric features
        numeric_features = [
            "Gender",
            "Age",
            "Sleep duration",
            "Quality of Sleep",
            "Physical Activity Level",
            "Stress Level",
            "BMI Category",
            "Heart Rate",
            "Daily Steps",
            "Systolic Pressure",
            "Diastolic Pressure",
        ]

        scaler = StandardScaler()
        scaled_input = scaler.fit_transform(input_array)

        # Convert the scaled input back to a dictionary
        scaled_input_dict = dict(zip(numeric_features, scaled_input.flatten()))

        # Update the user input dictionary with the scaled values
        user_input.update(scaled_input_dict)

        # Convert user input to a 1D array
        input_array = np.array(list(user_input.values()))

        # Reshape the input array to 2D
        input_array = input_array.reshape(1, -1)

        # Perform the prediction using the trained KNN model
        predicted_sleep_apnea = model2.predict(input_array)
        predicted_sleep_insomnia = model3.predict(input_array)

        # Get the predicted probabilities
        if hasattr(model2, "predict_proba"):
            sleep_apnea_probabilities = model2.predict_proba(input_array)
        else:
            # Handle case when predict_proba is not available
            sleep_apnea_probabilities = None

        if hasattr(model3, "predict_proba"):
            insomnia_probabilities = model3.predict_proba(input_array)
        else:
            # Handle case when predict_proba is not available
            insomnia_probabilities = None

        # Check if probabilities are available and display the probability of class 1
        if sleep_apnea_probabilities is not None:
            sleep_apnea_probability = sleep_apnea_probabilities[0][1]
            st.write("Risk of Sleep Apnea: {:.2%}".format(sleep_apnea_probability))
            if sleep_apnea_probability > 0.5:
                st.write(
                    "High risk of Sleep Apnea. It is recommended to consult a healthcare professional for further evaluation and appropriate treatment."
                )
            else:
                st.write("Low risk of Sleep Apnea.")

        if insomnia_probabilities is not None:
            insomnia_probability = insomnia_probabilities[0][1]
            st.write("Risk of Insomnia: {:.2%}".format(insomnia_probability))
            if insomnia_probability > 0.5:
                st.write(
                    "High risk of Insomnia. It is advisable to consult a healthcare professional or sleep specialist to discuss your sleep difficulties and explore potential treatment options."
                )
            else:
                st.write("Low risk of Insomnia.")
