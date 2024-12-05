
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to preprocess images
def scale_images(data_frame):
    image_list = []
    for index, data_row in data_frame.iterrows():
        # Reshape and normalize bands
        band1 = np.array(data_row['band_1']).reshape(75, 75)
        band2 = np.array(data_row['band_2']).reshape(75, 75)
        combined_band = band1 + band2

        normalized_band1 = (band1 - band1.mean()) / (band1.max() - band1.min())
        normalized_band2 = (band2 - band2.mean()) / (band2.max() - band2.min())
        normalized_combined_band = (combined_band - combined_band.mean()) / (combined_band.max() - combined_band.min())

        # Stack the normalized bands
        image_list.append(np.dstack((normalized_band1, normalized_band2, normalized_combined_band)))

    return np.array(image_list)

# Streamlit UI
st.title("Iceberg vs Ship Classifier")

# Initialize session state
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "index" not in st.session_state:
    st.session_state.index = None

# File uploader for the dataset
uploaded_file = st.file_uploader("Upload train.json or test.json", type="json", key="file_uploader")

if uploaded_file:
    # Store uploaded file in session state
    st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file:
    # Load and preprocess the dataset
    data = pd.read_json(st.session_state.uploaded_file)
    st.write(f"Dataset loaded: {len(data)} entries.")
    images = scale_images(data)

    # Load the trained model
    model = load_model("Iceberg_Ship_Classification.keras")

    # Check if it's train.json or test.json based on column presence
    if "is_iceberg" in data.columns:
        # Handle train.json
        st.write("Detected train.json: Processing labeled data.")
        labels = np.array(data['is_iceberg'])

        # Perform train-test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            images, labels, test_size=0.2, random_state=1
        )

        # Enter an index to classify
        col1, col2 = st.columns(2)
        with col1:
            index = st.number_input(
                "Enter an index to classify (0-based, corresponding to the test set):",
                min_value=0,
                max_value=len(X_test) - 1,
                value=st.session_state.index if st.session_state.index is not None else 0,
                key="index_input"
            )
        with col2:
            if st.button("Clear Index"):
                st.session_state.index = None

        # Display prediction
        if st.button("Classify Image"):
            st.session_state.index = index  # Save index in session state
            selected_image = np.expand_dims(X_test[index], axis=0)
            prediction = model.predict(selected_image)[0][0]

            predicted_label = "Iceberg" if prediction >= 0.5 else "Ship"
            actual_label = "Iceberg" if Y_test[index] == 1 else "Ship"

            # Display results
            st.write(f"Index {index}:")
            st.write(f"Predicted as: {predicted_label}")
            st.write(f"Actual: {actual_label}")

            # Check correctness
            if predicted_label == actual_label:
                st.success(f"The prediction is CORRECT.")
            else:
                st.error(f"The prediction is INCORRECT.")

            # Show the image
            fig, ax = plt.subplots()
            ax.imshow(X_test[index])
            ax.axis('off')
            ax.set_title(f"Prediction: {predicted_label}")
            st.pyplot(fig)

    else:
        # Handle test.json (unlabeled data)
        st.write("Detected test.json: Processing unlabeled data.")

        # Enter an index to classify
        col1, col2 = st.columns(2)
        with col1:
            index = st.number_input(
                "Enter an index to classify (0-based, corresponding to the dataset):",
                min_value=0,
                max_value=len(images) - 1,
                value=st.session_state.index if st.session_state.index is not None else 0,
                key="index_input_test"
            )
        with col2:
            if st.button("Clear Index"):
                st.session_state.index = None

        # Display prediction
        if st.button("Classify Image"):
            st.session_state.index = index  # Save index in session state
            selected_image = np.expand_dims(images[index], axis=0)
            prediction = model.predict(selected_image)[0][0]

            probability_of_iceberg = prediction
            probability_of_ship = 1 - prediction
            predicted_label = "Iceberg" if probability_of_iceberg >= 0.5 else "Ship"

            # Display results
            st.write(f"Index {index}:")
            st.write(f"Predicted as: {predicted_label}")
            st.write(f"Probability of Iceberg: {probability_of_iceberg:.2f}")
            st.write(f"Probability of Ship: {probability_of_ship:.2f}")

            # Show the image
            fig, ax = plt.subplots()
            ax.imshow(images[index])
            ax.axis('off')
            ax.set_title(f"Prediction: {predicted_label}")
            st.pyplot(fig)
    