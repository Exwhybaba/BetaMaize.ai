import os
import base64
import io
import pickle  # Ensure pickle is imported
import random
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from tensorflow.keras.models import load_model
import plotly.express as px

import requests

# Function to download a model or file from a URL
def download_model(url, destination):
    try:
        print(f"Downloading from {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        os.makedirs(os.path.dirname(destination), exist_ok=True)  # Ensure the directory exists
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"Saved to {destination}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")

# Define URLs and paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
model1_path = os.path.join(MODEL_DIR, "maizeReco.h5")
model2_path = os.path.join(MODEL_DIR, "maizeReco2.h5")
encoder_path = os.path.join(MODEL_DIR, "encoder_maize.sav")

model1_url = "https://github.com/Exwhybaba/BetaMaize.ai/raw/main/models/maizeReco.h5"
model2_url = "https://github.com/Exwhybaba/BetaMaize.ai/raw/main/models/maizeReco2.h5"
encoder_url = "https://github.com/Exwhybaba/BetaMaize.ai/raw/main/models/encoder_maize.sav"

# Download models and encoder
download_model(model1_url, model1_path)
download_model(model2_url, model2_path)
download_model(encoder_url, encoder_path)

# Load models
try:
    maize_model1 = load_model(model1_path, compile=False)
    print("Model 1 loaded successfully.")
except Exception as e:
    print(f"Error loading Model 1: {e}")

try:
    maize_model2 = load_model(model2_path, compile=False)
    print("Model 2 loaded successfully.")
except Exception as e:
    print(f"Error loading Model 2: {e}")

# Load encoder
try:
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    print("Encoder loaded successfully.")
except Exception as e:
    print(f"Error loading Encoder: {e}")

# Log paths
print(f"Model 1 Path: {model1_path}")
print(f"Model 2 Path: {model2_path}")
print(f"Encoder Path: {encoder_path}")


# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
server = app.server
sidebar_content = [
    html.H2("Description", className="display-6", style={"color": "#006400"}),
    html.Hr(style={"borderTop": "3px solid #006400"}),
    html.H4("Blight", style={"color": "#228B22"}),
    html.P("Blight is a general term for plant diseases caused by fungi or bacteria, leading to rapid tissue death, particularly affecting the leaves and stems."),  # Removed extra parenthesis
    html.H4("Gray Leaf Spot", style={"color": "#228B22"}),
    html.P("Gray Leaf Spot is a fungal disease that causes grayish lesions on maize leaves, leading to reduced photosynthesis and yield loss."),  # Removed extra parenthesis
    html.H4("Common Rust", style={"color": "#228B22"}),
    html.P("Common Rust is a fungal infection characterized by reddish-brown pustules on maize leaves, which can reduce plant vigor and yield."),  # Removed extra parenthesis
    html.H4("Healthy", style={"color": "#228B22"}),
    html.P("A healthy maize plant is free from disease, showing vibrant green leaves and strong growth.")  # Removed extra parenthesis
]


# Sidebar layout
sidebar = html.Div(
    [
        dbc.Button(
            "Toggle Sidebar",
            id="toggle-sidebar",
            color="success",
            className="mb-3",
            style={"width": "100%"},
        ),
        dbc.Collapse(
            dbc.Col(sidebar_content, style={"padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "8px"}),
            id="sidebar-collapse",
            is_open=True,
        ),
    ],
    style={"height": "100vh", "overflowY": "auto"},
)

# Main content layout
content = dbc.Col(
    [
        html.H1(
            "Maize Disease Recognition System",
            style={"textAlign": "center", "marginBottom": "30px", "color": "#006400"},
        ),
        html.Div(
            "Upload an image of a maize plant leaf to analyze for disease.",
            style={"textAlign": "center", "marginBottom": "20px"},
        ),
        dcc.Upload(
            id="upload-image",
            children=html.Div(
["Drag and Drop or ", html.A("Select Files")],
style = {"color": "#006400", "cursor": "pointer"},
),
style = {
"width": "100%",
"height": "100px",
"lineHeight": "100px",
"borderWidth": "1px",
"borderStyle": "dashed",
"borderColor": "#006400",
"textAlign": "center",
"marginBottom": "20px",
},
multiple = False,
),
html.Div(id="output-image-upload"),

    # GPS Location Map
html.H3("Drone GPS Tracker", style={"textAlign": "center", "marginTop": "30px"}),
dcc.Graph(id='drone-gps-map'),
],
style = {"padding": "20px"},
)

# App layout
app.layout = dbc.Container(
    dbc.Row([dbc.Col(sidebar, width=4), dbc.Col(content, width=8)]),
    fluid=True,
)

# Define classifier function


def classifier(image):
    def recogMaize(image):
        # Resize the image to match the input shape expected by the models
        image_resized = cv2.resize(image, (224, 224))  # Models expect 224x224x3 input

        # Normalize the image (scale pixel values to [0, 1])
        image_normalized = image_resized / 255.0

        # Add a batch dimension (shape: [1, 224, 224, 3])
        image_reshaped = np.expand_dims(image_normalized, axis=0)

        # Get predictions from both models
        predictions1 = maize_model1.predict(image_reshaped, verbose=0)
        predictions2 = maize_model2.predict(image_reshaped, verbose=0)

        # Combine predictions using weighted average
        combined_predictions = (0.5 * predictions1) + (0.5 * predictions2)

        # Get the class index with the highest probability
        predicted_class = np.argmax(combined_predictions, axis=1)

        # Get the class name using the label encoder
        class_name = encoder.inverse_transform(predicted_class)[0]

        # Get the confidence of the predicted class
        confidence_level = combined_predictions[0][predicted_class[0]]

        return class_name, confidence_level

    return recogMaize(image)


# Simulate GPS data (In real application, this data will come from drone's GPS)
def get_gps_coordinates():
    # Randomly generate latitude and longitude for demo
    latitude = random.uniform(-90, 90)
    longitude = random.uniform(-180, 180)
    return latitude, longitude


# Callback for sidebar toggle
@app.callback(
    Output("sidebar-collapse", "is_open"),
    [Input("toggle-sidebar", "n_clicks")],
    [State("sidebar-collapse", "is_open")],
)
def toggle_sidebar(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


# Callback to handle image upload and predictions
@app.callback(
    Output("output-image-upload", "children"),
    [Input("upload-image", "contents")],
    [State("upload-image", "filename")],
)
def update_output(content, filename):
    if content is not None:
        # Decode the uploaded image
        _, content_string = content.split(",")
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))

        # Convert image to array
        image_np = np.array(image)

        # Get prediction and confidence
        prediction_name, confidence = classifier(image_np)

        # Display the result
        return html.Div(
            [
                html.H5(f"Uploaded File: {filename}", style={"color": "#006400", "textAlign": "center"}),
                html.Img(src=content, style={"maxWidth": "100%", "marginTop": "20px"}),
                html.Div(
                    [
                        html.H3(f"Prediction: {prediction_name}", style={"color": "#228B22", "textAlign": "center"}),
                        html.H4(f"Confidence: {confidence * 100:.2f}%",
                                style={"color": "#006400", "textAlign": "center"}),
                    ],
                    style={"marginTop": "20px", "padding": "10px", "border": "2px solid #006400",
                           "borderRadius": "8px"},
                ),
            ]
        )
    return html.Div("No image uploaded yet.", style={"color": "#006400", "textAlign": "center"})


# Callback to update GPS location on map
@app.callback(
    Output('drone-gps-map', 'figure'),
    Input('upload-image', 'contents')  # This is just a trigger to update the map
)
def update_gps_map(content):
    latitude, longitude = get_gps_coordinates()

    # Create a DataFrame with the GPS coordinates
    gps_data = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'location': ['Drone Location']
    })

    # Plot the map using Plotly Express
    fig = px.scatter_geo(
        gps_data,
        lat='latitude',
        lon='longitude',
        hover_name='location',
        projection="natural earth",
        title="Drone GPS Location"
    )
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8030)
