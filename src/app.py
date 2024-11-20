import os
import base64
import io
import pickle
import random
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from tensorflow.keras.models import load_model
import plotly.express as px
import tempfile
import shutil

# Prevent TensorFlow from using the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the trained models
maize_model1 = load_model("maizeReco.h5", compile=False)
maize_model2 = load_model("maizeReco2.h5", compile=False)

# Load the encoder
with open("encoder_maize.sav", "rb") as f:
    encoder = pickle.load(f)

# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
server = app.server

# Sidebar content
sidebar_content = [
    html.H2("Description", className="display-6", style={"color": "#006400"}),
    html.Hr(style={"borderTop": "3px solid #006400"}),
    html.H4("Blight", style={"color": "#228B22"}),
    html.P("Blight is a general term for plant diseases caused by fungi or bacteria, leading to rapid tissue death."),
    html.H4("Gray Leaf Spot", style={"color": "#228B22"}),
    html.P("Gray Leaf Spot is a fungal disease that causes grayish lesions on maize leaves, leading to yield loss."),
    html.H4("Common Rust", style={"color": "#228B22"}),
    html.P("Common Rust is a fungal infection characterized by reddish-brown pustules on maize leaves."),
    html.H4("Healthy", style={"color": "#228B22"}),
    html.P("A healthy maize plant is free from disease, showing vibrant green leaves and strong growth."),
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
                style={"color": "#006400", "cursor": "pointer"},
            ),
            style={
                "width": "100%",
                "height": "100px",
                "lineHeight": "100px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderColor": "#006400",
                "textAlign": "center",
                "marginBottom": "20px",
            },
            multiple=False,
        ),
        html.Div(id="output-image-upload"),
        html.H3("Drone GPS Tracker", style={"textAlign": "center", "marginTop": "30px"}),
        dcc.Graph(id="drone-gps-map"),
    ],
    style={"padding": "20px"},
)

# App layout
app.layout = dbc.Container(
    dbc.Row([dbc.Col(sidebar, width=4), dbc.Col(content, width=8)]),
    fluid=True,
)

# Classifier function
def classifier(image_path):
    def recogMaize(image_path):
        image = cv2.imread(image_path)  # Read the image from the file path
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized / 255.0
        image_reshaped = np.expand_dims(image_normalized, axis=0)
        predictions1 = maize_model1.predict(image_reshaped, verbose=0)
        predictions2 = maize_model2.predict(image_reshaped, verbose=0)
        combined_predictions = (0.5 * predictions1) + (0.5 * predictions2)
        predicted_class = np.argmax(combined_predictions, axis=1)
        class_name = encoder.inverse_transform(predicted_class)[0]
        confidence_level = combined_predictions[0][predicted_class[0]]
        return class_name, confidence_level

    return recogMaize(image_path)

# Simulate GPS data
def get_gps_coordinates():
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

# Callback for image upload
@app.callback(
    Output("output-image-upload", "children"),
    [Input("upload-image", "contents")],
    [State("upload-image", "filename")],
)
def update_output(content, filename):
    if content is not None:
        try:
            content_type, content_string = content.split(",")
            decoded = base64.b64decode(content_string)

            # Create a temporary directory to save the uploaded image
            temp_dir = tempfile.mkdtemp()
            image_path = os.path.join(temp_dir, filename)

            # Save the uploaded image to the temporary directory
            with open(image_path, "wb") as f:
                f.write(decoded)

            # Now use the saved image path for prediction
            prediction_name, confidence = classifier(image_path)

            # Clean up temporary directory after processing
            shutil.rmtree(temp_dir)

            return html.Div(
                [
                    html.H5(f"Uploaded File: {filename}", style={"color": "#006400", "textAlign": "center"}),
                    html.Img(src=f"data:{content_type};base64,{content_string}", style={"maxWidth": "100%", "marginTop": "20px"}),
                    html.Div(
                        [
                            html.H3(f"Prediction: {prediction_name}", style={"color": "#228B22", "textAlign": "center"}),
                            html.H4(f"Confidence: {confidence * 100:.2f}%", style={"color": "#006400", "textAlign": "center"}),
                        ],
                        style={"marginTop": "20px", "padding": "10px", "border": "2px solid #006400", "borderRadius": "8px"},
                    ),
                ]
            )
        except Exception as e:
            return html.Div(f"Error processing image: {e}", style={"color": "red", "textAlign": "center"})
    return html.Div("No image uploaded yet.", style={"color": "#006400", "textAlign": "center"})

# Callback to update GPS map
@app.callback(
    Output("drone-gps-map", "figure"),
    Input("upload-image", "contents"),
)
def update_gps_map(content):
    latitude, longitude = get_gps_coordinates()
    gps_data = pd.DataFrame({"latitude": [latitude], "longitude": [longitude], "location": ["Drone Location"]})
    fig = px.scatter_geo(
        gps_data,
        lat="latitude",
        lon="longitude",
        hover_name="location",
        projection="natural earth",
        title="Drone GPS Location",
    )
    return fig

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host="0.0.0.0", port=port)
