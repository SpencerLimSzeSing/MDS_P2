import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import folium
import math

@st.cache_resource
def load_knn_model():
    return joblib.load('best_knn_model.joblib')

@st.cache_resource
def load_rf_model():
    return joblib.load('best_rf_model.joblib')

@st.cache_resource
def load_xgb_model():
    return joblib.load('best_xgb_model.joblib')

@st.cache_resource
def load_meta_ann_model():
    return tf.keras.models.load_model('Tuned_meta_ann_model.keras', compile=False)
    


import base64
def get_base64_image(file_path):
    """Convert image file to Base64."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Model loading with caching
@st.cache_resource
def load_knn_model():
    return joblib.load('models/best_knn_model.joblib')

@st.cache_resource
def load_rf_model():
    return joblib.load('models/best_rf_model.joblib')

@st.cache_resource
def load_xgb_model():
    return joblib.load('models/best_xgb_model.joblib')

@st.cache_resource
def load_meta_ann_model():
    return tf.keras.models.load_model('models/Tuned_meta_ann_model.keras', compile=False)
    
# Pre-load models (will only load once due to caching)
knn_model = load_knn_model()
rf_model = load_rf_model()
xgb_model = load_xgb_model()
meta_ann = load_meta_ann_model()



base_models = [knn_model, rf_model, xgb_model]
# Category mapping
category_mapping = {
    0: 'Rainfall_Category_No Rain',
    1: 'Rainfall_Category_Moderate Rain',
    2: 'Rainfall_Category_Heavy Rain',
    3: 'Rainfall_Category_Very Heavy Rain'
}

# Streamlit app title
st.title("Rainfall Prediction App")
st.write("Predict rainfall category based on climatic factors.")
st.header("Input Features")


# Function to add a background image
def add_bg_from_local(image_file):
    """
    Adds a background image to a Streamlit app.

    Parameters:
    - image_file: Path to the local image file
    """
    with open(image_file, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()
    
    # Inject CSS to set the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add the background image (call the function)
add_bg_from_local("assets/pexels.jpg")  


# Helper function for styling sections
def styled_section(header, color):
    st.markdown(
        f"""
        <div style="background-color: {color}; padding: 2px 10px; border-radius: 5px; margin-bottom: 10px;">
            <h4 style="margin: 0; color: black;">{header}</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Date Inputs
styled_section("Date", "#f0f8ff")  # Light blue background
col1, col2 = st.columns(2)  # Create two columns for month and day
with col1:
    month = st.number_input("Month (1-12):", min_value=1, max_value=12, step=1)
with col2:
    day = st.number_input("Day (1-31):", min_value=1, max_value=31, step=1)
# Weather Features
styled_section("Weather Features", "#fafad2")  # Light yellow background
# Sunshine and Evaporation
col3, col4 = st.columns(2)
with col3:
    sunshine = st.number_input("Sunshine (hours):", min_value=0.0, max_value=15.0, step=0.1)
    st.caption("The number of hours of bright sunshine in the day.")
with col4:
    evaporation = st.number_input("Evaporation (mm):", min_value=0.0, max_value=150.0, step=0.1)
    st.caption("The evaporation rate in millimeters (mm) for the day.")
# Temperature Features
styled_section("Temperature Features", "#f5f5dc")  # Beige background
col5, col6 = st.columns(2)
with col5:
    min_temp = st.number_input("Min Temperature (°C):", min_value=-10.0, max_value=50.0, step=0.1)
    temp_9am = st.number_input("Temperature at 9am (°C):", min_value=-10.0, max_value=50.0, step=0.1)
with col6:
    max_temp = st.number_input("Max Temperature (°C):", min_value=-10.0, max_value=50.0, step=0.1)
    temp_3pm = st.number_input("Temperature at 3pm (°C):", min_value=-10.0, max_value=50.0, step=0.1)
# Wind Features
styled_section("Wind Features", "#e6e6fa")  # Lavender background
col7, col8, col9 = st.columns(3)
with col7:
    wind_gust_speed = st.number_input("Wind Gust Speed (km/h):", min_value=0.0, max_value=200.0, step=1.0)
    st.caption("The speed (km/h) of the strongest wind gust in the 24 hours to midnight.")
with col8:
    wind_speed_9am = st.number_input("Wind Speed at 9am (km/h):", min_value=0.0, max_value=150.0, step=1.0)
    st.caption("Wind speed (km/hr) averaged over 10 minutes prior to 9am.")
with col9:
    wind_speed_3pm = st.number_input("Wind Speed at 3pm (km/h):", min_value=0.0, max_value=150.0, step=1.0)
    st.caption("Wind speed (km/hr) averaged over 10 minutes prior to 3pm.")
# Humidity Features
styled_section("Humidity Features", "#f0fff0")  # Honeydew background
col10, col11 = st.columns(2)
with col10:
    humidity_9am = st.number_input("Humidity at 9am (%):", min_value=0.0, max_value=100.0, step=1.0)
with col11:
    humidity_3pm = st.number_input("Humidity at 3pm (%):", min_value=0.0, max_value=100.0, step=1.0)
# Atmospheric Pressure Features
styled_section("Atmospheric Pressure Features", "#fff0f5")  # Lavender blush background
col12, col13 = st.columns(2)
with col12:
    pressure_9am = st.number_input("Atmospheric Pressure at 9am (hPa):", min_value=900.0, max_value=1100.0, step=0.1)
with col13:
    pressure_3pm = st.number_input("Atmospheric Pressure at 3pm (hPa):", min_value=900.0, max_value=1100.0, step=0.1)

# Target-encoded mapping for locations
location_target_encoded = {
    'Adelaide': 1.5663539307667422, 'Albany': 2.2638594164456234, 'Albury': 1.9141149119893721,
    'AliceSprings': 0.8828496042216359, 'BadgerysCreek': 2.1931010928961747, 'Ballarat': 1.7400264200792603,
    'Bendigo': 1.6193803559657218, 'Brisbane': 3.144890857323632, 'Cairns': 5.742034805890228,
    'Canberra': 1.7417203042715037, 'Cobar': 1.1273092369477913, 'CoffsHarbour': 5.061496782932611,
    'Dartmoor': 2.1465669612508496, 'Darwin': 5.092452239273411, 'GoldCoast': 3.7693959731543623,
    'Hobart': 1.601819322459222, 'Katherine': 3.2010897435897436, 'Launceston': 2.011988110964333,
    'Melbourne': 1.8700616016427105, 'MelbourneAirport': 1.4519774011299436, 'Mildura': 0.9450615231127371,
    'Moree': 1.6302032235459005, 'MountGambier': 2.087561860772022, 'MountGinini': 3.292260061919505,
    'Newcastle': 3.183891708967851, 'Nhil': 0.9348629700446144, 'NorahHead': 3.387299419597132,
    'NorfolkIsland': 3.127665317139001, 'Nuriootpa': 1.3903429903429902, 'PearceRAAF': 1.66908037653874,
    'Penrith': 2.1753036437246966, 'Perth': 1.906295020357031, 'PerthAirport': 1.761648388168827,
    'Portland': 2.5303738317757007, 'Richmond': 2.1384615384615384, 'Sale': 1.5101666666666667,
    'SalmonGums': 1.0343824027072759, 'Sydney': 3.324543002697033, 'SydneyAirport': 3.009916805324459,
    'Townsville': 3.485591823277283, 'Tuggeranong': 2.1640426951300866, 'Uluru': 0.7843626806833114,
    'WaggaWagga': 1.7099462365591398, 'Walpole': 2.9068463994324225, 'Watsonia': 1.860820273424475,
    'Williamtown': 3.591108499804152, 'Witchcliffe': 2.8956639566395665, 'Wollongong': 3.594902749832327,
    'Woomera': 0.49040454697425606
}
# locations with their corresponding latitudes and longitudes.
location_coordinates = {
    'Albury': (-36.0737, 146.9135),
    'BadgerysCreek': (-33.9209, 150.7253),
    'Cobar': (-31.4980, 145.8383),
    'CoffsHarbour': (-30.2963, 153.1141),
    'Moree': (-29.4632, 149.8419),
    'Newcastle': (-32.9267, 151.7789),
    'NorahHead': (-33.2820, 151.5703),
    'NorfolkIsland': (-29.0333, 167.9500),
    'Penrith': (-33.7516, 150.6958),
    'Richmond': (-33.6000, 150.7500),
    'Sydney': (-33.8688, 151.2093),
    'SydneyAirport': (-33.9399, 151.1753),
    'WaggaWagga': (-35.1175, 147.3671),
    'Williamtown': (-32.7924, 151.8345),
    'Wollongong': (-34.4278, 150.8931),
    'Canberra': (-35.2809, 149.1300),
    'Tuggeranong': (-35.4245, 149.0664),
    'MountGinini': (-35.5292, 148.7750),
    'Ballarat': (-37.5622, 143.8503),
    'Bendigo': (-36.7587, 144.2802),
    'Sale': (-38.1067, 147.0680),
    'MelbourneAirport': (-37.6690, 144.8410),
    'Melbourne': (-37.8136, 144.9631),
    'Mildura': (-34.2049, 142.1376),
    'Nhil': (-36.3313, 141.6547),
    'Portland': (-38.3420, 141.6040),
    'Watsonia': (-37.7169, 145.0842),
    'Dartmoor': (-37.9181, 141.2836),
    'Brisbane': (-27.4698, 153.0251),
    'Cairns': (-16.9186, 145.7781),
    'GoldCoast': (-28.0167, 153.4000),
    'Townsville': (-19.2580, 146.8183),
    'Adelaide': (-34.9285, 138.6007),
    'MountGambier': (-37.8297, 140.7822),
    'Nuriootpa': (-34.4685, 138.9964),
    'Woomera': (-31.1501, 136.8255),
    'Albany': (-35.0270, 117.8847),
    'Witchcliffe': (-34.0185, 115.1013),
    'PearceRAAF': (-31.6674, 116.0148),
    'PerthAirport': (-31.9403, 115.9677),
    'Perth': (-31.9505, 115.8605),
    'SalmonGums': (-32.9833, 121.6333),
    'Walpole': (-34.9762, 116.7333),
    'Hobart': (-42.8821, 147.3272),
    'Launceston': (-41.4332, 147.1441),
    'AliceSprings': (-23.6980, 133.8807),
    'Darwin': (-12.4634, 130.8456),
    'Katherine': (-14.4611, 132.2648),
    'Uluru': (-25.3444, 131.0369)
}
# Create a folium map centered on Australia
m = folium.Map(location=[-25.2744, 133.7751], zoom_start=4)

# Add markers for each location
for location, coord in location_coordinates.items():
    folium.Marker(
        location=coord,
        popup=location,  # Display the location name
        tooltip="Click to select",
    ).add_to(m)
# Embed Google Maps iframe for location selection or display
st.header("Location")
st.write("Use the map below to select your location:")
# Display the map in Streamlit
map_output = st_folium(m, width=700, height=500)
# Get the selected location
selected_location = None
# Default value for location_encoded
location_encoded = None  # Initialize with None to ensure it's defined

# Function to find the closest location
def get_closest_location(lat, lon, location_coordinates, max_distance=0.5):
    """
    Find the closest location within the given max_distance.

    Parameters:
    - lat: Latitude of the clicked location
    - lon: Longitude of the clicked location
    - location_coordinates: Dictionary of locations with their coordinates
    - max_distance: Maximum allowed distance for a match (default is 0.5)

    Returns:
    - The closest location name or None if no match is within max_distance
    """
    min_distance = float("inf")
    closest_location = None
    for loc, coord in location_coordinates.items():
        # Compute Euclidean distance
        distance = math.sqrt((lat - coord[0])**2 + (lon - coord[1])**2)
        if distance < min_distance and distance <= max_distance:  # Check max distance threshold
            min_distance = distance
            closest_location = loc
    return closest_location

# Ensure map_output is valid and contains 'last_clicked'
if map_output and map_output.get('last_clicked'):
    lat, lon = map_output['last_clicked']['lat'], map_output['last_clicked']['lng']
    st.write(f"Clicked coordinates: Latitude = {lat}, Longitude = {lon}")  # Debugging

    # Find the closest location within max distance
    selected_location = get_closest_location(lat, lon, location_coordinates, max_distance=0.5)

    if selected_location:
        st.write(f"**Selected Location:** {selected_location}")
        st.write(f"**Encoded Value:** {location_target_encoded[selected_location]}")
        location_encoded = location_target_encoded[selected_location]
    else:
        st.write("No valid location selected. Try clicking closer to a marker.")
else:
    st.write("Click on a marker to select a location.")

if location_encoded is None:
    st.write("Please select a valid location on the map before proceeding.")
else:
    # Continue processing with the valid location_encoded value
    st.write(f"Using encoded location value: {location_encoded}")


# Dropdown for selecting location
#location = st.selectbox("Select Location:", list(location_target_encoded.keys()))
#location_encoded = location_target_encoded[location]  # Get the target-encoded value for the selected location

# Derived features
dew_point_estimate = min_temp * (humidity_9am / 100)
avg_humidity = (humidity_9am + humidity_3pm) / 2
avg_temperature = (min_temp + max_temp) / 2
temp_range = max_temp - min_temp
temp_sunshine_interaction = max_temp * sunshine
pressure_difference = pressure_3pm - pressure_9am
temp_difference = temp_3pm - temp_9am

# Combine all raw features into a single array for base model predictions
input_features = np.array([[min_temp, max_temp, evaporation, wind_gust_speed, wind_speed_9am,
                            wind_speed_3pm, humidity_9am, humidity_3pm, pressure_9am, pressure_3pm,
                            temp_9am, temp_3pm, month, day, location_encoded, dew_point_estimate,
                            avg_humidity, avg_temperature, temp_range, temp_sunshine_interaction,
                            pressure_difference, temp_difference]])


display_mapping = {
    'Rainfall_Category_No Rain': 'No Rain (0 mm to 5 mm)',
    'Rainfall_Category_Moderate Rain': 'Moderate Rain (5 mm to 20 mm)',
    'Rainfall_Category_Heavy Rain': 'Heavy Rain (20 mm to 50 mm)',
    'Rainfall_Category_Very Heavy Rain': 'Very Heavy Rain (50 mm and higher)'
}



if st.button("Predict"):
    st.write("### Predicting Rainfall Category...")


    # Load models lazily
    #knn_model = load_knn_model()
    #rf_model = load_rf_model()
    #xgb_model = load_xgb_model()
    #meta_ann = load_meta_ann_model()

    # Base model predictions
    knn_pred = knn_model.predict(input_features)[0]
    rf_pred = rf_model.predict(input_features)[0]
    xgb_pred = xgb_model.predict(input_features)[0]

    # Simplified meta-features
    meta_features = np.array([[knn_pred, rf_pred, xgb_pred]])
    meta_pred = meta_ann.predict(meta_features)
    meta_pred_class = np.argmax(meta_pred, axis=1)[0]
    meta_pred_label = category_mapping[meta_pred_class]
    meta_pred_display = display_mapping[meta_pred_label]

    # Load Base64 icon dynamically
    icon_mapping = {
        'Rainfall_Category_No Rain': get_base64_image('assets/rain1.png'),
        'Rainfall_Category_Moderate Rain': get_base64_image('assets/rain2.png'),
        'Rainfall_Category_Heavy Rain': get_base64_image('assets/rain3.png'),
        'Rainfall_Category_Very Heavy Rain': get_base64_image('assets/rain4.png')
    }
    icon_file = icon_mapping[meta_pred_label]  # Get corresponding icon file

    # Display result with icon
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: flex-start;">
            <h3 style="margin: 0; color: white;">Predicted Category: {meta_pred_display}</h3>
            <img src="data:image/png;base64,{icon_file}" width="100" style="margin-left: 10px;"/>
        </div>
        """,
        unsafe_allow_html=True
    )

