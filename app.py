import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.neighbors import BallTree
import os
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="Singapore HDB Resale Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- OneMap API Authentication & Functions ---
@st.cache_resource
def get_onemap_token():
    try:
        email = "troykueh@gmail.com"
        password = "Itstroy5834@"
    except KeyError:
        st.error("OneMap credentials not found. Please add ONEMAP_EMAIL and ONEMAP_PASSWORD to your Streamlit secrets.")
        return None
    url = "https://www.onemap.gov.sg/api/auth/post/getToken"
    payload = {"email": email, "password": password}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get('access_token')
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get OneMap token: {e}")
        return None

def search_location(location, token):
    if not token: return None, None
    url = "https://www.onemap.gov.sg/api/common/elastic/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"searchVal": location, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1}
    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results: return None, None
        first = results[0]
        return float(first["LATITUDE"]), float(first["LONGITUDE"])
    except (requests.exceptions.RequestException, KeyError, ValueError):
        return None, None

# --- Caching Data Loading ---
@st.cache_data
def load_data():
    base_path = 'data/'
    return {
        "main": pd.read_csv(os.path.join(base_path, 'output.csv')),
        "bus_stops": pd.read_csv(os.path.join(base_path, 'bus_stop_locations.csv')),
        "pei": pd.read_csv(os.path.join(base_path, 'cpe_pei_premises.csv')),
        "jc": pd.read_csv(os.path.join(base_path, 'jc_locations.csv')),
        "kindergartens": pd.read_csv(os.path.join(base_path, 'kindergartens.csv')),
        "primary_schools": pd.read_csv(os.path.join(base_path, 'primary_school_locations.csv')),
        "secondary_schools": pd.read_csv(os.path.join(base_path, 'secondary_school_locations.csv')),
        "polys": pd.read_csv(os.path.join(base_path, 'poly_locations.csv')),
        "libraries": pd.read_csv(os.path.join(base_path, 'libraries.csv')),
        "malls": pd.read_csv(os.path.join(base_path, 'mall_locations.csv')),
        "hospitals": pd.read_csv(os.path.join(base_path, 'moh_hospitals.csv')),
        "mrt_stations": pd.read_csv(os.path.join(base_path, 'mrt_stations.csv')),
        "sports_facilities": pd.read_csv(os.path.join(base_path, 'sportsg_sport_facilities.csv')),
        "hawker_centres": pd.read_csv(os.path.join(base_path, 'ssot_hawkercentres.csv'))
    }

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('models/model_pipeline.joblib')
    except FileNotFoundError:
        return None

# --- Main App Logic ---
ACCESS_TOKEN = get_onemap_token()
dataframes = load_data()
model_pipeline = load_model()

if model_pipeline is None:
    st.error("Model file not found. Please ensure `models/model_pipeline.joblib` exists.")
    st.stop()

# --- Helper Functions for POI Calculation ---
def find_lat_lon_cols(df):
    lower = {c.lower(): c for c in df.columns}
    lat = lower.get("latitude") or lower.get("lat")
    lon = lower.get("longitude") or lower.get("long") or lower.get("lon")
    if not lat or not lon: raise KeyError(f"Could not find lat/long in {list(df.columns)}")
    return lat, lon

def add_nearest_poi_info(df_flats, df_poi, name_col, poi_prefix):
    lat_col, lon_col = find_lat_lon_cols(df_poi)
    df_poi_clean = df_poi.dropna(subset=[lat_col, lon_col]).copy()
    poi_rad = np.deg2rad(df_poi_clean[[lat_col, lon_col]].values)
    tree = BallTree(poi_rad, metric="haversine")
    flats_rad = np.deg2rad(df_flats[["latitude", "longitude"]].values)
    dist_rad, idx = tree.query(flats_rad, k=1)
    
    nearest_indices = idx.flatten()
    df_flats[f"nearest_{poi_prefix}"] = df_poi_clean.iloc[nearest_indices][name_col].values
    df_flats[f"dist_{poi_prefix}_m"] = dist_rad.flatten() * 6_371_000
    df_flats[f"lat_{poi_prefix}"] = df_poi_clean.iloc[nearest_indices][lat_col].values
    df_flats[f"lon_{poi_prefix}"] = df_poi_clean.iloc[nearest_indices][lon_col].values
    return df_flats

# *** THIS IS THE FIX ***
# Manually define the POIs with the correct prefixes to match the trained model
ALL_POIS = [
    (dataframes["bus_stops"], "name", "bus_stop"),
    (dataframes["pei"], "Name", "pei"),
    (dataframes["jc"], "name", "jc"),
    (dataframes["kindergartens"], "Name", "kindergarten"),
    (dataframes["primary_schools"], "name", "primary_school"),
    (dataframes["secondary_schools"], "name", "secondary_school"),
    (dataframes["polys"], "name", "poly"),
    (dataframes["libraries"], "Name", "library"),
    (dataframes["malls"], "name", "mall"),
    (dataframes["hospitals"], "Name", "hospital"),
    (dataframes["mrt_stations"], "name", "mrt_station"),
    (dataframes["sports_facilities"], "Name", "sports_facility"),
    (dataframes["hawker_centres"], "Name", "hawker_centre"),
]

# --- UI Design ---
st.title("🏠 Singapore HDB Resale Price Predictor")
st.markdown("Enter HDB flat details to get an estimated resale price and see a map of nearby amenities.")

with st.form("prediction_form"):
    st.header("Flat Details")
    form_col1, form_col2, form_col3 = st.columns(3)
    with form_col1:
        town = st.selectbox("Town/Estate", options=sorted(dataframes['main']['town'].unique()))
        street_name = st.text_input("Street Name", value="ANG MO KIO AVE 10")
        block = st.text_input("Block Number", value="406")
    with form_col2:
        flat_model = st.selectbox("Flat Model", options=sorted(dataframes['main']['flat_model'].unique()))
        storey_range = st.selectbox("Storey Range", options=sorted(dataframes['main']['storey_range'].unique()))
        flat_type = st.selectbox("Flat Type", options=sorted(dataframes['main']['flat_type'].unique()))
    with form_col3:
        floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=20.0, max_value=300.0, value=67.0)
        lease_commence_date = st.number_input("Lease Commence Date (Year)", min_value=1960, max_value=2025, value=1978)
        st.markdown("<h6>Remaining Lease</h6>", unsafe_allow_html=True)
        lease_years = st.number_input("Years", min_value=0, max_value=99, value=60)
        lease_months = st.number_input("Months", min_value=0, max_value=11, value=7)

    submitted = st.form_submit_button("Predict & Generate Map")

# --- Prediction & Map Generation Logic ---
if submitted:
    location_query = f"{block} {street_name}"
    with st.spinner(f"Getting coordinates for {location_query}..."):
        lat, lon = search_location(location_query, ACCESS_TOKEN)

    if lat is None or lon is None:
        st.error(f"Could not find coordinates for '{location_query}'. Please check the address.")
    else:
        st.success(f"Found coordinates: Latitude={lat:.5f}, Longitude={lon:.5f}")
        
        # --- Create DataFrame for Display ---
        df_for_display = pd.DataFrame({'latitude': [lat], 'longitude': [lon]})
        with st.spinner('Finding nearest amenities...'):
            for poi_df, name_col, prefix in ALL_POIS:
                df_for_display = add_nearest_poi_info(df_for_display, poi_df, name_col, prefix)

        # --- Create DataFrame for Prediction ---
        df_for_prediction = pd.DataFrame()
        
        # Add all features the model was trained on
        df_for_prediction['town'] = [town]
        df_for_prediction['flat_type'] = [flat_type]
        df_for_prediction['storey_range'] = [storey_range]
        df_for_prediction['flat_model'] = [flat_model]
        df_for_prediction['floor_area_sqm'] = [floor_area_sqm]
        df_for_prediction['lease_commence_date'] = [lease_commence_date]
        df_for_prediction['remaining_lease_years'] = [lease_years + lease_months / 12.0]
        
        # Add all distance features from the display dataframe
        for _, _, prefix in ALL_POIS:
            dist_col_name = f"dist_{prefix}_m"
            df_for_prediction[dist_col_name] = df_for_display[dist_col_name]

        with st.spinner('Calculating the estimated price...'):
            prediction = model_pipeline.predict(df_for_prediction)
            predicted_price = prediction[0]

        st.header("Prediction Results")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Estimated Resale Price (SGD)", value=f"${predicted_price:,.2f}")
            st.subheader("Nearest Amenities")
            poi_results = []
            for _, _, prefix in ALL_POIS:
                poi_results.append({
                    "Amenity": prefix.replace('_', ' ').title(),
                    "Name": df_for_display.iloc[0][f"nearest_{prefix}"],
                    "Distance (m)": f"{df_for_display.iloc[0][f'dist_{prefix}_m']:.0f}"
                })
            st.dataframe(pd.DataFrame(poi_results), height=400)

        with res_col2:
            st.subheader("Location Map")
            flat_marker = f'[{lat},{lon},"blue","H"]'
            poi_markers = '|'.join([f'[{df_for_display.iloc[0][f"lat_{prefix}"]},{df_for_display.iloc[0][f"lon_{prefix}"]},"red",""]' for _, _, prefix in ALL_POIS])
            all_markers = f"{flat_marker}|{poi_markers}"
            
            map_url = f"https://www.onemap.gov.sg/api/staticmap/getStaticImage?layerchosen=default&lat={lat}&lng={lon}&zoom=16&width=600&height=512&points={all_markers}"
            st.image(map_url, caption="Blue: HDB Flat, Red: Nearest Amenities")
            
# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info("This app predicts HDB resale prices and shows nearby amenities using a machine learning model and the OneMap API.")
