import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.neighbors import BallTree
import os
import requests
import folium
from streamlit_folium import st_folium

# --- Page Configuration ---
st.set_page_config(
    page_title="Singapore HDB Resale Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
# This ensures the variables exist across re-runs
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# --- OneMap API Authentication & Functions ---
@st.cache_resource
def get_onemap_token():
    try:
        email = st.secrets["ONEMAP_EMAIL"]
        password = st.secrets["ONEMAP_PASSWORD"]
    except (KeyError, FileNotFoundError):
        email = "troykueh@gmail.com"
        password = "Itstroy5834@"
        
    url = "https://www.onemap.gov.sg/api/auth/post/getToken"
    payload = {"email": email, "password": password}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get('access_token')
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get OneMap token: {e}")
        return None

@st.cache_data
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
    if df_poi_clean.empty: return df_flats # Return if no valid POIs
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

# --- Calculation Logic (when form is submitted) ---
if submitted:
    location_query = f"{block} {street_name}"
    with st.spinner(f"Getting coordinates for {location_query}..."):
        lat, lon = search_location(location_query, ACCESS_TOKEN)

    if lat is None or lon is None:
        st.error(f"Could not find coordinates for '{location_query}'. Please check the address.")
        st.session_state.prediction_results = None # Clear previous results on error
    else:
        with st.spinner('Finding nearest amenities and making prediction...'):
            # Create DataFrame for Display and POI calculation
            df_for_display = pd.DataFrame({'latitude': [lat], 'longitude': [lon]})
            for poi_df, name_col, prefix in ALL_POIS:
                df_for_display = add_nearest_poi_info(df_for_display, poi_df, name_col, prefix)

            # Create DataFrame for Model Prediction
            df_for_prediction = pd.DataFrame()
            df_for_prediction['town'] = [town]
            df_for_prediction['flat_type'] = [flat_type]
            df_for_prediction['storey_range'] = [storey_range]
            df_for_prediction['flat_model'] = [flat_model]
            df_for_prediction['floor_area_sqm'] = [floor_area_sqm]
            df_for_prediction['lease_commence_date'] = [lease_commence_date]
            df_for_prediction['remaining_lease_years'] = [lease_years + lease_months / 12.0]
            
            for _, _, prefix in ALL_POIS:
                dist_col_name = f"dist_{prefix}_m"
                df_for_prediction[dist_col_name] = df_for_display[dist_col_name]

            prediction = model_pipeline.predict(df_for_prediction)
            
            # *** SAVE RESULTS TO SESSION STATE ***
            st.session_state.prediction_results = {
                "price": prediction[0],
                "display_df": df_for_display,
                "location_query": location_query,
                "primary_coords": (lat, lon)
            }

# --- Display Logic (runs on every script re-run) ---
if st.session_state.prediction_results:
    results = st.session_state.prediction_results
    predicted_price = results["price"]
    df_for_display = results["display_df"]
    location_query = results["location_query"]
    lat, lon = results["primary_coords"]

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
        st.dataframe(pd.DataFrame(poi_results), height=500)

    with res_col2:
        st.subheader("Interactive Location Map")
        
        m = folium.Map(location=(lat, lon), zoom_start=16, tiles=None)

        folium.TileLayer(
            tiles='https://www.onemap.gov.sg/maps/tiles/Default/{z}/{x}/{y}.png',
            attr='<img src="https://www.onemap.gov.sg/web-assets/images/logo/om_logo.png" style="height:20px;width:20px;"/> <a href="https://www.onemap.gov.sg/" target="_blank">OneMap</a> © contributors',
            name='OneMap Default'
        ).add_to(m)

        folium.Marker(
            location=(lat, lon),
            popup=folium.Popup(f"<b>HDB Location</b><br>{location_query}", max_width=250),
            icon=folium.Icon(color='blue', icon='home')
        ).add_to(m)

        for _, _, prefix in ALL_POIS:
            if f"lat_{prefix}" in df_for_display.columns:
                poi_name = df_for_display.iloc[0][f"nearest_{prefix}"]
                poi_lat = df_for_display.iloc[0][f"lat_{prefix}"]
                poi_lon = df_for_display.iloc[0][f"lon_{prefix}"]
                poi_dist = df_for_display.iloc[0][f"dist_{prefix}_m"]
                
                popup_html = f"<b>{poi_name}</b><br>({prefix.replace('_', ' ').title()})<br>Distance: {poi_dist:.0f} m"
                
                folium.Marker(
                    location=[poi_lat, poi_lon],
                    popup=folium.Popup(popup_html, max_width=250),
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)

        st_folium(m, width=700, height=512, key="map_results")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info("This app predicts HDB resale prices and shows nearby amenities using a machine learning model and the OneMap API.")
