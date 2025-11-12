import streamlit as st
from utils import load_data, plot_ghi_boxplot, top_regions_table

st.set_page_config(page_title="Solar Dashboard", layout="wide")
st.title("🌞 Solar Challenge Dashboard")
st.write("Explore Global Horizontal Irradiance (GHI) data by country and region.")

# Sidebar filters
st.sidebar.header("Filters")
country_selected = st.sidebar.selectbox("Select a country", ["Togo", "Benin", "Sierra Leone"])
ghi_threshold = st.sidebar.slider("Minimum GHI", min_value=0, max_value=1000, value=200)

# Load CSV data (local)
data = load_data("data/ghi_data.csv")
filtered_data = data[data["GHI"] >= ghi_threshold]

# Boxplot of GHI
st.subheader(f"GHI Distribution for {country_selected}")
fig = plot_ghi_boxplot(filtered_data, country_selected)
st.pyplot(fig)

# Top regions table
st.subheader("Top Regions by Average GHI")
top_regions = top_regions_table(filtered_data, top_n=5)
st.table(top_regions)

# Display count
st.write(f"Number of data points with GHI ≥ {ghi_threshold}: {len(filtered_data)}")
