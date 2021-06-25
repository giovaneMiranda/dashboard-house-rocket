import pandas as pd
import streamlit as st
import numpy as np
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster
import geopandas
import plotly.express as px
from datetime import datetime


@st.cache(allow_output_mutation=True)
def get_geodata(url):
    geofile = geopandas.read_file(url)

    return geofile


@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)

    return data


def transformation_data(data):
    data['date'] = pd.to_datetime(data['date']).dt.strftime("%m/%d/%y")
    data['price_sqm'] = data['price'] / sqm_converter(data['sqft_lot'])

    return data


def sqm_converter(foot):
    return foot/10.764

# Filter source data
def apply_filter(data, filter_columns, filter_zip):
    if (filter_zip != []) & (filter_columns != []):
        df_filtered = data.loc[data['zipcode'].isin(filter_zip), filter_columns]

    elif (filter_zip != []) & (filter_columns == []):
        df_filtered = data.loc[data['zipcode'].isin(filter_zip), :]

    elif (filter_zip == []) & (filter_columns != []):
        df_filtered = data.loc[:, filter_columns]

    else:
        df_filtered = data.copy()

    return df_filtered


def generator_base_map(data):
    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                             default_zoom_star=15)

    return density_map


def show_map(df_house_map, geofile_areas):
    c1, c2 = st.beta_columns((1, 1))

    # Map Density overview
    c1.header('House Density')
    df_test = df_house_map.sample(10)
    # base map
    density_map = generator_base_map(df_house_map)
    make_cluster = MarkerCluster().add_to(density_map)
    for index, row in df_house_map.iterrows():
        folium.Marker(location=[row['lat'], row['long']],
                      popup='Price: {}. Data: {}. Bedrooms: {}.'.format(row['price'],
                                                                        row['date'],
                                                                        row['bedrooms'])).add_to(make_cluster)
    with c1:
        folium_static(density_map)
    # Region Price Map
    c2.header('Price Density')
    df_region = df_house_map[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df_region.columns = ['ZIP', 'PRICE']

    geofile_areas = geofile_areas[geofile_areas['ZIP'].isin(df_region['ZIP'].unique())]
    region_price_map = generator_base_map(df_house_map)
    folium.Choropleth(data=df_region,
                      name="choropleth",
                      geo_data=geofile_areas,
                      columns=['ZIP', 'PRICE'],
                      key_on='feature.properties.ZIP',
                      fill_color='YlGnBu',
                      fill_opacity=0.7,
                      line_opacity=0.5,
                      legend_name='AVG Price').add_to(region_price_map)
    with c2:
        folium_static(region_price_map)

        return None


# Generate line of price per year and days
def generate_graph_attributes(df_house):
    st.title('Commercial Attributes')
    temporal_serie = st.selectbox('Temporal Serie', ['Year', 'Days'], index=0)

    if temporal_serie == 'Year':
        min_yr_built = int(df_house['yr_built'].min())
        max_yr_built = int(df_house['yr_built'].max())

        filter_year_built = st.slider('Year Built', min_value=min_yr_built, max_value=max_yr_built, value=max_yr_built)
        df_house_filtered = df_house[df_house['yr_built']<= filter_year_built]

        price_by_year = df_house_filtered[['yr_built', 'price']].groupby('yr_built').mean().reset_index()
        fig_year = px.line(price_by_year, x='yr_built', y='price')

        st.header('Average Price per Year Built')
        st.plotly_chart(fig_year, use_container_width=True)

    else:
        min_date = datetime.strptime(df_house['date'].min(), '%m/%d/%y')
        max_date = datetime.strptime(df_house['date'].max(), '%m/%d/%y')

        df_house['date'] = pd.to_datetime(df_house['date'])
        filter_date = st.slider('Date', min_value=min_date, max_value=max_date, value=min_date)

        df_house_filtered = df_house[df_house['date'] <= filter_date]

        price_by_day = df_house_filtered[['date', 'price']].groupby('date').mean().reset_index()
        fig_day = px.line(price_by_day, x='date', y='price')

        st.header('Average Price per Date')
        st.plotly_chart(fig_day, use_container_width=True)


# Generate histogram based on house attributes
def plot_hist_house(df_house):
    st.title('House Attributes')
    c1, c2 = st.beta_columns((1, 1))

    # house per bedroom
    c1.header('House per Bedrooms')
    fig_bedroom = px.histogram(df_house, x='bedrooms', nbins=19)
    c1.plotly_chart(fig_bedroom, use_container_width=True)

    # House per bathrooms
    c2.header('House per Bathrooms')
    fig_bathroom = px.histogram(df_house, x='bathrooms', nbins=19)
    c2.plotly_chart(fig_bathroom, use_container_width=True)

    c1, c2 = st.beta_columns((1, 1))

    # House per floors
    c1.header('House per Floors')
    fig_floors = px.histogram(df_house, x='floors', nbins=10)
    c1.plotly_chart(fig_floors, use_container_width=True)

    # House per water view
    c2.header('House Water View')
    fig_water_view = px.histogram(df_house, x='waterfront')
    c2.plotly_chart(fig_water_view, use_container_width=True)

    return None


def generate_ui(df_house, geofile_areas):

    st.title('Data Overview')
    # generate filter side bar
    st.sidebar.title('Source Data')
    filter_zip = st.sidebar.multiselect('Enter Zipcode', df_house['zipcode'].unique())
    filter_columns = st.sidebar.multiselect('Enter Columns', df_house.columns)

    st.sidebar.write("""
            This tool supports analysis of United States county level data from a variety of data sources. It is 
            intended to exercise knowledge in both storytelling and data science.

             You can also use our Python code in a scripting environment or query our database directly. Details are at our 
             [GitHub](https://github.com/giovaneMiranda/dashboard_house_rocket). If you find bugs, please reach out or create an issue on our 
             GitHub repository. 

            More documentation and contribution details are at our [GitHub Repository](https://github.com/giovaneMiranda/dashboard_house_rocket).
            """)

    with st.sidebar.beta_expander("Credits"):
        """
        This app is the result of hard work by our team:
        - [Giovane Miranda 👨🏻‍💻](https://www.linkedin.com/in/giovane-miranda-galindo-junior-653a3b17a) 
        
        The analysis and underlying data are provided as open source [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction). 
            """

    checkbox_df = st.checkbox('Show Source Data', value=False)

    # Show source data
    if checkbox_df:
        df_filtered = apply_filter(df_house, filter_columns, filter_zip)

        st.dataframe(df_filtered)

    c1, c2 = st.beta_columns((1, 2))

    # Average metrics
    df_sumorized = df_house.groupby('zipcode').agg({'id': 'count', 'price': np.mean,
                                                'sqft_living': np.mean,
                                                'price_sqm': np.mean}).reset_index()
    df_sumorized.columns = ['Zipcode', 'Total House', 'Price', 'Sqft Living', 'Price/m2']
    c1.header('Average Values')
    c1.dataframe(df_sumorized)

    # Statistic Descriptive
    num_attributes = df_house.select_dtypes(include=['int64', 'float64'])
    df_describe = num_attributes.describe().transpose().reset_index()
    c2.header('Descriptive Analysis')
    c2.dataframe(df_describe)

    # show the house density map and the price density map
    show_map(df_house, geofile_areas)

    generate_graph_attributes(df_house)

    plot_hist_house(df_house)

    return None


if __name__ == '__main__':
    st.set_page_config(
        page_title="House Rocket",
        page_icon="🏠",
        initial_sidebar_state="expanded",
        layout='wide')

    # data extration
    url_geofile = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    path = 'dataset/kc_house_data.csv'
    data_raw = get_data(path)
    geofile = get_geodata(url_geofile)

    data = transformation_data(data_raw)

    generate_ui(data, geofile)
