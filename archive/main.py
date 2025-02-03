# Assortment Analysis

import pandas as pd
import streamlit as st
import os
import numpy as np

from pathlib import Path
from zipfile import ZipFile
from PIL import Image
import random

import plotly.figure_factory as ff
import plotly.express as px

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from libraries import *
from charts import *
from transformation import *
from streamlit_helper import *


import json
import time
import datetime


def brand_visualisation():
    streamlit_setup(_title="Product Intelligence",
                    _description="Learn Brands assortment and Product Differentiation Factors")

    uploaded_file = st.sidebar.file_uploader("Upload data file (.csv format) ", type=["csv"])
    
    try:

        df = pd.read_csv(uploaded_file)
        df = data_initialise(df)
        dff = streamlit_sidebar_selections_A(df)

        pricingfield_inp1 = 'Tentative Pricing'
        pricingfield_oup1 = 'Price_Tentative'
        dff, df_percentile = pricing_initialise(dff, pricingfields = (pricingfield_inp1, pricingfield_oup1))

        pricing_percentile = st.sidebar.selectbox("Percentile Benchmark for Pricing",("0.95", "0.75", "0.50", "0.25"))
        pricing_cluster_field = pricingfield_oup1+"_"+pricing_percentile
        pricing_values = st.sidebar.slider("Select a range of values", 
                                           df_percentile[pricing_cluster_field].min(), 
                                           df_percentile[pricing_cluster_field].max(), 
                                           (df_percentile[pricing_cluster_field].min(), df_percentile[pricing_cluster_field].max())
                                          )

        clusterName = 'AUR_cluster'
        dff, df_percentile = cluster_initialise(dff, df_percentile, clusterName = clusterName, pricing_cluster_field = pricing_cluster_field, 
                                _pricerange = (pricing_values[0], pricing_values[1]))


        image_paths = list(Path.cwd().joinpath("brandLogo").glob('*.jpg'))
        fig = plot_images_side_by_side(image_paths)
        st.pyplot(fig)
        add_line()

        dict = {'Category': df['Category'].unique()[0], 
                'Brands': df['Brand_D2C'].nunique(), 
                'Products': df['Product Image'].nunique(), 
                '': '', 
                '': '', 
               }
        add_markdowns_side_by_side(dict)
        add_line()

        col='new_Sub-category'
        chart_df = transformLevel0(dff, col)
        _title = 'A. Distribution for ' + col
        fig = single_pie_chart_distibution(chart_df, col, 'product_count', _title)

        with st.container(height=500):
            st.plotly_chart(fig, use_container_width=True)

        add_line()


        col='new_Sub-category'
        chart_df = transformLevel1(dff, col)
        fig = display_scatter_chart(chart_df, _description="B. Brand Assortment Comparison for " + col, x='Brand_D2C',  y=col, z='product_count%', w='circle', v='Brand_D2C', width=1200, height=600)

        with st.container(height=700):
            st.plotly_chart(fig, use_container_width=True)
        add_line()


        col = 'Dominant colour'
        _title = 'C. Distribution for ' + col
        fig = single_pie_chart_color(dff, col, _title, height=500, width=500)

        with st.container(height=500):
            st.plotly_chart(fig, use_container_width=True)
        add_line()



        _title = "E. Tentative Price - Complete Distrbution"
        st.markdown(f'<p style="color:black;font-size:16px;font-weight:bold;border-radius:2%;"> '+_title+'</p>', unsafe_allow_html=True)
        st.dataframe(df_percentile.style.apply(highlight_dataframe_cells, axis=1),width=1000, height=600)



        context = 'Brand_D2C'
        chart_df = df_percentile[[context, pricing_cluster_field, clusterName]]
        chart_df = chart_df.sort_values(context, ascending=True).reset_index()
        chart_df.drop(columns='index', inplace=True)
        chart_df['size'] = 1
        fig = display_scatter_chart(chart_df, _description="D. Brand Cluster basis Pricing for " + pricing_cluster_field, x=pricing_cluster_field, 
                                    y=context, z='size' , w='square-open', v=None, width=1200, height=800, color_discrete_sequence=['white'])
        fig = add_brand_image_to_scatter(fig, chart_df=chart_df, context=context, measure_field=pricing_cluster_field, clusterName=clusterName, add_vline='Yes')

        with st.container(height=800):
            st.plotly_chart(fig, use_container_width=True)
        add_line()




        dff2 = streamlit_sidebar_selections_B(dff)

        col='new_Type'
        chart_df = transformLevel2(dff2, col)
        chart_df_product_count = chart_df.groupby('Brand_D2C')['product_count'].sum().reset_index()
        st.dataframe(chart_df_product_count,width=400, height=220)
        

        fig = display_scatter_chart(chart_df, _description="F. Brand Assortment Comparison for " + col, x='Brand_D2C', y=col, z='product_count%', w='circle', v='Brand_D2C', width=1200, height=600)

        with st.container(height=600):
            st.plotly_chart(fig, use_container_width=True)

        add_line()



        _list = list(dff2['Brand_D2C'].unique())


        _title = "G : Color Distrbution Brand wise"
        if len(_list)>=4:
            _list = _list[:4]
            col = 'Dominant colour'
            dff_brand = dff2[dff2['Brand_D2C'].isin(_list)]
            fig = multi_pie_chart_color(dff_brand, col, _list, _title = _title, height=800, width=800)

            with st.container(height=800):
                st.plotly_chart(fig, use_container_width=True)

            add_line()

        elif len(_list)>=2:
            _list = _list[:2]
            col = 'Dominant colour'
            dff_brand = dff2[dff2['Brand_D2C'].isin(_list)]
            fig = multi_pie_chart_color(dff_brand, col, _list, _title = _title, height=500, width=500)

            with st.container(height=500):
                st.plotly_chart(fig, use_container_width=True)
            add_line()

        elif len(_list)==1:
            _title = "G : Color Distrbution for " + _list[0]
            col = 'Dominant colour'
            dff_brand = dff2[dff2['Brand_D2C'].isin(_list)]
            fig = single_pie_chart_color(dff_brand, col, _title, height=500, width=500)

            with st.container(height=500):
                st.plotly_chart(fig, use_container_width=True)

            add_line()
        else:
            pass



        _title = "H : Product Image Snapshot"
        col1 = 'Product Image'
        col2 = 'Product URL'
        product_image = list(dff2[col1].unique())[:100]

        st.markdown(f'<p style="color:black;font-size:16px;font-weight:bold;border-radius:2%;"> '+_title+'</p>', unsafe_allow_html=True)
        show_product_image_and_URL(dff2, col1, col2, product_image)

        add_line()



        col_list = ['Design Elements', 'Aesthetic Type', 'Silhouette', 'Branding Style']
        i=1
        for col in col_list:

            _title = 'I.'+str(i)+'. Distribution for ' + col
            st.markdown(f'<p style="color:green;font-size:16px;font-weight:bold;border-radius:2%;"> '+_title+'</p>', unsafe_allow_html=True)
            df_table = table_view(dff2, col, _title)
            st.dataframe(df_table,width=2000, height=200)
            add_line()
            i=i+1



        col_list = ['Consumer type', 'Target consumer age group', 'Target consumer gender', 'Target consumer socioeconomic background',
                    'Target consumer Lifestyle', 'Target consumer Fashion Style'
                   ]
        i=1
        for col in col_list:

            _title = 'J.'+str(i)+'. Distribution for - ' + col
            st.markdown(f'<p style="color:purple;font-size:16px;font-weight:bold;border-radius:2%;"> '+_title+'</p>', unsafe_allow_html=True)
            df_table = table_view(dff2, col, _title)
            st.dataframe(df_table,width=2000, height=200)
            add_line()
            i=i+1



    except:
        st.write('Please make a selection!')



        
        
# AUTHENTICATE AND VIEW APP
        
with open("authentication.json", 'r') as json_file:
    USER_CREDENTIALS = json.load(json_file)
    
with open("login_logs.json", 'r') as json_file:
    LOGS = json.load(json_file)

def authenticate(username, password):
    return USER_CREDENTIALS.get(username) == password

def main_app():
    brand_visualisation()

def login_page():
    st.title("Authentication")
    st.write("Please contact Team Pulse for login details.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(username, password):
            st.success(f"You have successfully logged in!")
            st.session_state["authenticated"] = True
            time.sleep(1)
            
            user_logged_in_at = str(datetime.datetime.now())[:16]
            LOGS.update({username : user_logged_in_at})
            
            with open("login_logs.json", 'w') as json_file:
                json.dump(LOGS, json_file)
            
            st.write('Press Login again to enter page!')

        else:
            st.error("Invalid username or password")


    
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_page()
else:
    main_app()

 