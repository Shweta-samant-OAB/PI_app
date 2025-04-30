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
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from libraries import *
from charts import *
from transformation import *
from streamlit_helper import *

# Import specific functions needed
from charts import (
    single_pie_chart_distibution,
    display_scatter_chart,
    single_pie_chart_color,
    calculate_brand_positioning,
    calculate_relative_scores,
    plot_brand_positioning
)

from transformation import (
    transformLevel0,
    transformLevel1,
    transformLevel2
)

from streamlit_helper import (
    streamlit_setup,
    streamlit_sidebar_selections_A,
    streamlit_sidebar_selections_B,
    add_line,
    add_brand_image_to_scatter,
    show_product_image_and_URL,
    sustainability_cluster,
    add_brand_image_to_sustainability,
    plot_images_side_by_side,
    data_initialise,
    pricing_initialise,
    cluster_initialise
)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    streamlit_setup(_title="Product Intelligence",
                    _description="Learn Brands assortment and Product Differentiation Factors")
    
    filename = "allbrands_gemini_1_5_pro_002_2025_02_20.csv"
    df = pd.read_csv(filename)
    df = df.fillna('')
    df['Price'] = df['Price'].astype(float)
    df['Price'] = df['Price'].astype(str)

    # try:
    df = data_initialise(df)
    dff = streamlit_sidebar_selections_A(df)

    pricingfield_inp1 = 'Price'
    pricingfield_oup1 = 'Actual Price'
    dff, df_percentile = pricing_initialise(dff, pricingfields = (pricingfield_inp1, pricingfield_oup1))
    pricing_percentile = st.sidebar.selectbox("Percentile Benchmark for Pricing",("0.95", "0.75", "0.50", "0.25"))
    pricing_cluster_field = pricingfield_oup1+"_"+pricing_percentile
    pricing_values = st.sidebar.slider("Select a range of values", 
                                        df_percentile[pricing_cluster_field].min(), 
                                        df_percentile[pricing_cluster_field].max(), 
                                        (df_percentile[pricing_cluster_field].min(), df_percentile[pricing_cluster_field].max())
                                        )


    clusterName = 'AUR_cluster'
    dff_price, df_percentile = cluster_initialise(dff, df_percentile, clusterName = clusterName, pricing_cluster_field = pricing_cluster_field, _pricerange = (pricing_values[0], pricing_values[1]))
    

    image_paths = list(Path.cwd().joinpath("brandLogo").glob('*.jpg'))
    fig = plot_images_side_by_side(image_paths)
    st.pyplot(fig)
    add_line()

    # Sustainability data preparation
    cluster_name = 'AUR_cluster' 

    df["Sustainability"] = pd.to_numeric(df["Sustainability"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Sustainability"])

    # Calculate average sustainability score per brand
    df_sustainability = df.groupby('Brand_D2C').agg({
        'Sustainability': 'mean'
    }).reset_index()
    df_sustainability[cluster_name] = "Yes"  # All brands are initially in the cluster

    # Sidebar slider for selecting sustainability range
    min_sust, max_sust = 0, 7  
    sustainability_range = st.sidebar.slider(
        "Select a Sustainability Score Range:",
        min_sust, max_sust, (min_sust, max_sust)
    )

    # Update cluster based on selected range
    df_sustainability[cluster_name] = np.where(
        (df_sustainability["Sustainability"] >= sustainability_range[0]) & 
        (df_sustainability["Sustainability"] <= sustainability_range[1]),
        "Yes",
        "No"
    )

    # Header section with category, brands, and product count
    dict = {
        'Category': f"<span style='color:black'>Category: </span><span style='color:#FF8C00'>{df['Category'].unique()[0]}</span>",
        'Brands': f"<span style='color:black'>Brands: </span><span style='color:#FF8C00'>{df['Brand_D2C'].nunique()}</span>",
        'Products': f"<span style='color:black'>Products: </span><span style='color:#FF8C00'>{df['Product Image'].nunique()}</span>"
    }
    
    # Display summary in a single row using columns with minimal spacing
    cols = st.columns(len(dict), gap="small")
    for i, (key, value) in enumerate(dict.items()):
        with cols[i]:
            st.markdown(f'<p style="font-size:16px;margin:0;padding:0;">{value}</p>', unsafe_allow_html=True)
    
    add_line()

    # Initialize dff2 before using it in sections
    dff2 = streamlit_sidebar_selections_B(dff)
    
    # Apply brand filter to dff2 if any brands are selected
    if 'selected_brands' in st.session_state and st.session_state.selected_brands:
        dff2 = dff2[dff2['Brand_D2C'].isin(st.session_state.selected_brands)]
    
    df_brand_scores = calculate_brand_positioning(dff2)
    df_relative_scores = calculate_relative_scores(df_brand_scores)

    # Define category pairs for brand positioning analysis
    category_pairs = [
        ("Fashion-forward", "Function-forward"),
        ("Minimalistic", "Bold"),
        ("Modern", "Classic"),
        ("Streetwear", "Luxury-Premium")
    ]

    # Add dropdown sections for different analyses
    with st.expander("**Brand Analysis**", expanded=True):
        col='new_Sub-category'
        chart_df = transformLevel0(dff2, col)
        _title = 'A. Distribution for Sub-category'
        fig = single_pie_chart_distibution(chart_df, col, 'product_count', _title)

        with st.container(height=500):
            st.plotly_chart(fig, use_container_width=True)
        add_line()

        col='new_Sub-category'
        chart_df = transformLevel1(dff2, col)
        fig = display_scatter_chart(chart_df, _description="B. Brand Assortment Comparison for Sub-category", x='Brand_D2C',  y=col, z='product_count%', w='circle', v='Brand_D2C', width=1200, height=600)

        with st.container(height=600):
            st.plotly_chart(fig, use_container_width=True)
    add_line()

    with st.expander("**Color Analysis**", expanded=True):
        # Brand selection for individual color analysis
        selected_brands_color = st.sidebar.multiselect(
            "Select Brands for Color Analysis",
            options=dff['Brand_D2C'].unique(),
            default=[]
        )
        
        col = 'Dominant colour'
        _title = 'C. Distribution for ' + col
        fig = single_pie_chart_color(dff, col, _title, height=500, width=500)

        with st.container(height=500):
            st.plotly_chart(fig, use_container_width=True)

        # Show individual Color distributions if selected
        if selected_brands_color:
            add_line()
            st.markdown("##### Individual Brand Color Distributions")
            num_brands = len(selected_brands_color)
            num_rows = (num_brands + 1) // 2
            
            for row in range(num_rows):
                cols = st.columns(2)
                for col_idx in range(2):
                    brand_idx = row * 2 + col_idx
                    if brand_idx < num_brands:
                        brand = selected_brands_color[brand_idx]
                        with cols[col_idx]:
                            dff_brand_color = dff[dff['Brand_D2C'] == brand]
                            _title = f"{brand}"
                            fig = single_pie_chart_color(dff_brand_color, 'Dominant colour', _title, height=400, width=400)
                            st.plotly_chart(fig, use_container_width=True)
    add_line()

    with st.expander("**Sustainability Analysis**", expanded=True):
        # Section E1: Brand Sustainability Score
        context = 'Brand_D2C'
        sustainability_field = "Sustainability"
        chart_df = df_sustainability[[context, sustainability_field, cluster_name]]
        chart_df = chart_df.sort_values(context, ascending=True).reset_index(drop=True)
        chart_df['size'] = 1

        # Generate scatter plot with fixed x-axis range
        fig = display_scatter_chart(
            chart_df, _description="Brand Sustainability Score", 
            x=sustainability_field, y=context, z='size', w='square-open', 
            v=None, width=1200, height=800, color_discrete_sequence=['white']
        )

        # Set fixed x-axis range from 0 to 7
        fig.update_xaxes(range=[0, 7])
        fig.update_layout(yaxis={"showticklabels": False})
        
        # Add brand images to the plot with fixed positions
        fig = add_brand_image_to_sustainability(
            fig, chart_df=chart_df, context=context, measure_field=sustainability_field, 
            clusterName=cluster_name, add_vline='Yes', selected_range=sustainability_range
        )

        # Display the plot
        with st.container(height=800):
            st.plotly_chart(fig, use_container_width=True)
    add_line()

    with st.expander("**Pricing Analysis**", expanded=True):
        # Section E: Brand Cluster basis Pricing
        context = 'Brand_D2C'
        chart_df = df_percentile[[context, pricing_cluster_field, clusterName]]
        chart_df = chart_df.sort_values(context, ascending=True).reset_index()
        chart_df.drop(columns='index', inplace=True)
        chart_df['size'] = 1
        fig = display_scatter_chart(chart_df, _description="E. Brand Cluster basis Pricing for " + pricing_cluster_field, x=pricing_cluster_field, 
                                    y=context, z='size' , w='square-open', v=None, width=1200, height=800, color_discrete_sequence=['white'])

        fig.update_layout(yaxis={"showticklabels": False})
        fig = add_brand_image_to_scatter(fig, chart_df=chart_df, context=context, measure_field=pricing_cluster_field, clusterName=clusterName, add_vline='Yes')

        with st.container(height=800):
            st.plotly_chart(fig, use_container_width=True)
    add_line()

    with st.expander("**Product Type Analysis**", expanded=True):
        col='new_Type'
        chart_df = transformLevel2(dff2, col)
        fig = display_scatter_chart(chart_df, _description="F. Brand Assortment Comparison for " + col, x='Brand_D2C', y=col, z='product_count%', w='circle', v='Brand_D2C', width=1200, height=600)

        with st.container(height=600):
            st.plotly_chart(fig, use_container_width=True)
    add_line()

    with st.expander("**Demographic Analysis**", expanded=True):
        # Brand selection for individual brand view
        selected_brands_individual = st.sidebar.multiselect(
            "Select Brands for Gender & Collaborations Analysis",
            options=dff['Brand_D2C'].unique(),
            default=[]
        )
        
        # Gender mix section
        dff_gender = processed_gender_mix(dff)
        _title = "G1: Overall Gender-Mix Distribution"
        col = 'Gender-Mix'

        fig = single_pie_chart_color(dff_gender, col, _title, height=500, width=500)
        for trace in fig.data:
            trace.marker.colors = ['#FF6347', '#4682B4', '#32CD32']

        with st.container(height=500):
            st.plotly_chart(fig, use_container_width=True)

        # Show individual Gender-Mix distributions if selected
        if selected_brands_individual:
            add_line()
            st.markdown("##### Individual Brand Gender-Mix Distributions")
            num_brands = len(selected_brands_individual)
            num_rows = (num_brands + 1) // 2
            
            for row in range(num_rows):
                cols = st.columns(2)
                for col_idx in range(2):
                    brand_idx = row * 2 + col_idx
                    if brand_idx < num_brands:
                        brand = selected_brands_individual[brand_idx]
                        with cols[col_idx]:
                            dff_brand_gender = processed_gender_mix(dff[dff['Brand_D2C'] == brand])
                            _title = f"{brand}"
                            fig = single_pie_chart_color(dff_brand_gender, 'Gender-Mix', _title, height=400, width=400)
                            for trace in fig.data:
                                trace.marker.colors = ['#FF6347', '#4682B4', '#32CD32']
                            st.plotly_chart(fig, use_container_width=True)

        # Collaborations section
        add_line()
        dff_collab = processed_collaborations(dff)
        _title = "G2: Overall Collaborations Distribution"
        col = 'Collaborations'

        fig = single_pie_chart_color(dff_collab, col, _title, height=500, width=500)
        for trace in fig.data:
            trace.marker.colors = ['#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#FF5722']

        with st.container(height=500):
            st.plotly_chart(fig, use_container_width=True)

        # Show individual Collaborations distributions if selected
        if selected_brands_individual:
            add_line()
            st.markdown("##### Individual Brand Collaborations Distributions")
            num_brands = len(selected_brands_individual)
            num_rows = (num_brands + 1) // 2
            
            for row in range(num_rows):
                cols = st.columns(2)
                for col_idx in range(2):
                    brand_idx = row * 2 + col_idx
                    if brand_idx < num_brands:
                        brand = selected_brands_individual[brand_idx]
                        with cols[col_idx]:
                            dff_brand_collab = processed_collaborations(dff[dff['Brand_D2C'] == brand])
                            _title = f"{brand}"
                            fig = single_pie_chart_color(dff_brand_collab, 'Collaborations', _title, height=400, width=400)
                            for trace in fig.data:
                                trace.marker.colors = ['#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#FF5722']
                            st.plotly_chart(fig, use_container_width=True)
    add_line()

    with st.expander("**Brand Positioning Analysis**", expanded=True):
        i = 1
        for (x_col, y_col) in category_pairs:
            _title = f'J.{i}.{x_col} vs {y_col}'
            
            st.markdown(f'<p style="color:purple;font-size:16px;font-weight:bold;border-radius:2%;"> {_title}</p>', unsafe_allow_html=True)
            
            fig = plot_brand_positioning(df_relative_scores, x_col, y_col)
            st.plotly_chart(fig, use_container_width=True)
            if i < len(category_pairs):
                add_line()
            i += 1
    add_line()

    # Move Section H here
    _title = "H : Product Image Snapshot"
    col1 = 'Product Image'
    col2 = 'Product URL'
    product_image = list(dff2[col1].unique())[:100]

    st.markdown(f'<p style="color:black;font-size:16px;font-weight:bold;border-radius:2%;"> '+_title+'</p>', unsafe_allow_html=True)
    with st.container():
        show_product_image_and_URL(dff2, col1, col2, product_image)

    add_line()

    category_pairs = [
        ("Fashion-forward", "Function-forward"),
        ("Minimalistic", "Bold"),
        ("Modern", "Classic"),
        ("Streetwear", "Luxury-Premium")
    ]

    plot_titles = [
        "Brand Positioning: Fashion vs Function",
        "Brand Positioning: Minimalist vs Bold",
        "Brand Positioning: Modern vs Classic",
        "Brand Positioning: Streetwear vs Luxury"
    ]

    
    df_relative_scores = calculate_relative_scores(df_brand_scores)

    st.markdown('<h3 style="font-size:16px; font-weight:bold;">📊 Tables and Data</h3>', unsafe_allow_html=True)

    with st.expander("Expand to view tables", expanded=False):
        _title = "I. Price - Complete Distrbution"
        st.markdown(f'<p style="color:black;font-size:16px;font-weight:bold;border-radius:2%;"> '+_title+'</p>', unsafe_allow_html=True)
        st.dataframe(df_percentile.style.apply(highlight_dataframe_cells, axis=1),width=1000, height=600)

        chart_df_product_count = chart_df.groupby('Brand_D2C')['product_count'].sum().reset_index()
        st.dataframe(chart_df_product_count,width=270, height=220)


        col_list = ['Design Elements', 'Aesthetic Type', 'Silhouette', 'Branding Style']
        i = 1
        for col in col_list:
            _title = 'I.' + str(i) + '. Distribution for ' + col
            st.markdown(f'<p style="color:green;font-size:16px;font-weight:bold;border-radius:2%;"> ' + _title + '</p>', unsafe_allow_html=True)
            df_table = table_view(dff2, col, _title)

            st.dataframe(df_table, width=2000, height=200)
            add_line()
            i += 1

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

    st.markdown("---")  

    # except:
    #     st.write('Please make a selection!')
