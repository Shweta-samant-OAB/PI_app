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
        fig.update_layout(yaxis_title="Sub-category")

        with st.container(height=600):
            st.plotly_chart(fig, use_container_width=True)
    add_line()

    with st.expander("**Product Type Analysis**", expanded=True):
        col='new_Type'
        chart_df = transformLevel2(dff2, col)
        fig = display_scatter_chart(chart_df, _description="F. Brand Assortment Comparison for Product Type" , x='Brand_D2C', y=col, z='product_count%', w='circle', v='Brand_D2C', width=1200, height=600)
        fig.update_layout(yaxis_title="Product Type")

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

        keywords = ["collaborat", "endorse", "celebrity", "athlete", "influencer"]
        brand_list = [b.lower() for b in dff['Brand_D2C'].unique()]

        def normalize_collaboration(val):
            val_lower = str(val).strip().lower()
            if (
                val in ["", "none", "none evident", "none evident in the image.", "none evident in the provided images.", "-", "nan"]
                or "none evident" in val
                or val == "none evident in the image"
                or val == "none evident in the provided images"
            ):
                return "None"
            return str(val).strip()

        dff_collab['CollabGroup'] = dff_collab['Collaborations'].apply(normalize_collaboration)
        collab_counts = dff_collab['CollabGroup'].value_counts()
        categories = collab_counts.index
        values = collab_counts.values

        fig = go.Figure(data=[go.Pie(labels=categories, values=values, hole=0.2)])
        fig.update_layout(title_text=_title, height=500, width=500)

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

        # Create product count summary using dff2
        col='new_Sub-category'
        chart_df = transformLevel1(dff2, col)
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

    # AI Generated Summary Section
    st.markdown('<h3 style="font-size:16px; font-weight:bold;">📝 Summary of the Report</h3>', unsafe_allow_html=True)
    
    # Generate detailed summary based on the data
    summary_points = []
    
    # Market Overview
    brand_count = len(df['Brand_D2C'].unique())
    sku_count = df['Product Image'].nunique()
    subcat_count = len(df['new_Sub-category'].unique())
    summary_points.append(f"• Market Overview: The report showcases {brand_count} brands in the UK market region. {sku_count} SKUs are observed across their D2C websites, spanning {subcat_count} distinct sub-categories.")
    
    # Product Type Analysis
    product_types = df['new_Type'].value_counts()
    top_type = product_types.index[0]
    top_type_pct = (product_types.iloc[0] / product_types.sum() * 100).round(1)
    second_type = product_types.index[1]
    second_type_pct = (product_types.iloc[1] / product_types.sum() * 100).round(1)
    third_type = product_types.index[2]
    third_type_pct = (product_types.iloc[2] / product_types.sum() * 100).round(1)
    summary_points.append(f"• Product Mix: The assortment primarily consists of {top_type} ({top_type_pct}%), {second_type} ({second_type_pct}%), and {third_type} ({third_type_pct}%), indicating a strong focus on these categories. This distribution suggests a balanced approach to product diversification.")
    
    # Pricing Analysis
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    median_price = df['Price'].median()
    min_price = df['Price'].min()
    max_price = df['Price'].max()
    price_range = f"${min_price:.0f} - ${max_price:.0f}"
    price_std = df['Price'].std()
    summary_points.append(f"• Price Positioning: Products are median priced at ${median_price:.0f}, with a range spanning {price_range}. The standard deviation of ${price_std:.0f} indicates {('significant' if price_std > 100 else 'moderate')} price variation across the market, reflecting diverse market positioning and target segments.")
    
    # Color Analysis
    color_dist = df['Dominant colour'].value_counts()
    top_colors = color_dist.head(3)
    color_pct = (top_colors.sum() / color_dist.sum() * 100).round(1)
    color_list = ', '.join(top_colors.index)
    color_variety = len(color_dist)
    summary_points.append(f"• Color Trends: {color_list} dominate the color palette, collectively representing {color_pct}% of the assortment. The market offers {color_variety} distinct color options, demonstrating {('a diverse' if color_variety > 10 else 'a focused')} color strategy.")
    
    # Gender Analysis
    gender_dist = df['Gender-Mix'].value_counts()
    top_gender = gender_dist.index[0]
    gender_pct = (gender_dist.iloc[0] / gender_dist.sum() * 100).round(1)
    second_gender = gender_dist.index[1]
    second_gender_pct = (gender_dist.iloc[1] / gender_dist.sum() * 100).round(1)
    third_gender = gender_dist.index[2] if len(gender_dist) > 2 else None
    third_gender_pct = (gender_dist.iloc[2] / gender_dist.sum() * 100).round(1) if len(gender_dist) > 2 else 0
    gender_text = f"• Target Audience: The product mix is primarily aimed at {top_gender} ({gender_pct}%), followed by {second_gender} ({second_gender_pct}%)"
    if third_gender:
        gender_text += f" and {third_gender} ({third_gender_pct}%)"
    gender_text += ". This distribution reflects the market's focus on inclusive product offerings."
    summary_points.append(gender_text)
    
    # Design Language Analysis
    design_columns = ['Modern', 'Bold', 'Minimalistic', 'Classic']
    for col in design_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    design_scores = df[design_columns].mean()
    top_design = design_scores.idxmax()
    second_design = design_scores.sort_values(ascending=False).index[1]
    third_design = design_scores.sort_values(ascending=False).index[2]
    design_text = f"• Design Language: Brands predominantly embrace {top_design} and {second_design} design elements, with {third_design} as a significant secondary influence. This combination reflects contemporary market preferences and evolving consumer tastes."
    summary_points.append(design_text)
    
    # Fashion vs Function Analysis
    df['Fashion-forward'] = pd.to_numeric(df['Fashion-forward'], errors='coerce')
    df['Function-forward'] = pd.to_numeric(df['Function-forward'], errors='coerce')
    fashion_score = df['Fashion-forward'].mean()
    function_score = df['Function-forward'].mean()
    if abs(fashion_score - function_score) < 0.2:
        summary_points.append("• Brand Positioning: Brands demonstrate a balanced approach between fashion and function, with equal emphasis on both aspects. This equilibrium suggests a market that values both aesthetic appeal and practical utility.")
    else:
        dominant = "fashion" if fashion_score > function_score else "function"
        summary_points.append(f"• Brand Positioning: The market shows a {('strong' if abs(fashion_score - function_score) > 0.5 else 'slight')} inclination towards {dominant}-oriented products, while maintaining a balanced portfolio. This indicates a market that prioritizes {dominant} while ensuring versatility in product offerings.")
    
    # Sustainability Analysis
    df['Sustainability'] = pd.to_numeric(df['Sustainability'], errors='coerce')
    sustainability_scores = df.groupby('Brand_D2C')['Sustainability'].mean()
    sustainable_brands = sustainability_scores[sustainability_scores > 5].index.tolist()
    avg_sustainability = df['Sustainability'].mean()
    if sustainable_brands:
        sustainability_text = f"• Sustainability Focus: Brands like {', '.join(sustainable_brands[:3])} lead in sustainability initiatives, while others maintain a neutral stance. With an average sustainability score of {avg_sustainability:.1f}, the market shows {('strong' if avg_sustainability > 5 else 'moderate')} commitment to eco-conscious practices."
    else:
        sustainability_text = f"• Sustainability Focus: The market shows varying levels of commitment to sustainability, with an average score of {avg_sustainability:.1f}. This indicates {('room for improvement' if avg_sustainability < 4 else 'a growing awareness')} in sustainable practices."
    summary_points.append(sustainability_text)

    # Market Insights
    # Price-Value Analysis
    price_value_ratio = df['Price'].mean() / df['Sustainability'].mean()
    summary_points.append(f"• Price-Value Proposition: The market demonstrates a {('strong' if price_value_ratio < 50 else 'moderate')} price-value relationship, with brands balancing premium pricing against product features and sustainability initiatives.")

    # Brand Concentration
    top_brands = df['Brand_D2C'].value_counts().head(3)
    top_brands_pct = (top_brands.sum() / len(df) * 100).round(1)
    summary_points.append(f"• Market Concentration: The top 3 brands account for {top_brands_pct}% of the market, indicating {('a concentrated' if top_brands_pct > 50 else 'a diverse')} market structure with {('established market leaders' if top_brands_pct > 50 else 'opportunities for new entrants')}.")

    # Product Innovation
    unique_combinations = len(df[['new_Type', 'Dominant colour', 'Gender-Mix']].drop_duplicates())
    summary_points.append(f"• Product Innovation: With {unique_combinations} unique product combinations across types, colors, and gender categories, the market shows {('high' if unique_combinations > 50 else 'moderate')} levels of product innovation and customization.")

    # Concluding Analysis
    summary_points.append("\nMarket Outlook and Strategic Implications:")
    summary_points.append("1. Market Maturity: The UK footwear market demonstrates a balanced mix of established players and emerging brands, with a strong focus on product diversification and innovation.")
    summary_points.append("2. Consumer Preferences: The market shows a clear preference for versatile, fashion-forward products that maintain functional aspects, indicating a sophisticated consumer base.")
    summary_points.append("3. Competitive Landscape: The presence of multiple price points and design approaches suggests a healthy competitive environment with room for both premium and value-oriented brands.")
    summary_points.append("4. Future Opportunities: Areas for growth include enhanced sustainability initiatives, expanded gender-neutral offerings, and further innovation in product design and materials.")
    summary_points.append("5. Strategic Recommendations: Brands should focus on balancing fashion and function, expanding sustainable practices, and maintaining competitive pricing while delivering value through innovation and quality.")
    
    # Display the summary
    summary_text = "\n".join(summary_points)
    st.markdown(f'<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; line-height: 1.6;">{summary_text}</div>', unsafe_allow_html=True)

    # except:
    #     st.write('Please make a selection!')
