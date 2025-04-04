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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from libraries import *
from charts import *
from transformation import *
from streamlit_helper import *


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

#####################################################################################################
    cluster_name = 'AUR_cluster' 

    df["Sustainability"] = pd.to_numeric(df["Sustainability"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Sustainability"])

    # Sidebar slider for selecting sustainability range
    min_sust, max_sust = 0, 7  
    sustainability_range = st.sidebar.slider(
        "Select a Sustainability Score Range:",
        min_sust, max_sust, (min_sust, max_sust)
    )

    # Apply clustering and get grouped brand scores
    cluster_name = "AUR_cluster"
    df_grouped = sustainability_cluster(df, cluster_name, sustainability_range)

    # Filter brands based on the selected sustainability range
    df_sustainability = df_grouped[
        (df_grouped["Sustainability"] >= sustainability_range[0]) & 
        (df_grouped["Sustainability"] <= sustainability_range[1])
    ]
    
# print(df_filtered[[context, sustainability_field, cluster_name]])

######################################################################################################
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

    with st.container(height=600):
        st.plotly_chart(fig, use_container_width=True)
    add_line()

    
    col = 'Dominant colour'
    _title = 'C. Distribution for ' + col
    fig = single_pie_chart_color(dff, col, _title, height=500, width=500)

    with st.container(height=500):
        st.plotly_chart(fig, use_container_width=True)
    add_line()

    dff2 = streamlit_sidebar_selections_B(dff)
    df_brand_scores = calculate_brand_positioning(dff2)
    _list = list(dff2['Brand_D2C'].unique())
    
    _title = "D : Color Distribution Brand wise"
  # Get filtered brands

    if len(_list) == 0:
        st.warning("No data available after filtering. Please adjust your selections.")
    else:
        # st.subheader("D : Color Distribution Brand wise")

        # Define layout dynamically (2 columns for better view)
        num_cols = min(2, len(_list))  # Max 2 columns
        cols = st.columns(num_cols)
        col = 'Dominant colour'
        num_rows = (len(_list) + num_cols - 1) // num_cols
        
        fig2 = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=_list,
                                 specs=[[{"type": "domain"} for _ in range(num_cols)] for _ in range(num_rows)],

                             )

        for idx, brand in enumerate(_list):
            with cols[idx % num_cols]:  # Distribute charts across columns
                df_filtered = dff2[dff2['Brand_D2C'] == brand]
                pie = single_pie_chart_color(df_filtered, col, f"{brand}", 400, 400, trace=True)
                
                # Add the pie trace directly to the subplot
                fig2.add_trace(
                    pie,
                    row=idx // num_cols + 1,
                    col=idx % num_cols + 1,
                )

        with st.container(height=400*num_rows):
            # st.plotly_chart(fig2, use_container_width=True)
            fig2.update_layout(height=400*num_rows,title_text = _title )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")  


            
    # _title = "E. Price - Complete Distrbution"
    # st.markdown(f'<p style="color:black;font-size:16px;font-weight:bold;border-radius:2%;"> '+_title+'</p>', unsafe_allow_html=True)
    # st.dataframe(df_percentile.style.apply(highlight_dataframe_cells, axis=1),width=1000, height=600)



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

###########################################################################################################################
    
    context = 'Brand_D2C'
    sustainability_field = "Sustainability"
    chart_df = df_sustainability[[context, sustainability_field, cluster_name]]
    chart_df = chart_df.sort_values(context, ascending=True).reset_index(drop=True)
    chart_df['size'] = 1

    # Generate scatter plot with fixed x-axis range
    fig = display_scatter_chart(
        chart_df, _description="E1. Brand Sustainability Score", 
        x=sustainability_field, y=context, z='size', w='square-open', 
        v=None, width=1200, height=800, color_discrete_sequence=['white']
    )

    fig.update_xaxes(range=[sustainability_range[0], sustainability_range[1]])
    fig.update_layout(yaxis={"showticklabels": False})
    # Add brand images to the plot
    fig = add_brand_image_to_sustainability(
        fig, chart_df=chart_df, context=context, measure_field=sustainability_field, 
        clusterName=cluster_name, add_vline='Yes'
    )

    # Display the plot
    with st.container(height=800):
        st.plotly_chart(fig, use_container_width=True)

    add_line()

    
    
    # dff2 = streamlit_sidebar_selections_B(dff)
    
    
    col='new_Type'
    chart_df = transformLevel2(dff, col)
    # chart_df_product_count = chart_df.groupby('Brand_D2C')['product_count'].sum().reset_index()
    # st.dataframe(chart_df_product_count,width=270, height=220)

    fig = display_scatter_chart(chart_df, _description="F. Brand Assortment Comparison for " + col, x='Brand_D2C', y=col, z='product_count%', w='circle', v='Brand_D2C', width=1200, height=600)

    with st.container(height=600):
        st.plotly_chart(fig, use_container_width=True)
        
    add_line()
    
    # _list = list(dff2['Brand_D2C'].unique())
    
    # _title = "G : Color Distribution Brand wise"
    # if len(_list)>=4:
    #     # print("list : ",_list)
    #     _list = _list[:4]
    #     col = 'Dominant colour'
    #     dff2 = dff2[dff2['Brand_D2C'].isin(_list)]
    #     fig = multi_pie_chart_color(dff2, col, _list, _title = _title, height=800, width=800)
        
    #     with st.container(height=800):
    #         st.plotly_chart(fig, use_container_width=True)
            
    #     add_line()
        
    # elif len(_list)>=2:
    #     _list = _list[:2]
    #     col = 'Dominant colour'
    #     dff2 = dff2[dff2['Brand_D2C'].isin(_list)]
    #     fig = multi_pie_chart_color(dff2, col, _list, _title = _title, height=500, width=500)

    #     with st.container(height=500):
    #         st.plotly_chart(fig, use_container_width=True)
    #     add_line()
        
    # elif len(_list)==1:
    #     _title = "G : Color Distrbution for " + _list[0]
    #     col = 'Dominant colour'
    #     dff2 = dff2[dff2['Brand_D2C'].isin(_list)]
    #     fig = single_pie_chart_color(dff2, col, _title, height=500, width=500)
        
    #     with st.container(height=500):
    #         st.plotly_chart(fig, use_container_width=True)
        
    #     add_line()
    # else:
    #     pass

    ##############-------Gender mix------------------#####################
    
    dff2 = processed_gender_mix(dff)

    _title = "G1 : Gender-Mix Distribution Brand wise"

    if len(_list) == 0:
        st.warning("No data available after filtering. Please adjust your selections.")
    else:
        # st.subheader("G1 : Gender-Mix Distribution Brand wise")

        # Define layout dynamically (2 columns for better view)
        num_cols = min(2, len(_list))  # Max 2 columns
        cols = st.columns(num_cols)
        col = 'Gender-Mix'
        num_rows = (len(_list) + num_cols - 1) // num_cols
        
        fig2 = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=_list,
                                 specs=[[{"type": "domain"} for _ in range(num_cols)] for _ in range(num_rows)],

                             )

        for idx, brand in enumerate(_list):
            with cols[idx % num_cols]:  # Distribute charts across columns
                df_filtered = dff2[dff2['Brand_D2C'] == brand]
                pie = single_pie_chart_color(df_filtered, col, f"{brand}", 400, 400, trace=True)
                
                pie.update(marker={'colors':['#FF6347', '#4682B4', '#32CD32']})

                
                # Add the pie trace directly to the subplot
                fig2.add_trace(
                    pie,
                    row=idx // num_cols + 1,
                    col=idx % num_cols + 1,
                )

        with st.container(height=400*num_rows):
            # st.plotly_chart(fig2, use_container_width=True)
            fig2.update_layout(height=400*num_rows,title_text=_title)
            st.plotly_chart(fig2, use_container_width=True)

        # st.markdown("---")  
        
        add_line()

 
    dff2 = processed_collaborations(dff)

    _title = "G2 : Collaborations"

    if len(_list) == 0:
        st.warning("No data available after filtering. Please adjust your selections.")
    else:
        # st.subheader("G2 : Collaborations")

        # Define layout dynamically (2 columns for better view)
        num_cols = min(2, len(_list))  # Max 2 columns
        cols = st.columns(num_cols)
        col = 'Collaborations'
        num_rows = (len(_list) + num_cols - 1) // num_cols
        
        fig2 = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=_list,
                                 specs=[[{"type": "domain"} for _ in range(num_cols)] for _ in range(num_rows)],

                             )

        for idx, brand in enumerate(_list):
            with cols[idx % num_cols]:  # Distribute charts across columns
                df_filtered = dff2[dff2['Brand_D2C'] == brand]
                pie = single_pie_chart_color(df_filtered, col, f"{brand}", 400, 400, trace=True)
                
                pie.update(marker={'colors':['#FF6347', '#4682B4', '#32CD32']})

                
                # Add the pie trace directly to the subplot
                fig2.add_trace(
                    pie,
                    row=idx // num_cols + 1,
                    col=idx % num_cols + 1,
                )

        with st.container(height=400*num_rows):
            # st.plotly_chart(fig2, use_container_width=True)
            fig2.update_layout(height=400*num_rows,title_text=_title)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")  
        
    # if len(_list) >= 4:
    #     _list = _list[:4]
    #     col = 'Collaborations'
    #     dff2 = dff2[dff2['Brand_D2C'].isin(_list)]
    #     fig = multi_pie_chart_collaboration(dff2, col, _list, _title=_title, height=800, width=800)

    #     with st.container(height=800):
    #         st.plotly_chart(fig, use_container_width=True)
        
    #     add_line()

    # elif len(_list) >= 2:
    #     _list = _list[:2]
    #     col = 'Collaborations'
    #     dff2 = dff2[dff2['Brand_D2C'].isin(_list)]
    #     fig = multi_pie_chart_collaboration(dff2, col, _list, _title=_title, height=500, width=500)

    #     with st.container(height=500):
    #         st.plotly_chart(fig, use_container_width=True)
        
    #     add_line()

    # elif len(_list) == 1:
    #     _title = f"G : Collaborations Distribution for {_list[0]}"
    #     col = 'Collaborations'
    #     dff2 = dff2[dff2['Brand_D2C'].isin(_list)]
    #     fig = single_pie_chart_color(dff2, col, _title, height=500, width=500)

    #     with st.container(height=500):
    #         st.plotly_chart(fig, use_container_width=True)
        
    #     add_line()
            


    _title = "H : Product Image Snapshot"
    col1 = 'Product Image'
    col2 = 'Product URL'
    product_image = list(dff2[col1].unique())[:100]

    st.markdown(f'<p style="color:black;font-size:16px;font-weight:bold;border-radius:2%;"> '+_title+'</p>', unsafe_allow_html=True)
    with st.container():
        show_product_image_and_URL(dff2, col1, col2, product_image)

    add_line()


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

    st.markdown('<h3 style="font-size:16px; font-weight:bold;">Brand Positioning Analysis</h3>', unsafe_allow_html=True)
    i = 1
    for (x_col, y_col) in category_pairs:
        _title = f'J.{i}.{x_col} vs {y_col}'
        
        st.markdown(f'<p style="color:purple;font-size:16px;font-weight:bold;border-radius:2%;"> {_title}</p>', unsafe_allow_html=True)
        
        fig = plot_brand_positioning(df_relative_scores, x_col, y_col)
        st.plotly_chart(fig, use_container_width=True)

        context = 'Brand_D2C'
        chart_df = df_percentile[[context, pricing_cluster_field, clusterName]]
        chart_df = chart_df.sort_values(context, ascending=True).reset_index()
        chart_df.drop(columns='index', inplace=True)
        chart_df['size'] = 1
        fig = display_scatter_chart(chart_df, _description="E. Brand Cluster basis Pricing for " + pricing_cluster_field, x=pricing_cluster_field, 
                                    y=context, z='size' , w='square-open', v=None, width=1200, height=800, color_discrete_sequence=['white'])

        fig.update_layout(yaxis={"showticklabels": False})
        fig = add_brand_image_to_scatter(fig, chart_df=chart_df, context=context, measure_field=pricing_cluster_field, clusterName=clusterName, add_vline='Yes')


        add_line()
        i += 1
    
    # except:
    #     st.write('Please make a selection!')
