# Assortment Analysis

import pandas as pd
import streamlit as st
import os
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px


from libraries import *
from charts import *
from transformation import *
from streamlit_helper import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    streamlit_setup(_title="Product Intelligence",
                    _description="Learn Brands assortment and Product Differentiation Factors")

    filename = 'allBrands_gemini_1_5_pro_002_2024_10_12.csv'
    
    df, df_percentile, price_col  = initialise(filename = filename, pricingfield='Tentative Pricing', _percentile ='0.75', _pricerange = (176,299))

    filters= create_filters(df)


    try:
        dff, df_subcat, df_PT = streamlit_sidebar_selections(df)
        
        context = 'Brand_D2C'
        cluster = 'AUR_cluster'
        chart_df = df_percentile[[context, price_col, cluster]]
        fig1 = display_scatter_chart(chart_df, _description="Brand Cluster basis Pricing for " + price_col, x=price_col, y=context, z=price_col , w='circle', v=cluster, width=1200, height=600, color_discrete_sequence=['teal','coral'])
        st.plotly_chart(fig1, use_container_width=True)
        add_line()
    
        _title = "Tentative Price - Complete Distrbution"
        st.markdown(f'<p style="color:black;font-size:16px;font-weight:bold;border-radius:2%;"> '+_title+'</p>', unsafe_allow_html=True)
        st.dataframe(df_percentile,width=1000, height=600)

    except:
        print('Please make a selection!')