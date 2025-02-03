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
    
    filename = 'allBrands_gemini_1_5_pro_002_2024_10_08.csv'
    
    df, df_percentile,price_col  = initialise(filename = filename, pricingfield='Tentative Pricing', _percentile ='0.75', _pricerange = (176,299))

    filters= create_filters(df)

    try:
        dff, df_subcat, df_PT = streamlit_sidebar_selections(df)
        
        col='new_Type'
        # dff = apply_filters(dff, 'subcat_filters')
        chart_df = transformLevel2(df_subcat, col)
        fig1 = display_scatter_chart(chart_df, _description="Brand Assortment Comparison for " + col, x='Brand_D2C', y=col, z='product_count%', w='circle', v='Brand_D2C', width=1200, height=1000)
        st.plotly_chart(fig1, use_container_width=True)
        add_line()
        

    except:
        print('Please make a selection!')