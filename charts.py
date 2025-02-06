

import plotly.figure_factory as ff
import plotly.express as px

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from chart_dependencies import *
import matplotlib.pyplot as plt
import random
import streamlit as st


def display_scatter_chart(df_, _description, x, y, z, w, v, width, height, color_discrete_sequence=px.colors.qualitative.Light24):
    if df_.empty is False:
        fig = px.scatter(df_, x=x, y=y, title=_description, size=z, color=v, size_max=40,
                         color_discrete_sequence=color_discrete_sequence)
        fig.update_traces(marker=dict(symbol=w))
        fig.update_layout(autosize=False, width=width, height=height, plot_bgcolor='white',paper_bgcolor='white',
                          xaxis=dict(showgrid=True,  gridcolor='#EDF1F2'),yaxis=dict(showgrid=True,  gridcolor='#EDF1F2'))
        return fig


def display_multi_scatter_chart(df_, _description, x, y, z, w, v, width, height, color_discrete_sequence=px.colors.qualitative.Light24):
    if df_.empty is False:
        fig = px.scatter(df_, x=x, y=y, title=_description, size=z, color=v, size_max=40,
                         color_discrete_sequence=color_discrete_sequence)
        fig.update_traces(marker=dict(symbol=w))
        fig.update_layout(autosize=False, width=width, height=height, plot_bgcolor='white',paper_bgcolor='white',xaxis=dict(showgrid=True,  gridcolor='#EDF1F2'),yaxis=dict(showgrid=True,  gridcolor='#EDF1F2'))
        return fig



def display_bar_chart(df_, _description, x, y, v, height, width):
    if df_.empty is False:
        fig = px.bar(df_, x=x, y=y, title=_description, color=v, color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(autosize=False, width=width, height=height, plot_bgcolor='white',paper_bgcolor='white', xaxis=dict(showgrid=True,  gridcolor='#EDF1F2'),yaxis=dict(showgrid=True,  gridcolor='#EDF1F2'))
        return fig


def display_frequency_bar_chart(df_, _description, col, height, width):
    if df_.empty is False:
        dftemp = get_word_combination_frequency(df_, col)
        dftemp['color'] = 1
        x=col
        y= 'word_freq'
        v= 'color'
        fig = px.bar(dftemp, x=x, y=y, title=_description, color=v, color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(autosize=False, width=width, height=height, plot_bgcolor='white',paper_bgcolor='white', xaxis=dict(showgrid=True,  gridcolor='#EDF1F2'),yaxis=dict(showgrid=True,  gridcolor='#EDF1F2'))
        return fig



def single_pie_chart_color(df_, col, _title, height, width):
    
    if df_.empty is False:
        dftemp = color_frequency(df_, col)
    
        categories = dftemp[col]
        values = dftemp['word_freq']
        colors = dftemp['color_HEX']

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Pie(labels=categories, values=values, marker=dict(colors=colors), hole=0.1))      
        fig.update_traces(marker=dict(line=dict(color='black', width=0.1)))
        fig.update_layout(title_text=_title, height=height, width=width, plot_bgcolor='white',paper_bgcolor='white',
                          xaxis=dict(showgrid=True,  gridcolor='#EDF1F2'),yaxis=dict(showgrid=True,  gridcolor='#EDF1F2'))
        
        return fig



def table_view(df_, col, _title):
    
    if df_.empty is False:
        
        _brand_list = df_['Brand_D2C'].unique()

        dfcol = pd.DataFrame()
        for _brand in _brand_list:
            dftemp = get_word_combination_frequency(df_, col, _brand)
            dfcol = pd.concat([dfcol, dftemp], axis=0)

        dfcol = dfcol.reset_index()
        dfcol.drop(columns='index', inplace=True)
        print(dfcol)
            
        return dfcol
    
def aesthetic_table_view(df_, col, _title):
    if df_.empty is False:
        
        _brand_list = df_['Brand_D2C'].unique()

        dfcol = pd.DataFrame()

        for _brand in _brand_list:
            # Get the word combination frequency data for each brand
            dftemp = get_word_combination_frequency(df_, col, _brand)
            
            # if col == 'Aesthetic Type':  # Check if column is Aesthetic Type
                # Extract only the top 4 aesthetic categories and their percentages
            dftemp['Aesthetic Type'] = dftemp['Aesthetic Type'].apply(lambda x: extract_aesthetic_data(x))
            
            dfcol = pd.concat([dfcol, dftemp], axis=0)

        dfcol = dfcol.reset_index()
        dfcol.drop(columns='index', inplace=True)
            
        return dfcol

def extract_aesthetic_data(aesthetic_str):
    """Extract modern, bold, minimalist, casual, and their percentages from the string."""
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        aesthetic_dict = ast.literal_eval(aesthetic_str)
        
        # Extract only the relevant aesthetics
        categories = ['modern', 'bold', 'minimalist', 'casual']
        result = {category: aesthetic_dict.get(category, '0.0%') for category in categories}
        
        return str(result)  
    except Exception as e:
        print(f"Error in extracting aesthetic data: {e}")
        return str({})  # Return empty dictionary if there was an error

def single_pie_chart_term_distribution(df_, col, _title, height, width):
    
    if df_.empty is False:
        dftemp = get_word_combination_frequency(df_, col)
    
        categories = dftemp[col]
        values = dftemp['word_freq']

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Pie(labels=categories, values=values, marker=dict(colors=px.colors.qualitative.Light24), hole=0.5))      
        fig.update_traces(marker=dict(line=dict(color='black', width=0.1)))
        fig.update_layout(title_text=_title, height=height, width=width, plot_bgcolor='white',paper_bgcolor='white',
                          xaxis=dict(showgrid=True,  gridcolor='#EDF1F2'),yaxis=dict(showgrid=True,  gridcolor='#EDF1F2'))
        
        return fig


def single_pie_chart_distibution(dftemp, dimension, measure, _title):
    
    if dftemp.empty is False:

        categories = dftemp[dimension]
        values = dftemp[measure]
        total = str(int(dftemp[measure].sum()))
        
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Pie(labels=categories, values=values, marker=dict(colors=px.colors.qualitative.Bold), hole=0.6))      
        fig.update_traces(marker=dict(line=dict(color='black', width=0.2)))
        fig.update_layout(title_text=_title, height=500, width=500, plot_bgcolor='white',paper_bgcolor='white',
                          xaxis=dict(showgrid=True,  gridcolor='#EDF1F2'),yaxis=dict(showgrid=True,  gridcolor='#EDF1F2'))
        fig.update_layout(annotations=[dict(text=total, x=0.5, y=0.5, font_size=20, font_color="black", showarrow=False)])
        
        return fig



def multi_pie_chart_color(df_, col, _list, _title, height, width):
    
    if df_.empty is False:
        df_Dict = color_frequency_brand(df_, col, _list)
        len(df_Dict)
    
        categories = [None]*len(df_Dict)
        values = [None]*len(df_Dict)
        colors = [None]*len(df_Dict)
        
        i = 0
        for k in list(df_Dict.keys()):
            print(k)
            categories[i] = df_Dict[k][col]
            values[i] = df_Dict[k]['word_freq']
            colors[i] = df_Dict[k]['color_HEX']
            i=i+1
        
        
        n = int((i+1)/2)
        
        specs = []
        sp = [{'type':'domain'}, {'type':'domain'}]
        for s in range(n):
            specs.append(sp)
        
        fig = make_subplots(rows=n, cols=2, specs=specs, subplot_titles=tuple(df_Dict.keys()))
        p = 0
        for m in range(n):
            fig.add_trace(go.Pie(labels=categories[p], values=values[p], name=_list[p], marker=dict(colors=colors[p]), hole=0.1), row=m+1, col=1)
            p = p+1
            fig.add_trace(go.Pie(labels=categories[p], values=values[p], name=_list[p], marker=dict(colors=colors[p]), hole=0.1), row=m+1, col=2)
            p = p+1
        
        fig.update_traces(marker=dict(line=dict(color='black', width=0.1)))
        
        fig.update_layout(title_text=_title,height=height, width=width, plot_bgcolor='white',paper_bgcolor='white',
                          xaxis=dict(showgrid=True,  gridcolor='#EDF1F2'),yaxis=dict(showgrid=True,  gridcolor='#EDF1F2'))
        return fig



def histogram_pricing_chart(df_, BrandList, col):

    if df_.empty is False:

        df_Dict, values = pricing_frequency(df_, BrandList, col)
        
        n = int(len(BrandList)/2)

        fig = make_subplots(rows=n, cols=2, subplot_titles=tuple(df_Dict.keys()))
        p = 0
        for m in range(n):
            fig.add_trace(go.Histogram(x=df_Dict[BrandList[p]][col], name=BrandList[p], nbinsx=40,
                                       marker=dict(color=px.colors.qualitative.Set2[p])), row=m+1, col=1 )
            p = p+1
            fig.add_trace(go.Histogram(x=df_Dict[BrandList[p]][col], name=BrandList[p], nbinsx=40,
                                       marker=dict(color=px.colors.qualitative.Set2[p])), row=m+1, col=2 )
            p = p+1
        
        fig.update_layout(title_text="E : "+col,height=700, width=1000,
                          plot_bgcolor='white',paper_bgcolor='white')
    
        return fig

def calculate_brand_positions(df, col="Product Story"):
    """
    Calculate brand positioning based on keyword occurrences in the specified column.
    """
    if "Brand_D2C" not in df.columns or col not in df.columns:
        st.error(f"Required columns 'Brand_D2C' and '{col}' not found in dataset.")
        return {}

    df[col] = df[col].fillna("").astype(str).str.lower()

    categories = {
        "Fashion": ["fashion", "style", "trend", "elegance", "luxury", "chic", "glamour"],
        "Function": ["performance", "utility", "comfort", "sport", "active", "support", "durable"],
        "Design Story": ["design", "innovation", "story", "aesthetic", "art", "creativity", "signature"],
        "Seasonal Use": ["seasonal", "weather", "cold", "summer", "rain", "winter", "hot", "snow", "heat", "spring", "fall"]
    }

    brand_scores = {}

    for brand, group in df.groupby("Brand_D2C"):
        product_story = " ".join(group[col])
        category_counts = {cat: sum(product_story.count(word) for word in words) for cat, words in categories.items()}
        
        total_words = sum(category_counts.values())
        category_percentages = {cat: count / total_words if total_words > 0 else 0 for cat, count in category_counts.items()}

        x_value = (category_percentages["Fashion"] - category_percentages["Function"])
        y_value = (category_percentages["Design Story"] - category_percentages["Seasonal Use"])

        brand_scores[brand] = (x_value, y_value)

    if not brand_scores:
        st.warning("No valid brand positions calculated. Check your dataset.")
        return {}

    x_vals, y_vals = zip(*brand_scores.values())
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    brand_positions = {
        brand: (
            2 * (x - x_min) / (x_max - x_min) - 1 if x_max != x_min else 0,
            2 * (y - y_min) / (y_max - y_min) - 1 if y_max != y_min else 0
        )
        for brand, (x, y) in brand_scores.items()
    }

    return brand_positions


# Function to generate color-coded brand positioning plot
def plot_brand_positions(brand_positions):
    if not brand_positions:
        st.warning("No data available for plotting.")
        return

    unique_colors = list(plt.cm.tab10.colors)  
    brand_list = list(brand_positions.keys())
    random.shuffle(unique_colors)  
    brand_colors = {brand: unique_colors[i % len(unique_colors)] for i, brand in enumerate(brand_list)}

    x_coords, y_coords = zip(*brand_positions.values())

    fig, ax = plt.subplots(figsize=(10, 10))
    
    for brand, (x, y) in brand_positions.items():
        ax.scatter(x, y, color=brand_colors[brand], alpha=0.8, label=brand, s=100)  # s=100 makes points bigger
        ax.text(x, y, brand, fontsize=9, ha="right", color="black")

    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")

    ax.text(0.8, 1.05, "Fashion & Design Story", fontsize=12, fontweight="bold", color="black")
    ax.text(-1.0, 1.05, "Function & Design Story", fontsize=12, fontweight="bold", color="black")
    ax.text(-1.0, -1.05, "Function & Seasonal", fontsize=12, fontweight="bold", color="black")
    ax.text(0.8, -1.05, "Fashion & Seasonal", fontsize=12, fontweight="bold", color="black")

    ax.set_xlabel("Fashion (+) vs Function (-)")
    ax.set_ylabel("Design Story (+) vs Seasonal (-)")
    ax.set_title("Brand Positioning Matrix")

    # Add legend
    ax.legend(loc="upper right", fontsize=9, title="Brands", frameon=True, bbox_to_anchor=(1.3, 1.0))

    # Display plot
    st.pyplot(fig)


