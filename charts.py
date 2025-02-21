

import plotly.figure_factory as ff
import plotly.express as px

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from chart_dependencies import *
import matplotlib.pyplot as plt
import random
import streamlit as st
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import StandardScaler
import plotly.colors as pc



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
        # if col == 'Aesthetic Type':
        #     dfcol.to_excel('temp1.xlsx', index=False)  # Ensure index is False for cleaner output
        #     print("Excel file saved: temp1.xlsx")  # Add confirmation

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
            fig.add_trace(go.Pie(labels=categories[p], values=values[p], name=_list[p], marker=dict(colors=colors[p]), hole=0.1), row=m+1, col=2)
            p = p+1
        
        fig.update_traces(marker=dict(line=dict(color='black', width=0.1)))
        
        fig.update_layout(title_text=_title,height=height, width=width, plot_bgcolor='white',paper_bgcolor='white',
                          xaxis=dict(showgrid=True,  gridcolor='#EDF1F2'),yaxis=dict(showgrid=True,  gridcolor='#EDF1F2'))
        return fig


def multi_pie_charts(df_, col, _list, _title, height, width):
    if not df_.empty:
        df_Dict = color_frequency_brand(df_, col, _list)
        
        categories = []
        values = []
        colors = []
        
        unique_values = set()
        for k in df_Dict.keys():
            unique_values.update(df_Dict[k][col])
        
        color_palette = pc.qualitative.Dark24  # Alternative: pc.qualitative.D3
        color_map = {v: color_palette[i % len(color_palette)] for i, v in enumerate(sorted(unique_values))}
        
        for k in df_Dict.keys():
            categories.append(df_Dict[k][col])
            values.append(df_Dict[k]['word_freq'])
            
            # Assign dark colors based on the category
            category_colors = [color_map.get(c, '#333333') for c in df_Dict[k][col]]  # Default to dark gray
            colors.append(category_colors)
        
        n = (len(df_Dict) + 1) // 2
        specs = [[{'type': 'domain'}, {'type': 'domain'}] for _ in range(n)]
        
        fig = make_subplots(rows=n, cols=2, specs=specs, subplot_titles=tuple(df_Dict.keys()))
        
        p = 0
        for m in range(n):
            fig.add_trace(
                go.Pie(labels=categories[p], values=values[p], name=_list[p], marker=dict(colors=colors[p]), hole=0.1),
                row=m+1, col=1
            )
            fig.add_trace(
                go.Pie(labels=categories[p], values=values[p], name=_list[p], marker=dict(colors=colors[p]), hole=0.1),
                row=m+1, col=2
            )
            p += 1
        
        fig.update_traces(marker=dict(line=dict(color='white', width=1)))  
        fig.update_layout(
            title_text=_title, 
            height=height, 
            width=width, 
            plot_bgcolor='white',  
            paper_bgcolor='white',  
            font=dict(color="black")  
        )
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
    

def multi_pie_chart_collaboration(df, col, brands, _title, height=800, width=800):
    
    df_Dict = {brand: df[df["Brand_D2C"] == brand][col].value_counts() for brand in brands}
    
    unique_collaborations = set()
    for collabs in df_Dict.values():
        unique_collaborations.update(collabs.index.tolist())
    

    color_palette = px.colors.qualitative.D3  # Alternative: px.colors.qualitative.Dark24
    color_map = {v: color_palette[i % len(color_palette)] for i, v in enumerate(sorted(unique_collaborations))}

    n = (len(brands) + 1) // 2  
    fig = make_subplots(rows=n, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]] * n, subplot_titles=brands)

    for i, brand in enumerate(brands):
        row, col_num = divmod(i, 2)

        labels = df_Dict[brand].index.tolist()
        values = df_Dict[brand].values.tolist()
        color_list = [color_map[label] for label in labels if label in color_map]  

        fig.add_trace(
            go.Pie(labels=labels, values=values, name=brand, marker=dict(colors=color_list)),
            row=row + 1, col=col_num + 1
        )

    fig.update_layout(title_text=_title, height=height, width=width)
    return fig

def calculate_brand_positions(df, col="Product Story"):
    """
    Calculate brand positioning based on keyword occurrences in the specified column.
    Determines Fashion vs. Function and Strong Design vs. Weak Design.
    """
    if "Brand_D2C" not in df.columns or col not in df.columns:
        print(f"Required columns 'Brand_D2C' and '{col}' not found in dataset.")
        return {}

    df[col] = df[col].fillna("").astype(str).str.lower()

    categories = {
        "Fashion": ["fashion", "style", "trend", "elegance", "luxury", "chic", "glamour", "couture", "runway", "designer", "aesthetic", "statement"],
        "Function": ["performance", "utility", "comfort", "sport", "active", "support", "durable", "ergonomic", "weatherproof", "protective", "technical", "breathable"],
        "Strong Design": ["iconic", "signature", "logo", "recognizable", "distinct", "bold", "silhouette", "heritage", "craftsmanship", "pattern", "monogram", "statement piece", "branding", "custom", "unique", "artistic", "exclusive"],
        "Weak Design": ["basic", "standard", "plain", "subtle", "generic", "minimal", "simple", "classic", "neutral", "understated", "functional", "common", "versatile", "everyday"]
    }

    brand_scores = {}

    for brand, group in df.groupby("Brand_D2C"):
        product_story = " ".join(group[col])

        category_counts = {
            cat: sum(product_story.count(word) for word in words) 
            for cat, words in categories.items()
        }

        total_words = sum(category_counts.values())

        category_percentages = {
            cat: count / total_words if total_words > 0 else 0 
            for cat, count in category_counts.items()
        }

        x_value = category_percentages["Fashion"] - category_percentages["Function"]
        y_value = category_percentages["Strong Design"] - category_percentages["Weak Design"]

        brand_scores[brand] = (x_value, y_value)

    if not brand_scores:
        print("No valid brand positions calculated. Check your dataset.")
        return {}

    # Normalize values to fit within -1 to 1 range
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


def calculate_brand_positioning(df):
    """
    Groups dataset by Brand_D2C and calculates average scores for fashion attributes.
    Returns a DataFrame with rounded scores.
    """
    columns_to_avg = [
        "Fashion-forward", "Function-forward", "Minimalistic", "Bold", 
        "Modern", "Classic", "Streetwear", "Luxury-Premium"
    ]
    
    df[columns_to_avg] = df[columns_to_avg].apply(pd.to_numeric, errors='coerce')

    brand_scores = df.groupby("Brand_D2C")[columns_to_avg].mean().reset_index()
    
    brand_scores[columns_to_avg] = brand_scores[columns_to_avg].round(2)
    
    return brand_scores

# Step 2: Calculate Relative Scores
def calculate_relative_scores(df):
    """
    Standardizes the fashion attribute scores and rounds them to the nearest 0.5 step.
    """
    cols_to_scale = [
        "Fashion-forward", "Function-forward",
        "Minimalistic", "Bold",
        "Modern", "Classic",
        "Streetwear", "Luxury-Premium"
    ]
    
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    df[cols_to_scale] = np.round(df[cols_to_scale] * 2) / 2  
    
    return df

# Step 3: Generate Brand Positioning Plot
def plot_brand_positioning(df, x_col, y_col):
    """
    Generates an interactive brand positioning scatter plot using Plotly.
    """
    avg_x, avg_y = df[x_col].mean(), df[y_col].mean()

    categories = []
    colors = []

    for i in range(len(df)):
        if df.loc[i, x_col] > avg_x and df.loc[i, y_col] > avg_y:
            categories.append(f"High {x_col} & High {y_col}")
            colors.append("green")
        elif df.loc[i, x_col] > avg_x and df.loc[i, y_col] <= avg_y:
            categories.append(f"High {x_col} & Low {y_col}")
            colors.append("blue")
        elif df.loc[i, x_col] <= avg_x and df.loc[i, y_col] > avg_y:
            categories.append(f"Low {x_col} & High {y_col}")
            colors.append("yellow")
        else:
            categories.append(f"Low {x_col} & Low {y_col}")
            colors.append("red")

    df["category"] = categories
    df["color"] = colors

    fig = go.Figure()

    for category, color in zip(df["category"].unique(), ["green", "blue", "yellow", "red"]):
        subset = df[df["category"] == category]
        fig.add_trace(go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            mode="markers+text",
            text=subset["Brand_D2C"],
            textposition="top center",
            marker=dict(size=12, color=color, line=dict(width=1, color="black")),
            name=category,
            hovertemplate="<b>%{text}</b><br>" + x_col + ": %{x:.2f}<br>" + y_col + ": %{y:.2f}<extra></extra>"
        ))

    x_min, x_max = df[x_col].min() - 0.5, df[x_col].max() + 0.5
    y_min, y_max = df[y_col].min() - 0.5, df[y_col].max() + 0.5

    fig.add_hline(y=avg_y, line_dash="dash", line_color="gray")
    fig.add_vline(x=avg_x, line_dash="dash", line_color="gray")

    fig.update_layout(
        # title=title,
        xaxis=dict(zeroline=False, range=[x_min, x_max], 
                   showticklabels=False, showgrid=False),
        yaxis=dict(zeroline=False, range=[y_min, y_max], 
                   showticklabels=False, showgrid=False),
        width=900,
        height=700,
        annotations=[
            dict(x=x_max, y=avg_y, text=f"More {x_col}", showarrow=False, 
                 font=dict(size=12, color="black", family="Arial Black"), 
                 xanchor="right", yanchor="bottom"),
            
            dict(x=x_min, y=avg_y, text=f"Less {x_col}", showarrow=False, 
                 font=dict(size=12, color="black", family="Arial Black"), 
                 xanchor="left", yanchor="bottom"),
            
            dict(x=avg_x, y=y_max, text=f"More {y_col}", showarrow=False, 
                 font=dict(size=12, color="black", family="Arial Black"), 
                 xanchor="center", yanchor="bottom"),
            
            dict(x=avg_x, y=y_min, text=f"Less {y_col}", showarrow=False, 
                 font=dict(size=12, color="black", family="Arial Black"), 
                 xanchor="center", yanchor="top"),
        ],
        legend_title="Brand Categories"
    )

    return fig



def process_gender_mix(df):
    df['Gender-Mix'] = df['Gender-Mix'].str.lower().str.strip()
    
    df['Gender-Mix'] = df['Gender-Mix'].replace({
        'men': 'Men', 'male': 'Men',
        'women': 'Women', 'female': 'Women',
        'unisex': 'Unisex'
    })
    valid_categories = {'Men', 'Women', 'Unisex'}
    df['Gender-Mix'] = df['Gender-Mix'].astype(str).str.strip().str.title()
    df = df[df['Gender-Mix'].isin(valid_categories)]    
    return df


# Function to process collaboration data
def process_collaborations(df):
    valid_collaborations = df["Collaborations"].replace("", float("nan")).dropna()

    top_collaborations = valid_collaborations.value_counts().nlargest(3).index.tolist()

    df_filtered = df[df["Collaborations"].isin(top_collaborations)].copy()
    
    return df_filtered


def gender_mix_distribution(df):
    """Processes only the Gender-Mix column."""
    return process_gender_mix(df)

def collaborations_distribution(df):
    """Processes only the Collaborations column."""
    return process_collaborations(df)
