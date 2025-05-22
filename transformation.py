
import pandas as pd
import numpy as np
import re
import json

from utility import *

def load_data(filename):
    df = pd.read_csv(filename)
    return df


def get_sub_category(df_):
    
    col = 'Sub-category'
    
    df_[col] = df_[col].str.lower()
    df_[col] = df_[col].str.replace(',','')
    df_[col] = df_[col].str.replace("'",'')
    df_[col] = df_[col].str.strip()
    
    unique_words = get_unique_words(df_[col].unique())
    word_list = unique_words
    
    df_WF = apply_word_occurrences_to_df(df_, col, word_list)
    df_WF = pd.DataFrame(df_WF['word_occurrences'].tolist())
    
    WF = pd.DataFrame(df_WF[word_list].sum()).reset_index()
    WF.columns=[col, 'word_freq']
    WF = WF.sort_values('word_freq', ascending=False)
    WF = WF[WF['word_freq']>=50]
    
    word_list = WF[col].unique()
    df_ = apply_find_first_match_to_df(df_, col, word_list)
    subcat_list = list(df_['new_'+col].unique())
    
    return df_, subcat_list


def get_product_type(df_, subcat_list):
    
    col = 'Type'
    
    df_[col] = df_[col].str.lower()
    df_[col] = df_[col].str.replace(',','')
    df_[col] = df_[col].str.replace("'",'')
    df_[col] = df_[col].str.strip()
    
    unique_words = get_unique_words(df_[col].unique())
    word_list = unique_words

    input_list = word_list
    
    if '' in word_list:
         word_list.remove('')


    df_WF = apply_word_occurrences_to_df(df_, col, word_list)
    df_WF = pd.DataFrame(df_WF['word_occurrences'].tolist())
    
    WF = pd.DataFrame(df_WF[word_list].sum()).reset_index()
    WF.columns=[col, 'word_freq']
    WF = WF.sort_values('word_freq', ascending=False)

    WF = WF[WF['word_freq']>=50]
    
    word_list = WF[col].unique()
    word_list = remove_words_from_list(word_list, subcat_list)
    df_ = apply_find_first_match_to_df(df_, col, word_list)

    return df_




def transformLevel0(df_, col):
    field = col
    df_ = df_.groupby([field])['Product Image'].nunique(dropna=False).reset_index()
    df_.rename(columns={'Product Image': 'product_count'}, inplace=True)
    df_['product_count%'] = round((df_['product_count'] / df_['product_count'].sum()) * 100,1)
    return df_


def transformLevel1(df_, field):
    df_ = df_.groupby(['Brand_D2C',field])['Product Image'].nunique(dropna=False).reset_index()
    df_.rename(columns={'Product Image': 'product_count'}, inplace=True)
    df_['product_count_grp'] = df_.groupby(['Brand_D2C'])['product_count'].transform('sum')
    df_['product_count%'] = round((df_['product_count'] / df_['product_count_grp']) * 100,1)
    return df_


def transformLevel2(df_, field):
    df_ = df_.groupby(['Brand_D2C',field])['Product Image'].nunique(dropna=False).reset_index()
    df_.rename(columns={'Product Image': 'product_count'}, inplace=True)
    df_['product_count_grp'] = df_.groupby(['Brand_D2C'])['product_count'].transform('sum')
    df_['product_count%'] = round((df_['product_count'] / df_['product_count_grp']) * 100,1)
    return df_


def transfomLevel3(df_, inp_col, oup_col):
    df_['PricingList'] = df_[inp_col].apply(lambda x: extract_numeric_values_for_pricing(x))
    df_[['Min' + oup_col, 'Max'+ oup_col]] = df_['PricingList'].tolist()
    df_['Min' + oup_col] = df_['Min' + oup_col].astype('int')
    df_['Max' + oup_col] = df_['Max' + oup_col].astype('int')
    df_[oup_col] = (df_['Min' + oup_col]+df_['Max' + oup_col])/2
    return df_


def get_stats(df_, field):
    dfperc = pd.DataFrame()
    percentile = [0.05, 0.25, 0.50, 0.75, 0.95]
    for perc in percentile:
        dfnew = df_.groupby('Brand_D2C')[field].quantile(perc).reset_index().sort_values(field)
        dfnew[field] = dfnew[field].astype('int64')
        dfnew.rename(columns={field: field+"_"+str(perc)}, inplace=True)
        if perc != min(percentile):
            dfperc = pd.merge(dfperc, dfnew, on='Brand_D2C', how='inner')
        else:
            dfperc = dfnew

    dfperc = dfperc.sort_values(field+"_"+str(perc))
    dfperc = dfperc.reset_index()
    dfperc.drop(columns='index', inplace=True)
    
    return dfperc

# Function to process Gender Mix
# def process_gender_mix(df):
#     df['Gender Mix'] = df['Gender Mix'].str.lower().str.strip()
#     df['Gender Mix'] = df['Gender Mix'].replace(
#         {'male': 'Men', 'men': 'Men', 'female': 'Women', 'women': 'Women'}
#     )
#     df['Gender Mix'] = df['Gender Mix'].apply(lambda x: 'Unisex' if 'unisex' in str(x) else x)
#     return df

# # Function to process Sustainability Scores
# def process_sustainability(df):
#     df['Sustainability Score'] = pd.to_numeric(df['Sustainability Score'], errors='coerce')
#     sustainability_avg = df.groupby('Brand_D2C')['Sustainability Score'].mean().reset_index()

#     def categorize_sustainability(score):
#         if score >= 5:
#             return 'Sustainable'
#         elif 2 <= score < 5:
#             return 'Partially Sustainable'
#         else:
#             return 'Non-Sustainable'

#     sustainability_avg['Sustainability'] = sustainability_avg['Sustainability Score'].apply(categorize_sustainability)
#     df = df.merge(sustainability_avg[['Brand_D2C', 'Sustainability']], on='Brand_D2C', how='left')
#     return df

# # Function to process Collaborations
# def process_collaborations(df):
#     df['Collaborations'] = df['Collaborations'].str.lower().str.strip()
#     df['Collaborations'] = df['Collaborations'].apply(lambda x: 'None' if 'none' in str(x) else x)
#     return df

# # Main function to process data
# def preprocess_data(dff2):
#     dff2 = process_gender_mix(dff2)
#     dff2 = process_sustainability(dff2)
#     dff2 = process_collaborations(dff2)
#     return dff2
