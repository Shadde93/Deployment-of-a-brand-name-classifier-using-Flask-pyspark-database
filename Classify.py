# -*- coding: utf-8 -*-
import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.pipeline import Pipeline

def import_data_from_SQL(sel):
    dbcon = pymysql.connect(host='127.0.0.1' ,user='root',password='', port =3307, database ="company_names")
    try:
        pdf = pd.read_sql_query('SELECT name, country, industry FROM company_table WHERE industry IN ' + sel, dbcon)
        dbcon.close()
        return(pdf)
    except:
        dbcon.close()
        return(print("Error: unable to convert the data"))

def clean_df(df):
    df_copy = df.copy()
    df_copy  = df_copy[(df_copy !='nan').all(1)]
    df_copy['name'] = df_copy['name'].str.replace('[^a-zA-Z]+',' ', regex=True).str.strip()
    df_copy  = df_copy[df_copy['name'].str.isspace() == False]
    df_copy = df_copy.drop_duplicates()
    indust_cut = df_copy['industry'].value_counts() - df_copy['industry'].value_counts().min()
    n_cut_dict = indust_cut.to_dict()
    for k, v in n_cut_dict.items():
        df_class = df_copy[df_copy['industry'] == k]
        drop_indices = np.random.choice(df_class.index, v, replace=False)
        df_copy = df_copy.drop(drop_indices)
    return(df_copy)

def data_transform_save_model(df):
    df_copy = df.copy()
    pipe = Pipeline([('le_country', LabelEncoder()), ('le_industry', LabelEncoder()), ('vec', TfidfVectorizer(max_features=100, analyzer='word')), ('scaler', MaxAbsScaler()), 
    ('rf', RandomForestClassifier())])
    train,_ = train_test_split(df_copy, test_size=0.2, random_state=42)
    pipe['le_country'].fit(df_copy['country'])
    train_country_col = pipe['le_country'].transform(train['country'])
    train_T = pipe['le_industry'].fit_transform(train['industry'])
    train_name_sp = pipe['vec'].fit_transform(train['name'])
    train_X_unsc = hstack((train_name_sp ,train_country_col[:,None]))
    train_X_sc = pipe['scaler'].fit_transform(train_X_unsc)
    pipe['rf'].fit(train_X_sc, train_T)
    pipe_save = open(r'C:\Users\shadd\Desktop\Code\Company_name_proj\Model\pipe.pkl', "wb")
    pickle.dump(pipe,pipe_save)
    pipe_save.close()
    return()


def pred_input(name, country):
    name = name.lower()
    country = country.lower()
    input_df  = pd.DataFrame([[name, country]], columns=['name', 'country'])
    pipe_pkl = open(r'C:\Users\shadd\Desktop\Code\Company_name_proj\Model\pipe.pkl',"rb")
    pipe = pickle.load(pipe_pkl)
    input_country_col = pipe['le_country'].transform(input_df['country'])
    input_name_sp = pipe['vec'].transform(input_df['name'])
    input_X_unsc = hstack((input_name_sp, input_country_col [:,None]))
    input_X_sc = pipe['scaler'].transform(input_X_unsc)
    input_pred = pipe['rf'].predict(input_X_sc)
    input_pred_label = pipe['le_industry'].inverse_transform(input_pred)
    pipe_pkl.close()
    return(input_pred_label[0])



def main_function(sel):
    pdf = import_data_from_SQL(sel)
    pdf_clean = clean_df(pdf)
    data_transform_save_model(pdf_clean)
    return(print('Model ready'))
