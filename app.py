import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro ,kruskal
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit as st

# Titre de l'application
st.markdown("""
# Exploratory Data Analysis Application

This is EDA Application with can analyze data and help in these steps :
Data Understanding, Data Visualization, Missing Value Analysis,
Outliers Detection and Statistical Tests.

            """
    )

# Charger les données en format CSV
with st.sidebar.header("Upload your csv data here") :
    uploaded_file = st.sidebar.file_uploader("Upload your csv file",type=["csv"])
    
# Fonction de chargement des données
def load_data_csv():
    df = pd.read_csv(uploaded_file)
    return df

# Fonction de détection des outliers par la méthode de l'Intervalle InterQuartile
def detect_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR
    outliers  = data[(data[col] < borne_inf) | (data[col] > borne_sup)]
    return outliers

# Fonction de tracé de boxplots
def boxplot_csv(data,col):
    fig, ax = plt.subplots()
    sns.boxplot(data[col],ax=ax)
    ax.set_title(f"Boxplot of {col}")
    st.pyplot(fig)
        
# Fonction d'application du Test de Shapiro
def shapiro_test(data,var_quant):
    normalite = [] 
    p_values = []
    for col in var_quant :
        stats,p_value = shapiro(data[col])
        p_values.append(p_value)
        if p_value > 0.05 :
            normalite.append("OUI")
        else :
            normalite.append("NON")
    resultat = pd.DataFrame({"Variables":var_quant,"P-Valeur":p_values,"Normalité":normalite})
    return resultat


if uploaded_file is not None :
    # Charger le dataframe de données
    df = load_data_csv()
    # Rapport exploratoire
    pr = ProfileReport(df,explorative=True)
    # Variables quantitatives 
    var_quant = [col for col in df.columns if df[col].dtype!="object"]
    
    # Afficher le jeu de données
    st.header("Dataset upload")
    st.write(df)
    st.write('     ')
    
    # Afficher le Rapport Exploratoire
    st.header("Profiling Report")
    st_profile_report(pr)
    
    # Afficher les boxplots et les outliers
    st.header("Boxplots and Outliers")
    for i , col in enumerate(var_quant,1):
        # Affichier le boxplot associé à la variable
        boxplot_csv(df,col)
        # Affichier les outliers
        st.write(f'Outliers in {col}')
        st.write(detect_outliers_iqr(df, col))
    
    # Afficher les résultats du test de Shapiro
    test_result  = shapiro_test(df,var_quant)
    st.write(test_result)

