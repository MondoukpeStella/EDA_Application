import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro 
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Titre de l'application
st.markdown("""
# Exploratory Data Analysis Application by Stella Mondoukpè AGUEMON(https://www.linkedin.com/in/stella-aguemon)

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

# Fonction de tracé du cercle d'ACP à deux dimensions
def acp_buider(pca,vars,n):
    # Récupérer les deux premiers axes factoriels
    ax_fact = np.transpose(pca.components_[:2])

    # Crééer la figure
    fig, ax = plt.subplots(figsize=(20,20))

    # Projeter les vecteurs et leur annotation sur les axes factoriels
    for i in range(n):
        ax.arrow(0,0,ax_fact[i,0],ax_fact[i,1],head_width=0.02)
        ax.text(ax_fact[i,0]*1.15,ax_fact[i,1]*1.15,vars[i],va="center",ha="center")

    # Placer le cercle unitaire
    cercle = plt.Circle((0,0),1,color="black",linestyle='--',fill=False)
    
    # Ajouter le cercle à la figure
    ax.add_artist(cercle)

    # Ajuster les limites et les axes
    ax.axhline(0,color='black',linestyle='--')
    plt.axvline(0,color='black',linestyle='--')

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)

    # Noms des axes et figure
    ax.set_xlabel("Axe 1")
    ax.set_ylabel("Axe 2")
    ax.set_title("Cercle de corrélation de l'ACP")

    # Afficher la figure
    st.pyplot(fig)


if uploaded_file is not None :
    # Charger le dataframe de données
    df = load_data_csv()
    
    # Rapport exploratoire
    pr = ProfileReport(df,explorative=True)
    
    # Variables quantitatives 
    var_quant = [col for col in df.columns if df[col].dtype!="object"]
    
    # Afficher le jeu de données
    st.header("Dataset uploaded")
    st.write(df)
    st.write('     ')
    
    # Afficher les statistiques descriptives
    st.header("Descriptive Statistics")
    st.write(df.describe())
    
    # Afficher le Rapport Exploratoire
    st.header("Profiling Report")
    st_profile_report(pr)
    
    # Afficher les boxplots et les outliers
    st.header("Boxplots and Outliers, method=IQR")
    for i , col in enumerate(var_quant,1):
        # Affichier le boxplot associé à la variable
        boxplot_csv(df,col)
        
        # Affichier les outliers
        st.write(f'Outliers in {col}')
        st.write(detect_outliers_iqr(df, col))
    
    # Afficher les résultats du test de Shapiro
    st.header("Test de Normalité de Shapiro Wilk")
    test_result  = shapiro_test(df,var_quant)
    st.write(test_result)

    # Afficher les analyse multivariées 
    if len(var_quant) > 3:
        st.header("Dimensionality Reduction (ACP)")
        df_quant = df[var_quant]
        
        # Standardisation des données
        scaler = StandardScaler()
        df_quant_scaled = scaler.fit_transform(df_quant)
        
        # Reduction des dimensions des données
        pca = PCA()
        df_quant_pca = pca.fit_transform(df_quant_scaled)
        
        # Définir le nombre de vecteurs à placer
        n = len(var_quant)
        
        # Choix des axes factoriels
        st.write("Explained Variance of diferent factor axes")
        table = pd.DataFrame({"Axes_factoriels":["Axe"+str(x+1) for x in range(n)],
                        "%_variance_expliquee":pca.explained_variance_ratio_*100,
                        "%_cumulee_variance_expliquee":np.cumsum(pca.explained_variance_ratio_*100)})
        st.write(table)
        
        # Visualisation des axes factoriels par pourcentage de variance expliquée
        st.header("Visualisation of ACP Circle")
        fig,ax = plt.subplots()
        sns.barplot(data=table,x="Axes_factoriels",y="%_variance_expliquee",ax=ax)
        ax.set_title(f"Contributon of different factor axes")
        plt.xticks(rotation=90)
        st.pyplot(fig)
        
        acp_buider(pca,var_quant,n)
        
        

