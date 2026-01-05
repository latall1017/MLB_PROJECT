import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def purge_df_data(df : pd.DataFrame) -> pd.DataFrame : 
    """
    Fonction de faire le préprocessing du dataframe data_df.
    
    Paramètres :
    ------------
     
        df (pd.DataFrame) : DataFrame avant la purge.
    
    Retour :
    --------
     
        result_df (pd.DataFrame) : DataFrame après la purge.
    """
    
    # Mettre le sample name comme index
    df_clean = df.set_index('sample_name')
    
    # Garder age et diagnosis
    df_clean = df_clean.loc[:,["age","diagnosis"]]
    
    # Mise en miniscule des str
    df_clean["diagnosis"] = df_clean["diagnosis"].apply(lambda s : s.lower())
    
    # Suppression des lignes ayant au moins un nan     
    df_clean = df_clean[~df_clean.isna().any(axis=1)] 
    
    return df_clean


def purge_df_otu(df : pd.DataFrame) -> pd.DataFrame : 
    """
    
    Fonction de faire le préprocessing du dataframe otu_df.
    
    Paramètres : 
    ------------
    
        df (pd.DataFrame) : DataFrame avant la purge.
    
    Retour :
    --------
     
        result_df (pd.DataFrame) : DataFrame après la purge.
    """
    
    ## Enlever les échantillons à faible profondeur
    threshold = 12e3
    cleaned_otu_df = df.loc[df.sum(axis=1) >= threshold]
    
    ## Récupérer les OTU qui sont présents dans au moins 5% des échantillons
    
    first_cond = (cleaned_otu_df.gt(0).sum(axis=0) / cleaned_otu_df.shape[0]) >= 0.05
    
    ## Enlever les OTU ultra faibles aussi (au minimum au total sur tous les échantillons)
    
    second_cond = cleaned_otu_df.sum(axis=0) >= 50
    
    cleaned_otu_df = cleaned_otu_df.loc[:,first_cond & second_cond]
       
    
    ## Puis garder les plus variables parmi les otu (pour retirer les OTU qui sont quasi constants car peu informatifs)
    
    var = cleaned_otu_df.var(axis=0)
    thr = np.quantile(var, 0.05)
    
    cleaned_otu_df = cleaned_otu_df.loc[:, var > thr]
    
    return cleaned_otu_df


def mergin_otu_level(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Pour pré-processer et merger les dataframes sans agrégation taxonomique.
    Cette fonction applique la transformation CLR directement au niveau des OTUs.
    
    Paramètres : 
    ------------
        df1 (pd.DataFrame) : Le dataframe des métadonnées (data.Rdata).
        df2 (pd.DataFrame) : Le dataframe des OTUs (otu.Rdata).
        
    Retour : 
    --------
        merged_df (pd.DataFrame) : Le dataframe final avec les OTUs comme features.
    """
    ## 1. Nettoyer les tables de métadonnées et d'OTU (la table OTU n'est pas encore transformée CLR)
    cleaned_df1 = purge_df_data(df1)
    # La fonction purge_df_otu est appelée, elle ne fait pas encore de transformation CLR.
    cleaned_df2 = purge_df_otu(df2) # This function already filters low-abundance OTUs

    ## 2. Joindre les métadonnées nettoyées avec les données OTU nettoyées
    merged_df = cleaned_df1.join(cleaned_df2, how='inner')
    
    ## 3. Appliquer la transformation CLR sur les colonnes de features des données fusionnées finales
    # (Cette logique est également présente dans la fonction mergin)
    feature_cols = [col for col in merged_df.columns if col not in ['age', 'diagnosis']]
    abundance_df = merged_df[feature_cols].copy()
    abundance_df[abundance_df < 0] = 0 # S'assurer qu'il n'y a pas de valeurs négatives avant le pseudo-comptage

    # Ajouter un pseudo-comptage
    abundance_df += 1
    # Fermeture : diviser par la somme des lignes
    proportions = abundance_df.div(abundance_df.sum(axis=1), axis=0)
    # Transformation log et centrage
    log_proportions = np.log(proportions)
    clr_transformed = log_proportions.sub(log_proportions.mean(axis=1), axis=0)

    # Recombiner avec les métadonnées
    merged_df = pd.concat([merged_df[['age', 'diagnosis']], clr_transformed], axis=1)
    
    ## Changer la colonne cible 
    merged_df["diagnosis"] = merged_df["diagnosis"].apply(lambda x : 'healthy' if x == 'no' else 'disease')
    
    return merged_df

def get_best_features(X : pd.DataFrame, y : np.array, threshold : float) -> np.ndarray: 
    """
    Fonction utilisée pour sélectionner les meilleures caractéristiques (features) 
    basé sur l'importance d'un modèle RandomForest.
    
    Paramètres :
    ------------
    
        X (pd.DataFrame) : DataFrame contenant les caractéristiques.
        y (np.array) : Tableau numpy contenant la variable cible.
        threshold (float) : Seuil d'importance pour la sélection des caractéristiques.
        
    Retour :
    ---------
    
        np.ndarray : Un tableau des noms de colonnes des caractéristiques sélectionnées.
    """
    
    rf = RandomForestClassifier().fit(X, y)
    
    importances = rf.feature_importances_
    masque = importances >= threshold
    noms_caracteristiques = np.array(X.columns)
    
    return noms_caracteristiques[masque]
