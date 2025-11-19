import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
       
    ## Transformation clr pour normaliser les données
    copy_otu_df = cleaned_otu_df.copy()
    copy_otu_df += 1 # Ajout du 0 pour éviter d'avoir des erreurs sur le log
    copy_otu_df = copy_otu_df.div(copy_otu_df.sum(axis=1),axis=0) # Diviser pour avoir une proportion
    log_otu_df = np.log(copy_otu_df) # Chercher le log
    clr_otu_df =  log_otu_df.sub(log_otu_df.mean(axis=1),axis=0) # Pour retrancher contre la moyenne
    
    cleaned_otu_df = clr_otu_df
    
    ## Puis garder les plus variables parmi les otu (pour retirer les OTU qui sont quasi constants car peu informatifs)
    
    var = cleaned_otu_df.var(axis=0)
    thr = np.quantile(var, 0.05)
    
    cleaned_otu_df = cleaned_otu_df.loc[:, var > thr]
    
    return cleaned_otu_df

def mergin(df1 : pd.DataFrame,df2 : pd.DataFrame) -> pd.DataFrame : 
    """
    Pour merger les deux dataframes. (Faire attention à l'ordre)
    
    Paramètres : 
    ------------
    
        df1 (pd.DataFrame) = data.Rdata
        df2 (pd.DataFrame) = otu.Rdata
        
    Retour : 
    --------
    
        merged_df (pd.DataFrame) = le dataframe 
    """
    cleaned_df1 = purge_df_data(df1)
    cleaned_df2 = purge_df_otu(df2)
    
    merged_df = cleaned_df1.join(cleaned_df2, how='inner')
    
    X = merged_df.drop(columns=['diagnosis'])
    y = merged_df.loc[:, "diagnosis"].astype(str)

    X_train,_,y_train,_ = train_test_split(X,y,test_size=0.3,random_state=42)
    le_y = LabelEncoder().fit(y_train)
    # y_test = le_y.transform(y_test)
    y_train = le_y.transform(y_train)
    
    merged_df = merged_df.loc[:,get_best_features(X_train,y_train,0.003)]
    merged_df["diagnosis"] = y

    return merged_df

def get_best_features(X : np.array,y : np.array,threshold : float) : 
    """
    Fonction used to get best features.
    
    Parameters :
    ------------
    
        X (np.array) : Different features.
        y (np.array) : Feature to predict
        threshold (float) : Threshold for the mask.
        
    Returns :
    ---------
    
        features_names_filtered (List[str]) : Features kept afterward.
    """
    
    rf = RandomForestClassifier()
    
    rf = RandomForestClassifier().fit(X,y)
    
    list_imp = rf.feature_importances_
    list_mask = np.where(list_imp >= threshold,True,False)
    features_names = np.array(X.columns)
    
    return features_names[list_mask]
    
    
    
