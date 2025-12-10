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
    # copy_otu_df = cleaned_otu_df.copy()
    # copy_otu_df += 1 # Ajout du 0 pour éviter d'avoir des erreurs sur le log
    # copy_otu_df = copy_otu_df.div(copy_otu_df.sum(axis=1),axis=0) # Diviser pour avoir une proportion
    # log_otu_df = np.log(copy_otu_df) # Chercher le log
    # clr_otu_df =  log_otu_df.sub(log_otu_df.mean(axis=1),axis=0) # Pour retrancher contre la moyenne
    
    # cleaned_otu_df = clr_otu_df
    
    ## Puis garder les plus variables parmi les otu (pour retirer les OTU qui sont quasi constants car peu informatifs)
    
    var = cleaned_otu_df.var(axis=0)
    thr = np.quantile(var, 0.05)
    
    cleaned_otu_df = cleaned_otu_df.loc[:, var > thr]
    
    return cleaned_otu_df

def mergin(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame,label : str = "taxonomy4") -> pd.DataFrame:
    """
    Pour pré-processer et merger les dataframes de métadonnées, OTU et taxonomie.
    Cette fonction agrège les abondances d'OTU par le niveau 'taxonomy4'.
    
    Paramètres : 
    ------------

        df1 (pd.DataFrame) : Le dataframe des métadonnées (data.Rdata).
        df2 (pd.DataFrame) : Le dataframe des OTUs (otu.Rdata).
        df3 (pd.DataFrame) : Le dataframe de la taxonomie (tax.Rdata).
        label (str) : La colonne à choisir dans taxonomy.Rdata pour regrouper
        
    Retour : 
    --------
    
        merged_df (pd.DataFrame) : Le dataframe final avec les abondances agrégées par taxonomie.
    """
    ## Clean the metadata and OTU tables
    cleaned_df1 = purge_df_data(df1)
    cleaned_df2 = purge_df_otu(df2)
    
    ## Join the OTU table with the taxonomy df
    otu_plus_tax = df3.join(cleaned_df2.T, how='inner')
    
    ## Group by 'taxonomy4' and sum the abundances for each sample
    
    ## First, identify sample columns (all columns that are not taxonomy columns)
    sample_cols = [col for col in otu_plus_tax.columns if not col.startswith('taxonomy')]
    
    ## Group by taxonomy4 and sum the sample abundances
    if label not in df3.columns : 
        return "The choosen label is not in the taxonomy dataframe"
    
    tax_aggregated = otu_plus_tax.groupby(label)[sample_cols].sum()
    
    ## Transpose the aggregated table so samples are rows and join with the cleaned metadata
    merged_df = cleaned_df1.join(tax_aggregated.T, how='inner')

    ## Apply CLR transformation on the aggregated abundance data
    # Isolate feature columns (everything except metadata)
    feature_cols = [col for col in merged_df.columns if col not in ['age', 'diagnosis']]
    abundance_df = merged_df[feature_cols]

    # Add pseudocount
    abundance_df += 1
    # Closure: divide by row sum
    proportions = abundance_df.div(abundance_df.sum(axis=1), axis=0)
    # Log transform and center
    log_proportions = np.log(proportions)
    clr_transformed = log_proportions.sub(log_proportions.mean(axis=1), axis=0)

    # Recombine with metadata
    merged_df = pd.concat([merged_df[['age', 'diagnosis']], clr_transformed], axis=1)
    
    ## Change target column 
    merged_df["diagnosis"] = merged_df["diagnosis"].apply(lambda x : 'healthy' if x == 'no' else 'disease')
    
    return merged_df

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
    ## 1. Clean the metadata and OTU tables (OTU table is not CLR transformed yet)
    cleaned_df1 = purge_df_data(df1)
    # We call a modified purge_df_otu that doesn't do CLR yet
    cleaned_df2 = purge_df_otu(df2) # This function already filters low-abundance OTUs

    ## 2. Join the cleaned metadata with the cleaned OTU data
    merged_df = cleaned_df1.join(cleaned_df2, how='inner')
    
    ## 3. Apply CLR transformation on the final merged data's feature columns
    # (This logic is now inside the mergin function)
    feature_cols = [col for col in merged_df.columns if col not in ['age', 'diagnosis']]
    abundance_df = merged_df[feature_cols].copy()
    abundance_df[abundance_df < 0] = 0 # Ensure no negative values before pseudocount

    # Add pseudocount
    abundance_df += 1
    # Closure: divide by row sum
    proportions = abundance_df.div(abundance_df.sum(axis=1), axis=0)
    # Log transform and center
    log_proportions = np.log(proportions)
    clr_transformed = log_proportions.sub(log_proportions.mean(axis=1), axis=0)

    # Recombine with metadata
    merged_df = pd.concat([merged_df[['age', 'diagnosis']], clr_transformed], axis=1)
    
    ## Change target column 
    merged_df["diagnosis"] = merged_df["diagnosis"].apply(lambda x : 'healthy' if x == 'no' else 'disease')
    
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
    
    
    
