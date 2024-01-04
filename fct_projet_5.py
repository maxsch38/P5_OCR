#############################################################################################################################
### Fichier de fonction du projet 5 : Segmentez des clients d'un site e-commerce
#############################################################################################################################


#############################################################################################################################
# Importation des librairies : 
import numpy as np
import pandas as pd 
import copy

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances_argmin_min
from sklearn.metrics import adjusted_rand_score

from scipy.cluster.hierarchy import linkage, dendrogram

#############################################################################################################################
def transformation_df(df): 
    """
     Applique des transformations de prétraitement à un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.

    Returns:
        pd.DataFrame: DataFrame transformé.
    """
        
    # Création des sélecteurs de colonnes : 
    numerical_col = make_column_selector(dtype_include=np.number)
    categorical_col = make_column_selector(dtype_exclude=np.number)

    # Création du preprocessor : 
    preprocessor = make_column_transformer((StandardScaler(), numerical_col))
    
    # Récupération des noms de colonnes numériques et index : 
    numerical_col_name = list(numerical_col(df))
    index = df.index
    
    # Vérification de la présence de variables catégorielles : 
    if categorical_col(df):
        preprocessor = make_column_transformer((StandardScaler(), numerical_col),
                                               (OneHotEncoder(), categorical_col))
        
        # Application de la transformation :  
        df = preprocessor.fit_transform(df)
        
        # Récupération des noms de colonnes catégorielles : 
        categorical_col_name = list(preprocessor.named_transformers_['onehotencoder'].get_feature_names_out())
    else:
        df = preprocessor.fit_transform(df)
        categorical_col_name = []
    
    # Création de la liste des noms de colonnes : 
    col_name = numerical_col_name + categorical_col_name
    
    # Finalisation : 
    df = pd.DataFrame(df, columns=col_name, index=index)

    return df

#############################################################################################################################
def performances_modèle(dico, model, params, data):
    """
    Évalue les performances d'un modèle de clustering.

    Args:
        dico (dict): Dictionnaire contenant les informations sur le modèle.
        model (str): Clé du modèle dans le dictionnaire.
        params (dict): Paramètres à appliquer au modèle.
        data (pd.DataFrame): Les données d'entrée pour le modèle de clustering.

    Returns:
        dict: Le dictionnaire mis à jour avec les mesures de performances du modèle.
    """
    
    # Création du meilleur modèle :
    dico[model]['model'] = dico[model]['model'].set_params(**params)
    
    # Entrainement du modèle : 
    dico[model]['model'].fit(data)
    
    # Calcul de l'indice de Davies-Bouldin : 
    dico[model]['Davies-Bouldin'] = davies_bouldin_score(data, dico[model]['model'].labels_)
    
    # Calcul du coefficient de silhouette : 
    dico[model]['Silhouette'] = silhouette_score(data, dico[model]['model'].labels_)
    
    # Calcul du la stabilité des attributs : 
    
        # Calcul de cluster_assignements : 
    n_samples = len(data)
    n_bootstraps = 100 # Nombre de rééchantillonnages
    
    cluster_assignments = []
    
    for _ in range(n_bootstraps):
        bootstrap_indices = np.random.choice(n_samples, n_bootstraps, replace=True)
        bootstrap_data = data.iloc[bootstrap_indices]
        
        cluster_assignment = dico[model]['model'].fit_predict(bootstrap_data)
        cluster_assignments.append(cluster_assignment)
        
        # Calcul du score de stabilité : 
    n_samples = len(cluster_assignments[0])
    pairwise_stabilities = []

    for assignment1 in cluster_assignments:
        for assignment2 in cluster_assignments:
            if not np.array_equal(assignment1, assignment2):
                pairwise_stabilities.append(np.mean(assignment1 == assignment2))
    
    dico[model]['Score stabilité'] = np.mean(pairwise_stabilities)

    return dico

#############################################################################################################################
def best_k_KMeans(data):
    """
    Trouve le meilleur nombre de clusters K pour l'algorithme K-Means.

    Args:
        data (pd.DataFrame): Les données d'entrée pour le clustering.

    Returns:
        None: La fonction affiche des graphiques pour la méthode du coude
              et la méthode de la silhouette.
    """
    
    # Création des données : 
    k_range = range(2, 11)
    inertia= []
    silhouette = []

    # Test des valeurs : 
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        labels = kmeans.labels_
        silhouette.append(silhouette_score(data, labels))

    # Visualisation de la méthode du coude : 
    plt.figure(figsize=(12,8))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, marker='o')
    plt.xlabel('Nombre de clusters (K)')
    plt.ylabel('Inertie')
    plt.title('Méthode du Coude')

    # Visualisation de la méthode de la silhouette : 
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette, marker='o')
    plt.xlabel('Nombre de clusters (K)')
    plt.ylabel('Score Silhouette')
    plt.title('Méthode de la Silhouette')

    plt.tight_layout()
    plt.show()
    
#############################################################################################################################
def best_params_DBSCAN(data, eps_range, min_samples_range): 
    """
    Trouve les meilleurs paramètres pour l'algorithme DBSCAN en explorant des plages spécifiques.

    Args:
        data (pd.DataFrame): Les données d'entrée pour le clustering.
        eps_range (list): Liste des valeurs à tester pour le paramètre eps.
        min_samples_range (list): Liste des valeurs à tester pour le paramètre min_samples.

    Returns:
        pd.DataFrame: Un DataFrame contenant les résultats pour chaque combinaison de paramètres.
                      Colonnes : ['Eps', 'Min_samples', 'Nbre_clusters', 'Nbre_outliers', 
                                  'Taille_moy_clusters', 'Davies_Bouldin_score']
    """
    result = []
    
    for eps in eps_range: 
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)
            
            tailles_cluster = [len(labels[labels == i]) for i in set(labels) - {-1}]
            taille_outlier = len(labels[labels == -1])
            
            if len(tailles_cluster) > 0: 
                nbre_clusters = len(tailles_cluster)
                taille_moy_cluster = sum(tailles_cluster) / nbre_clusters
                db_score = davies_bouldin_score(data, labels)
            else: 
                nbre_clusters = 0
                taille_moy_cluster = 0
                db_score = None
                
            result.append([eps, min_samples, nbre_clusters, taille_outlier, taille_moy_cluster, db_score])
            
    # Création d'un dataframe avec les résultats : 
    col = ['Eps', 'Min_samples', 'Nbre_clusters', 'Nbre_outliers', 'Taille_moy_clusters', 'Davies_Bouldin_score']
    df = pd.DataFrame(result, columns=col)

    return df    

#############################################################################################################################
def dendrogramme_AC (data):
    """
    Affiche un dendrogramme pour l'analyse de clustering hierarchique.

    Args:
        data (pd.DataFrame): Les données d'entrée pour l'analyse de clustering hierarchique.
    """
    
    # Création de linkage_matrix : 
    linkage_matrix = linkage(data, method='ward')
    
    # Tracer du dendrogramme : 
    plt.figure(figsize=(12,6))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=20)
    plt.title(f'Dendrogramme sur X_sampled')
    plt.xlabel('Indices des échantillons')
    plt.ylabel('Distance')
    plt.xticks(rotation=90)
    plt.show()
    
#############################################################################################################################    
def clustersing_with_AC(sample, data_full, key, dico):
    """
    Effectue le clustering sur un échantillon et étend les labels au jeu de données complet.

    Args:
        sample (pd.DataFrame): L'échantillon sur lequel le modèle de clustering a été entraîné.
        data_full (pd.DataFrame): Le jeu de données complet sur lequel les labels seront étendus.
        key (str): La clé du modèle dans le dictionnaire 'dico'.
        dico (dict): Dictionnaire contenant les informations sur le modèle.

    Returns:
        np.ndarray: Les labels de clustering étendus au jeu de données complet.
    """

    # Entrainnement du modèle : 
    model = dico[key]['model']
    labels = model.fit_predict(sample)
    
    
    # Calcul des centres des clusters : 
    n_clusters = model.n_clusters_
    cluster_centers = np.zeros((n_clusters, sample.shape[1]))
    
    for cluster_idx in range(n_clusters):
        cluster_mask = labels == cluster_idx
        cluster_points = sample[cluster_mask]
        cluster_center = cluster_points.mean(axis=0)
        cluster_centers[cluster_idx] = cluster_center
  

    # Calul des labels sur le jeu de données complet : 
    labels_full, _ = pairwise_distances_argmin_min(data_full, cluster_centers)
    
    return labels_full

#############################################################################################################################
def transformation_df_2(df): 
    
    """
    Permet le passage du dataset par commande au dataset par client unique. 

    Création des variables utiles pour l'application des modèeles. 

    --> Fonction pour le feature engineering 2.
    """    
    
    # Calcul de la recence : 
    df['recence'] = (df['order_purchase_timestamp'].max() - df['order_purchase_timestamp']).dt.days

    # Regrouppement par client et calcul des variables : 
    df = df.groupby('customer_unique_id').agg({
        'recence' : 'min',
        'order_id' : 'nunique',
        'payment_total' : 'sum',
        'review_score_moy' : 'mean', 
        'payment_boleto' : 'sum',
        'payment_credit_card' : 'sum',
        'payment_debit_card' : 'sum',
        'payment_voucher' : 'sum',
        'payment_installments': 'first',
        'number_of_product' : 'sum',
        'temps_expedition' : 'mean',
        'jour_de_commande' : 'first',
        'moment_de_commande' : 'first',   
    })

    # Renommage des colonnes : 
    df = df.rename(columns={
        'recence' : 'recence',
        'order_id' : 'frequence',
        'payment_total' : 'montant',
        'review_score_moy' : 'satisfaction',
        'number_of_product' : 'number_of_product_tot',
        'payment_installments' : 'payment_installments_mode',
        'jour_de_commande' : 'jour_de_commande_mode',
        'temps_expedition' : 'temps_expedition_moy',
        'moment_de_commande' : 'moment_de_commande_mode'  
    })

    # Aronndi de la colonne satisfaction et typage : 
    df['satisfaction'] = round(df['satisfaction']).astype('int64')

    df = df.sort_index()

    return df

#############################################################################################################################
def transformation_df_RFMS(df): 
    
    """
    Permet le passage du dataset par commande au dataset par client unique. 

    Création des variables utiles pour l'application des modèeles. 

    --> Fonction pour le feature engineering 1.
   """
        
    # Calcul de la recence : 
    df['recence'] = (df['order_purchase_timestamp'].max() - df['order_purchase_timestamp']).dt.days

    # Regrouppement par client et calcul des variables : 
    df = df.groupby('customer_unique_id').agg({
        'recence' : 'min',
        'order_id' : 'nunique',
        'payment_total' : 'sum',
        'review_score_moy' : 'mean',    
    })

    # Renommage des colonnes : 
    df = df.rename(columns={
        'recence' : 'recence',
        'order_id' : 'frequence',
        'payment_total' : 'montant',
        'review_score_moy' : 'satisfaction',
    })

    # Aronndi de la colonne satisfaction et typage : 
    df['satisfaction'] = round(df['satisfaction']).astype('int64')
    
    df = df.sort_index()

    return df

#############################################################################################################################
def evo_ari(df, df_init, model, period, transformation_func):
    
      
    """
    Analyse l'évolution du comportement des clients au fil du temps en utilisant l'ARI.
    

    - paramètre df : DataFrame contenant les données historiques des commandes.
    - paramètre df_init : DataFrame contenant les données des commandes pour la période initiale.
    - paramètres model : modèle de clustering utilisé pour l'analyse.
    - paramètres periode : période de temps utilisée pour regrouper les données (par exemple, 'M' pour mois).
    - paramètre transformation_func : fonction de transformation des données.

     --> renvoie un DataFrame avec les résultats.
    """

    # 1. Preprocessor : 

        # Création des sélecteurs de colonnes : 
    numerical_col = make_column_selector(dtype_include=np.number)
    categorical_col = make_column_selector(dtype_exclude=np.number)

        # Création du preprocessor : 
    preprocessor = make_column_transformer((StandardScaler(), numerical_col),
                                           (OneHotEncoder(), categorical_col))
    
    # 2. Création des modèles :
    
    model_init = copy.deepcopy(model)
    model_current = copy.deepcopy(model)

    # 3. Initialisation du modèle initial sur la période initiale :

    model_init = model_init.fit(preprocessor.fit_transform(transformation_func(df_init)))

    # 4. Création des données : 

        # Initialisation des listes de résultats : 
    periods = []
    ari_scores = []

        # Création des périodes à partir de df : 
    unique_periods = df['order_purchase_timestamp'].dt.to_period(period).unique()
    
    # 5. Calcul du premier point : 
    p = df_init['order_purchase_timestamp'].max().to_period(period)
    labels_init = model_init.predict(preprocessor.transform(transformation_func(df_init)))
    current_labels = model_current.fit_predict(preprocessor.transform(transformation_func(df_init)))
    
    ari = adjusted_rand_score(labels_init, current_labels)
    
    periods.append(p-1)
    ari_scores.append(ari)

    # 6. Calcul des ari et des périodes de maintenance sur l'ensemble de df : 

    for indice, p in enumerate(unique_periods):

        # Vérification qu'au moins une des dates de la période n'est pas dans df_init :  
        if df.loc[df['order_purchase_timestamp'].dt.to_period(period) == p, 'order_purchase_timestamp'].max()\
        > df_init['order_purchase_timestamp'].max():

            current_df = df[df['order_purchase_timestamp'].dt.to_period(period) <= p]

            labels_init = model_init.predict(preprocessor.transform(transformation_func(current_df)))
            current_labels = model_current.fit_predict(preprocessor.transform(transformation_func(current_df)))

            ari = adjusted_rand_score(labels_init, current_labels)

            periods.append(p)
            ari_scores.append(ari)
           
    result_df = pd.DataFrame({'Period': periods, 'ARI': ari_scores})

    return result_df

#############################################################################################################################
def maintenance_seuil(df, df_init, model, ari_seuil, period, transformation_func):
     
    """
    Permet d'effectuer une analyse de l'évolution du comportement des clients au fil du temps en utilisan.
    Identification des périodes où le comportement des clients diffère significativement par rapport à 
    la période initiale, en se basant sur le score ARI.

    - paramètre df : DataFrame contenant les données historiques des commandes.
    - paramètre df_init : DataFrame contenant les données des commandes pour la période initiale.
    - paramètre model : modèle de clustering utilisé pour l'analyse.
    - paramètre ari_seuil: seuil de score ARI à partir duquel déclencher une maintenance.
    - paramètre periode : période de temps utilisée pour regrouper les données (par exemple, 'M' pour mois).
    - paramètre transformation_func :fonction de transformation des données.

    --> renvoie un DataFrame avec l'évolution de l'ARI en fonction des période et une liste des périodes de maintenances.
    """    
    
    # 1. Preprocessor : 

        # Création des sélecteurs de colonnes : 
    numerical_col = make_column_selector(dtype_include=np.number)
    categorical_col = make_column_selector(dtype_exclude=np.number)

        # Création du preprocessor : 
    preprocessor = make_column_transformer((StandardScaler(), numerical_col),
                                            (OneHotEncoder(), categorical_col))
    
    # 2. Création des modèles :
    
    model_init = copy.deepcopy(model)
    model_current = copy.deepcopy(model)

    # 3. Initialisation du modèle initial sur la période initiale :

    model_init = model.fit(preprocessor.fit_transform(transformation_func(df_init)))

    # 4. Création des données : 

        # Initialisation des listes de résultats : 
    periods = []
    ari_scores = []
    maintenance = []

        # Création des périodes à partir de df : 
    unique_periods = df['order_purchase_timestamp'].dt.to_period(period).unique()
    
    # 5. Calcul du premier point : 
    p = df_init['order_purchase_timestamp'].max().to_period(period)
    labels_init = model_init.predict(preprocessor.transform(transformation_func(df_init)))
    current_labels = model.fit_predict(preprocessor.transform(transformation_func(df_init)))
    
    ari = adjusted_rand_score(labels_init, current_labels)
    
    periods.append(p-1)
    ari_scores.append(ari)

    # 6. Calcul des ari et des périodes de maintenance sur l'ensemble de df : 

    for p in unique_periods:

        # Vérification qu'au moins une des dates de la période n'est pas dans df_init :  
        if df.loc[df['order_purchase_timestamp'].dt.to_period(period) == p, 'order_purchase_timestamp'].max()\
        > df_init['order_purchase_timestamp'].max():

            current_df = df[df['order_purchase_timestamp'].dt.to_period(period) <= p]

            labels_init = model_init.predict(preprocessor.transform(transformation_func(current_df)))
            current_labels = model.fit_predict(preprocessor.transform(transformation_func(current_df)))

            ari = adjusted_rand_score(labels_init, current_labels)
            
            # Vérification du seuil d'ari + réinitialisation : 
            if ari < ari_seuil : 
                model_init = model.fit(preprocessor.fit_transform(transformation_func(current_df)))
                maintenance.append(p)
            
                labels_init = model_init.predict(preprocessor.transform(transformation_func(current_df)))
                current_labels = model.fit_predict(preprocessor.transform(transformation_func(current_df)))

                ari = adjusted_rand_score(labels_init, current_labels)

            periods.append(p)
            ari_scores.append(ari)

    result_df = pd.DataFrame({'Period': periods, 'ARI': ari_scores})

    return result_df, maintenance

#############################################################################################################################
def maintenance_tempo(df, df_init, model, transformation_func, retrain_period, period):
    """
    Analyse l'évolution de l'ARI en fonction de la période, en ré-entrainant le modèle à des intervalles spécifiques.

    - paramètre df : DataFrame contenant les données historiques des commandes.
    - paramètre df_init : DataFrame contenant les données des commandes pour la période initiale.
    - paramètres model : modèle de clustering utilisé pour l'analyse.
    - paramètre transformation_func : fonction de transformation des données.
    - paramètre retrain_period : période de réentrainement du modèle initial.
    - paramètre period_type : type de période ('M' pour mois, 'W' pour semaine, etc.).

     --> renvoie un DataFrame avec les résultats.
    """
    
    # 1. Preprocessor : 

    # Création des sélecteurs de colonnes : 
    numerical_col = make_column_selector(dtype_include=np.number)
    categorical_col = make_column_selector(dtype_exclude=np.number)

    # Création du preprocessor : 
    preprocessor = make_column_transformer((StandardScaler(), numerical_col),
                                            (OneHotEncoder(), categorical_col))
    
    # 2. Création des modèles :
    
    model_init = copy.deepcopy(model)
    model_current = copy.deepcopy(model)

    # 3. Initialisation du modèle initial sur la période initiale :
    model_init = model.fit(preprocessor.fit_transform(transformation_func(df_init)))

    # 4. Création des données :
    periods = []
    ari_scores = []
    maintenance_dates = [] 

    # Création des périodes à partir de df :
    unique_periods = df['order_purchase_timestamp'].dt.to_period(period).unique()
    
    # 5. Calcul du premier point : 
    p = df_init['order_purchase_timestamp'].max().to_period(period)
    labels_init = model_init.predict(preprocessor.transform(transformation_func(df_init)))
    current_labels = model.fit_predict(preprocessor.transform(transformation_func(df_init)))
    
    ari = adjusted_rand_score(labels_init, current_labels)
    
    periods.append(p-1)
    ari_scores.append(ari)

    # 6. Calcul des ari et avec périodes de maintenance sur l'ensemble de df : 

    for p in unique_periods:
        
        current_df = df[df['order_purchase_timestamp'].dt.to_period(period) <= p]

        # Ré-entrainement du modèle aux intervalles spécifiques :
        if (p - df_init['order_purchase_timestamp'].max().to_period(period)).n % retrain_period == 0\
        and df.loc[df['order_purchase_timestamp'].dt.to_period(period) == p, 'order_purchase_timestamp'].max()\
        > df_init['order_purchase_timestamp'].max():
            
            model_init = model.fit(preprocessor.fit_transform(transformation_func(current_df)))
            maintenance_dates.append(p) 
        
        # Vérification qu'au moins une des dates de la période n'est pas dans df_init :  
        if df.loc[df['order_purchase_timestamp'].dt.to_period(period) == p, 'order_purchase_timestamp'].max()\
        > df_init['order_purchase_timestamp'].max():
            
            
            labels_init = model_init.predict(preprocessor.transform(transformation_func(current_df)))
            current_labels = model.fit_predict(preprocessor.transform(transformation_func(current_df)))

            ari = adjusted_rand_score(labels_init, current_labels)

            periods.append(p)
            ari_scores.append(ari)
        
    result_df = pd.DataFrame({'Period': periods, 'ARI': ari_scores})

    return result_df, maintenance_dates

#############################################################################################################################
def visualisation_resultats_maintenance(df, maintenance_dates, date_ref, periode, title):
    
    """
    Permet de générer un graphique pour visualiser l'évolution de l'indice ARI sur différentes périodes.
    Mise en évidence des périodes de maintenance.

    - paramètre df : DataFrame contenant l'évolution de l'ARI en fonction des période
    - paramètre maintenance_dates : liste des périodes pour lequelles une maintenance est nécessaire.
    - paramètre date_ref: date de référence à partir de laquelle sont calculées les périodes.
    - paramètre periode: période temporelle utilisée, permet l'affichage de la légende x du graphique.
    - paramètre title: nom du DataFrame utiliser pour l'affichage dans le titre du graphique.

    --> Affiche le graphique.
    """

    # Création des données intermédiaires : 
    
    # Création de la liste des valeurs d'ARI : 
    ari_values = df['ARI'].to_list()
    
    # Création de la liste du nombre de période pour chaque valeur de ls_period par rapport à la date de référence : 
    ls_periods_numeric = [(p - pd.Period(date_ref, freq=periode)).n for p in df['Period']]
    
    # Création de la liste du nombre de période pour chaque valeur de maintenance_date par rapport à la date de référence : 
    maintenance_date_numerics = [(p - pd.Period(date_ref, freq=periode)).n for p in maintenance_dates]

    # Création du nom de la période à afficher dans la légende de l'axe x : 
    if periode == 'D': 
        name_periode = 'jours'
    elif periode == 'W-MON' or periode == 'W-SUN'or periode == "W": 
        name_periode = 'semaines'
    elif periode == 'M': 
        name_periode = 'mois'
    elif periode == 'Q': 
        name_periode = 'trimestres'
    elif periode == 'Y': 
        name_periode = 'années'
        
    # Création du graphique : 
    plt.figure(figsize=(10, 6))

    plt.plot(ls_periods_numeric, ari_values, marker='o', linestyle='-', label='Évolution de l\'ARI')

    # Création d'un artiste personnalisé pour la légende de la maintenance : 
    legend_artist = Line2D([], [], color='red', linestyle='--', label='Maintenance')

    # Tracé des dates de mainteanace :
    for maintenance_date_numeric in maintenance_date_numerics:
        plt.axvline(x=maintenance_date_numeric, color='red', linestyle='--')

    # Ajout de l'artiste personnalisé à la légende : 
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(legend_artist)
    labels.append('Maintenance')

    plt.legend(handles=handles, labels=labels, loc='lower left')

    plt.xlabel(f'Nbre de {name_periode} depuis le début de collecte des données')
    plt.ylabel('ARI')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
#############################################################################################################################  
def visualisation_resultats_evo(df, date_ref, periode, title): 
   
    """
    Crée un graphique pour visualiser l'évolution de l'ARI au fil du temps sans indications de maintenance.

    - paramètre df : DataFrame contenant l'évolution de l'ARI en fonction des période
    - paramètre date_ref: date de référence à partir de laquelle sont calculées les périodes.
    - paramètre periode: période temporelle utilisée, permet l'affichage de la légende x du graphique.
    - paramètre title: nom du DataFrame utiliser pour l'affichage dans le titre du graphique.

    --> Affiche un graphique montrant l'évolution de l'ARI sans indications de maintenance.
    """

    # Création des données intermédiaires : 
    
    # Création de la liste des valeurs d'ARI : 
    ari_values = df['ARI'].to_list()
    
    # Création de la liste du nombre de période pour chaque valeur de ls_period par rapport à la date de référence : 
    ls_periods_numeric = [(p - pd.Period(date_ref, freq=periode)).n for p in df['Period']]
    
    # Création du nom de la période à afficher dans la légende de l'axe x : 
    if periode == 'D': 
        name_periode = 'jours'
    elif periode == 'W-MON' or periode == 'W-SUN'or periode == "W": 
        name_periode = 'semaines'
    elif periode == 'M': 
        name_periode = 'mois'
    elif periode == 'Q': 
        name_periode = 'trimestres'
    elif periode == 'Y': 
        name_periode = 'années'
        
    # Création du graphique : 
    plt.figure(figsize=(10, 6))

    plt.plot(ls_periods_numeric, ari_values, marker='o', linestyle='-', label='Évolution de l\'ARI')

    plt.legend(loc='lower left')
    plt.xlabel(f'Nbre de {name_periode} depuis le début de collecte des données')
    plt.ylabel('ARI')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()