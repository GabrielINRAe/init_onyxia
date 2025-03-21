# Pour charger vos images dans le workspace tapez la commande en changeant votre nom d'utilisateur :
# mc cp -r s3/gabgab/diffusion/images /home/onyxia/work/data

import os
import seaborn as sns
import geopandas as gpd
from osgeo import gdal, ogr, gdal_array
import numpy as np
import pandas as pd
import glob
import matplotlib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from shapely.geometry import Point
import sys
import fonctions.read_and_write as rw
import fonctions.classification as cla
# from rasterstats import zonal_stats
from sklearn.metrics import (confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from collections import defaultdict

def filter_classes(dataframe, valid_classes):
    """
    Filtre les classes de la BD Forêt.
    """
    return dataframe[dataframe['TFV'].isin(valid_classes)]

def sel_classif_pixel(dataframe):
    """
    Sélectionne seulement les classes pour la classification à l'échelle des pixels
    """
    codes = [11, 12, 13, 14, 21, 22, 23, 24, 25]
    return dataframe[dataframe['Code'].isin(codes)]

def count_polygons_by_class(dataframe, class_column='classif_objet'):
    """
    Compte le nombre de polygones par classe.
    """
    return dataframe.groupby(class_column).size().reset_index(name='count')


def compute_ndvi(masque, ref_raster_path, l_traitements ):

    """
    Calcule le NDVI.

    Parameters:
        masque (array): Masque sous forme de tableau numpy.
        ref_raster_path (str): Chemin jusqu'à un raster de référence.
        l_traitements (list): Liste des prétraitements sur lesquels on va calculer le ndvi.

    Returns:
        Le tableau du ndvi masqué.
    """
    # Pour les 6 dates 
    x,y = rw.get_image_dimension(rw.open_image(ref_raster_path))[:2]
    bandes = 6

    dates = ["20220125","20220326","20220405","20220714","20220922","20221111"] # Liste des 6 dates
    nir_name = 'B8.'
    r_name = 'B4.'

    ndvi_blank = np.zeros((x,y,bandes), dtype=np.float32)  # Créer un array NDVI avec les mêmes dimensions que nir

    print("Calcul des NDVI")
    for i,date in enumerate(dates) :
        for img in l_traitements:
            if date in img and r_name in img :
                red = rw.load_img_as_array(img)[:,:,0].astype('float32')
            if date in img and nir_name in img :
                nir = rw.load_img_as_array(img)[:,:,0].astype('float32')
        nominator = nir-red
        nominator_masked = np.where(nominator >= 0, nominator, 0)
        denominator = nir+red
        ndvi_blank[:,:,i] = np.where(denominator != 0, nominator_masked/denominator, 0)
    ndvi_masked = np.where(masque == 1, ndvi_blank, int(-9999))
    return ndvi_masked

def plot_bar(data, title, xlabel, ylabel, output_path):
    """
    Génère un diagramme en bâtons.
    """
    plt.figure(figsize=(10, 6))
    data.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def violin_plot(
    df, x_col, y_col, output_file, title="", xlabel="", ylabel="", palette="muted", figsize=(12, 8)
):
    """
    Crée un graphique de type violin plot pour visualiser la distribution des données autour d'une valeur moyenne.

    Parameters:
        df (pd.DataFrame): DataFrame contenant les données à tracer.
        output_file (str): Chemin et nom du fichier pour enregistrer le graphique.
        title (str, optional): Titre du graphique.
        xlabel (str, optional): Étiquette de l'axe X.
        ylabel (str, optional): Étiquette de l'axe Y.
        palette (str, optional): Palette de couleurs pour le graphique.
        figsize (tuple, optional): Taille de la figure.

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    sns.violinplot(data=df, x=x_col, y=y_col, hue=x_col, palette=palette, legend=False)
    plt.xlabel(xlabel if xlabel else x_col, fontsize=12)
    plt.ylabel(ylabel if ylabel else y_col, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def clip_raster_with_shapefile(raster_path, shapefile_path, output_path):
    """
    Découpe un raster selon l'emprise d'un shapefile en utilisant GDAL.
    """
    gdal.UseExceptions()
    raster = gdal.Open(raster_path)
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()
    options = gdal.WarpOptions(
        cutlineDSName=shapefile_path,
        cropToCutline=True,
        dstNodata=0,
        outputBoundsSRS="EPSG:2154"
    )
    gdal.Warp(output_path, raster, options=options)


def reproject_and_resample(input_path, output_path, resolution=10):
    """
    Reprojette et rééchantillonne un raster en Lambert 93 à une résolution de 10 m.
    """
    gdal.UseExceptions()
    raster = gdal.Open(input_path)
    options = gdal.WarpOptions(
        xRes=resolution,
        yRes=resolution,
        dstSRS="EPSG:2154",
        resampleAlg="bilinear",
        dstNodata=0
    )
    gdal.Warp(output_path, raster, options=options)


def save_raster(data, ref_raster_path, output_path, dtype, nodata):
    """
    Sauvegarde d'une image raster en utilisant GDAL.
    """
    ref = gdal.Open(ref_raster_path)
    driver = gdal.GetDriverByName('GTiff')


def supprimer_dossier_non_vide(dossier):
    '''
    Permet de supprimer un dossier contenant des fichiers
    '''
    # Parcourir tout le contenu du dossier
    for element in os.listdir(dossier):
        chemin_element = os.path.join(dossier, element)
        # Vérifier si c'est un fichier
        if os.path.isfile(chemin_element) or os.path.islink(chemin_element):
            os.remove(chemin_element)  # Supprimer le fichier ou le lien
        elif os.path.isdir(chemin_element):
            supprimer_dossier_non_vide(chemin_element)  # Appel récursif pour les sous-dossiers
    # Supprimer le dossier une fois qu'il est vide
    os.rmdir(dossier)

def report_from_dict_to_df(dict_report):
    '''
    Permet de convertir en DataFrame un dictionnaire retourné par la fonction classification_report
    '''
    # convert report into dataframe
    report_df = pd.DataFrame.from_dict(dict_report)

    # drop unnecessary rows and columns
    try :
        report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=1)
    except KeyError:
        print(dict_report)
        report_df = report_df.drop(['micro avg', 'macro avg', 'weighted avg'], axis=1)

    report_df = report_df.drop(['support'], axis=0)

    return report_df


def create_bar_plot(data, output_path):
    """
    Crée un graphique en bâton pour les distances moyennes au centroïde.
    data: dict {class: avg_distance}
    output_path: str, chemin du fichier de sortie
    """
    classes = list(data.keys())
    distances = list(data.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, distances, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Distance moyenne au centroïde')
    plt.title('Distance moyenne au centroïde par classe')
    plt.savefig(output_path)
    plt.close()

def create_violin_plot(polygon_distances, violin_plot_dist_centroide_by_poly_by_class_path):
    """
    Crée un graphique en violon pour visualiser les distances moyennes au centroïde par classe.

    Parameters:
    - polygon_distances (dict): Dictionnaire où les clés sont les noms des classes et les valeurs
      sont des listes de distances moyennes des polygones de chaque classe.
    - violin_plot_dist_centroide_by_poly_by_class_path (str): Chemin complet pour sauvegarder le graphique.

    """

    # Créer les données pour le graphique
    class_names = list(polygon_distances.keys())
    distances = [polygon_distances[cls] for cls in class_names]

    # Créer le graphique en violon
    plt.figure(figsize=(12, 8))
    
    # Vérifiez la version de Matplotlib et appliquez le paramètre approprié
    if matplotlib.__version__ >= "3.4.0":
        plt.violinplot(distances, showmeans=True, showextrema=True, showmedians=True)
    else:
        plt.violinplot(distances, showmeans=True, showextrema=True, showmedians=True)

    # Ajouter des labels et un titre
    plt.xticks(ticks=range(1, len(class_names) + 1), labels=class_names, rotation=45, fontsize=10)
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("Distances moyennes au centroïde", fontsize=12)
    plt.title("Distribution des distances moyennes au centroïde par polygone et par classe", fontsize=14)

    # Sauvegarder le graphique
    plt.tight_layout()
    plt.savefig(violin_plot_dist_centroide_by_poly_by_class_path, dpi=300)
    plt.close()


def masque_shp(path_input, path_output):
    """
    Permet la création du masque en format shp à partir du fichier formation végétale shp.

    Parameters:
        path_input (str): Chemin du fichier pour accéder au fichier formation végétale
        path_output (str) : Chemin du fichier pour enregistrer le masque

    Returns:
        None
    """
    f_vege = gpd.read_file(path_input)    # Mettre le path en paramètre
    L_mask = ['Lande',
          'Formation herbacée',
          'Forêt ouverte de conifères purs',
          'Forêt ouverte de feuillus purs',
          'Forêt ouverte sans couvert arboré',
          'Forêt ouverte à mélange de feuillus et conifères',
          'Forêt fermée sans couvert arboré']   # Liste des classes à masquer
    ones = np.ones((24041,1),dtype=int)      # Création d'un vecteur de 1
    f_vege.loc[:,'value'] = ones             # Ajout de la colonne value remplis de 1
    # Valeur 0 pour les classes à masquer
    for i,j in zip(f_vege['TFV'],range(len(f_vege['value']))):
        if i in L_mask:
            f_vege.loc[j,'value'] = 0
    # Ajout de la colonne Classe
    for i in range(len(f_vege['value'])):
        if f_vege['value'][i] == 1:
            f_vege.loc[i,'Classe'] = 'Zone de forêt'
        else:
            f_vege.loc[i,'Classe'] = 'Zone hors forêt'

    Masque = f_vege[['ID','Classe','value','geometry']]    # Sélections des colonnes d'intérêt
    Masque.loc[:,'value'] = Masque['value'].astype('uint8')   # Conversion de la colonne value en uint8

    Masque.to_file(path_output)  # Enregistrement du masque
    return None


def rasterization (
    in_vector,
    out_image,
    field_name,
    sp_resol = None,
    emprise = None,
    ref_image = None,
    dtype = None):
    """
    Rasterise un fichier vectoriel.

    Parameters:
        in_vector (str): Chemin du fichier vectoriel à rasteriser.
        out_image (str): Chemin du fichier raster en sortie.
        field_name (str): Nom de la colonne du vecteur à rasteriser.
        sp_resol (str,optional): Résolution spatiale du fichier à rasteriser.
        emprise (str, optional): Chemin du fichier emprise sur lequel rasteriser.
        ref_image (str, optional): Chemin du fichier image référence pour la rasterisation.
        dtype (str, optional) : Type de données en sortie.

    Returns:
        None
    """
    if emprise is not None :
        xmin,ymin,xmax,ymax=emprise.total_bounds
    else :
        ref_image_open = rw.open_image(ref_image)
        if sp_resol is None :
            sp_resol = rw.get_pixel_size(ref_image_open)[0]
        if dtype is None :
            band = ref_image_open.GetRasterBand(1)
            dtype = gdal.GetDataTypeName(band.DataType)
        xmin,ymax = rw.get_origin_coordinates(ref_image_open)
        y,x = rw.get_image_dimension(ref_image_open)[0:2]
        xmax,ymin = xmin+x*10,ymax-y*10
    
    # Créer le répertoire de sortie si nécessaire
    out_dir = os.path.dirname(out_image)
    os.makedirs(out_dir, exist_ok=True)  # Crée les répertoires manquants

    # define command pattern to fill with parameters
    cmd_pattern = ("gdal_rasterize -a {field_name} "
                "-tr {sp_resol} {sp_resol} "
                "-te {xmin} {ymin} {xmax} {ymax} -ot {dtype} -of GTiff "
                "{in_vector} {out_image}")

    # fill the string with the parameter thanks to format function
    cmd = cmd_pattern.format(in_vector=in_vector, xmin=xmin, ymin=ymin, xmax=xmax,
                            ymax=ymax, out_image=out_image, field_name=field_name,
                            sp_resol=sp_resol, dtype = dtype)

    # execute the command in the terminal
    os.system(cmd)
    return None

def apply_decision_rules(class_percentages, samples_path):
    
    """
    Applique des règles de décision pour déterminer la classe prédominante de chaque polygone.

    Arguments :
    - class_percentages : DataFrame contenant une colonne `class_percentages` avec des dictionnaires.
    - samples_path : Chemin vers le fichier des échantillons.

    Retourne :
    - Une liste `code_predit` avec les codes prédits pour chaque polygone.

    """
    code_predit = []  # Liste pour stocker les classes prédites
    samples = gpd.read_file(samples_path)  # Charger les données des échantillons
    samples["Surface"] = samples.geometry.area  # Calculer la surface des polygones

    for index, row in class_percentages.iterrows():
        # Récupérer le dictionnaire des pourcentages pour ce polygone
        class_dict = row["class_percentages"]

        # Surface du polygone
        surface = samples.loc[index, "Surface"] if index in samples.index else 0

        # Identifier la classe dominante et son pourcentage
        if class_dict:  # Vérifier que le dictionnaire n'est pas vide
            dominant_class_name = max(class_dict, key=class_dict.get)  # Classe avec le plus grand pourcentage
            dominant_class_percentage = class_dict[dominant_class_name]  # Pourcentage de cette classe
        else:  # Si le dictionnaire est vide
            dominant_class_name = None
            dominant_class_percentage = 0

    for index, row in class_percentages.iterrows():

        # Calcul des proportions
        sum_feuillus = row.get("11", 0) + row.get("16", 0) + row.get("15", 0)+row.get("12", 0)+row.get("14", 0)+row.get("13", 0)
        sum_coniferes = row.get("21", 0) + row.get("27", 0) + row.get("26", 0)+ row.get("23", 0)+ row.get("25", 0)+ row.get("24", 0)+ row.get("22", 0)

        # Décisions
        if surface < 20000:  # Cas surface < 2 ha
            if sum_feuillus > 75 and sum_coniferes < sum_feuillus: 
                code_predit.append("Feuillus_en_ilots")
            elif sum_coniferes > 75 and sum_coniferes > sum_feuillus: 
                code_predit.append("coniferes_en_ilots")
            elif sum_coniferes > sum_feuillus: 
                code_predit.append("Melange_de_coniferes_preponderants_et_feuillus")
            else:
                code_predit.append("Melange_de_feuillus_preponderants_et_coniferes")
        else:  # Cas surface >= 2 ha
            if dominant_class_percentage > 75:
                code_predit.append(dominant_class_name)
            elif sum_feuillus > 75 and sum_coniferes < 75: 
                code_predit.append("Melange_feuillus")
            elif sum_coniferes > 75 and sum_feuillus < 75: 
                code_predit.append("Melange_coniferes")
            elif sum_coniferes > sum_feuillus:
                code_predit.append("Melange_de_coniferes_preponderants_et_feuillus")
            else:
                code_predit.append("Melange_de_feuillus_preponderants_et_coniferes")
    return code_predit


def compute_confusion_matrix_with_plots(polygons, label_col, prediction_col):
    """
    Calcule la matrice de confusion, affiche les métriques et génère les graphiques demandés.
    :param polygons: GeoDataFrame ou DataFrame contenant les labels et prédictions.
    :param label_col: Nom de la colonne pour les labels vrais.
    :param prediction_col: Nom de la colonne pour les prédictions.
    :param output_dir: Répertoire où sauvegarder les graphiques.
    """
    # Vérification des colonnes
    if label_col not in polygons.columns or prediction_col not in polygons.columns:
        raise ValueError(f"Les colonnes {label_col} et/ou {prediction_col} sont introuvables dans les données.")

    # Suppression des lignes avec des valeurs manquantes dans les colonnes d'intérêt
    polygons = polygons.dropna(subset=[label_col, prediction_col])

    # Récupération des labels vrais et prédits
    y_true = polygons[label_col].astype(str)  # Conversion en chaîne pour éviter les comparaisons avec None
    y_pred = polygons[prediction_col].astype(str)
    print(polygons[[label_col, prediction_col]].head(10))
    # Calcul de la matrice de confusion
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Classification report
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    
    # Normalisation pour les pourcentages
    cm_sum = cm.sum(axis=1)
    cm_sum[cm_sum == 0] = 1  # Évite la division par zéro
    cm_normalized = cm.astype('float') / cm_sum[:, np.newaxis]

    # ---- Création de la heatmap de la matrice de confusion ----
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix with Normalized Values")
    plt.tight_layout()
    plt.show()

    # ---- Création du graphique des métriques (précision, rappel, F1) ----
    metrics = np.array([precision, recall, f1_score])
    metric_names = ["Precision", "Recall", "F1 Score"]

    plt.figure(figsize=(10, 8))
    bar_width = 0.25
    x = np.arange(len(labels))

    for i, metric in enumerate(metrics):
        plt.bar(x + i * bar_width, metric * 100, width=bar_width, label=metric_names[i])

    # Personnalisation des axes
    plt.xlabel("Classes")
    plt.ylabel("Score (%)")
    plt.title("Class quality estimation")
    plt.xticks(x + bar_width, labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Retour des métriques au besoin
    return {
        "confusion_matrix": cm,
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def pre_traitement_img(
    p_emprise,
    l_images,
    input_raster_dir,
    output_dir):

    """
    Rasterise un fichier vectoriel.

    Parameters:
        p_emprise (str): Chemin du fichier vectoriel pour l'emprise du clip.
        l_images (list): Liste des images à traiter.
        input_raster_dir (str): Dossier où les images brutes sont stockées.
        output_dir (str): Chemin du dossier temporaire des output pré-traités.

    Returns:
        None.
    """
    # Charger le vecteur avec Geopandas
    emprise = gpd.read_file(p_emprise).to_crs("EPSG:2154")
    # Extraire le GeoJSON sous forme de string
    print("Chargement du geojson en str")
    geojson_str = emprise.to_json()
    print("Chargement du geojson en str ok!")
    print("Traitements des images")
    for i,img in enumerate(l_images) :
        date = img[11:19]
        bande = img[53:56]
        ds_img = rw.open_image(os.path.join(input_raster_dir,img))
        name_file = f"traitement_{date}_{bande}"+".tif"
        output_file = os.path.join(output_dir,name_file)
        # Appliquer le clip avec GDAL
        resolution = 10  # Résolution (10 m)
        output_raster_test = gdal.Warp(
            output_file, # Chemin de fichier, car on utilise GTiff
            # "",  # Pas de chemin de fichier, car on utilise MEM
            ds_img,  # Fichier en entrée (chemin ou objet gdal)
            format = "GTiff", # Utiliser GTiff comme format
            # format = "MEM",  # Utiliser MEM comme format
            cutlineDSName = geojson_str,  # Passer directement le GeoJSON
            cropToCutline = True,
            outputType = gdal.GDT_UInt16, # UInt16
            dstSRS = "EPSG:2154",  # Reprojection
            xRes = resolution,  # Résolution X
            yRes = resolution,  # Résolution Y
            dstNodata = 0  # Valeur NoData
        )
        print(f"Image {i+1}/{len(l_images)} traitée")
    ds_img = None
    emprise = None
    geojson_str = None
    return None

def concat_numpy(
    ref_raster_path,
    L_images_clip,
    masque_path,
    output_path):

    """
    Concatène les arrays masqués.

    Parameters:
        ref_raster_path (str): Chemin du raster de référence.
        L_images_clip (list): Liste des images clipées.
        masque_path (str): Chemin du masque.
        output_path (str): Chemin de sortie.
    """
    x, y = rw.get_image_dimension(rw.open_image(ref_raster_path))[:2]
    bandes = len(L_images_clip)
    array_tot = np.zeros((x,y,bandes))
    masque = rw.load_img_as_array(masque_path)
    L_array_masqued = []
    for i, img in enumerate(L_images_clip[:bandes]):
        array = rw.load_img_as_array(img)
        array_masqued = np.where(masque == 1, array, 0)
        L_array_masqued.append(array_masqued)
    # Concaténation des arrays masqués
    print("Concaténation en cours")
    array_final_masqued = np.concatenate(L_array_masqued, axis=2)
    print("Tableau concaténé avec masque appliqué")
    print("Ecriture en cours")
    rw.write_image(out_filename=output_path, array=array_final_masqued,
        data_set=rw.open_image(ref_raster_path))
    print("Ecriture terminée")

def id_construction(sample_px, path_sample_px_id):
    """
    Construis un fichier shp avec une colonne "id" sur les polygones.

    Parameters:
        sample_px (str): Chemin du fichier où ajouter les id.
        path_sample_px_id (str): Chemin du fichier id en sortie.

    Returns:
        None
    """
    l_id = [i+1 for i in range(sample_px.shape[0])]
    sample_px_id = sample_px.copy()
    sample_px_id['id'] = l_id
    sample_px_id = sample_px_id[['id','geometry']]
    sample_px_id.to_file(path_sample_px_id)
    return None


def stratified_grouped_validation(
    nb_iter,
    nb_folds,
    sample_filename,
    image_filename,
    id_filename
):
    """
    Réalise l'entrainement et la validation du modèle.

    Parameters:
        nb_iter (int): Nombre d'itération.
        nb_folds (int): Nombre de folds.
        sample_filename (str): Chemin vers le raster des échantillons.
        image_filename (str): Chemin vers l'image sur laquelle entrainer le modèle.
        id_filename (str): Chemin vers le raster des id des polygones.

    Returns:
        rfc: Retourne le modèle.
        list_cm: Liste des matrices de confusions.
        list_accuracy: Liste des scores de précisions.
        list_report: Liste des rapports de classifications.
        Y_predict: Utilisé pour les labels pour le plot.
    """
    # Extraction des échantillons
    X, Y, t = cla.get_samples_from_roi(image_filename, sample_filename)
    _, groups, _ = cla.get_samples_from_roi(image_filename, id_filename)
    list_cm = []   # Stockage des matrices de confusions
    list_accuracy = []    # Stockage des OA
    list_report = []    # Stockage des rapports de classifications
    groups = np.squeeze(groups)
    # Iter on stratified K fold
    for i in range(nb_iter):
        print (f"Début de la {i+1} itération")
        kf = StratifiedGroupKFold(n_splits=nb_folds, shuffle=True)
        for train, test in kf.split(X, Y, groups=groups):
            X_train, X_test = X[train], X[test]
            Y_train, Y_test = Y[train], Y[test]

            # 3 --- Train
            rfc = RandomForestClassifier(
                max_depth = 50,
                oob_score = True,
                max_samples = 0.75,
                class_weight = 'balanced',
                n_jobs = -1
            )
            rfc.fit(X_train, Y_train[:,0])

            # 4 --- Test
            Y_predict = rfc.predict(X_test)

            # compute quality
            list_cm.append(confusion_matrix(Y_test, Y_predict))
            list_accuracy.append(accuracy_score(Y_test, Y_predict))
            report = classification_report(Y_test, Y_predict,
                                            labels=np.unique(Y_predict),
                                            output_dict=True,
                                            zero_division = 0)

            # store them
            list_report.append(report_from_dict_to_df(report))
    return rfc, list_cm, list_accuracy, list_report, Y_predict


def save_classif(
    image_filename,
    model,
    out_classif
):
    """
    Produit la carte finale de classification.

    Parameters:
        image_filename (str): Chemin du fichier vers l'image utilisée pour la classification.
        model (sklearn): Modèle utilisé lors de l'apprentissage.
        out_classif (str): Chemin de sauvegarde de la classif finale.

    Returns:
        None
    """
    X_img, _, t_img = cla.get_samples_from_roi(image_filename, image_filename)
    Y_predict = model.predict(X_img)
    # Get image dimension
    ds = rw.open_image(image_filename)
    nb_row, nb_col, _ = rw.get_image_dimension(ds)
    #initialization of the array
    img = np.zeros((nb_row, nb_col, 1), dtype='uint8')
    img[t_img[0], t_img[1], 0] = Y_predict
    rw.write_image(out_classif, img, data_set=ds, gdal_dtype=gdal.GDT_Byte,
                transform=None, projection=None, driver_name=None,
                nb_col=None, nb_ligne=None, nb_band=1)
    return None


def get_samples_from_roi(raster_name, sample_name, id_image_name, value_to_extract=None,
                         bands=None, output_fmt='full_matrix'):
    '''
    The function gets the set of pixel of an image according to an roi file (raster).
    In case of raster format, both map should be of same size.

    Parameters
    ----------
    raster_name : string
        The name of the raster file, could be any file GDAL can open.
    sample_name : string
        The path of the sample image.
    id_image_name : string
        The path of the raster file containing polygon IDs.
    value_to_extract : float, optional, defaults to None
        If specified, the pixels extracted will be only those which are equal to this value.
        By default, all the pixels different from zero are extracted.
    bands : list of integer, optional, defaults to None
        The bands of the raster_name file whose value should be extracted.
        Indexation starts at 0. By default, all the bands will be extracted.
    output_fmt : {'full_matrix', 'by_label'}, optional
        By default, the function returns a matrix with all pixels present in the sample dataset.
        With option 'by_label', a dictionary containing as many array as labels present in the sample dataset,
        i.e., the pixels are grouped in matrices corresponding to one label, the keys of the dictionary corresponding to the labels.
        The coordinates 't' will also be in dictionary format.

    Returns
    -------
    X : ndarray or dict of ndarray
        The sample matrix. A nXd matrix, where n is the number of referenced pixels and d is the number of variables.
        Each line of the matrix is a pixel.
    Y : ndarray
        The label of the pixel.
    t : tuple or dict of tuple
        Tuple of the coordinates in the original image of the pixels extracted. Allows rebuilding the image from `X` or `Y`.
    P : ndarray
        The ID of the polygon to which each pixel belongs.
    '''

    # lecture des rasters 
    raster = gdal.Open(raster_name)
    sample = gdal.Open(sample_name)
    id_image = gdal.Open(id_image_name)

    # Vérification si les rasters sont bien chargées 
    if raster is None or sample is None or id_image is None:
        raise FileNotFoundError("One or more of the specified raster files could not be opened.")

    # Vérification si le raster de référence et le raster des échantillons ont la même dimension
    if raster.RasterXSize != sample.RasterXSize or raster.RasterYSize != sample.RasterYSize:
        raise ValueError("Images should be of the same size")

    if not bands:
        nb_band = raster.RasterCount
        bands = list(range(nb_band))
    else:
        nb_band = len(bands)

    # Lecture de matrice des échantillons et des identifiants 
    sample_array = sample.GetRasterBand(1).ReadAsArray()
    id_array = id_image.GetRasterBand(1).ReadAsArray()

    if value_to_extract:
        t = np.where(sample_array == value_to_extract)
    else:
        t = np.nonzero(sample_array)

    Y = sample_array[t].reshape((t[0].shape[0], 1)).astype('int32')
    P = id_array[t].reshape((t[0].shape[0], 1)).astype('int32')

    del sample_array
    del id_array
    sample = None  #  fermuture de fichier des échantillons 
    id_image = None  # Fermuture de fichier des idendifiants 

    try:
        X = np.empty((t[0].shape[0], nb_band), dtype=gdal_array.GDALTypeCodeToNumericTypeCode(raster.GetRasterBand(1).DataType))
    except MemoryError:
        print('Impossible to allocate memory: sample too large')
        return

    # chargement des données 
    for i in bands:
        temp = raster.GetRasterBand(i + 1).ReadAsArray()
        X[:, i] = temp[t]
        del temp
    raster = None  # fermeture de raster 

    if output_fmt == 'by_label':
        labels = np.unique(Y)
        dict_X = {}
        dict_t = {}
        dict_P = {}
        for lab in labels:
            coord = np.where(Y == lab)[0]
            dict_X[lab] = X[coord]
            dict_t[lab] = (t[0][coord], t[1][coord])
            dict_P[lab] = P[coord]
        return dict_X, Y, dict_t, dict_P
    else:
        return X, Y, t, P

def main(in_vector, image_filename, sample_filename, id_image_filename):

    X, Y, t, P = get_samples_from_roi(image_filename, sample_filename, id_image_filename)
    points = [Point(x, y) for x, y in zip(t[1], t[0])]
    
    gdf_points = gpd.GeoDataFrame(geometry=points)
    gdf_points['class'] = Y
    gdf_points['polygon_id'] = P

    for i in range(X.shape[1]):
        gdf_points[f'band_{i+1}'] = X[:, i]

    gdf_polygons = gpd.read_file(in_vector)
    gdf_points.set_crs(gdf_polygons.crs, inplace=True)
    gdf_joined = gpd.sjoin(gdf_points, gdf_polygons, how="left", predicate="intersects")

    List_classes = [11, 12, 13, 14, 15, 23, 24, 25, 26, 28, 29]
    gdf_filtered = gdf_joined[gdf_joined['class'].isin(List_classes)]

    return gdf_filtered  

def calcul_distance(group, band_columns):
    difference = group[band_columns].values - group[[f'{band}_centroid' for band in band_columns]].values
    distance = np.sqrt((difference ** 2).sum(axis=1))
    group['distance_euclidienne'] = distance
    return group


def calculate_surface(samples_path, index):
    """
    Calcul la surface de chaque polygone présent dans la couche vecteur d'entrée 
    Paramatères :  sample_path :  chemin vers la couche vecteur d'entrée 
                    index = l'indice associé à chaque polygone 
    """
    samples = gpd.read_file(samples_path)
    if index in samples.index:
        return samples.loc[index, "geometry"].area
    else:
        return 0


def get_dominant_class(class_dict):
    """
    Identifier la classe dominante et son pourcentage à partir du dictionnaire des pourcentages de recouvrement de chaque classe 

    Arguments :

    class_dict : Dictionnaire contenant les classes et leurs pourcentages.
    Retourne : Le code de la classe dominante ainsi que son pourcentage.

    """
    if class_dict:
        dominant_class_code = max(class_dict, key=class_dict.get)
        dominant_class_percentage = class_dict[dominant_class_code]
        return dominant_class_code, dominant_class_percentage
    else:
        return None, 0

def calculate_proportions(class_percentages):
    """
    Calcul la somme des proportions de feuillus et de conifères à partir des données d'une ligne.

    Arguments :

    class_percentages : Dictionnaire contenant les pourcentages des différentes classes.
    Retourne : La somme des pourcentages des feuillus et des conifères.

    """
    feuillus_classes = [11, 12, 13, 14, 15, 16]  # liste des codes associés aux classes feuillus 
    coniferes_classes = [21, 22, 23, 24, 25, 26, 27]  # liste de codes associées aux classe conifère 
    sum_feuillus = 0
    sum_coniferes = 0
    
    for class_value, percentage in class_percentages.items():
        if class_value in feuillus_classes:
            sum_feuillus += percentage
        elif class_value in coniferes_classes:
            sum_coniferes += percentage
    
    return sum_feuillus, sum_coniferes


def make_decision(surface, sum_feuillus, sum_coniferes, dominant_class_percentage, dominant_class_code):
    """
        Applique les règles de décision basées sur la surface et les proportions de classes 

    Arguments :

    surface : Surface du polygone.
    sum_feuillus : Somme des proportions de feuillus.
    sum_coniferes : Somme des proportions de conifères.
    dominant_class_percentage : Pourcentage de la classe dominante.
    dominant_class_code : Code de la classe dominante.
    Retourne :

    Le code prédit pour chaque polygone.

  """
    if surface < 20000:  # superficie < 2 ha
        if sum_feuillus > 75: 
            return 16
        elif sum_coniferes > 75: 
            return 27
        elif sum_coniferes > sum_feuillus: 
            return 28
        else:
            return 29
    else:  # superficie >= 2 ha
        if dominant_class_percentage > 75:
            return int(dominant_class_code)  
        elif sum_feuillus > 75: 
            return 15
        elif sum_coniferes > 75: 
            return 26
        elif sum_coniferes > sum_feuillus:
            return 28
        else:
            return 29

