import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder,StandardScaler,label_binarize
from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, make_scorer, f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_ind
 

# Small plotting helpers

_PLOT_CONTEXT = None  # holds {'dir': Path, 'run_id': str, 'model': str}


def _get_plot_dir(model_name: str, reduction: bool) -> Path:
    """Return directory where plots should be saved for a given model + reduction flag."""
    folder_map = {
        'lr': 'Logistic Regression',
        'logistic regression': 'Logistic Regression',
        'logistic_regression': 'Logistic Regression',
        'rf': 'Random Forest',
        'random forest': 'Random Forest',
        'random_forest': 'Random Forest',
        'xgb': 'XGBoost',
        'xgboost': 'XGBoost',
        'nn': 'Neural Network',
        'neural network': 'Neural Network',
        'neural_network': 'Neural Network',
    }
    key = model_name.lower()
    model_folder = folder_map.get(key, model_name)
    sub = "Reduction" if reduction else "No_reduction"
    base = Path(__file__).resolve().parent
    out_dir = base / model_folder / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _set_plot_context(model_name: str, reduction: bool):
    """Set global plotting context for subsequent helper plots."""
    global _PLOT_CONTEXT
    plot_dir = _get_plot_dir(model_name, reduction)
    _PLOT_CONTEXT = {'dir': plot_dir, 'model': model_name}
    return plot_dir


def plot_confusion(cm, class_names, title=None, save_path=None):
    """Tracer une matrice de confusion (carte de chaleur).

    Parametres
    ----------
    cm : array-like de forme (n_classes, n_classes)
        Comptes de la matrice de confusion.
    class_names : list[str]
        Ordre d'affichage des etiquettes de classes.
    title : str, optionnel
        Titre optionnel du graphique.
    """
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    # Derive default save path from global context if none provided
    if save_path is None and _PLOT_CONTEXT is not None:
        ctx = _PLOT_CONTEXT
        base = (title or 'confusion').replace(' ', '_').replace('/', '_')
        save_path = ctx['dir'] / f"{base}.png"
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_roc(fpr, tpr, auc_val=None, title=None, point=None, point_label=None, save_path=None):
    """Tracer une courbe ROC avec AUC et point optionnels.

    Parametres
    ----------
    fpr, tpr : array-like
        Taux de faux positifs / vrais positifs.
    auc_val : float, optionnel
        Valeur d'AUC a afficher dans la legende.
    title : str, optionnel
        Titre du graphique.
    point : tuple(float, float), optionnel
        Point (FPR, TPR) a mettre en evidence.
    point_label : str, optionnel
        Libelle du point mis en evidence.
    """
    fig, ax = plt.subplots(figsize=(5,4))
    lbl = (f'AUC = {auc_val:.3f}' if auc_val is not None else None)
    ax.plot(fpr, tpr, label=lbl)
    if point is not None:
        px, py = point
        ax.scatter([px], [py], c='red', edgecolor='k', s=60,
                   label=point_label if point_label else 'Selected')
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    if title:
        ax.set_title(title)
    if lbl or point_label:
        ax.legend()
    fig.tight_layout()
    # Derive default save path from global context if none provided
    if save_path is None and _PLOT_CONTEXT is not None:
        ctx = _PLOT_CONTEXT
        base = (title or 'roc').replace(' ', '_').replace('/', '_')
        save_path = ctx['dir'] / f"{base}.png"
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_fold_performance(fold_scores, model_name, reduction):
    """Tracer les scores F1 pour chaque fold (pli)."""
    if not fold_scores:
        print(f"[{model_name}] Pas de scores de fold à tracer.")
        return
    
    plot_dir = _get_plot_dir(model_name, reduction)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    folds = range(1, len(fold_scores) + 1)
    ax.plot(folds, fold_scores, marker='o', linestyle='-', label='F1-Score par Fold')
    ax.set_xlabel('Fold')
    ax.set_ylabel('F1-Score')
    ax.set_title(f'Performance F1 par Fold - {model_name}')
    ax.set_xticks(folds)
    ax.set_ylim(0, 1.05)
    
    # Ligne pour la moyenne
    mean_score = np.mean(fold_scores)
    ax.axhline(mean_score, color='r', linestyle='--', label=f'Moyenne = {mean_score:.3f}')
    ax.legend()
    
    fig.tight_layout()
    save_path = plot_dir / f"fold_performance_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_star_chart(stats, model_name, reduction):
    """Tracer un diagramme radar (star chart) des metriques du modele avec Plotly."""
    df = pd.DataFrame(dict(
        r=list(stats.values()),
        theta=list(stats.keys())
    ))

    plot_dir = _get_plot_dir(model_name, reduction)
    fig = px.line_polar(df, r='r', theta='theta', line_close=True,
                        title=f'Radar des Metriques - {model_name}')
    fig.update_traces(fill='toself')

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )
      ),
      showlegend=False
    )
    
    save_path = plot_dir / f"star_chart_{model_name.lower().replace(' ', '_')}.png"
    try:
        fig.write_image(str(save_path))
    except ValueError as e:
        if 'kaleido' in str(e).lower():
             print("Sauvegarde de l'image echouee: le package 'kaleido' est requis.")
             print("Veuillez l'installer avec : pip install kaleido")
        else:
            print(f"Erreur lors de la sauvegarde du star chart pour {model_name}: {e}")
    try:
        fig.show()
    except ValueError as e:
        if 'nbformat' in str(e).lower():
            print("\n--- ERREUR D'AFFICHAGE PLOTLY ---")
            print("Pour afficher ce graphique, le package 'nbformat' doit etre mis a jour.")
            print("Veuveuillez exécuter la commande suivante dans une cellule de votre notebook :")
            print("!pip install --upgrade nbformat")
            print("Apres l'installation, redemarrez le noyau (Kernel -> Restart) et re-executez les cellules.")
            print("---------------------------------\n")
        else:
            # Re-raise other ValueErrors
            raise e

# Fonctions utilisées pour créer et entrainer les réseaux de neurones.
def build_model(input_dim: int, num_classes: int, lr: float = 1e-3, l2_rate: float = 1e-2) -> keras.Model:
    """
    
    Fonction utilisee pour construire un reseau de neurones dense pour la classification.

    Paramètres
    ----------
    input_dim : int
        Nombre de variables d'entrée (features).
    num_classes : int
        Nombre de classes cibles. Sigmoïde si 2 classes, sinon softmax.
    lr : float, par défaut 3e-4
        Taux d'apprentissage de l'optimiseur Adam.
    l2_rate : float, par défaut 1e-2
        Facteur de regularisation L2 (Ridge) pour eviter le surapprentissage.

    Retourne
    --------
    callable
        Un modele Keras compile.
    """
    
    # Le réseau de neurones
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, kernel_regularizer=keras.regularizers.l2(l2_rate)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(32, kernel_regularizer=keras.regularizers.l2(l2_rate)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') if num_classes == 2 else layers.Dense(num_classes, activation='softmax'),
    ])
    
    # Fonction perte
    loss = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'
    
    # Metriques de suivie
    metrics = ['accuracy'] + ([keras.metrics.AUC(name='auc')] if num_classes == 2 else [])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=metrics)
    return model

class PrefitNN:
    """ Wrapper pour utiliser CalibratedClassifierCV avec un reseau de neurones keras pre-entraine."""
    _estimator_type = 'classifier'
    def __init__(self, keras_model, verbose=0):
        self.model = keras_model
        self.classes_ = np.array([0,1])
        self.verbose = verbose
    def fit(self, X, y):
        return self
    def predict_proba(self, X):
        p_healthy = self.model.predict(X, verbose=self.verbose).ravel()
        p_disease = 1.0 - p_healthy
        return np.vstack([p_healthy, p_disease]).T
    def predict(self, X) : 
        proba = self.predict_proba(X)
        return (proba[:,1] >= 0.5).astype(int)

def _evaluate_nn(model, X_val, y_val_int, X_test, y_test_int, plot_dir, verbose=True, random_state=42, reduction=False):
    """Fonction interne pour evaluer, calibrer et tracer les resultats du reseau de neurones."""

    # Preparation des labels (healthy=0, disease=1)
    class_names = ['healthy', 'disease']
    y_test_lbl = np.where(y_test_int == 1, 'disease', 'healthy')
    y_test_bin = y_test_int  # La classe positive (disease) est déjà 1
    y_val_bin = y_val_int    # La classe positive (disease) est déjà 1

    # 1. Evaluation Non Calibree (Test)
    # La sortie sigmoïde du modèle est P(disease) car disease=1
    proba_pos_uncal_test = model.predict(X_test, verbose=verbose).ravel()
    
    preds_uncal = (proba_pos_uncal_test >= 0.5).astype(int)
    y_pred_uncal_lbl = np.where(preds_uncal == 1, 'disease', 'healthy')
    
    cm_uncal = confusion_matrix(y_test_lbl, y_pred_uncal_lbl, labels=class_names)
    print("Report NN (uncalibrated):\n", classification_report(y_test_lbl, y_pred_uncal_lbl, target_names=class_names))

    if verbose:
        plot_confusion(cm_uncal, class_names, title='Confusion matrix NN (uncalibrated, test)', save_path=plot_dir / "cm_uncal.png")
    
    auc_uncal = roc_auc_score(y_test_bin, proba_pos_uncal_test)
    fpr_uncal, tpr_uncal, _ = roc_curve(y_test_bin, proba_pos_uncal_test)
    
    if verbose:
        plot_roc(fpr_uncal, tpr_uncal, auc_val=auc_uncal, title='ROC NN (uncalibrated, test)', save_path=plot_dir / "roc_uncal.png")

    # 2. Calibration (sur Validation)
    base_est = PrefitNN(model, verbose=verbose)
    calibrator_nn = CalibratedClassifierCV(estimator=base_est, method='sigmoid', cv='prefit')
    calibrator_nn.fit(X_val, y_val_bin)
    
    # Visualisation Calibration
    # L'index de la classe positive (disease=1) est 1
    pos_idx_cal = 1
    proba_pos_cal_val = calibrator_nn.predict_proba(X_val)[:, pos_idx_cal]
    
    if verbose:
        # Histogramme
        fig_hist2, ax_hist2 = plt.subplots(figsize=(5,4))
        ax_hist2.hist(proba_pos_cal_val, bins=20, range=(0,1), alpha=0.8, edgecolor='k')
        ax_hist2.set_xlabel('Predicted probability (disease)')
        ax_hist2.set_ylabel('Frequency')
        ax_hist2.set_title('Histogram of calibrated probabilities (calibration set)')
        fig_hist2.tight_layout()
        fig_hist2.savefig(plot_dir / "hist_calib.png", bbox_inches='tight')
        plt.show()

        # Courbe de fiabilite
        proba_pos_uncal_val = model.predict(X_val, verbose=verbose).ravel()
        frac_cal, mean_cal = calibration_curve(y_val_bin, proba_pos_cal_val, n_bins=10, strategy='uniform')
        frac_unc, mean_unc = calibration_curve(y_val_bin, proba_pos_uncal_val, n_bins=10, strategy='uniform')
        
        fig_cal, ax_cal = plt.subplots(figsize=(5,4))
        ax_cal.plot(mean_unc, frac_unc, 'o--', label='Uncalibrated')
        ax_cal.plot(mean_cal, frac_cal, 'o-', label='Calibrated')
        ax_cal.plot([0,1],[0,1],'k--', label='Ideal')
        ax_cal.set_xlabel('Mean predicted probability'); ax_cal.set_ylabel('Fraction of positives')
        ax_cal.set_title('Calibration curve NN (validation/calibration set)')
        ax_cal.legend(); fig_cal.tight_layout()
        fig_cal.savefig(plot_dir / "calibration_curve.png", bbox_inches='tight')
        plt.show()

    # 3. Evaluation Calibrée (Test)
    proba_pos_cal_test = calibrator_nn.predict_proba(X_test)[:, pos_idx_cal]
    auc_cal = roc_auc_score(y_test_bin, proba_pos_cal_test)
    fpr_cal, tpr_cal, _ = roc_curve(y_test_bin, proba_pos_cal_test)
    
    if verbose:
        plot_roc(fpr_cal, tpr_cal, auc_val=auc_cal, title='ROC NN (calibrated, test)', save_path=plot_dir / "roc_cal.png")

    preds_cal = (proba_pos_cal_test >= 0.5).astype(int)
    y_pred_cal_lbl = np.where(preds_cal == 1, 'disease', 'healthy')
    cm_cal = confusion_matrix(y_test_lbl, y_pred_cal_lbl, labels=class_names)
    print("Report NN (calibrated, 0.5):\n", classification_report(y_test_lbl, y_pred_cal_lbl, target_names=class_names))
    
    if verbose:
        plot_confusion(cm_cal, class_names, title='Confusion matrix NN (calibrated, test)', save_path=plot_dir / "cm_cal.png")

    # 4. Seuil Optimal sur la courbe ROC
    fpr_v, tpr_v, thr_v = roc_curve(y_val_bin, proba_pos_cal_val)
    J = tpr_v - fpr_v
    best_thresh = float(thr_v[np.argmax(J)])
    print(f"Best threshold (Youden J) NN on calib: {best_thresh:.4f}")

    if verbose:
        auc_val_set = roc_auc_score(y_val_bin, proba_pos_cal_val)
        plot_roc(fpr_v, tpr_v, auc_val=auc_val_set, title='ROC NN (calibrated, validation) + Youden',
                point=(fpr_v[np.argmax(J)], tpr_v[np.argmax(J)]), point_label=f'Th={best_thresh:.3f}', save_path=plot_dir / "roc_cal_threshold.png")
    
    preds_thr = (proba_pos_cal_test >= best_thresh).astype(int)
    y_pred_thr_lbl = np.where(preds_thr == 1, 'disease', 'healthy')
    cm_thr = confusion_matrix(y_test_lbl, y_pred_thr_lbl, labels=class_names)
    print("Report NN (calibrated + threshold):\n", classification_report(y_test_lbl, y_pred_thr_lbl, target_names=class_names))
    
    if verbose:
        plot_confusion(cm_thr, class_names, title='Confusion matrix NN (calibrated + threshold, test)', save_path=plot_dir / "cm_cal_threshold.png")

    # 5. Choix du meilleur modele (calibre vs non-calibre) base sur le rappel 'disease'
    report_uncal_dict = classification_report(y_test_lbl, y_pred_uncal_lbl, output_dict=True)
    report_thr_dict = classification_report(y_test_lbl, y_pred_thr_lbl, output_dict=True)

    recall_uncal = report_uncal_dict['disease']['recall']
    recall_thr = report_thr_dict['disease']['recall']

    print(f"\n[INFO] Comparaison du rappel (disease): Non-calibré={recall_uncal:.3f} vs Calibré+Seuil={recall_thr:.3f}")

    use_calibrated = recall_thr >= recall_uncal

    if use_calibrated:
        print("[INFO] Le modèle calibré est meilleur ou équivalent. Utilisation des résultats calibrés.")
        report_final = report_thr_dict
        cm_final = cm_thr
        final_auc, final_fpr, final_tpr, final_threshold = auc_cal, fpr_cal, tpr_cal, best_thresh
    else:
        print("[INFO] Le modèle non-calibré est meilleur. Utilisation des résultats non-calibrés.")
        report_final = report_uncal_dict
        cm_final = cm_uncal
        final_auc, final_fpr, final_tpr, final_threshold = auc_uncal, fpr_uncal, tpr_uncal, 0.5

    fold_scores = []
    if use_calibrated:
        print("[INFO] Calcul des scores de fold sur le modèle calibré.")
        try:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            for tr_idx, va_idx in skf.split(X_val, y_val_bin):
                cal_cv = CalibratedClassifierCV(estimator=base_est, method='sigmoid', cv='prefit')
                cal_cv.fit(X_val[tr_idx], y_val_bin[tr_idx])
                pos_idx_cv = 1 # disease
                proba_tr_cv = cal_cv.predict_proba(X_val[tr_idx])[:, pos_idx_cv]
                fpr_tr_cv, tpr_tr_cv, thr_tr_cv = roc_curve(y_val_bin[tr_idx], proba_tr_cv)
                th_cv = float(thr_tr_cv[np.argmax(tpr_tr_cv - fpr_tr_cv)])
                proba_va_cv = cal_cv.predict_proba(X_val[va_idx])[:, pos_idx_cv]
                y_va_pred_cv = (proba_va_cv >= th_cv).astype(int)
                fold_scores.append(f1_score(y_val_bin[va_idx], y_va_pred_cv))
        except Exception as e:
            print(f"Cross-validation for fold_scores failed: {e}")
    else:
        print("[INFO] Le modèle non-calibré a été choisi, le calcul des scores de fold est ignoré.")

    mean_fold_scores = np.mean(fold_scores) if fold_scores else 0.0
    
    if verbose:
        # Graphique des performances par fold
        plot_fold_performance(fold_scores, 'Neural Network', reduction=reduction)
        
        # Radar chart des métriques
        star_stats = {
            'AUC': final_auc,
            'Précision': report_final["disease"]["precision"],
            'Rappel': report_final["disease"]["recall"],
            'F1-Score': report_final["disease"]["f1-score"],
            'Mean Fold F1': mean_fold_scores
        }
        plot_star_chart(star_stats, 'Neural Network', reduction=reduction)

    return {
        'auc_uncalibrated': auc_uncal, 'auc_calibrated': final_auc, 'best_threshold': final_threshold,
        'cm_uncalibrated': cm_uncal, 'cm_calibrated': cm_cal, 'cm_threshold': cm_thr, 'cm_final': cm_final,
        'precision': report_final["macro avg"]["precision"], 'accuracy': report_final["accuracy"],
        'recall': report_final["macro avg"]["recall"], 'f1-score': report_final["macro avg"]["f1-score"],
        'fold_scores': fold_scores,
        'mean_fold_scores': mean_fold_scores,
        'fpr': final_fpr,
        'tpr': final_tpr
    }

def train_model(model: callable, df: pd.DataFrame, label_col: str = 'diagnosis',
                test_size: float = 0.3, batch_size: int = 16, epochs: int = 200,
                random_state: int = 42, reduction: bool = False,
                verbose: bool = True, reduction_threshold: float = None):
    """
    Fonction permettant d'entrainer le reseau de neurones.
    
    
    Paramètres : 
    ------------
    
    model : callable
        Un modèle Keras compilé.
    df : pd.DataFrame
        Matrice de données
    label_col : str, par défaut 'diagnosis'
        Nom de la colonne contenant les etiquettes
    test_size : float, par défaut 0.3
        Proportion utilisée pour le test
    batch_size : int, par défaut 16
        Taille du lot
    epochs : int, par défaut 200
        Nombre d'epoques
    random_state : int, par défaut 42
        Graine aleatoire pour les decoupages
    reduction : bool, par défaut False
        Si on doit faire la reduction de dimensions ou pas
    verbose : bool, para défaut True
        Si True, affiche les graphiques; sinon les supprime.
    reduction_threshold : float, par défaut None 
        Le seuil d'importance pour la selection (Random Forest). Si None, utilise la moyenne.
        
    Retourne : 
    ----------
        result_dict : dict 
            Un dictionnaire contenant l'ensemble des informations nécessaires pour l'étude.
    """
    # Mise en place du dossier pour la sauvegarde
    plot_dir = _set_plot_context('Neural Network', reduction)

    # Récupération des features et des labels
    X_df = df.drop(columns=[label_col])
    y = df[label_col].astype(str).values

    # Encodage en binaire : 'healthy'->0, 'disease'->1
    le = LabelEncoder().fit(['healthy', 'disease'])
    y_int = le.transform(y)

    # 1. Premier Split : Train+Val vs Test
    # Le Test set est mis de côté définitivement pour l'évaluation finale impartiale.
    X_tr_df, X_test_df, y_tr_int, y_test_int = train_test_split(
        X_df, y_int, test_size=test_size, stratify=y_int, random_state=random_state
    )

    # Si reduction --> Sélection de features via Random Forest pour récupérer les colonnes les plus importantes
    if reduction:
        selected_features = get_best_features(X_tr_df, y_tr_int, p_thresh=reduction_threshold)
        X_tr_df = X_tr_df.loc[:, selected_features]
        X_test_df = X_test_df.loc[:, selected_features]
        if verbose:
            th_str = f"{reduction_threshold:.4f}" if reduction_threshold is not None else "mean"
            print(f"[train_model] Réduction de variables activée (seuil={th_str}): "
                  f"{len(selected_features)} / {X_df.shape[1]} colonnes conservées")
            
        # Recréation du réseau de neurones avec le bon nombre d'entrée
        model = build_model(len(selected_features), 2, lr=1e-3, l2_rate=1e-2)
        
    # 2. Deuxième Split : Train vs Validation
    # Le Validation set sert à l'Early Stopping (éviter le surapprentissage)
    # ET à la calibration (ajuster les probabilités sur des données non vues par l'entraînement).
    X_tr_df, X_val_df, y_tr_int, y_val_int = train_test_split(
        X_tr_df, y_tr_int, test_size=0.2, stratify=y_tr_int, random_state=random_state
    )

    # Scale using train only
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_df.values)
    X_val = scaler.transform(X_val_df.values)
    X_test = scaler.transform(X_test_df.values)

    # Class weights
    classes = np.array([0,1])
    cw_vals = compute_class_weight('balanced', classes=classes, y=y_tr_int)
    class_weights = {int(c): float(w) for c, w in zip(classes, cw_vals)}

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5),
    ]

    # Train
    history = model.fit(
        X_tr, y_tr_int,
        validation_data=(X_val, y_val_int),
        epochs=epochs, batch_size=batch_size,
        class_weight=class_weights, shuffle=True, verbose=verbose,
        callbacks=callbacks
    )

    # Training history
    history_df = pd.DataFrame(history.history)
    
    if verbose : 
        ax_hist_1 = history_df[['loss','val_loss']].plot(title='Training history')
        fig_hist_1 = ax_hist_1.get_figure()
        fig_hist_1.tight_layout()
        fig_hist_1.savefig(plot_dir / f"training_history.png", bbox_inches='tight')
        plt.show()
        
        
        

    # Evaluation et Calibration
    results = _evaluate_nn(
        model, X_val, y_val_int, X_test, y_test_int, 
        plot_dir=plot_dir, verbose=verbose, random_state=random_state, reduction=reduction
    )
    
    # Ajout des objets d'entraînement aux résultats
    results.update({
        'model': model,
        'history': history_df,
    })
    
    return results


# Entrainement avec les autres modèles (classical ML)
def train_with_calibration(
    model_key: str,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    calib_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1,
    method: str = 'sigmoid',
    verbose: bool = True,
    reduction: bool = False,
    reduction_threshold: float = None,
):
    """
    Pipeline de bout en bout pour l'entraînement, la calibration et l'évaluation.

    Hypothèses:
      - Classification binaire avec étiquettes exactement {'disease','healthy'}
      - 'disease' est la classe positive (1) pour les métriques et le seuillage
      - Le même random_state garantit les mêmes découpages pour comparer les modèles

    Déroulé: split train/val/test, recherche d'hyperparamètres (modèles classiques),
    évaluation, calibration sur un set de calibration, ré‑évaluation, calcul du
    seuil optimal de Youden sur la calibration et application au test.

      - 'lr' : Régression Logistique (StandardScaler + LogisticRegression)
      - 'xgb': XGBoost (XGBClassifier; labels encodés en interne)
      - 'rf' : Random Forest (RandomForestClassifier)

    Le seuil optimal (Youden) est calculé uniquement sur le set de calibration,
    puis appliqué au test.

    Paramètres
    ----------
    model_key : str
        L'un de {'lr','xgb','rf'}.
    X : array-like de forme (n_samples, n_features)
        Matrice de caracteristiques.
    y : array-like de forme (n_samples,)
        Etiquettes cibles ('disease'/'healthy').
    test_size : float, par défaut 0.3
        Proportion utilisée pour le test.
    calib_size : float, par défaut 0.2
        Proportion du trainval reservee a la calibration.
    random_state : int, par défaut 42
        Graine aleatoire pour les splits.
    n_jobs : int, par défaut -1
        Parallelisme pour les modeles scikit-learn.
    method : str, par défaut 'sigmoid'
        Methode de calibration scikit-learn.
    verbose : bool, par défaut True
        Si True, affiche les graphiques; sinon les supprime.
    """

    key = model_key.lower().strip()
    # Directory for saving plots (and set global context)
    plot_dir = _set_plot_context(key, reduction)
    # Ordre coherent pour les labels et matrices de confusion
    class_names = ['healthy', 'disease']
    if set(np.unique(y)) != set(class_names):
        raise ValueError("y must contain exactly {'disease','healthy'}")

    # Ensure we keep feature names if X is a DataFrame
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(np.asarray(X))

    # 1) Split global: trainval / test
    X_trainval_df, X_test_df, y_trainval, y_test = train_test_split(
        X_df, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 2) Split trainval -> search_train / calib (calibration holdout)
    X_search_df, X_calib_df, y_search, y_calib = train_test_split(
        X_trainval_df, y_trainval, test_size=calib_size, stratify=y_trainval, random_state=random_state
    )

    # Reduction de features optionnelle sur le set de recherche via Random Forest
    if reduction:
        selected_features = get_best_features(X_search_df, y_search, p_thresh=reduction_threshold)
        X_search_df = X_search_df.loc[:, selected_features]
        X_calib_df = X_calib_df.loc[:, selected_features]
        X_test_df = X_test_df.loc[:, selected_features]
        if verbose:
            th_str = f"{reduction_threshold:.4f}" if reduction_threshold is not None else "moyenne"
            print(f"[train_with_calibration] Reduction de features activee (seuil={th_str}): "
                  f"{len(selected_features)} / {X_df.shape[1]} features conservees")

    # Convert back to numpy arrays for scikit-learn
    X_search = X_search_df.values
    X_calib = X_calib_df.values
    X_test = X_test_df.values

    # XGBoost requiert des labels numeriques. On force l'ordre healthy=0, disease=1.
    if key == 'xgb':
        le_y = LabelEncoder().fit(['healthy', 'disease'])
        y_search_enc = le_y.transform(y_search)
        y_calib_enc = le_y.transform(y_calib)
        y_test_enc = le_y.transform(y_test)
        pos_label_xgb = int(le_y.transform(['disease'])[0])

    # 3) Build pipeline and param grid per model
    if key == 'lr':
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, random_state=random_state))
        ])
        param_grid = {
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'clf__solver': ['saga'],
                'clf__multi_class': ['ovr', 'multinomial'],
                'clf__class_weight': [None, 'balanced'],
            }
        scoring = make_scorer(f1_score, pos_label='disease')

    elif key == 'xgb':
        pipe = Pipeline([
            ('clf', XGBClassifier(objective='binary:logistic',eval_metric='logloss',use_label_encoder=False))
        ])
        
        param_grid = {
            'clf__n_estimators' : [100,300,500], 
            'clf__max_depth' : [3,5,7], 
            'clf__learning_rate' : [0.01,0.05,0.1], 
            'clf__subsample' : [0.8,1.0], 
            'clf__colsample_bytree' : [0.8,1.0]
        }
        # Use encoded positive label for XGB scoring
        scoring = make_scorer(f1_score, pos_label=pos_label_xgb) 

    elif key == 'rf':
        pipe = Pipeline([
            ('clf', RandomForestClassifier(random_state=random_state, n_jobs=n_jobs))
        ])
        param_grid = {
            'clf__n_estimators': [200, 500],
            'clf__max_depth': [None, 5, 10],
            'clf__max_features': ['sqrt', 'log2'],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__bootstrap': [True],
            'clf__class_weight': [None, 'balanced'],
        }
        scoring = make_scorer(f1_score, pos_label='disease') 

    else:
        raise ValueError("model_key must be one of {'lr','xgb','rf'}")

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=3,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )
    
    
    # Fit grid with encoded y for XGB, otherwise original y
    if key == 'xgb' and le_y is not None:
        grid.fit(X_search, y_search_enc)
    else:
        grid.fit(X_search, y_search)
    
    
    best_pipe = grid.best_estimator_

    # Initial evaluation (uncalibrated) on test
    y_pred_test_uncal = best_pipe.predict(X_test)
    # Inverse-transform numeric predictions to strings for XGB
    if key == 'xgb' and le_y is not None:
        y_pred_test_uncal = le_y.inverse_transform(y_pred_test_uncal)
    cm_uncal = confusion_matrix(y_test, y_pred_test_uncal, labels=class_names)
    print(f"Report {key.upper()} (uncalibrated):\n", classification_report(y_test, y_pred_test_uncal, labels=class_names))

    proba_test_uncal = None
    auc_uncal = None
    if hasattr(best_pipe, 'predict_proba'):
        proba_test_uncal = best_pipe.predict_proba(X_test)
        if key == 'xgb' and le_y is not None:
            # Dynamically find the index of the positive class (disease, encoded as 1)
            pos_idx = list(best_pipe.classes_).index(pos_label_xgb)
            y_test_bin = (y_test_enc == pos_label_xgb).astype(int)
        else:
            pos_idx = list(best_pipe.classes_).index("disease")
            y_test_bin = (y_test == "disease").astype(int)
        fpr_uncal, tpr_uncal, _ = roc_curve(y_test_bin, proba_test_uncal[:, pos_idx])
        auc_uncal = roc_auc_score(y_test_bin, proba_test_uncal[:, pos_idx])
        if verbose:
            plot_roc(fpr_uncal, tpr_uncal, auc_val=auc_uncal, title=f'ROC {key.upper()} (uncalibrated, test)')
    
    if verbose:
        plot_confusion(cm_uncal, class_names, title=f'Confusion matrix {key.upper()} (uncalibrated, test)')

    # Calibration on X_calib
    calibrator = CalibratedClassifierCV(estimator=best_pipe, method=method, cv='prefit')
    # Fit calibration on encoded y for XGB
    if key == 'xgb' and le_y is not None:
        calibrator.fit(X_calib, y_calib_enc)
    else:
        calibrator.fit(X_calib, y_calib)

    # Calibration histogram + reliability curve (calibration set)
    if key == 'xgb' and le_y is not None:
        pos_idx = list(calibrator.classes_).index(pos_label_xgb)
        proba_calib = calibrator.predict_proba(X_calib)[:, pos_idx]
        y_calib_bin = (y_calib_enc == pos_label_xgb).astype(int)
    else:
        pos_idx = list(calibrator.classes_).index('disease')
        proba_calib = calibrator.predict_proba(X_calib)[:, pos_idx]
        y_calib_bin = (y_calib == 'disease').astype(int)

    if verbose:
        plt.figure(figsize=(5,4))
        plt.hist(proba_calib, bins=20, range=(0,1), alpha=0.8, edgecolor='k')
        plt.xlabel('Probabilite predite (positive)'); plt.ylabel('Frequence')
        plt.title(f'Histogramme des probabilites {key.upper()} (calibration set)')
        plt.savefig(plot_dir / f"hist_calib_{key}.png", bbox_inches='tight')
        plt.tight_layout(); plt.show()

    # Reliability curves: uncalibrated vs calibrated on the calibration set
    # Uncalibrated probabilities from best_pipe
    if key == 'xgb' and le_y is not None:
        pos_idx_uncal = list(best_pipe.classes_).index(pos_label_xgb)
        proba_calib_uncal = best_pipe.predict_proba(X_calib)[:, pos_idx_uncal]
    else:
        pos_idx_uncal = list(best_pipe.classes_).index('disease')
        proba_calib_uncal = best_pipe.predict_proba(X_calib)[:, pos_idx_uncal]

    frac_pos_cal, mean_pred_cal = calibration_curve(y_calib_bin, proba_calib, n_bins=10, strategy='uniform')
    frac_pos_unc, mean_pred_unc = calibration_curve(y_calib_bin, proba_calib_uncal, n_bins=10, strategy='uniform')

    if verbose:
        plt.figure(figsize=(5,4))
        plt.plot(mean_pred_unc, frac_pos_unc, 'o--', label='Uncalibrated')
        plt.plot(mean_pred_cal, frac_pos_cal, 'o-', label='Calibrated')
        plt.plot([0,1], [0,1], 'k--', label='Ideal')
        plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction positives')
        plt.title(f'Calibration curve {key.upper()} (calibration set)')
        plt.legend(); plt.tight_layout()
        plt.savefig(plot_dir / f"calibration_curve_{key}.png", bbox_inches='tight') 
        plt.show()
    
    # Re-evaluation (calibrated) on test
    y_pred_test_cal = calibrator.predict(X_test)
    
    
    # Inverse-transform predictions for XGB
    if key == 'xgb' and le_y is not None:
        y_pred_test_cal = le_y.inverse_transform(y_pred_test_cal)
    cm_cal = confusion_matrix(y_test, y_pred_test_cal, labels=class_names)
    print(
        f"Rapport {key.upper()} (calibrated):\n",
        classification_report(
            y_test,
            y_pred_test_cal,
            labels=class_names
        )
    )

    auc_cal = None
    proba_test_cal = None
    if hasattr(calibrator, 'predict_proba'):
        proba_test_cal = calibrator.predict_proba(X_test)
        if key == 'xgb' and le_y is not None:
            pos_idx = list(calibrator.classes_).index(pos_label_xgb)
            y_test_bin = (y_test_enc == pos_label_xgb).astype(int)
        else:
            pos_idx = list(calibrator.classes_).index('disease')
            y_test_bin = (y_test == 'disease').astype(int)
        fpr_cal, tpr_cal, _ = roc_curve(y_test_bin, proba_test_cal[:, pos_idx])
        auc_cal = roc_auc_score(y_test_bin, proba_test_cal[:, pos_idx])
        if verbose:
            plot_roc(fpr_cal, tpr_cal, auc_val=auc_cal, title=f'ROC {key.upper()} (calibrated, test)')
        
    if verbose:
        plot_confusion(
            cm_cal,
            class_names=class_names,
            title=f'Confusion matrix {key.upper()} (calibrated, test)'
        )

    # Optimal threshold (Youden) on calibration, applied to test
    best_thresh = None
    cm_thresh = None
    if proba_test_cal is not None:
        if key == 'xgb' and le_y is not None:
            pos_idx = list(calibrator.classes_).index(pos_label_xgb)
            y_calib_bin = (y_calib_enc == pos_label_xgb).astype(int)
            proba_calib_pos = calibrator.predict_proba(X_calib)[:, pos_idx]
        else:
            pos_idx = list(calibrator.classes_).index('disease')
            y_calib_bin = (y_calib == 'disease').astype(int)
            proba_calib_pos = calibrator.predict_proba(X_calib)[:, pos_idx]
        fpr_c, tpr_c, thr_c = roc_curve(y_calib_bin, proba_calib_pos)
        J = tpr_c - fpr_c
        ix = np.argmax(J)
        best_thresh = float(thr_c[ix])
        
        # Ajout d'une protection contre un seuil infini
        if np.isinf(best_thresh):
            print(f"Attention: Le meilleur seuil (Youden J) est infini. Le modele est peu performant. Seuil par defaut 0.5 utilise.")
            best_thresh = 0.5
        else:
            print(f"Meilleur seuil (Youden J) {key.upper()} sur calib: {best_thresh:.4f} | J={J[ix]:.4f} | TPR={tpr_c[ix]:.4f} | FPR={fpr_c[ix]:.4f}")

        y_pred_test_thr = (proba_test_cal[:, pos_idx] >= best_thresh).astype(int)
        if key == 'xgb' and le_y is not None:
            # y_pred_test_thr est déjà encodé (0/1), on peut inverse_transform
            y_pred_test_thr_lbl = le_y.inverse_transform(y_pred_test_thr)
            cm_thresh = confusion_matrix(y_test, y_pred_test_thr_lbl, labels=class_names)
            print(
                f"Rapport {key.upper()} (calibrated + threshold):\n",
                classification_report(y_test, y_pred_test_thr_lbl, labels=class_names)
            )
        else:
            y_pred_test_thr_lbl = np.where(y_pred_test_thr == 1, "disease", "healthy")
            cm_thresh = confusion_matrix(y_test, y_pred_test_thr_lbl, labels=class_names)
            print(
                f"Rapport {key.upper()} (calibrated + threshold):\n",
                classification_report(y_test, y_pred_test_thr_lbl, labels=class_names)
            )

        # ROC identique mais on marque le point (FPR, TPR) au seuil choisi
        if key == 'xgb' and le_y is not None:
            fpr_c2, tpr_c2, _ = roc_curve((y_test_enc == pos_label_xgb).astype(int), proba_test_cal[:, pos_idx])
            tn, fp, fn, tp = confusion_matrix((y_test_enc == pos_label_xgb).astype(int), y_pred_test_thr, labels=[0,1]).ravel()
        else:
            fpr_c2, tpr_c2, _ = roc_curve((y_test == 'disease').astype(int), proba_test_cal[:, pos_idx])
            tn, fp, fn, tp = confusion_matrix((y_test == 'disease').astype(int), y_pred_test_thr, labels=[0,1]).ravel()
        fpr_pt = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_pt = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if verbose:
            plot_roc(
                fpr_c2, tpr_c2, auc_val=auc_cal,
                title=f'ROC {key.upper()} (calibrated, test) + Youden threshold',
                point=(fpr_pt, tpr_pt), point_label=f'Th={best_thresh:.3f}'
            )
            plot_confusion(
                cm_thresh,
                class_names=class_names,
                title=f'Confusion matrix {key.upper()} (calibrated + threshold, test)'
            )

    # 5. Choix du meilleur modele (calibre vs non-calibre) base sur le rappel 'disease'
    report_uncal_dict = classification_report(y_test, y_pred_test_uncal, output_dict=True)
    
    use_calibrated = False
    if 'y_pred_test_thr_lbl' in locals() and y_pred_test_thr_lbl is not None:
        report_thr_dict = classification_report(y_test, y_pred_test_thr_lbl, output_dict=True)
        recall_uncal = report_uncal_dict['disease']['recall']
        recall_thr = report_thr_dict['disease']['recall']
        print(f"\n[INFO] Comparaison du rappel (disease): Non-calibre={recall_uncal:.3f} vs Calibre+Seuil={recall_thr:.3f}")
        use_calibrated = recall_thr >= recall_uncal

    if use_calibrated:
        print("[INFO] Le modele calibre est meilleur ou equivalent.")
        y_pred_final_lbl = y_pred_test_thr_lbl
        cm_final = cm_thresh
        classification_report_final = report_thr_dict
        final_auc, final_fpr, final_tpr, final_threshold = auc_cal, fpr_cal, tpr_cal, best_thresh
    else:
        print("[INFO] Le modele non-calibre est meilleur (ou le seul disponible).")
        y_pred_final_lbl = y_pred_test_uncal
        cm_final = cm_uncal
        classification_report_final = report_uncal_dict
        final_auc = auc_uncal
        final_fpr = fpr_uncal if 'fpr_uncal' in locals() else None
        final_tpr = tpr_uncal if 'tpr_uncal' in locals() else None
        final_threshold = 0.5 # implicite pour .predict()

    # Cross-validated fold scores on the calibration set
    fold_scores = []
    print("[INFO] Calcul des scores de fold (F1-score) par validation croisee.")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    y_calib_cv = y_calib_enc if key == 'xgb' and le_y is not None else y_calib

    if use_calibrated:
        for tr_idx, va_idx in skf.split(X_calib, y_calib_cv):
            cal_cv = CalibratedClassifierCV(estimator=best_pipe, method=method, cv='prefit')
            if key == 'xgb' and le_y is not None:
                cal_cv.fit(X_calib[tr_idx], y_calib_enc[tr_idx])
                pos_idx_cv = 1 # disease
                # Determine threshold on train fold
                proba_tr = cal_cv.predict_proba(X_calib[tr_idx])[:, pos_idx_cv]
                y_tr_bin = (y_calib_enc[tr_idx] == pos_label_xgb).astype(int)
                fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr_bin, proba_tr)
                J = tpr_tr - fpr_tr
                th = float(thr_tr[np.argmax(J)]) if len(thr_tr) > 1 else 0.5
                # Validate on val fold
                proba_va = cal_cv.predict_proba(X_calib[va_idx])[:, pos_idx_cv]
                y_va_pred = (proba_va >= th).astype(int)
                y_va_bin = (y_calib_enc[va_idx] == pos_label_xgb).astype(int)
                fold_scores.append(f1_score(y_va_bin, y_va_pred))
            else:
                cal_cv.fit(X_calib[tr_idx], y_calib[tr_idx])
                pos_idx_cv = list(cal_cv.classes_).index('disease')
                proba_tr = cal_cv.predict_proba(X_calib[tr_idx])[:, pos_idx_cv]
                y_tr_bin = (y_calib[tr_idx] == 'disease').astype(int)
                fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr_bin, proba_tr)
                J = tpr_tr - fpr_tr
                th = float(thr_tr[np.argmax(J)]) if len(thr_tr) > 1 else 0.5
                proba_va = cal_cv.predict_proba(X_calib[va_idx])[:, pos_idx_cv]
                y_va_pred = (proba_va >= th).astype(int)
                y_va_bin = (y_calib[va_idx] == 'disease').astype(int)
                fold_scores.append(f1_score(y_va_bin, y_va_pred))
    else:
        y_calib_bin_cv = (y_calib_enc == pos_label_xgb).astype(int) if key == 'xgb' and le_y is not None else (y_calib == 'disease').astype(int)
        for _, va_idx in skf.split(X_calib, y_calib_cv):
            # The model is the already-fitted `best_pipe`, which predicts with a 0.5 threshold
            y_va_pred = best_pipe.predict(X_calib[va_idx])
            
            # Convert predictions to binary (0/1) for f1_score
            if key == 'xgb' and le_y is not None:
                y_va_pred_bin = (y_va_pred == pos_label_xgb).astype(int)
            else:
                y_va_pred_bin = (y_va_pred == 'disease').astype(int)
            
            fold_scores.append(f1_score(y_calib_bin_cv[va_idx], y_va_pred_bin))
    
    mean_fold_scores = np.mean(fold_scores) if fold_scores else 0.0

    if verbose:
        # Graphique des performances par fold
        plot_fold_performance(fold_scores, key.upper(), reduction=reduction)
        
        # Radar chart des métriques
        star_stats = {
            'AUC': final_auc,
            'Précision': classification_report_final["disease"]["precision"],
            'Rappel': classification_report_final["disease"]["recall"],
            'F1-Score': classification_report_final["disease"]["f1-score"],
            'Mean Fold F1': mean_fold_scores
        }
        plot_star_chart(star_stats, key.upper(), reduction=reduction)

    results = {
        'cm_final': cm_final,
        'auc_calibrated': final_auc,
        'best_threshold': final_threshold,
        'precision': classification_report_final["macro avg"]["precision"],
        'accuracy': classification_report_final["accuracy"],
        'recall': classification_report_final["macro avg"]["recall"],
        'f1-score': classification_report_final["macro avg"]["f1-score"],
        'fold_scores': fold_scores,
        'mean_fold_scores': mean_fold_scores,
        'fpr': final_fpr,
        'tpr': final_tpr
    }
    
    return results

def get_best_features(X: pd.DataFrame, y: np.ndarray, p_thresh: float = None) -> np.ndarray:
    """
    Selectionne les features en se basant sur l'importance des variables d'un Random Forest.
    Cette methode est choisie car elle peut capturer les interactions non lineaires entre les features, 
    ce qui est souvent pertinent dans les donnees du microbiote.
    """
    # On utilise un Random Forest pour capturer les interactions
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X, y)
    
    # Recuperation des importances
    importances = rf.feature_importances_
    
    # Strategie de selection : on garde ceux qui sont au-dessus de la moyenne (ou un autre seuil)
    threshold = np.mean(importances) if p_thresh is None else p_thresh
    
    mask = importances > threshold
    selected = X.columns[mask]
    
    # print(f"Threshold is : {threshold}")
    # print(f"[get_best_features] Random Forest a sélectionné {len(selected)} / {X.shape[1]} features (interactions prises en compte)")
    return selected



# def plot_roc_comparison(results_dict: dict, title: str = "Comparaison des courbes ROC des meilleurs modèles", save_path=None):
#     """
#     Trace les courbes ROC de plusieurs modèles sur un seul graphique pour comparaison.

#     Paramètres
#     ----------
#     results_dict : dict
#         Un dictionnaire où les clés sont les noms des modèles et les valeurs sont les
#         dictionnaires de résultats contenant 'fpr', 'tpr', et 'auc_calibrated'.
#     title : str, optionnel
#         Le titre du graphique.
#     save_path : str ou Path, optionnel
#         Chemin pour sauvegarder le graphique. Si None, le chemin est déduit.
#     """
#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     for model_name, results in results_dict.items():
#         if results and results.get('fpr') is not None and results.get('tpr') is not None:
#             label = f"{model_name} (AUC = {results['auc_calibrated']:.3f})"
#             ax.plot(results['fpr'], results['tpr'], label=label)
#         else:
#             print(f"Skipping {model_name}: missing 'fpr' or 'tpr' data.")

#     ax.plot([0, 1], [0, 1], 'k--')
#     ax.set_xlabel('Taux de Faux Positifs (FPR)')
#     ax.set_ylabel('Taux de Vrais Positifs (TPR)')
#     ax.set_title(title)
#     ax.legend()
#     fig.tight_layout()

#     if save_path is None:
#         # Save in a dedicated 'model_comparison' directory
#         base = Path(__file__).resolve().parent
#         save_dir = base / "model_comparison"
#         save_dir.mkdir(exist_ok=True)
#         save_path = save_dir / "roc_curves_comparison.png"
        
#     fig.savefig(save_path, bbox_inches='tight')
#     plt.show()


def plot_model_comparison_boxplot(results_dict: dict, title: str = "Comparaison des F1-Scores (CV) par modèle", save_path=None):
    """
    Crée un boxplot pour comparer la distribution des scores de validation croisée (fold scores)
    entre différents modèles.

    Paramètres
    ----------
    results_dict : dict
        Dictionnaire où les clés sont les noms de modèles et les valeurs sont les
        dictionnaires de résultats contenant la clé 'fold_scores'.
    title : str, optionnel
        Le titre du graphique.
    save_path : str ou Path, optionnel
        Chemin pour sauvegarder le graphique. Si None, déduit du contexte.
    """
    plot_data = []
    for model_name, results in results_dict.items():
        if results and 'fold_scores' in results and results['fold_scores']:
            for score in results['fold_scores']:
                plot_data.append({'Modèle': model_name, 'F1-Score': score})
    
    if not plot_data:
        print("Aucune donnée de fold_score à tracer pour le boxplot.")
        return

    df_plot = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(x='Modèle', y='F1-Score', data=df_plot, ax=ax)
    sns.stripplot(x='Modèle', y='F1-Score', data=df_plot, ax=ax, color=".25", jitter=0.2)
    
    ax.set_title(title)
    ax.set_ylabel('F1-Score (sur les folds de validation)')
    ax.set_xlabel('Modèle')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()

    if save_path is None:
        base = Path(__file__).resolve().parent
        save_dir = base / "model_comparison"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "fold_scores_boxplot_comparison.png"
        
    fig.savefig(save_path, bbox_inches='tight')
    plt.show()
