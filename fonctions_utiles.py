import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
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
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
 

# Small plotting helpers
def plot_confusion(cm, class_names, title=None):
    """Tracer une matrice de confusion (carte de chaleur).

    Paramètres
    ----------
    cm : array-like de forme (n_classes, n_classes)
        Comptes de la matrice de confusion.
    class_names : list[str]
        Ordre d'affichage des étiquettes de classes.
    title : str, optionnel
        Titre optionnel du graphique.
    """
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    if title: ax.set_title(title)
    plt.tight_layout(); plt.show()

def plot_roc(fpr, tpr, auc_val=None, title=None, point=None, point_label=None):
    """Tracer une courbe ROC avec AUC et point optionnels.

    Paramètres
    ----------
    fpr, tpr : array-like
        Taux de faux positifs / vrais positifs.
    auc_val : float, optionnel
        Valeur d'AUC à afficher dans la légende.
    title : str, optionnel
        Titre du graphique.
    point : tuple(float, float), optionnel
        Point (FPR, TPR) à mettre en évidence.
    point_label : str, optionnel
        Libellé du point mis en évidence.
    """
    plt.figure(figsize=(5,4))
    lbl = (f'AUC = {auc_val:.3f}' if auc_val is not None else None)
    plt.plot(fpr, tpr, label=lbl)
    if point is not None:
        px, py = point
        plt.scatter([px], [py], c='red', edgecolor='k', s=60,
                    label=point_label if point_label else 'Selected')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    if title: plt.title(title)
    if lbl or point_label: plt.legend()
    plt.tight_layout(); plt.show()

# Functions to build and train a neural network model
def build_model(input_dim: int, num_classes: int, lr: float = 3e-4) -> callable:
    """Construire un réseau de neurones dense pour la classification.

    Paramètres
    ----------
    input_dim : int
        Nombre de variables d'entrée (features).
    num_classes : int
        Nombre de classes cibles. Sigmoïde si 2 classes, sinon softmax.
    lr : float, par défaut 3e-4
        Taux d'apprentissage de l'optimiseur Adam.

    Retourne
    --------
    callable
        Un modèle Keras compilé.
    """
    
    # Le réseau de neurones
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid') if num_classes == 2 else layers.Dense(num_classes, activation='softmax'),
    ])
    
    # Fonction perte
    loss = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'
    
    # Métriques de suivie
    metrics = ['accuracy'] + ([keras.metrics.AUC(name='auc')] if num_classes == 2 else [])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=metrics)
    return model


def train_model(model: callable, df: pd.DataFrame, label_col: str = 'diagnosis',
                test_size: float = 0.3, batch_size: int = 16, epochs: int = 200,
                random_state: int = 42,verbose : bool = True):
    """Train a neural network (binary: 'disease' positive, 'healthy' negative),
    calibrate with Platt scaling, and plot before/after results."""
    # Features/labels
    X = df.drop(columns=[label_col]).values
    y = df[label_col].astype(str).values

    # Fixed encoding: 'disease'->0, 'healthy'->1
    le = LabelEncoder().fit(['disease','healthy'])
    y_int = le.transform(y)

    # Splits: train/test, then train/val (val used for calibration)
    X_tr, X_test, y_tr_int, y_test_int = train_test_split(
        X, y_int, test_size=test_size, stratify=y_int, random_state=random_state
    )
    X_tr, X_val, y_tr_int, y_val_int = train_test_split(
        X_tr, y_tr_int, test_size=0.2, stratify=y_tr_int, random_state=random_state
    )

    # print(f"X_train shape: {X_tr.shape}")
    # print(f"X_test shape: {X_test.shape}")
    # print(f"X_val (calib) shape: {X_val.shape}")

    # Scale using train only
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

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
        history_df[['loss','val_loss']].plot(title='Training history')
        plt.tight_layout(); plt.show()

    # Uncalibrated on test (P(disease) = 1 - sigmoid)
    raw_test = model.predict(X_test).ravel()                 # P(healthy)
    proba_pos_uncal_test = 1.0 - raw_test                    # P(disease)
    y_test_bin = (y_test_int == 0).astype(int)
    preds_uncal = (proba_pos_uncal_test >= 0.5).astype(int)
    y_pred_uncal_lbl = np.where(preds_uncal == 1, 'disease', 'healthy')
    class_names = ['disease','healthy']
    y_test_lbl = np.where(y_test_int==0,'disease','healthy')
    cm_uncal = confusion_matrix(y_test_lbl, y_pred_uncal_lbl, labels=class_names)
    print("Report NN (uncalibrated):\n", classification_report(y_test_lbl, y_pred_uncal_lbl, target_names=class_names))
    
    if verbose : 
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm_uncal, annot=True, fmt='d', cbar=False, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion matrix NN (uncalibrated, test)')
        plt.tight_layout(); plt.show()
    auc_uncal = roc_auc_score(y_test_bin, proba_pos_uncal_test)
    fpr_uncal, tpr_uncal, _ = roc_curve(y_test_bin, proba_pos_uncal_test)
    
    if verbose : 
        plot_roc(fpr_uncal, tpr_uncal, auc_val=auc_uncal, title='ROC NN (uncalibrated, test)')

    # Calibration with CalibratedClassifierCV (sigmoid) on validation set using a prefit wrapper
    raw_val = model.predict(X_val).ravel()                   # P(healthy)
    proba_pos_uncal_val = 1.0 - raw_val                      # P(disease)
    y_val_bin = (y_val_int == 0).astype(int)

    class PrefitNN:
        _estimator_type = 'classifier'
        def __init__(self, keras_model):
            self.model = keras_model
            self.classes_ = np.array([0,1])
        def fit(self, X, y):
            return self
        def predict_proba(self, X):
            p_healthy = self.model.predict(X, verbose=0).ravel()
            p_disease = 1.0 - p_healthy
            return np.vstack([p_healthy, p_disease]).T
        def predict(self, X):
            # Return class labels 0/1 based on 0.5 threshold on P(disease)
            proba = self.predict_proba(X)
            return (proba[:,1] >= 0.5).astype(int)

    base_est = PrefitNN(model)
    calibrator_nn = CalibratedClassifierCV(estimator=base_est, method='sigmoid', cv='prefit')
    calibrator_nn.fit(X_val, y_val_bin)
    pos_idx_cal = list(calibrator_nn.classes_).index(1)
    proba_pos_cal_val = calibrator_nn.predict_proba(X_val)[:, pos_idx_cal]
    # Histogram of calibrated probabilities on calibration set (validation)
    plt.figure(figsize=(5,4))
    plt.hist(proba_pos_cal_val, bins=20, range=(0,1), alpha=0.8, edgecolor='k')
    plt.xlabel('Predicted probability (disease)'); plt.ylabel('Frequency')
    plt.title('Histogram of calibrated probabilities (calibration set)')
    plt.tight_layout(); plt.show()

    # Reliability curves (validation)
    frac_cal, mean_cal = calibration_curve(y_val_bin, proba_pos_cal_val, n_bins=10, strategy='uniform')
    frac_unc, mean_unc = calibration_curve(y_val_bin, proba_pos_uncal_val, n_bins=10, strategy='uniform')
    
    if verbose : 
        plt.figure(figsize=(5,4))
        plt.plot(mean_unc, frac_unc, 'o--', label='Uncalibrated')
        plt.plot(mean_cal, frac_cal, 'o-', label='Calibrated')
        plt.plot([0,1],[0,1],'k--', label='Ideal')
        plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction positives')
        plt.title('Calibration curve NN (validation/calibration set)')
        plt.legend(); plt.tight_layout(); plt.show()

    # Calibrated on test
    proba_pos_cal_test = calibrator_nn.predict_proba(X_test)[:, pos_idx_cal]
    auc_cal = roc_auc_score(y_test_bin, proba_pos_cal_test)
    fpr_cal, tpr_cal, _ = roc_curve(y_test_bin, proba_pos_cal_test)
    plot_roc(fpr_cal, tpr_cal, auc_val=auc_cal, title='ROC NN (calibrated, test)')
    preds_cal = (proba_pos_cal_test >= 0.5).astype(int)
    y_pred_cal_lbl = np.where(preds_cal == 1, 'disease', 'healthy')
    cm_cal = confusion_matrix(y_test_lbl, y_pred_cal_lbl, labels=class_names)
    print("Report NN (calibrated, 0.5):\n", classification_report(y_test_lbl, y_pred_cal_lbl, target_names=class_names))
    
    if verbose : 
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm_cal, annot=True, fmt='d', cbar=False, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion matrix NN (calibrated, test)')
        plt.tight_layout(); plt.show()

    # Threshold (Youden) on calibrated validation, applied to test
    fpr_v, tpr_v, thr_v = roc_curve(y_val_bin, proba_pos_cal_val)
    J = tpr_v - fpr_v
    ix = np.argmax(J)
    best_thresh = float(thr_v[ix])
    print(f"Best threshold (Youden J) NN on calib: {best_thresh:.4f} | J={J[ix]:.4f} | TPR={tpr_v[ix]:.4f} | FPR={fpr_v[ix]:.4f}")
    preds_thr = (proba_pos_cal_test >= best_thresh).astype(int)
    y_pred_thr_lbl = np.where(preds_thr == 1, 'disease', 'healthy')
    
    cm_thr = confusion_matrix(y_test_lbl, y_pred_thr_lbl, labels=class_names)
    # Confusion matrix after calibration + optimal threshold
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm_thr, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion matrix NN (calibrated + threshold, test)')
    plt.tight_layout(); plt.show()
    print("Report NN (calibrated + threshold):\n", classification_report(y_test_lbl, y_pred_thr_lbl, target_names=class_names))
    fpr_c2, tpr_c2, _ = roc_curve(y_test_bin, proba_pos_cal_test)
    tn, fp, fn, tp = confusion_matrix(y_test_bin, preds_thr, labels=[0,1]).ravel()
    fpr_pt = fp/(fp+tn) if (fp+tn)>0 else 0.0
    tpr_pt = tp/(tp+fn) if (tp+fn)>0 else 0.0
    plot_roc(fpr_c2, tpr_c2, auc_val=auc_cal, title='ROC NN (calibrated, test) + Youden threshold', point=(fpr_pt,tpr_pt), point_label=f'Th={best_thresh:.3f}')

    # Final metrics and fold-wise calibration CV on validation set
    classification_report_final = classification_report(y_test_lbl, y_pred_thr_lbl, output_dict=True)
    fold_scores = []
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        for tr_idx, va_idx in skf.split(X_val, y_val_bin):
            cal_cv = CalibratedClassifierCV(estimator=base_est, method='sigmoid', cv='prefit')
            cal_cv.fit(X_val[tr_idx], y_val_bin[tr_idx])
            pos_idx_cv = list(cal_cv.classes_).index(1)
            proba_tr_cv = cal_cv.predict_proba(X_val[tr_idx])[:, pos_idx_cv]
            fpr_tr_cv, tpr_tr_cv, thr_tr_cv = roc_curve(y_val_bin[tr_idx], proba_tr_cv)
            th_cv = float(thr_tr_cv[np.argmax(tpr_tr_cv - fpr_tr_cv)])
            proba_va_cv = cal_cv.predict_proba(X_val[va_idx])[:, pos_idx_cv]
            y_va_pred_cv = (proba_va_cv >= th_cv).astype(int)
            fold_scores.append(f1_score(y_val_bin[va_idx], y_va_pred_cv))
    except Exception:
        pass

    return {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'history': history_df,
        'auc_uncalibrated': auc_uncal,
        'auc_calibrated': auc_cal,
        'best_threshold': best_thresh,
        'cm_uncalibrated': cm_uncal,
        'cm_calibrated': cm_cal,
        'cm_threshold': cm_thr,
        'cm_final': cm_thr,
        'precision': classification_report_final["macro avg"]["precision"],
        'accuracy': classification_report_final["accuracy"],
        'recall': classification_report_final["macro avg"]["recall"],
        'f1-score': classification_report_final["macro avg"]["f1-score"],
        'fold_scores': fold_scores,
    }


# Training with other models (classical ML)
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
):
    """
    Pipeline de bout en bout pour l'entraînement, la calibration et l'évaluation.

    Hypothèses:
      - Classification binaire avec étiquettes exactement {'disease','healthy'}
      - 'disease' est la classe positive pour les métriques et le seuillage
      - Le même random_state garantit les mêmes découpages pour comparer les matrices

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
        Matrice de caractéristiques.
    y : array-like de forme (n_samples,)
        Étiquettes cibles ('disease'/'healthy').
    test_size : float, par défaut 0.3
        Proportion utilisée pour le test.
    calib_size : float, par défaut 0.2
        Proportion du trainval réservée à la calibration.
    random_state : int, par défaut 42
        Graine aléatoire pour les splits.
    n_jobs : int, par défaut -1
        Parallélisme pour les modèles scikit-learn.
    method : str, par défaut 'sigmoid'
        Méthode de calibration scikit-learn.
    verbose : bool, par défaut True
        Si True, affiche les graphiques; sinon les supprime.
    """

    key = model_key.lower().strip()
    X = np.asarray(X)
    y = np.asarray(y)
    class_names = ['disease', 'healthy']
    if set(np.unique(y)) != set(class_names):
        raise ValueError("y must contain exactly {'disease','healthy'}")

    # y = (y == pos_label).astype(int)
    # 1) Split global: trainval / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 2) Split trainval -> search_train / calib (calibration holdout)
    X_search, X_calib, y_search, y_calib = train_test_split(
        X_trainval, y_trainval, test_size=calib_size, stratify=y_trainval, random_state=random_state
    )

    # Optional: XGBoost requires numeric labels; keep strings for others.
    le_y = None
    if key == 'xgb':
        le_y = LabelEncoder().fit(y_trainval)
        y_search_enc = le_y.transform(y_search)
        y_calib_enc = le_y.transform(y_calib)
        y_test_enc = le_y.transform(y_test)
        pos_label_xgb = int(le_y.transform(['disease'])[0])

    # Neural network support removed from this function — use train_model separately.

    # print(np.unique(y_search))
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
            ('clf', XGBClassifier(objective='binary:logistic',eval_metric='logloss',use_label_encoder=True))
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
    # Use fixed class order for comparability
    class_names = ['disease', 'healthy']

    # Initial evaluation (uncalibrated) on test
    y_pred_test_uncal = best_pipe.predict(X_test)
    # Inverse-transform numeric predictions to strings for XGB
    if key == 'xgb' and le_y is not None:
        y_pred_test_uncal = le_y.inverse_transform(y_pred_test_uncal)
    cm_uncal = confusion_matrix(y_test, y_pred_test_uncal, labels=class_names)
    print(f"Report {key.upper()} (uncalibrated):\n", classification_report(y_test, y_pred_test_uncal, labels=class_names, target_names=class_names))

    proba_test_uncal = None
    auc_uncal = None
    if hasattr(best_pipe, 'predict_proba'):
        proba_test_uncal = best_pipe.predict_proba(X_test)
        if key == 'xgb' and le_y is not None:
            # Columns ordered by encoded classes
            pos_idx = list(best_pipe.named_steps['clf'].classes_).index(pos_label_xgb)
            y_test_bin = (y_test_enc == pos_label_xgb).astype(int)
        else:
            pos_idx = class_names.index("disease")
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

    # Cross-validated fold scores on the calibration set (post-calibration)
    fold_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for tr_idx, va_idx in skf.split(X_calib, (y_calib_enc if key == 'xgb' and le_y is not None else y_calib)):
        cal_cv = CalibratedClassifierCV(estimator=best_pipe, method=method, cv='prefit')
        if key == 'xgb' and le_y is not None:
            cal_cv.fit(X_calib[tr_idx], y_calib_enc[tr_idx])
            pos_idx_cv = list(cal_cv.classes_).index(pos_label_xgb)
            # Determine threshold on train fold
            proba_tr = cal_cv.predict_proba(X_calib[tr_idx])[:, pos_idx_cv]
            y_tr_bin = (y_calib_enc[tr_idx] == pos_label_xgb).astype(int)
            fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr_bin, proba_tr)
            J = tpr_tr - fpr_tr
            th = float(thr_tr[np.argmax(J)])
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
            th = float(thr_tr[np.argmax(J)])
            proba_va = cal_cv.predict_proba(X_calib[va_idx])[:, pos_idx_cv]
            y_va_pred = (proba_va >= th).astype(int)
            y_va_bin = (y_calib[va_idx] == 'disease').astype(int)
            fold_scores.append(f1_score(y_va_bin, y_va_pred))

    # Calibration histogram + reliability curve (calibration set)
    if key == 'xgb' and le_y is not None:
        pos_idx = list(calibrator.classes_).index(pos_label_xgb)
        proba_calib = calibrator.predict_proba(X_calib)[:, pos_idx]
        y_calib_bin = (y_calib_enc == pos_label_xgb).astype(int)
    else:
        pos_idx = list(calibrator.classes_).index('disease') if hasattr(calibrator, 'classes_') else 1
        proba_calib = calibrator.predict_proba(X_calib)[:, pos_idx]
        y_calib_bin = (y_calib == 'disease').astype(int)

    if verbose:
        plt.figure(figsize=(5,4))
        plt.hist(proba_calib, bins=20, range=(0,1), alpha=0.8, edgecolor='k')
        plt.xlabel('Probabilité prédite (positive)'); plt.ylabel('Fréquence')
        plt.title(f'Histogramme des probabilités {key.upper()} (calibration set)')
        plt.tight_layout(); plt.show()

    # Reliability curves: uncalibrated vs calibrated on the calibration set
    # Uncalibrated probabilities from best_pipe
    if key == 'xgb' and le_y is not None:
        pos_idx_uncal = list(best_pipe.named_steps['clf'].classes_).index(pos_label_xgb)
        proba_calib_uncal = best_pipe.predict_proba(X_calib)[:, pos_idx_uncal]
    else:
        pos_idx_uncal = list(best_pipe.named_steps['clf'].classes_).index('disease')
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
        plt.legend(); plt.tight_layout(); plt.show()
    
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
            target_names=(list(le_y.classes_) if key == 'xgb' and le_y is not None else list(calibrator.classes_))
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
            class_names=(list(le_y.classes_) if key == 'xgb' and le_y is not None else list(calibrator.classes_)),
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
        print(f"Meilleur seuil (Youden J) {key.upper()} sur calib: {best_thresh:.4f} | J={J[ix]:.4f} | TPR={tpr_c[ix]:.4f} | FPR={fpr_c[ix]:.4f}")

        y_pred_test_thr = (proba_test_cal[:, pos_idx] >= best_thresh).astype(int)
        if key == 'xgb' and le_y is not None:
            # Build encoded labels then inverse-transform to strings
            neg_label_enc = [c for c in calibrator.classes_ if c != pos_label_xgb][0]
            y_pred_test_thr_enc = np.where(y_pred_test_thr == 1, pos_label_xgb, neg_label_enc)
            y_pred_test_thr_lbl = le_y.inverse_transform(y_pred_test_thr_enc)
            cm_thresh = confusion_matrix(y_test, y_pred_test_thr_lbl, labels=list(le_y.classes_))
            print(
                f"Rapport {key.upper()} (calibrated + threshold):\n",
                classification_report(y_test, y_pred_test_thr_lbl, target_names=list(le_y.classes_))
            )
        else:
            neg_label = [c for c in calibrator.classes_ if c != "disease"][0]
            y_pred_test_thr_lbl = np.where(y_pred_test_thr == 1, "disease", neg_label)
            cm_thresh = confusion_matrix(y_test, y_pred_test_thr_lbl, labels=list(calibrator.classes_))
            print(
                f"Rapport {key.upper()} (calibrated + threshold):\n",
                classification_report(y_test, y_pred_test_thr_lbl, target_names=list(calibrator.classes_))
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
                class_names=(list(le_y.classes_) if key == 'xgb' and le_y is not None else list(calibrator.classes_)),
                title=f'Confusion matrix {key.upper()} (calibrated + threshold, test)'
            )


    y_pred_final_lbl = y_pred_test_thr_lbl if 'y_pred_test_thr_lbl' in locals() else y_pred_test_cal
    cm_final = cm_thresh if cm_thresh is not None else cm_cal
    classification_report_final = classification_report(y_test, y_pred_final_lbl, output_dict=True)
    
    results = {
        'cm_final': cm_final,
        'auc_calibrated': auc_cal,
        'best_threshold': best_thresh,
        'precision': classification_report_final["macro avg"]["precision"],
        'accuracy': classification_report_final["accuracy"],
        'recall': classification_report_final["macro avg"]["recall"],
        'f1-score': classification_report_final["macro avg"]["f1-score"],
        'fold_scores': fold_scores,
    }
    
    return results

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
    
    rf = RandomForestClassifier().fit(X,y)
    
    list_imp = rf.feature_importances_
    list_mask = np.where(list_imp >= threshold,True,False)
    features_names = np.array(X.columns)
    
    return features_names[list_mask]
    
    
    
