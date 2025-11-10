import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder,StandardScaler,label_binarize
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, make_scorer, f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Fonctions utiliser pour entrainer le modèle avec un réseau de neurones
def build_model(input_dim : int, num_classes : int, lr : float=3e-4) -> callable:
    """
    Fonction utilisé pour créer notre réseau de neurones.
    
    Paramètres : 
        input_dim (int) : Taille d'entrée du réseau de neurones
        num_classes (int) : Nombre de classes 
        lr (float) : Learning rate pour l'optimization
        
    Retour : 
        model (callable) : Le réseau de neurones fabriqué
    """
    
    # Le réseau de neurones
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid') if num_classes == 2 else layers.Dense(num_classes, activation='softmax'),
    ])
    
    # Fonction perte
    loss = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'
    
    # Métriques de suivie
    metrics = ['accuracy'] + ([keras.metrics.AUC(name='auc')] if num_classes == 2 else [])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=metrics)
    return model


def train_model(model : callable, df : pd.DataFrame, label_col : str ='diagnosis', test_size : float =0.3, batch_size : int =16, epochs : int =200,random_state : int =42):
    """
    Fonction utilisé pour entrainer le réseau de neurones.
    
    Paramètres : 
        model (callable) : Le réseau de neurones
        df (pd.DataFrame) : Le dataframe à extraire le X et le Y
        label_col (str) : Le label de la colonne correspondant au Y
        test_size (float) : Proportion du set de test
        batch_size (int) : Taille du batch
        epochs (int) : Nombre d'epochs
    """
    # X, y
    X = df.drop(columns=[label_col]).values
    y = df[label_col].astype(str).values

    # Transformation Y en variable numérique
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    K = len(le.classes_)
    y_encoded = y_int if K == 2 else keras.utils.to_categorical(y_int, num_classes=K)
    # print(y_encoded)
    
    # Split (Train/test)
    X_tr, X_test, y_tr, y_test, y_tr_int, y_test_int = train_test_split(
        X, y_encoded, y_int, test_size=test_size, stratify=y_int, random_state=random_state
    )
    
    # Split (Train/val)
    X_tr, X_val, y_tr, y_val, y_tr_int, y_val_int = train_test_split(
        X_tr, y_tr, y_tr_int, test_size=0.2, stratify=y_tr_int, random_state=random_state
    )

    print(f"La taille de X_train est : {X_tr.shape}")
    print(f"La taille de X_test est : {X_test.shape}")
    print(f"La taille de X_val est : {X_val.shape}")
    
    # Standardisation (On fit que sur le train pour éviter le biais)
    scaler = StandardScaler()

    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Réduction de dimensions (On prend que 0.95*X.shape[1])
    # if do_acp : 
    #     pca = PCA(n_components=0.95, svd_solver='full', random_state=random_state)
    #     X_tr = pca.fit_transform(X_tr)
    #     X_val = pca.transform(X_val)
    #     X_test = pca.transform(X_test)
    #     model = build_model(input_dim=pca.n_components_,num_classes=K,lr=float(keras.backend.get_value(model.optimizer.learning_rate)))
    
    # Pour ajuster le poids des classes 
    classes = np.arange(K) 
    cw_vals = compute_class_weight('balanced', classes=classes, y=y_tr_int)
    class_weights = {int(c): float(w) for c, w in zip(classes, cw_vals)}

    # Callbacks 
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5),
    ]

    # Entrainement
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        class_weight=class_weights, shuffle=True, verbose=1,
        callbacks=callbacks
    )

    # Plot training history
    history_df = pd.DataFrame(history.history)
    history_df[['loss','val_loss']].plot(title='Training history')
    plt.tight_layout(); plt.show()

    # Evaluation sur le set de test
    
    # Dans le cas d'un problème à 2 classes
    if K == 2:
        probs = model.predict(X_test).ravel()
        preds = (probs >= 0.5).astype(int)

        # Confusion matrix + classification report
        labels = np.arange(K)
        cm = confusion_matrix(y_test_int, preds, labels=labels)
        report_str = classification_report(y_test_int, preds, target_names=le.classes_)
        print(report_str)
        # print('Confusion matrix:\n', cm)

        # Plot confusion matrix 
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion matrix (test)')
        plt.tight_layout(); plt.show()

        # courbe ROC + AUC
        auc = roc_auc_score(y_test_int, probs)
        fpr, tpr, thresholds = roc_curve(y_test_int, probs)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f'ROC AUC = {auc:.3f}')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.title('ROC curve'); plt.legend(); plt.tight_layout(); plt.show()
        
        # Indice de Youden pour le seuil optimal
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        print(f"Best threshold (Youden J): {best_thresh:.4f}  | J={J[ix]:.4f}  | TPR={tpr[ix]:.4f}  | FPR={fpr[ix]:.4f}")

        # Application du seuil optimal et predictions
        preds_opt = (probs >= best_thresh).astype(int)
        print(classification_report(y_test_int, preds_opt))
        print("Confusion matrix:\n", confusion_matrix(y_test_int, preds_opt))
        
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.scatter(fpr[ix], tpr[ix], color="red", edgecolor="k", s=60, label=f"Best Th = {best_thresh:.3f}")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("Courbe ROC avec seuil optimal"); plt.legend(); plt.tight_layout(); plt.show()
    
    # Dans le cas d'un problème multiclasses
    else:
        probs = model.predict(X_test)
        preds = probs.argmax(axis=1)
        labels = np.arange(K)
        cm = confusion_matrix(y_test_int, preds, labels=labels)
        report_str = classification_report(y_test_int, preds, target_names=le.classes_)
        print(report_str)
        # print('Confusion matrix:\n', cm)

        # Plot confusion matrix (counts)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion matrix (test)')
        plt.tight_layout(); plt.show()

        
        auc = None  # Pas besoin 

    # return {
    #     'model': model,
    #     'scaler': scaler,
    #     'label_encoder': le,
    #     'history': history_df,
    #     'X_test': X_test,
    #     'y_test_int': y_test_int,
    #     'y_pred': preds,
    #     'y_proba': probs,
    #     'auc': auc,
    #     'cm': cm,
    #     'report': report_str,
    # }


# Fonctions utiliser pour entrainer avec les autres modèles
def train_with_calibration(
    model_key: str,
    X: np.ndarray,
    y: np.ndarray,
    pos_label: str = 'no',
    test_size: float = 0.2,
    calib_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1,
    method: str = 'sigmoid',
):
    """
    Fonction générique qui exécute le workflow complet (GridSearchCV → évaluation →
    calibration sur holdout → ré-évaluation → seuil de Youden sur calibration → plots finaux)
    pour différents modèles, selon l'acronyme fourni:

      - 'lr' : Logistic Regression (StandardScaler + LogisticRegression)
      - 'xgb': XGBoost (XGBClassifier)
      - 'rf' : Random Forest (RandomForestClassifier)
      - 'knn': K Nearest Neighbors (StandardScaler + KNeighborsClassifier)
      - 'dt' : Decision Tree (DecisionTreeClassifier)

    Le seuil optimal (Youden) est calculé uniquement sur le set de calibration en binaire,
    puis appliqué tel quel au test (pas de recalcul sur test). En multiclasses, ROC/AUC et
    calibration sont traités one-vs-rest.
    """

    key = model_key.lower().strip()

    X = np.asarray(X)
    y = np.asarray(y)

    classes = np.unique(y)
    K = len(classes)
    if K == 2 and pos_label not in classes:
        raise ValueError(f"pos_label='{pos_label}' n'est pas dans les classes: {list(classes)}")

    # 1) Split global: trainval / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 2) Split trainval -> search_train / calib (calibration holdout)
    X_search, X_calib, y_search, y_calib = train_test_split(
        X_trainval, y_trainval, test_size=calib_size, stratify=y_trainval, random_state=random_state
    )

    # 3) Build pipeline and param grid per model
    if key == 'lr':
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, random_state=random_state))
        ])
        if K == 2:
            param_grid = {
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'clf__solver': ['liblinear', 'saga'],
                'clf__class_weight': [None, 'balanced'],
            }
            scoring = make_scorer(f1_score, pos_label=pos_label)
        else:
            param_grid = {
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'clf__solver': ['saga'],
                'clf__multi_class': ['ovr', 'multinomial'],
                'clf__class_weight': [None, 'balanced'],
            }
            scoring = 'f1_macro'

    elif key == 'xgb':
        base_params = dict(
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric='logloss' if K == 2 else 'mlogloss',
            objective='binary:logistic' if K == 2 else 'multi:softprob',
            num_class=None if K == 2 else K,
            tree_method='auto',
            verbosity=0,
        )
        pipe = Pipeline([
            ('clf', XGBClassifier(**base_params))
        ])
        param_grid = {
            'clf__n_estimators': [200, 500],
            'clf__max_depth': [3, 5, 7],
            'clf__learning_rate': [0.03, 0.1],
            'clf__subsample': [0.7, 1.0],
            'clf__colsample_bytree': [0.7, 1.0],
            'clf__reg_lambda': [1.0, 10.0],
            'clf__min_child_weight': [1, 5],
        }
        scoring = make_scorer(f1_score, pos_label=pos_label) if K == 2 else 'f1_macro'

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
        scoring = make_scorer(f1_score, pos_label=pos_label) if K == 2 else 'f1_macro'

    elif key == 'knn':
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier())
        ])
        param_grid = {
            'clf__n_neighbors': [3, 5, 7, 11, 15],
            'clf__weights': ['uniform', 'distance'],
            'clf__p': [1, 2],  # Manhattan vs Euclidean
        }
        scoring = make_scorer(f1_score, pos_label=pos_label) if K == 2 else 'f1_macro'

    elif key == 'dt':
        pipe = Pipeline([
            ('clf', DecisionTreeClassifier(random_state=random_state))
        ])
        param_grid = {
            'clf__max_depth': [None, 5, 10, 20],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__max_features': [None, 'sqrt', 'log2'],
            'clf__class_weight': [None, 'balanced'],
        }
        scoring = make_scorer(f1_score, pos_label=pos_label) if K == 2 else 'f1_macro'

    else:
        raise ValueError("model_key doit être parmi {'lr','xgb','rf','knn','dt'}")

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=5,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )
    grid.fit(X_search, y_search)
    best_pipe = grid.best_estimator_
    class_names = list(best_pipe.named_steps['clf'].classes_)

    # 4) Évaluation initiale (uncalibrated) sur test
    y_pred_test_uncal = best_pipe.predict(X_test)
    cm_uncal = confusion_matrix(y_test, y_pred_test_uncal, labels=class_names)
    print(f"Rapport {key.upper()} (uncalibrated):\n", classification_report(y_test, y_pred_test_uncal, target_names=class_names))

    proba_test_uncal = None
    auc_uncal = None
    if hasattr(best_pipe, 'predict_proba'):
        proba_test_uncal = best_pipe.predict_proba(X_test)
        if K == 2:
            pos_idx = class_names.index(pos_label)
            y_test_bin = (y_test == pos_label).astype(int)
            fpr_uncal, tpr_uncal, _ = roc_curve(y_test_bin, proba_test_uncal[:, pos_idx])
            auc_uncal = roc_auc_score(y_test_bin, proba_test_uncal[:, pos_idx])
            plt.figure(figsize=(5,4))
            plt.plot(fpr_uncal, tpr_uncal, label=f'AUC = {auc_uncal:.3f}')
            plt.plot([0,1], [0,1], 'k--')
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC {key.upper()} (uncalibrated, test)')
            plt.legend(); plt.tight_layout(); plt.show()
        else:
            y_test_bin = label_binarize(y_test, classes=class_names)
            auc_uncal = roc_auc_score(y_test_bin, proba_test_uncal, multi_class='ovr', average='macro')
            for i, cls in enumerate(class_names):
                fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], proba_test_uncal[:, i])
                plt.plot(fpr_i, tpr_i, label=str(cls))
            plt.plot([0,1], [0,1], 'k--')
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC OVR {key.upper()} (uncalibrated, test) | AUC_macro={auc_uncal:.3f}')
            plt.legend(); plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm_uncal, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'Confusion matrix {key.upper()} (uncalibrated, test)')
    plt.tight_layout(); plt.show()

    # 5) Calibration sur X_calib
    calibrator = CalibratedClassifierCV(base_estimator=best_pipe, method=method, cv='prefit')
    calibrator.fit(X_calib, y_calib)

    # Histogramme + courbe de calibration (sur calib)
    if K == 2:
        pos_idx = calibrator.classes_.tolist().index(pos_label)
        proba_calib = calibrator.predict_proba(X_calib)[:, pos_idx]
        y_calib_bin = (y_calib == pos_label).astype(int)

        plt.figure(figsize=(5,4))
        plt.hist(proba_calib, bins=20, range=(0,1), alpha=0.8, edgecolor='k')
        plt.xlabel('Probabilité prédite (positive)'); plt.ylabel('Fréquence')
        plt.title(f'Histogramme des probabilités {key.upper()} (calibration set)')
        plt.tight_layout(); plt.show()

        frac_pos, mean_pred = calibration_curve(y_calib_bin, proba_calib, n_bins=10, strategy='uniform')
        plt.figure(figsize=(5,4))
        plt.plot(mean_pred, frac_pos, 'o-', label='Calibrated')
        plt.plot([0,1], [0,1], 'k--', label='Idéal')
        plt.xlabel('Probabilité moyenne prédite'); plt.ylabel('Fraction positifs')
        plt.title(f'Courbe de calibration {key.UPPER()} (calibration set)')
        plt.legend(); plt.tight_layout(); plt.show()
    else:
        proba_calib = calibrator.predict_proba(X_calib)
        y_calib_bin_all = label_binarize(y_calib, classes=calibrator.classes_)
        plt.figure(figsize=(6,5))
        for i, cls in enumerate(calibrator.classes_):
            frac_pos, mean_pred = calibration_curve(y_calib_bin_all[:, i], proba_calib[:, i], n_bins=10, strategy='uniform')
            plt.plot(mean_pred, frac_pos, 'o-', label=str(cls))
        plt.plot([0,1], [0,1], 'k--', label='Idéal')
        plt.xlabel('Probabilité moyenne prédite'); plt.ylabel('Fraction positifs')
        plt.title(f'Courbes de calibration OVR {key.upper()} (calibration set)')
        plt.legend(); plt.tight_layout(); plt.show()

    # 6) Ré-évaluation (CALIBRÉ) sur test
    y_pred_test_cal = calibrator.predict(X_test)
    cm_cal = confusion_matrix(y_test, y_pred_test_cal, labels=list(calibrator.classes_))
    print(f"Rapport {key.upper()} (calibrated):\n", classification_report(y_test, y_pred_test_cal, target_names=list(calibrator.classes_)))

    auc_cal = None
    proba_test_cal = None
    if hasattr(calibrator, 'predict_proba'):
        proba_test_cal = calibrator.predict_proba(X_test)
        if K == 2:
            pos_idx = list(calibrator.classes_).index(pos_label)
            y_test_bin = (y_test == pos_label).astype(int)
            fpr_cal, tpr_cal, _ = roc_curve(y_test_bin, proba_test_cal[:, pos_idx])
            auc_cal = roc_auc_score(y_test_bin, proba_test_cal[:, pos_idx])
            plt.figure(figsize=(5,4))
            plt.plot(fpr_cal, tpr_cal, label=f'AUC = {auc_cal:.3f}')
            plt.plot([0,1], [0,1], 'k--')
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC {key.upper()} (calibrated, test)')
            plt.legend(); plt.tight_layout(); plt.show()
        else:
            y_test_bin = label_binarize(y_test, classes=list(calibrator.classes_))
            auc_cal = roc_auc_score(y_test_bin, proba_test_cal, multi_class='ovr', average='macro')
            for i, cls in enumerate(calibrator.classes_):
                fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], proba_test_cal[:, i])
                plt.plot(fpr_i, tpr_i, label=str(cls))
            plt.plot([0,1], [0,1], 'k--')
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC OVR {key.upper()} (calibrated, test) | AUC_macro={auc_cal:.3f}')
            plt.legend(); plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm_cal, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=list(calibrator.classes_), yticklabels=list(calibrator.classes_), ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'Confusion matrix {key.UPPER()} (calibrated, test)')
    plt.tight_layout(); plt.show()

    # 7) Seuil optimal (Youden) calculé SUR CALIB (et appliqué au test)
    best_thresh = None
    cm_thresh = None
    if K == 2 and proba_test_cal is not None:
        pos_idx = list(calibrator.classes_).index(pos_label)
        y_calib_bin = (y_calib == pos_label).astype(int)
        proba_calib_pos = calibrator.predict_proba(X_calib)[:, pos_idx]
        fpr_c, tpr_c, thr_c = roc_curve(y_calib_bin, proba_calib_pos)
        J = tpr_c - fpr_c
        ix = np.argmax(J)
        best_thresh = float(thr_c[ix])
        print(f"Best threshold (Youden J) {key.upper()} sur calib: {best_thresh:.4f} | J={J[ix]:.4f} | TPR={tpr_c[ix]:.4f} | FPR={fpr_c[ix]:.4f}")

        y_pred_test_thr = (proba_test_cal[:, pos_idx] >= best_thresh).astype(int)
        neg_label = [c for c in calibrator.classes_ if c != pos_label][0]
        y_pred_test_thr_lbl = np.where(y_pred_test_thr == 1, pos_label, neg_label)
        cm_thresh = confusion_matrix(y_test, y_pred_test_thr_lbl, labels=list(calibrator.classes_))
        print(f"Rapport {key.upper()} (calibrated + threshold):\n", classification_report(y_test, y_pred_test_thr_lbl, target_names=list(calibrator.classes_)))

        # ROC identique mais on marque le point (FPR, TPR) au seuil choisi
        fpr_c2, tpr_c2, _ = roc_curve((y_test == pos_label).astype(int), proba_test_cal[:, pos_idx])
        tn, fp, fn, tp = confusion_matrix((y_test == pos_label).astype(int), y_pred_test_thr, labels=[0,1]).ravel()
        fpr_pt = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_pt = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        plt.figure(figsize=(5,4))
        plt.plot(fpr_c2, tpr_c2, label=f'AUC = {auc_cal:.3f}')
        plt.scatter([fpr_pt], [tpr_pt], c='red', edgecolor='k', s=60, label=f'Th={best_thresh:.3f}')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC {key.upper()} (calibrated, test) + Youden threshold')
        plt.legend(); plt.tight_layout(); plt.show()

        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm_thresh, annot=True, fmt='d', cbar=False, cmap='Blues',
                    xticklabels=list(calibrator.classes_), yticklabels=list(calibrator.classes_), ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'Confusion matrix {key.upper()} (calibrated + threshold, test)')
        plt.tight_layout(); plt.show()

    results = {
        'model_key': key,
        'classes': class_names,
        'grid_best_params': grid.best_params_,
        'grid_best_score': grid.best_score_,
        'best_estimator_uncalibrated': best_pipe,
        'calibrated_model': calibrator,
        'auc_uncalibrated': auc_uncal,
        'auc_calibrated': auc_cal,
        'best_threshold': best_thresh,
        'cm_uncalibrated': cm_uncal,
        'cm_calibrated': cm_cal,
        'cm_calibrated_threshold': cm_thresh,
    }
    return results
