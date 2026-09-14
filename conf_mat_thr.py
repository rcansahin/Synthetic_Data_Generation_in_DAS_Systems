import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import mlflow
plt.style.use('fivethirtyeight');


def save_conf_mat_thr_ano(y_trues, y_preds, classes, model_name_ano, model_folder_ano, model_epoch_ano, thr_ano):
    """Confusion matrix."""
    cm = confusion_matrix(y_trues, y_preds)
    cm_norm = cm / cm.sum(axis=1)
    annot = np.array([annot_format(i, row) for i, row in enumerate(cm)])
    sns.set_style("dark")
    sns.set(font_scale=1.0)
    sns.heatmap(cm, annot=annot, fmt="",
                xticklabels=classes,
                yticklabels=classes,
                cmap='Reds', square=True, cbar=True,
                linewidths=4, linecolor='Black',
                vmin=0, vmax=1000)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f'{model_name_ano}\n {model_epoch_ano}_threshold_{thr_ano}')
    
    path = model_folder_ano / f"confusion_matrix_threshold_{thr_ano}_{model_epoch_ano}.png"
    #path = model_folder / f"confusion_matrix_threshold_{th}_{model_epoch}_oldtestset.png"
    
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()
    


def save_conf_mat_thr_clf(y_trues, y_preds, classes, model_name_clf, model_folder_clf, model_epoch_clf, thr_clf):
    """Confusion matrix."""
    cm = confusion_matrix(y_trues, y_preds)
    cm_norm = cm / cm.sum(axis=1)
    annot = np.array([annot_format(i, row) for i, row in enumerate(cm)])
    sns.set_style("dark")
    sns.set(font_scale=1.0)
    plt.figure(figsize=(8, 8)) 
    sns.heatmap(cm, annot=annot, fmt="",
                xticklabels=classes,
                yticklabels=classes,
                cmap='Reds', square=True, cbar=True,
                linewidths=4, linecolor='Black',
                vmin=0, vmax=100)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f'{model_name_clf}\n {model_epoch_clf}_threshold_{thr_clf}')
    
    path = model_folder_clf / f"confusion_matrix_threshold_{thr_clf}_{model_epoch_clf}.png"
    #path = model_folder / f"confusion_matrix_threshold_{th}_{model_epoch}_newtestset.png"
    
    plt.savefig(path, dpi=100, bbox_inches="tight")

    mlflow.log_artifact(str(path), "evaluation")
    
    
    
def save_conf_mat_thr_anoclf(y_trues, y_preds, classes, model_name_clf, model_folder_clf, model_epoch_clf, thr_clf, model_name_ano, model_epoch_ano, thr_ano):
    """Confusion matrix."""
    cm = confusion_matrix(y_trues, y_preds)
    cm_norm = cm / cm.sum(axis=1)
    annot = np.array([annot_format(i, row) for i, row in enumerate(cm)])
    sns.set_style("dark")
    sns.set(font_scale=1.0)
    plt.figure(figsize=(8, 8)) 
    sns.heatmap(cm, annot=annot, fmt="",
                xticklabels=classes,
                yticklabels=classes,
                cmap='Reds', square=True, cbar=True,
                linewidths=4, linecolor='Black',
                vmin=0, vmax=100)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f'{model_name_ano}_{model_epoch_ano}_ano_thr_{thr_ano}\n {model_name_clf}_{model_epoch_clf}_clf_thr_{thr_clf}')
    
    path = model_folder_clf / f"confusion_matrix_{model_name_ano}_{model_epoch_ano}_ano_thr_{thr_ano}_{model_name_clf}_{model_epoch_clf}_clf_thr_{thr_clf}.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()   
    

    
def annot_format(i, row):
    """Confusion matrix annotation format."""
    res = []
    s = np.sum(row)
    for j, val in enumerate(row):
        if i == j:
            res.append(f'{ val*100/s:.1f}%\n{val}/{s}')
        else:
            res.append(f'{ val*100/s:.1f}%\n{val}')
    return res