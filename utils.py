import logging
import os
import datetime
from pathlib import Path
import mlflow
import absl.logging
import dotenv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import (
    Callback,
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
)
from tensorflow.keras.models import save_model
from tqdm import tqdm

from conf_mat_thr import save_conf_mat_thr_clf
from sequences import (
    SequenceConfig,
    TrainSequence,
    ValidationSequence,
    TestSequence,
)
from threshold import threshold_clf

# Setup callbacks
def setup_callbacks(best_model_folder, epochs, model_folder, patience):
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=50,
        min_lr=1e-7,
        verbose=1
    )
    custom_model_checkpoint = CustomModelCheckpoint(best_model_folder, epochs)
    csv_logger = CSVLogger(model_folder / "model_history_log.csv", append=True)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    return [reduce_lr, custom_model_checkpoint, csv_logger, early_stopping]


# Prepare data sequences
def prepare_data_loader(
    data_folder,
    batch_size,
    nb_classes,
    window_size,
    augmentor=None,
    augment_prob: float = 0.0,
):
    base_cfg = dict(
        batch_size=batch_size,
        n_classes=nb_classes,
        window_size=window_size,
        augmentor=augmentor,
        augment_prob=augment_prob,
    )

    train_seq = TrainSequence(
        SequenceConfig(
            csv_path=Path(data_folder) / "train.csv",
            **base_cfg,
        )
    )
    val_seq = ValidationSequence(
        SequenceConfig(
            csv_path=Path(data_folder) / "validation.csv",
            **base_cfg,
        )
    )
    test_seq = TestSequence(
        SequenceConfig(
            csv_path=Path(data_folder) / "test.csv",
            **base_cfg,
        )
    )

    return train_seq, val_seq, test_seq


# Create model folders
def create_model_folders(model_name):
    date = datetime.date.today().strftime("%Y-%m-%d")
    model_folder_name = f"{date}-{model_name}"
    model_folder = Path("./", model_folder_name)

    try:
        os.mkdir(model_folder)
    except FileExistsError:
        print("Model folder exists")

    best_model_folder = model_folder / 'best_model'
    try:
        os.mkdir(best_model_folder)
    except FileExistsError:
        print("Best model folder exists")

    return model_folder, best_model_folder


# Configure GPU settings
def setup_gpu(memory_fraction=0.3):
    gpus = tf.config.list_physical_devices('GPU')
    print("gpus:", gpus)
    if gpus:

        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_fraction * 19294)]
            )
        except RuntimeError as e:
            print(e)
    return gpus


def evaluate_model(model_folder_clf, model_name, model_epoch_clf, custom_objects, thresholds, test_seq, classes):

    model = tf.keras.models.load_model(model_folder_clf/f"best_model/model_epoch_{model_epoch_clf}.keras", custom_objects=custom_objects)

    seq_files_clf = []
    seq_channels_clf = []
    y_trues_clf, y_scores_clf = [], []

    for x, y_true, file, channel in tqdm(test_seq, total=test_seq.n_batches):
        seq_files_clf.append(file)
        seq_channels_clf.append(channel)
        y_scores_clf.append(model.predict(x, verbose=0))
        y_trues_clf.append(y_true)

    for threshold in thresholds:
        y_trues_clf2, y_scores_clf3 = threshold_clf(y_scores_clf, y_trues_clf, threshold)

        y_trues_lc = np.argmax(y_trues_clf2, -1)
        y_preds_lc = np.argmax(y_scores_clf3, -1)

        acc = accuracy_score(y_trues_lc, y_preds_lc)
        mlflow.log_metric(f"test_acc_thr_{threshold}", acc)

        save_conf_mat_thr_clf(y_trues_lc, y_preds_lc, classes, model_name, model_folder_clf, model_epoch_clf, threshold)


# Custom model checkpoint callback
class CustomModelCheckpoint(Callback):    
	def __init__(self, folder, total_epochs, *args, **kwargs):
		super(CustomModelCheckpoint, self).__init__(*args, **kwargs)
		self.folder = folder  # Path object
		self.total_epochs = total_epochs
		self.minimum_val_loss = float('inf')
		self.best_model = None
		self.best_epoch = 0

	def on_epoch_end(self, epoch, logs=None):
		
		path = os.path.join(self.folder, f'model_epoch_{epoch}.keras')
		save_model(self.model, path)
		# logs is a dictionary
		if logs['val_loss'] < self.minimum_val_loss:  # your custom condition
			self.minimum_val_loss = logs['val_loss']
			self.best_epoch = epoch
			self.best_model = self.model
			self.model.save(self.folder/'best_model.keras', overwrite=True)
		if epoch == self.total_epochs - 1:
			self.best_model.save(self.folder / f'best_model_epoch_{self.best_epoch}.keras', overwrite=True)
