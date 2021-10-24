import tensorflow as tf
import os

class BestLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name, folder_name):
      super(BestLossCallback, self).__init__()
      self.best_score = 1
      self.folder_name = folder_name
      self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        if self.best_score > logs['val_loss']: 
          self.best_score = logs['val_loss']
          self.model.save(f"drive/MyDrive/MTD/Models/{self.model_name}/{self.folder_name}/best_loss.hdf5", overwrite=True)
          print("saving model for best loss.")

class BestAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name, folder_name):
      super(BestAccuracyCallback, self).__init__()
      self.best_score = 0
      self.folder_name = folder_name
      self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        if self.best_score < logs['val_categorical_accuracy']: 
          self.best_score = logs['val_categorical_accuracy']
          self.model.save(f"drive/MyDrive/MTD/Models/{self.model_name}/{self.folder_name}/best_accuracy.hdf5", overwrite=True)
          print("saving model for best accuracy.")

class LoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name, folder_name):
      super(LoggingCallback, self).__init__()
      self.folder_name = folder_name
      self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        model_name = f"{self.model_name}/{self.folder_name}"
        path = f"./drive/MyDrive/MTD/Models/{model_name}"
        if (epoch == 0):
          os.makedirs(path, exist_ok=True)
          f = open(path + "/logs.txt", "w")
          f.write(f"loss: {logs['loss']:.4f}, categorical_accuracy: {logs['categorical_accuracy']:.4f}, val_loss: {logs['val_loss']:.4f}, val_categorical_accuracy: {logs['val_categorical_accuracy']:.4f}\n")
          f.close()
        else:
          f = open(path + "/logs.txt", "a")
          f.write(f"loss: {logs['loss']:.4f}, categorical_accuracy: {logs['categorical_accuracy']:.4f}, val_loss: {logs['val_loss']:.4f}, val_categorical_accuracy: {logs['val_categorical_accuracy']:.4f}\n")
          f.close()

class LastModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name, folder_name):
      super(LastModelCallback, self).__init__()
      self.folder_name = folder_name
      self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        if logs['categorical_accuracy'] > 0.90:
          self.model.save(f"drive/MyDrive/MTD/Models/{self.model_name}/{self.folder_name}/last.hdf5", overwrite=True)
          print("saving last model.")