import tensorflow as tf

class BestLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name, folder_name):
      super(BestLossCallback, self).__init__()
      self.best_score = 1
      self.folder_name = folder_name
      self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        if (logs['categorical_accuracy'] >= 0.90 and self.best_score > logs['val_loss']): 
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
        if (logs['categorical_accuracy'] >= 0.90 and self.best_score < logs['val_categorical_accuracy']): 
          self.best_score = logs['val_categorical_accuracy']
          self.model.save(f"drive/MyDrive/MTD/Models/{self.model_name}/{self.folder_name}/best_accuracy.hdf5", overwrite=True)
          print("saving model for best accuracy.")