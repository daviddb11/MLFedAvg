from tensorflow_addons.metrics import HammingLoss
import tensorflow as tf
import keras


class Metrica:

    def __init__(self, threshold=0.5):
        
         self.threshold = threshold

    def HammingLoss(self, mode='multilabel'):
        
        return HammingLoss(threshold=self.threshold, mode=mode)

    def f1_macro(self):
        return tf.keras.metrics.F1Score(threshold=self.threshold,average='macro',name='f1_score_macro')

    def f1_micro(self):
        return tf.keras.metrics.F1Score(threshold=self.threshold,average='micro',name='f1_score_micro')
    
    def JS(self):
        return keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='JS')


