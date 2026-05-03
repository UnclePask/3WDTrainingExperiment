'''
Created on Dec 22, 2025

@author: pasquale
'''
from keras.metrics import Recall

class RecallMalicious(Recall):
    
    def __init__(self, **kwargs):
        super().__init__(class_id=1, **kwargs)