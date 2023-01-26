from tensorflow.keras.utils import to_categorical
import numpy as np

class LabelingUtility:
    __instance = None
    @staticmethod 
    def getInstance():
        """ Static access method. """
        if LabelingUtility.__instance == None:
            LabelingUtility()
        
        return LabelingUtility.__instance
    
    def __init__(self):
        if LabelingUtility.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            LabelingUtility.__instance = self
        
        print ("LabelingUtility __instance created")
    
    def full_label_switch(self, label):
        switch={
            'neutral': 1,
            'anger': 2,
            'fear': 3,
            'sadness': 4,
            'joy': 5,
            'surprise': 6,
            'love': 7,
           }
        return switch.get(label,0)
    
    def reverse_full_label_switch(self, label):
        switch={
            1: 'neutral',
            2: 'anger',
            3: 'fear',
            4: 'sadness',
            5: 'joy',
            6: 'surprise',
            7: 'love',
           }
        return switch.get(label,'unknown')
    
    def simple_label_switch(self, label):
        switch={
            'neutral': 1,
            'negative': 2,
            'positive': 3,
           }
        return switch.get(label,0)
    
    def reverse_simple_label_switch(self, label):
        switch={
            1: 'neutral',
            2: 'negative',
            3: 'positive',
           }
        return switch.get(label,'unknown')
    
    def prepare_labels(self, label, num_labels, method):
        y = []
        for i in range(len(label)):
            y.append(method(label[i]))
            
        y = np.array(y)
        label = to_categorical(y, num_labels+1, dtype="float32")
        del y
        return label