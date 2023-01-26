import numpy as np

class PredictionService:
    __instance = None
    @staticmethod 
    def getInstance():
        """ Static access method. """
        if PredictionService.__instance == None:
            PredictionService(model = None)
        
        return PredictionService.__instance
    
    def __init__(self, model, max_len = 100):
        if PredictionService.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            PredictionService.__instance = self
        
        self.model = model
        self.max_len = max_len
        print ("PredictionService __instance created")
    
    def reload(self,model):
        self.model = model

    def process(self, text_list):
        if self.model == None:
            return None
        
        result_list = list()
        result_list_percentage = list()
        for text_to_process in text_list:
            prediction = self.model.predict(text_to_process)
            result = np.around(prediction, decimals=0).argmax(axis=1)[0] #we might really have to change this one
            result_list.append(result)
            result_percentage = np.around(prediction, decimals=3)
            result_list_percentage.append(result_percentage)

        return result_list, result_list_percentage