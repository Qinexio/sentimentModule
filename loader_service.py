import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

class LoaderService:
    __instance = None
    @staticmethod 
    def getInstance():
        """ Static access method. """
        if LoaderService.__instance == None:
            print("LoaderService Singleton Init")
            LoaderService()
            
        return LoaderService.__instance
    
    def __init__(self, model_name = "data/defaultModel", tokenizer_name = "data/defaultTokenizer"):
        if LoaderService.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            LoaderService.__instance = self
        
        self.model = None
        self.tokenizer = Tokenizer()
        
        try:
            self.tokenizer = self.load_pickle(tokenizer_name)
        except:
            print("LoaderService failed to find tokenizer")
        
        try:
            self.model = load_model(model_name + ".hdf5")
        except:
            print("LoaderService failed to find model")
            
        print ("LoaderService __instance created")
    
    def reload(self, model_name = "data/defaultModel", tokenizer_name = "data/defaultTokenizer"):
        try:
            self.tokenizer = self.load_pickle(tokenizer_name)
        except:
            print("LoaderService failed to find tokenizer")
        
        try:
            self.model = load_model(model_name + ".hdf5")
        except:
            print("LoaderService failed to find model")
            
        print ("LoaderService __instance created")
        
    def save_pickle(self, object_save, filename):
        with open(filename+'.pickle', 'wb') as handle:
            pickle.dump(object_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, filename):
        with open(filename+'.pickle', 'rb') as handle:
            return pickle.load(handle)
    
    def save_tokenizer(self, tokenizer):
        self.save_pickle(tokenizer, "data/defaultTokenizer")
    