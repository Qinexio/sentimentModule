import numpy as np
from flask import Flask
#here be two lines that fix a weird import issue with flask_restful
import flask.scaffold
from flask.json import jsonify
from pickle import NONE
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func

from collections import namedtuple
from flask_restful import Resource
from flask_restful import Api
from flask_restful import reqparse
from flask_restful import request
from pandas import read_csv
from glove_utility import GloVeUtility
from loader_service import LoaderService
from prediction_service import PredictionService
from train_service import TrainService
from processing_utility import TextPreprocessUtility
from labeling_utility import LabelingUtility
from threading import Thread


class WebRouting:
    app = Flask(__name__)  # creating the Flask class object
    api = Api(app)
    
    __instance = None
    @staticmethod 
    def getInstance():
        """ Static access method. """
        if WebRouting.__instance == None:
            WebRouting()
            
        return WebRouting.__instance
    
    def __init__(self):
        if WebRouting.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            WebRouting.__instance = self
        
        self.loader_service = LoaderService()
        self.prediction_service = PredictionService(model = self.loader_service.model)
        self.train_service = TrainService()
        self.processing_utility = TextPreprocessUtility(tokenizer = self.loader_service.tokenizer)
        self.label_utility = LabelingUtility()
        
        print ("WebRouting __instance created")
    
    #did you know you can't access outer class features from within an inner class? 
    #the more you know
    #I had to bootleg this using singletons
    class Predict(Resource):

        def post(self):
            parser = reqparse.RequestParser()
            parser.add_argument('txt', required=True)
            args = parser.parse_args()   
            
            processing_utility = TextPreprocessUtility.getInstance()
            label_utility = LabelingUtility.getInstance()
            prediction_service = PredictionService.getInstance()
            
            text_list = processing_utility.split_text(args['txt'])

            processed_text = processing_utility.process_data(text_list)
            
            #again, can't stress enough how important this is, empty rogue elements are not good
            processed_text = list(filter(None, processed_text))
            
            #something odd might happen here see: sentence_to_process = [ processing_utility.depure_data(text_to_process)]
            #very likely inside the function
            processed_text_to_numbers = processing_utility.process_text_to_numbers(processed_text)
            
            #print(processed_text_to_numbers)
            
            result, result_percentage = prediction_service.process(processed_text_to_numbers)
            
            if result == None:
                return {}, 404
                        
            result_translated = []
            result_translated_percentage = []
            
            #print(result)
            #print(result_percentage)
            
            for to_translate in result:
                result_translated.append(label_utility.reverse_full_label_switch(to_translate))
            
            for to_translate in result_percentage:
                toAdd = ""
                for i in range(0,len(to_translate[0])):
                    toAdd = toAdd + str(np.around(to_translate[0][i]*100, decimals=3)) +"% " + label_utility.reverse_full_label_switch(i) + " "
                result_translated_percentage.append(toAdd)
                    
            #print(result_translated_percentage)
            predNamedTouple = namedtuple('prediction', ['sentenceText', 'sentencePrediction', 'sentencePercentage'])
            
            mixed_result_list = tuple(zip(text_list, result_translated,result_translated_percentage)) 
            to_return = []
            
            for toTranscribe in mixed_result_list:
                to_return.append(predNamedTouple(toTranscribe[0],toTranscribe[1],toTranscribe[2])._asdict())
            
            # 200 is the OK code, 404 however, well you probably know already what it does
            return { 'result' : to_return }, 200
        
        pass
    
    # make sure to test if predict works wihle training is in session
    # also make sure you send back a response before training starts
    class Train(Resource):

        def post(self):
            
            parser = reqparse.RequestParser()
            #parser.add_argument('batches', required=True)
            parser.add_argument('useCSV', required=True)
            parser.add_argument('useGloVE', required=True)
            
            args = parser.parse_args()
            use_csv = str.capitalize(args['useCSV'])
            use_glove = str.capitalize(args['useGloVE'])
            
            print(use_csv)
            print(use_glove)
            labels = []
            text_to_train = []
            
            if use_csv == "True":
                loaded_csv = read_csv("data/defaultCSV.csv")
                text_to_train = loaded_csv['text'].values.tolist()
                labels = loaded_csv['emotions'].values.tolist()
            
            #parser won't work,yay
            body = request.get_json()
            
            sentences = body['sentences']
            for sentence in sentences:
                text_to_train.append(sentence['sentenceText'])
                labels.append(sentence['sentencePrediction'])
            
            loader_service = LoaderService.getInstance()
            processing_utility = TextPreprocessUtility.getInstance()
            label_utility = LabelingUtility.getInstance()
            train_service = TrainService.getInstance()
            
            #for testing purposes
            #text_to_train = text_to_train[:20000]
            #labels = labels[:20000]
            
            processed_text =  processing_utility.process_data(text_to_train)
            #something odd might happen here see: sentence_to_process = [ processing_utility.depure_data(text_to_process)]
            #very likely inside the function
            processed_text_to_numbers =  processing_utility.process_text_to_numbers_train(processed_text)
            
            loader_service.save_tokenizer(processing_utility.tokenizer_train)

            labels_translated = label_utility.prepare_labels(labels, 7, label_utility.full_label_switch)
            
            vocabulary = None
            vocab_size = len(processing_utility.tokenizer.word_index)+1
            
            if use_glove == "True":
                glove_utility = GloVeUtility(processing_utility.tokenizer)
                vocabulary = glove_utility.create_dictionary()
            
            thread = Thread(target=train_service.process, args= (processed_text_to_numbers, labels_translated, vocabulary, vocab_size, use_glove))
            thread.start()
            
            #might need to add the glove option or not here too
            #but it comes with a problem with an already trained model getting new data
            #actually doesn't that mean a glove layer already has a limit
            #I feel like this is a bad idea
            #we should just always train a new model
            #we also need a variable check here to make sure the app can't train more than once
            #change the value back to 0 on callback finish
            #train_service.process(text = processed_text_to_numbers, labels = labels_translated, vocab_size = vocab_size, use_weights = True, glove_layer = vocabulary )
  
            #also make a callback function to reload the model for the prediction element
            return {}, 200
        
        pass
    
    class Reload(Resource):

        def post(self):

            prediction_service = PredictionService.getInstance()
            processing_utility = TextPreprocessUtility.getInstance() 
            loader_service = LoaderService.getInstance()
            loader_service.reload()
            prediction_service.reload(loader_service.model)
            processing_utility.reload(loader_service.tokenizer)
            return {}, 200
        
        pass
    
    api.add_resource(Predict, '/predict') 
    api.add_resource(Train, '/train')
    api.add_resource(Reload, '/reload')

