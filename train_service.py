from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

class TrainService:
    __instance = None
    @staticmethod 
    def getInstance():
        """ Static access method. """
        if TrainService.__instance == None:
            TrainService()
        
        return TrainService.__instance
    
    def __init__(self, dropout_rate = 0.5, epochs_to_train = 1, max_len = 100, use_weights = False):
        if TrainService.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            TrainService.__instance = self
            
        self.model = Sequential()
        self.dropout_rate = dropout_rate
        self.max_len = max_len
        self.epochs_to_train = epochs_to_train
        self.weights = None #add here a boolean to initialize the methods without a weight matrix if so desired
        self.use_weights = use_weights
        self.vocab_size = None
        self.is_training = False
        print ("TrainService __instance created")
    
    #did this with 20 and 6 before, LSTM cell method, one layer only
    def model_init_meth_one(self,num_cell_LSTM,num_cell_dense):
        if self.use_weights:
            self.model.add(layers.Embedding(input_dim = self.vocab_size, output_dim = self.max_len, weights = [self.weights],input_length=self.max_len,trainable = False))
        else:
            self.model.add(layers.Embedding(input_dim = self.vocab_size, output_dim = self.max_len, input_length= self.max_len))            
        
        self.model.add(layers.Bidirectional(layers.LSTM(num_cell_LSTM,dropout= self.dropout_rate)))
        self.model.add(layers.Dense(num_cell_dense,activation='softmax'))
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
    
    #previous method but more layers
    def model_init_meth_two(self, num_cell_LSTM_one, num_cell_LSTM_two , num_cell_dense):
        if self.use_weights:
            self.model.add(layers.Embedding(input_dim = self.vocab_size, output_dim = self.max_len, weights = [self.weights],input_length=self.max_len,trainable = False))
        else:
            self.model.add(layers.Embedding(input_dim = self.vocab_size, output_dim = self.max_len, input_length= self.max_len))         
        
        self.model.add(layers.Bidirectional(layers.LSTM(num_cell_LSTM_one, dropout= self.dropout_rate, return_sequences=True)))
        self.model.add(layers.Bidirectional(layers.LSTM(num_cell_LSTM_two, dropout= self.dropout_rate)))
        self.model.add(layers.Dense(num_cell_dense,activation='softmax'))
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
    
    #previous method but with GRU cells
    def model_init_meth_three(self, num_cell_GRU_one, num_cell_GRU_two, num_cell_dense):
        if self.use_weights:
            self.model.add(layers.Embedding(input_dim = self.vocab_size, output_dim = self.max_len, 
                                            weights = [self.weights],input_length=self.max_len,trainable = False))
        else:
            self.model.add(layers.Embedding(input_dim = self.vocab_size, output_dim = self.max_len, input_length= self.max_len))         
        
        self.model.add(layers.Bidirectional(layers.GRU(num_cell_GRU_one*2, dropout= self.dropout_rate, return_sequences=True)))
        self.model.add(layers.Bidirectional(layers.GRU(num_cell_GRU_one, dropout= self.dropout_rate, return_sequences=True)))
        self.model.add(layers.Bidirectional(layers.GRU(num_cell_GRU_two, dropout= self.dropout_rate, return_sequences=True)))
        self.model.add(layers.Bidirectional(layers.GRU(int(num_cell_GRU_two/2), dropout= self.dropout_rate)))
        self.model.add(layers.Dense(num_cell_dense,activation='softmax'))
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
    
    def model_init_meth_four(self, num_cell_GRU_one, num_cell_GRU_two, num_cell_dense):
        if self.use_weights:
            self.model.add(layers.Embedding(input_dim = self.vocab_size, output_dim = self.max_len, 
                                            weights = [self.weights],input_length=self.max_len,trainable = False))
        else:
            self.model.add(layers.Embedding(input_dim = self.vocab_size, output_dim = self.max_len, input_length= self.max_len))         
        
        self.model.add(layers.Bidirectional(layers.LSTM(num_cell_GRU_one*2, dropout= self.dropout_rate, return_sequences=True)))
        self.model.add(layers.Bidirectional(layers.LSTM(num_cell_GRU_one, dropout= self.dropout_rate, return_sequences=True)))
        self.model.add(layers.Bidirectional(layers.LSTM(num_cell_GRU_two, dropout= self.dropout_rate, return_sequences=True)))
        self.model.add(layers.Bidirectional(layers.LSTM(int(num_cell_GRU_two/2), dropout= self.dropout_rate)))
        self.model.add(layers.Dense(num_cell_dense,activation='softmax'))
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train_model(self, X_train,X_test,y_train,y_test):
        #Implementing model checkpoins to save the best metric and do not lose it on training.
        filename = "data/defaultModel.hdf5"
        checkpoint = ModelCheckpoint(filename, monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
        history = self.model.fit(X_train, y_train, epochs=self.epochs_to_train ,validation_data=(X_test, y_test),callbacks=[checkpoint])

    def start_train_simple(self, data_to_train, labels):
        X_train, X_test, y_train, y_test = train_test_split(data_to_train,labels, random_state=0, shuffle=True)
        self.train_model(X_train,X_test,y_train,y_test)
    
    
    def start_train_complex(self, data_to_train,labels, model_split_method = KFold(n_splits= 3, shuffle=True)):
        for train_index, test_index in model_split_method.split(data_to_train,labels):
            X_train, X_test = data_to_train[train_index], data_to_train[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            self.train_model(X_train,X_test,y_train,y_test)
    
    #did you know python doesn't have implicit multi parameter version functions?
    #I'm starting to regret my choice of language for this part of the project
    #well granted, there are alternatives like forcing a 
    def process(self, text, labels, glove_layer, vocab_size, use_weights = False):
        
        if self.is_training == True:
            return
        
        self.is_training = True
        self.vocab_size = vocab_size
        
        if use_weights == True:
            self.weights = glove_layer
            self.use_weights = True
        else:
            self.use_weights = False
        
        #don't do this, always train a new model
        #if is_new == False and model != None:
        #   self.model = load_model(model)  
        
        #self.model_init_meth_four(64,32,8)
        #self.model_init_meth_three(128,64,8)
        self.model_init_meth_four(64,32,8)
        self.start_train_complex(text, labels)
        self.is_training = False
    
        
        
        
            