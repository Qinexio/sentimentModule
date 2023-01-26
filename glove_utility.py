import numpy as np

from tqdm import tqdm


class GloVeUtility:
    def __init__(self, tokenizer, max_len = 100):
        self.tokenizer = tokenizer
        self.max_len = max_len
        print ("GloveUtility __instance created")
        
    def create_dictionary(self, gloveFileName = "glove.6B.100d.txt"):
        vocab_size = len(self.tokenizer.word_index)+1 #in case of other surprises we always leave an open space
        
        dictionary_vector = {}
        #while accuaracy does take a hit please keep the dimension numbers at the moderate for faster training times and less space occupied
        #we are not training rockets here
        gloveFileNameNew = 'data/glove/'+gloveFileName
        glove_file = open(gloveFileNameNew, encoding='utf-8')
        
        for line in tqdm(glove_file):
            value = line.split(' ')
            word = value[0] 
            coef = np.array(value[1:],dtype = 'float32')
            dictionary_vector[word] = coef
        
        glove_file.close()
        #Second value is dependent on the glove word vector (that 50d before .txt for example) on the np.zeros func. , we use the maxlen
        #that d defines the dimensions it is, they are usually coincidental, I had issues with matrix shapes before because of that
        #We pre-initailize the matrix, after that it gets loaded in, if a word is missing it will just be a lot of 0's sadly that won't be trained
        dictionary_matrix = np.zeros((vocab_size,self.max_len)) #this used to be double paranthesis
        
        #This essentially loads the data based on the words the tokenizer met throughout this process that are unique
        for word,i in tqdm(self.tokenizer.word_index.items()):
            dictionary_value = dictionary_vector.get(word)
            if dictionary_value is not None:
                dictionary_matrix[i] = dictionary_value
        
        return dictionary_matrix
    