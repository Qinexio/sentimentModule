import re
import itertools
#from autocorrect import Speller
import nltk
from nltk import pos_tag
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from tensorflow.keras.preprocessing.sequence import pad_sequences
#why use two tokenizer means? because I'm lazy 
#and don't want to purge the tokenizer to learn words it will actually use later on
from tensorflow.keras.preprocessing.text import Tokenizer

class TextPreprocessUtility:
    __instance = None
    @staticmethod 
    def getInstance():
        """ Static access method. """
        if TextPreprocessUtility.__instance == None:
            TextPreprocessUtility(tokenizer = Tokenizer())
        return TextPreprocessUtility.__instance
    
    #these two should be imported from a loader component at start
    negation_cues = word_tokenize("aint cannot cant darent didnt doesnt dont hadnt hardly hasnt havent havnt isnt lack lacking lacks neither never no nobody none nor not nothing nowhere mightnt mustnt neednt oughtnt shant shouldnt wasnt without wouldnt")
    #negation_targets = word_tokenize("JJ JJR JJS RB RBR RBS VB VBD VBG VBN VBP VBZ IN") #IN is an odd case when a word like 'like' shows up (yes i'm aware of the irony)
    adjective_targets = word_tokenize("JJ JJR JJS")
    adverb_targets = word_tokenize ("RB RBR RBS IN")
    verb_targets = word_tokenize("VB VBD VBG VBN VBP VBZ")
    #dictonary for transforming a value into another one
    apostrophe_dictonary = {"'s":" is","n't":" not","'m":" am","'ll":" will", "'d":" would","'ve":" have","'re":" are"}
    def __init__(self, tokenizer, word_length_tresh = 2, max_lemma_length = 2,  max_len = 100):
        if TextPreprocessUtility.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            TextPreprocessUtility.__instance = self
            
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer_train = None
        #self.spell = Speller(lang='en')
        self.tokenizer = tokenizer
        self.stopword_list = [i for i in stopwords.words('english') if i not in self.negation_cues]
        self.max_len = max_len
        self.word_length_tresh = word_length_tresh
        self.max_lemma_length = max_lemma_length
        print ("TextPreprocessUtility __instance created")
    
    def reload(self,tokenizer):
        self.tokenizer = tokenizer
        
    def detokenize(self, text):
        return TreebankWordDetokenizer().detokenize(text)

    def tokenize(self, text):
        return word_tokenize(text)
     
    def lemmatize_text(self, tokens):
        new_tokens = []
        
        for word in tokens:
            lemma = self.lemmatizer.lemmatize(word, pos="v")
            # exclude if lenght of lemma is smaller than max_lemma_length
            if len(lemma) > self.max_lemma_length:
                new_tokens.append(lemma)
    
        return new_tokens
    
    def remove_stopwords(self, tokens):
        new_tokens = []
        for word in tokens:
            # remove words with length under treshold AND stopwords
            if len(word) > self.word_length_tresh and word not in self.stopword_list:
                new_tokens.append(word)
        
        return new_tokens
    
    def lower_sentence(self, tokens):
        new_tokens = []
        for word in tokens:
            new_tokens.append(word.lower())
        
        return new_tokens
    
    def transform_apos(self, text):
        #replace the contractions
        for key,value in self.apostrophe_dictonary.items():
            if key in text:
                text = text.replace(key,value)
        
        return text
    
    def clear_tags(self, data):
        to_return = data
        
        #Remove URLs using a regular expression
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        to_return = url_pattern.sub(r'', to_return)
        
        #Remove e-mails
        to_return = re.sub('\S*@\S*\s?', '', to_return)
        
        #Remove html tags
        html_pattern = re.compile(r'<.*?>')
        to_return = html_pattern.sub(r'', to_return)
        
        return to_return
    
    #this section is a tad slow, probably the spell checker
    def fix_text(self, data):
        to_return = data
        
        #Remove long strings of vocal letters (or even consonants in rare cases)
        to_return = ''.join(''.join(s)[:2] for _, s in itertools.groupby(to_return))
        
        #Spellcheck the data, this takes too long, hence commented unless you want to do it to a whole dataset in one go or a single sentence
        #to_return = self.spell(to_return)
        
        #Transform text by changing apostrophe forms to split words
        data = self.transform_apos(to_return)
        
        return to_return
    
    def remove_impurities(self, data):
        to_return = data
        #Remove new line characters
        to_return = re.sub('\s+', ' ', to_return)
    
        #Remove single quotes
        to_return = re.sub('\'', '', to_return)
        
        #Remove double quotes
        to_return = re.sub('\"', '', to_return)
        
        #Remove numbers
        to_return = re.sub(r'\d+', '', to_return)
        
        #Remove other weird characters
        to_return = re.sub(r'[^\w]', ' ', to_return)
        
        return to_return
    
    def simplify_text(self,data):
        to_return = data
        
        #Transform data into tokens for the next sequence
        to_return = self.tokenize(to_return)
        
        #Lowercase the sentence
        to_return = self.lower_sentence(to_return)
        
        #Lemantize the sentence
        to_return = self.lemmatize_text(to_return)
        
        #Remove stopwords (essentially they are not relevant)
        to_return = self.remove_stopwords(to_return)
        
        #Deal with negations
        to_return = self.handle_negation(to_return)
        
        #Merge the data back into text form
        to_return = self.detokenize(to_return)
        
        return to_return
    
    def misc_preprocessing(self,data):
        to_return = data
        
        #Remove a common word (probably should not do this)
        to_return = re.sub('feel','',to_return)
        
        #Transform not feel into a special word
        #This does not work well (not sure if it's being ignored completely by the tokenizer)
        #It might work better if we train the word matrix
        #Or just do the classic if you find not you shall negativize it to the punctuation, 
        #and therefore move this before any punctuation is removed
        #data = re.sub('not feel','not_feel',to_return)
        
        return to_return
    
    # sent_tokenize is one of instances of
    # PunktSentenceTokenizer from the nltk.tokenize.punkt module
    def get_sent_pos(self, data):
        #tag the sentence with what every word is like an adverb or an adjective
        return pos_tag(data)
    
    def has_negation(self, data):
        return any(item in data for item in self.negation_cues)
    
    def remove_pos(self, data):
        return [first[0] for first in data]
    
    def translate_pos(self, data_type):
        if data_type in self.adjective_targets:
            return "a"
        
        if data_type in self.adverb_targets:
            return "r"
        
        if data_type in self.verb_targets:
            return "v"
        
        return "n"
    
    #okay I'm going to be real but this method is not the greatest 
    #for example great is an adjective satelite with no antonyms
    def get_antonym(self, data, data_type):
        antonyms = []
        data_type_mod = self.translate_pos(data_type) #added checking for the type
        #the pos = data_type_mod doesn't work so well with training
        for synset in wordnet.synsets(data, pos = data_type_mod):
            #print(synset)
            for lemma in synset.lemmas():
                if lemma.antonyms():    #When antonyms are available, add them into the list
                    antonyms.append(lemma.antonyms()[0].name())
                    
        #believe it or not, an empty list returns false
        if not antonyms:
            return None
        else:
            return antonyms[0]
    
    def antonymize_sent(self, data):
        to_process = data
        negation_signal = False
        new_sent = []
        #print(to_process)
        for word in to_process:
            if word[0] in self.negation_cues:
                negation_signal = True
                continue
            
            if negation_signal is True:
                if word[1] in self.adjective_targets or self.adverb_targets or self.verb_targets:
                    toCheck = self.get_antonym(word[0],word[1])
                    if toCheck is not None:
                        new_sent.append(toCheck)
                        negation_signal = False
                        continue
            
            new_sent.append(word[0])
        return new_sent
    
    #not bullet proof as it might fail to get the target of an adverb or adjective if there is a verb in the way that isn't related to sentiments
    def handle_negation(self, data):
        to_process = data
        
        if self.has_negation(to_process):
            to_process = self.get_sent_pos(to_process)
            to_process = self.antonymize_sent(to_process)
    
        return to_process
        
    def depure_data(self, data):    
        data = self.clear_tags(data)
        data = self.fix_text(data)
        data = self.remove_impurities(data)
        data = self.simplify_text(data)
        #data = self.misc_preprocessing(data)
        print("Depure data call")
        return data
    
    #there may be a problem regarding stuff like '.....' used in speech when it comes to splitting, but it shouldn't show up if you just delete it during the split
    def split_text(self, text):
        delimiters = r'[.;!?,]'
        
        #believe it or not this is an issue, which bothers me to no end of why it is one
        #my sanity is slowly dwindling
        text_copy = re.sub('\"', '', text);
        
        split_list = re.split(delimiters, text_copy.strip())
        #this part deals with removing any sort of empty strings
        split_list = list(filter(None, split_list))
        #we still need to deal with sentences that are too short, maybe removing something like "hi" or a lone ','
        return split_list
    
    def process_data(self, data_list):
        temp = list()
        for data in data_list:
            temp.append(self.depure_data(data))
        
        return temp
    
    def process_text_to_numbers(self, data_list):
        temp = list()
        
        #do not do this, we don't want to modify the tokenizer unless we are training a new model
        #self.tokenizer.fit_on_texts(data_list)
        
        #the powers that be compel me to put data in a list, I don't know why, the prediction mechanism cries otherwise
        #okay you want the actual explanation? the padding needs to be done to the whole sentence, otherwise it will pad it word
        #by word, that's why this must be done to the data element
        #arguably it could've been done in process_data too but eh, I'd rather do it here for safety measures
        for data in data_list:
            temp.append(pad_sequences(self.tokenizer.texts_to_sequences([data]), maxlen= self.max_len))

        return temp
    
    def process_text_to_numbers_train(self, data_list):
        #we kinda need a new tokenizer, we still use the old one for predictions until the new one is loaded
        self.tokenizer_train = Tokenizer()
        self.tokenizer_train.fit_on_texts(data_list)
        sequences = self.tokenizer.texts_to_sequences(data_list)
        data_to_train = pad_sequences(sequences, maxlen=self.max_len)
        return data_to_train
    
