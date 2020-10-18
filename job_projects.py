import numpy as np
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import joblib
from nltk.tokenize import RegexpTokenizer
import nltk
from sklearn.model_selection import cross_val_score
from imblearn.combine import SMOTETomek
import string
from sklearn.model_selection import RandomizedSearchCV


os.chdir(os.path.dirname(__file__))


# nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english')
default_tokenizer=RegexpTokenizer(r"\w+")

# pip install transformers
# pip install tensorflow==2.1.0
# pip install simpletransformers
# pip install tokenizers==0.8.1.rc1
# export CUDA_HOME=/usr/local/cuda-10.1
# git clone https://github.com/NVIDIA/apex
# %cd apex
# pip install -v --no-cache-dir ./

import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# -*- coding: utf-8 -*-
import pickle

# Load dict
type_dict = np.load('type.npy', allow_pickle=True).item()
experi_dict = np.load('experi.npy', allow_pickle=True).item()
educa_dict = np.load('educa.npy', allow_pickle=True).item()
industry_dict = np.load('industry.npy', allow_pickle=True).item()
function_dict = np.load('function.npy', allow_pickle=True).item()

embeddings_index = np.load('glove_dict.npy', allow_pickle=True).item()

# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+|url\S+|URL\S+')
    return url.sub(r'',str(text))


def remove_emoji(text):
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', str(text))


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',str(text))


def remove_punctuation(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

# general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def final_preprocess(text):
    text = text.replace('\\r', ' ')
    text = text.replace('\\"', ' ')
    text = text.replace('\\n', ' ')
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = ' '.join(e for e in text.split() if e.lower() not in stopwords)
    text = text.lower()
    ps = PorterStemmer()
    text = ps.stem(text)
    return text

def final_preprocess_b(text):
    text = text.replace('\\r', ' ')
    text = text.replace('\\"', ' ')
    text = text.replace('\\n', ' ')
    return text


def softmax(x):
    e_x = np.exp(x - np.max(x))
    e_y = e_x / e_x.sum()
    return e_y

##=======================================

def fliter(data):
    text_lr = data.loc[0,'text']
    str_list = text_lr.split()

    a = 0
    b = len(str_list)

    for i in str_list:
        if i in embeddings_index.keys():
            a = a+1
    c = a/b*100

    if c < 75:
        d = False
    else:
        d = True
    return d


def lr(data):
    #=============================lr====================================#
    #=========================lr=====data===================================#
    # data['text'] = data[['title', 'department','company_profile','description',
    #                  'requirements','benefits']].apply(lambda x: ' '.join(x), axis = 1)
    # data.drop(['location','title','salary_range' ,'department','salary_range',
    #        'company_profile','description','requirements','benefits'], axis=1, inplace=True)
    # data.columns
    # data_columns = data.columns.tolist()
    #import pickle

    # Load dict
    # type_dict = np.load('type.npy', allow_pickle=True).item()
    # experi_dict = np.load('experi.npy', allow_pickle=True).item()
    # educa_dict = np.load('educa.npy', allow_pickle=True).item()
    # industry_dict = np.load('industry.npy', allow_pickle=True).item()
    # function_dict = np.load('function.npy', allow_pickle=True).item()

    #from sklearn.preprocessing import LabelEncoder
    label_columns = ['employment_type','required_experience', 'required_education', 'industry', 'function']
    label_dict = [type_dict, experi_dict, educa_dict, industry_dict, function_dict]

    for i,k in zip(label_columns,label_dict):
        if data.loc[0,i] in k.keys():
            data.loc[0,i] = k[data.loc[0,i]]
        else:
            data.loc[0,i] = 0
    # print(data)
    
    ## Defining the utility functions


    
    data['text']=data['text'].map(remove_URL)
    data['text']=data['text'].map(remove_emoji)
    data['text']=data['text'].map(remove_html)
    data['text']=data['text'].map(remove_punctuation)
    data['text']=data['text'].map(decontracted)
    data['text']=data['text'].map(final_preprocess)

    # embeddings_index = np.load('glove_dict.npy', allow_pickle=True).item()
    # f = open('glove.840B.300d.txt')
    # for line in f:
    #     values = line.split(' ')
    #     word = values[0] ## The first entry is the word
    #     coefs = np.asarray(values[1:], dtype='float32') ## These are the vectors representing the embedding for the word
    #     embeddings_index[word] = coefs
    # f.close()
    glove_words =  set(embeddings_index.keys())

    '''
    Below is a uliity function that takes sentenes as a input and return the vector representation of the same
    Method adopted is similar to average word2vec. Where i am summing up all the vector representation of the words from the glove and 
    then taking the average by dividing with the number of words involved
    '''

    converted_data = []

    for i in range(0, data.shape[0]):
            vector = np.zeros(300) # as word vectors are of zero length
            cnt_words =0; # num of words with a valid vector in the sentence
            for word in data['text'][i].split():
                if word in glove_words:
                    vector += embeddings_index[word]
                    cnt_words += 1
            if cnt_words != 0:
                vector /= cnt_words
            converted_data.append(vector)


    _1 = pd.DataFrame(converted_data)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    data[['required_education', 'required_experience', 'employment_type', 'industry', 'function']] = StandardScaler().fit_transform(data[['required_education', 'required_experience', 'employment_type', 'industry', 'function']])

    data.drop(["text"], axis=1, inplace=True)
    main_data = pd.concat([_1,data], axis=1)


    #=======================lr=====model=====================================#
    from sklearn.linear_model import LogisticRegression
    import pickle
    # Load from file
    pkl_filename = "pickle_lr_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
        

    y_predict = pickle_model.predict_proba(main_data)
    #np.savetxt("proba_lr.txt",y_predict)
    possi_lr = y_predict
    possi_lr=possi_lr.tolist()
    possi_lr = possi_lr[0]
    return possi_lr
    

def be(df):
     #===========================bert======================================#
    #=======================bert==========data================================#   
    # df['text'] = df['title'] + " " + df['department'] + \
    #             " " + df['company_profile'] + " " + \
    #             df['description'] + " " + \
    #             df['requirements'] + " " +\
    #             df['benefits'] + " " +\
    #             df['function'] + " " \

    # delete_list=['title','department',
    #         'company_profile','description','requirements',
    #         'benefits','employment_type','required_experience','required_education',
    #         'industry','function']
    # for val in delete_list:
    #     del df[val]
#    import spacy, re
    #Data Cleanup
#    df['text']=df['text'].str.replace('\n','')
#    df['text']=df['text'].str.replace('\r','')
#    df['text']=df['text'].str.replace('\t','')
#
#    #This removes unwanted texts
#    df['text'] = df['text'].apply(lambda x: re.sub(r'[0-9]','',str(x)))
#    df['text'] = df['text'].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]',' ',str(x)))
#
#    #Converting all upper case to lower case
#    df['text']= df['text'].apply(lambda s:s.lower() if type(s) == str else s)
#
#
#    #Remove un necessary white space
#    df['text']=df['text'].str.replace('  ',' ')
#
#    #Remove Stop words
#    nlp=spacy.load("en_core_web_sm")
#    df['text'] =df['text'].apply(lambda x: ' '.join([word for word in x.split() if nlp.vocab[word].is_stop==False ]))
    df['text']=df['text'].map(remove_URL)
    df['text']=df['text'].map(remove_emoji)
    df['text']=df['text'].map(remove_html)
    df['text']=df['text'].map(final_preprocess_b)

    #===========================bert======model================================#
    # """Compute softmax values for each sets of scores in x.""" #


    from simpletransformers.classification import ClassificationModel
    model = ClassificationModel('bert', './bert/', num_labels=2, args={'fp16': False,'overwrite_output_dir': True,'output_dir':'bert_classifier_model',"train_batch_size": 64, "save_steps": 10000, "save_model_every_epoch":False,'num_train_epochs': 1}, use_cuda=False)

    df = df.values.tolist()
    df = df[0]
    result, model_outputs = model.predict(df)

    import numpy as np
    preds = [np.argmax(tuple(m)) for m in model_outputs]
    possibility = [tuple(k) for k in model_outputs]

    possi_b = []
    for m in possibility:
        mm = softmax(m)
        possi_b.append(mm)
    # print(type(possi_be))
    # print(possi_be[0])

    possi_b = possi_b[0]
    possi_b=possi_b.tolist()

    # print(possi_b)
    #possi_b.reverse()
    # print(possi_b)
    return possi_b

def ga_job(possi_lr,possi_b):
    #====================================ga=ensembel============================# 
    #====Best solution :  [[[1.17620871 1.18895564 0.62855357 1.2274774 ]]]=====#
    gene = [1.17620871, 1.18895564, 0.62855357, 1.2274774 ]
    possi = np.append(possi_b,possi_lr)
    equation_inputs = possi
    # print(type(equation_inputs))
    # print(equation_inputs)
    k = 0
    product = np.zeros((1,equation_inputs.shape[0]), dtype = float, order = 'C') #create an empty array
    # print(product)
    while k<4:
        prod = equation_inputs[k]*gene[k]
        product[0][k] = prod
        #product = np.insert(product,k,values=prod,axis=0)
        #print(prod)
        k=k+1
    #product = np.delete(product, 4, 0)  #4*3324
    # print("1",product)
    product = product.T
    # pjob = np.zeros(0,1),dtype = float,order = 'C')

    l_p = product[0]+product[2]
    r_p = product[1]+product[-1]
    pjob = np.append(l_p,r_p)
    # print(pjob)

    job_last = np.argmax(pjob)

    return(job_last) 
