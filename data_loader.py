import nltk
import random
from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp_
import numpy as np
import re
from nltk.corpus import brown
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from nltk.tokenize import RegexpTokenizer

def load_dataset():
    data = brown.tagged_sents(tagset='universal')
    return data

def get_sent_and_tags(data):
    sentence_dataset = []
    tag_dataset = []
    for sentence in data:
        sent_list = []
        tag_list = []
        for tagged_word in sentence:
            sent_list.append(tagged_word[0])
            tag_list.append(tagged_word[1])
        sentence_dataset.append(sent_list)
        tag_dataset.append(tag_list)

    # ~~~~~~~~  In case you do not want to train your model on unknown tag 'X'. ~~~~~~~~ #

    # to_be_removed_indexes = []
    # for i,tag_list in enumerate(tag_dataset):
    #     for tag in tag_list:
    #         if tag=='X':
    #             to_be_removed_indexes.append(i)
    #             break

    # sentence_dataset = [sent for i,sent in enumerate(sentence_dataset) if i not in to_be_removed_indexes]
    # tag_dataset = [sent for i,sent in enumerate(tag_dataset) if i not in to_be_removed_indexes]

    return (sentence_dataset,tag_dataset)

def get_embedder():
    glove_6b50d = nlp_.embedding.create('glove', source='glove.6B.50d')
    emb = nlp_.Vocab(nlp_.data.Counter(glove_6b50d.idx_to_token))
    emb.set_embedding(glove_6b50d)
    return emb

def get_embedding_matrix(emb):
    embedding_matrix = []
    for word in emb.idx_to_token:
        embedding_matrix.append(emb.embedding[word].asnumpy())
    embedding_matrix = np.asarray(embedding_matrix)

    return embedding_matrix

def trim_data(sentence_dataset,tag_dataset,max_length):
    to_be_removed_indexes = []
    for i,sentence in enumerate(sentence_dataset):
        if(len(sentence) > max_length): to_be_removed_indexes.append(i)

    sentence_dataset = [sent for i,sent in enumerate(sentence_dataset) if i not in to_be_removed_indexes]
    tag_dataset = [sent for i,sent in enumerate(tag_dataset) if i not in to_be_removed_indexes]

    return (sentence_dataset,tag_dataset)

def add_eos(sentence_dataset,tag_dataset):
    for sentence in  sentence_dataset:
        sentence.append('<eos>')

    for tag_list in tag_dataset:
        tag_list.append('EOS')

    # MAX_LENGTH changes on adding <eos>
    MAX_LENGTH = np.max(np.asarray([len(s) for s in sentence_dataset]))

    return (sentence_dataset,tag_dataset,MAX_LENGTH)

def pad_data(sentence_dataset,tag_dataset,MAX_LENGTH):
    padded_sent_dataset = pad_sequences(sequences=sentence_dataset,
                                        dtype=object,
                                        maxlen=MAX_LENGTH,
                                        padding='post',
                                        value='<pad>')
    padded_tag_dataset =  pad_sequences(sequences=tag_dataset,
                                        dtype=object,
                                        maxlen=MAX_LENGTH,
                                        padding='post',
                                        value='PAD')

    return (padded_sent_dataset,padded_tag_dataset)


def get_tag_dicts(padded_tag_dataset):
    set_of_tags = set()
    for tag_list in padded_tag_dataset:
        for tag in tag_list:
            set_of_tags.add(tag)

    tag2id = {tag: i for i,tag in enumerate(sorted(set_of_tags))}
    id2tag = {i: tag for i,tag in enumerate(sorted(set_of_tags))}

    return (tag2id, id2tag)

def get_feeding_sequences(padded_sent_dataset,padded_tag_dataset,emb,MAX_LENGTH):

    X_data = padded_sent_dataset.copy()
    for sentence in X_data:
        for i in range(MAX_LENGTH):
            sentence[i] = emb[sentence[i].lower()]
    X_data = np.asarray(X_data)

    Y_data = [] 
    tag2id,_ = get_tag_dicts(padded_tag_dataset)
    for tag_list in padded_tag_dataset:
        temp = []
        for tag in tag_list:
            temp.append(tag2id[tag])
        Y_data.append(np.asarray(temp))
    Y_data = np.asarray(Y_data)
    Y_data = to_categorical(Y_data, num_classes=len(tag2id))

    return X_data,Y_data

def preprocess_for_model_testing(sentence,MAX_LENGTH,emb,number_format=1):
    tokenizer = RegexpTokenizer(pattern=r'\w+|\$[\d\.]+|\S+')
    if(isinstance(sentence, str)):
        sentence = tokenizer.tokenize(sentence)
    sentence.append('<eos>')
    sequence = [sentence]
    sequence = pad_sequences(sequences=sequence,dtype=object,maxlen=MAX_LENGTH,padding='post',value='<pad>')

    if(number_format==0): return sequence

    for sentence in sequence:
        for i in range(MAX_LENGTH):
            sentence[i] = emb[sentence[i].lower()]

    return sequence

def get_features(sentence):
    '''
    In features we will have:
    1) Boolean value for: Is token the first index?
    2) Boolean value for: Is token the last index?
    3) Boolean value for: Is token in CAPS?
    4) Boolean value for: Is first character in CAPS?
    5) Boolean value for: Is first character a number?
    6) Boolean value for: Is token a symbol?
    '''
    sentence_features = []
    for index,word in enumerate(sentence):
        feature_list = []
        if(word=='<pad>' or word=='<eos>'): feature_list += [0,0,0,0,0,0]
        else:
            feature_list.append(int(index==0))
            feature_list.append(int(index==len(sentence)-1))
            feature_list.append(int(word.isalpha() and word.upper()==word))
            feature_list.append(int(word.isalpha() and word[0].upper()==word[0]))
            feature_list.append(int(word[0].isnumeric()))
            feature_list.append(int(word[0] in ['!','@','#','$','%','^','&','*','(',')',';',
                                                '"',"'",':','{','}','[',']','-','_','+','=',
                                                '~','``','`','.',',','/','?','<','>','|']))
        sentence_features.append(feature_list)
    
    return (np.asarray(sentence_features)).reshape(-1,6)

def get_feature_map(data,max_length):
    s_dataset,_,max_length = preprocess_data(data,number_format=0,max_length=max_length)
    feature_map = []
    for sentence in s_dataset:
        feature_map.append(get_features(sentence))

    feature_map = np.asarray(feature_map)
    return feature_map

def preprocess_data(data,number_format = 1,max_length = 50):
    emb = get_embedder()
    s_data, t_data = get_sent_and_tags(data)
    s_data, t_data = trim_data(s_data,t_data, max_length)
    s_data, t_data, MAX_SEQ_LEN = add_eos(s_data,t_data)
    s_data, t_data = pad_data(s_data,t_data, MAX_SEQ_LEN)

    if(number_format==0):
        return s_data,t_data,MAX_SEQ_LEN

    X,Y = get_feeding_sequences(s_data,t_data,emb, MAX_SEQ_LEN)
    return s_data,t_data,X,Y,MAX_SEQ_LEN
# Code provided by Shreyansh Chordia