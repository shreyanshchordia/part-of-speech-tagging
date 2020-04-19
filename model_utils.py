import data_loader as dl
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    return ignore_accuracy
        
################################################################################

def predict(model,sentence,MAX_LENGTH,id2tag,emb):
    sentence = dl.preprocess_for_model_testing(sentence,MAX_LENGTH,emb)
    pred = model.predict(sentence)
    pred = pred[0]

    output = []
    for tag in pred:
        id = np.argmax(tag)
        if(id==9): break
        output.append(id2tag[id])

    return output 

################################################################################

def plot_model_history(model,history):
    # summarize history for accuracy
    plt.figure(figsize=(10,8))
    plt.plot(history.history['ignore_accuracy'])
    plt.plot(history.history['val_ignore_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure(figsize=(10,8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
################################################################################

# ~~~~~~~~~~~~~~~~~~ Following function is for model3 ~~~~~~~~~~~~~~~~~~~~~~ #

def predict_v2(model,sentence,MAX_LENGTH,emb,id2tag):
    padded_sentence = dl.preprocess_for_model_testing(sentence,MAX_LENGTH,emb,number_format=0)
    f_input = dl.get_features(padded_sentence[0])
    f_input = f_input.reshape(-1,f_input.shape[0],f_input.shape[1])
    sentence = dl.preprocess_for_model_testing(sentence,MAX_LENGTH,emb)
    pred = model.predict([sentence,f_input])
    pred = pred[0]

    output = []
    for tag in pred:
        id = np.argmax(tag)
        if(id==9): break
        output.append(id2tag[id])

    return output

################################################################################
# Code provided by Shreyansh Chordia