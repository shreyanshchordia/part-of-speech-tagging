import data_loader as dl
import model_utils as M 
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed,Embedding, Activation,Flatten,Bidirectional,Input,concatenate
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

MAX_LENGTH = 50

data = dl.load_dataset()
embedder = dl.get_embedder()
padded_sent_dataset, padded_tag_dataset, X, Y, MAX_LENGTH = dl.preprocess_data(data)
tag2id,id2tag = dl.get_tag_dicts(padded_tag_dataset)
embedding_matrix = dl.get_embedding_matrix(embedder)
feature_map = dl.get_feature_map(data,MAX_LENGTH)

# Building inputs to the model

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
(train_feature,test_feature) = feature_map[:X_train.shape[0]],feature_map[X_train.shape[0]:]

# Model3 architecture

feature_input = Input(shape=(MAX_LENGTH,feature_map.shape[2]),name='features')
sequence_input = Input(shape=(MAX_LENGTH,),name='sequence')

A = Embedding(input_dim=embedding_matrix.shape[0],output_dim=embedding_matrix.shape[1],
                    weights=[embedding_matrix],trainable=False,input_length=MAX_LENGTH,name='embedding')(sequence_input)
A = Model(inputs=sequence_input, outputs=A)

M = concatenate([A.output,feature_input])
M = Bidirectional(LSTM(64, return_sequences=True),name='bi_lstm')(M)
M = TimeDistributed(Dense(len(tag2id),activation='softmax',name='dense_1'),name='timedist')(M)
model3 = Model(inputs=[A.input, feature_input], outputs=M)

model3.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy',M.ignore_class_accuracy(9)])

model3.summary()
plot_model(model3)

# Training
history3 = model3.fit([X_train,train_feature],Y_train,epochs=20,batch_size=128,validation_split=0.2,verbose=1)
M.plot_model_history(model3,history3)

# Evaluation
model3.evaluate([X_test,test_feature],Y_test)

# Code provided by Shreyansh Chordia