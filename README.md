# Part Of Speech Tagging Using Sequential Neural Networks

In this repository I have taught my models, the task of part of speech tagging. I have been able to achieve an accuracy of 96%. 3 different models have been trained.

Since part of speech tagging of a word does not simply depend on the word, but its tagging has a dependence on the nearby words as well, hence solving this problem requires us to have information, even of the past words when we are tagging a particular word. This kind of problem cannot be solved effectively by a Simple Neural Network.

Such problems can be solved by training less number of parameters and hence in a more robust way using Sequential Neural Networks. A sequential network is the one that uses sequential layers like the simple recurrent neural layer, or the LSTM layer or the GRU layer. A sequential network is so called because it can effectively understand the relation between two phases of the same sequence virtually inputted at different time frames.

In this repository I have architectured 3 simple Sequential Models:

1) model1- Uni-directional LSTM Layer Model (93.20% ACC)

2) model2- Bidirectional LSTM Layer Model (96% ACC)

3) model3- Multiple Input Bidirectional LSTM Layer Model (inputted with certain features along with the sequence) (96% ACC)

Detailed overview can be done by refering to the colab notebook first.

If you directly want to see the models giving outputs, then:
  
1. clone the repository
  
2. run model1_in_action.py / model2_in_action.py / model3_in_action.py

I have used a lot of libraries in this project. You must install them before trying to run the code.

1) nltk

2) numpy

3) tensorflow

4) keras

5) sklearn

6) matplotlib

7) regex

8) mxnet

9) gluonnlp

10) h5py
