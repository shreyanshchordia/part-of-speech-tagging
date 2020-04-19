from keras.models import load_model
import data_loader as dl
import model_utils as M

embedder = dl.get_embedder()
MAX_LENGTH = 50
data = dl.load_dataset()
_, padded_tag_dataset,MAX_LENGTH  = dl.preprocess_data(data,number_format=0,max_length=MAX_LENGTH)
tag2id,id2tag = dl.get_tag_dicts(padded_tag_dataset)

model = load_model('pre_trained_models/model2.h5',custom_objects={'ignore_accuracy': M.ignore_class_accuracy(9)})

while(1):
    print("\nType a sentence!")
    string = input()
    print(M.predict(model,string,MAX_LENGTH,id2tag,embedder))

# Code provided by Shreyansh Chordia