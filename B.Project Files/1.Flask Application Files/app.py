from flask import Flask,render_template,request
import cv2
from keras.models import load_model
import numpy as np
import pickle
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense,Flatten,Input,Convolution2D,Dropout,LSTM,TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential,Model
from keras.utils import to_categorical
from keras.preprocessing import image,sequence
from keras import models
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from keras.applications import ResNet50
resnet=ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
print("model loaded")

with open('features.pkl','rb') as f:

	vocab=pickle.load(f)

inv_vocab={v:k for k,v in vocab.items()}


output_dims=128
MAX_LEN=36
vocab_size=8912
num_classes=vocab_size+1




model=models.load_model("proj_model.h5")


app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1

@app.route('/')

def index():

    return render_template('index.html')

@app.route('/after',methods=['GET', 'POST'])
def after():

    global model,vocab,inv_vocab,resnet


    file=request.files['file2']

    file.save('static/file.jpg')

    img=cv2.imread('static/file.jpg')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(224,224,))
    img=np.reshape(img,(1,224,224,3))
    

    features=resnet.predict(img).reshape(1,2048)

    text_in=['startofseq']
    final= ''
    
    count=0
    while(tqdm(count <25)):

        count=count+1

        encoded=[]
        for i in text_in:

            encoded.append(vocab[i])

        paded=pad_sequences([encoded],maxlen=MAX_LEN,padding='post',truncating='post').reshape(1,MAX_LEN)

        sampled_index=np.argmax(model.predict([features,paded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word=='.':

        	final=final+'.'
        	break

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)

    return render_template('predict.html',final=final)

if __name__ == "__main__":
    app.run(debug=True)
