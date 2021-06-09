#!/usr/bin/python
# -*- coding: utf-8 -*-
from tensorflow.keras.models import load_model
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template
import pickle
import random

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.reshape(img,(1,299,299,3))
    return img, image_path

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output


image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
infile = open("tokenizer_CBIR.pkl",'rb')
tokenizer = pickle.load(infile)
infile.close()
model = load_model("model-Sergio.hdf5")
app = Flask(__name__)
cors = CORS(app)
print("ready")

@app.route('/')
@cross_origin()
def home():
    return render_template('./home.html')
@app.route('/sergio',methods=['POST'])
@cross_origin()
def uploadLabel():
    isthisFile=request.files.get('fileupload')
    file_path = "./Images/"+isthisFile.filename
    print(file_path)
    isthisFile.save(file_path)
    img, _ = load_image(file_path)
    val = 0
    batch_features = image_features_extract_model([img])
    batch_feature = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3]))
    F = batch_feature
    C = np.zeros((1,25,1))
    end = tokenizer.word_index['<end>']
    r = 2
    for i in range(25):
        val = -1
        x = model.predict([F,C])
        pred = np.argmax(x)
        if random.random() < 0.3 and i>=3 and r>2 and pred!=end:
            pred = np.argsort(np.max(x, axis=0))[val-1]
            val = val - 1
            r = r - 1
        if pred == 2:
            pred = np.argsort(np.max(x, axis=0))[val-1]
        C[0][i][0] = pred
        if int(pred) == end:
            break
    output = ""
    for i in C[0]:
        val = tokenizer.index_word[i[0]]
        if i[0] == 1:
            break
        if val == "<start>":
            continue
        if val == "s":
            val = "\b\'s"
        output += val+" "
    output+"\b?"
    return output
if __name__ == '__main__':
    app.run()