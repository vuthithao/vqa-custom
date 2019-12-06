from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
print("done")

import service

from gensim import *

from pickle import load
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model


try:
    from PIL import Image
except:
    from pil import Image
import cv2
from io import BytesIO
import base64

from flask import Flask, jsonify
from flask import request
import json
from gevent.pywsgi import WSGIServer
import time

app = Flask(__name__)



# Load model visual or non visual

unique_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
               'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
               'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'SP']

vocab_size = len(unique_tags)

model_visual = Sequential()
model_visual.add(Embedding(vocab_size, vocab_size))
model_visual.add(LSTM(output_dim=100, activation='sigmoid', inner_activation='hard_sigmoid'))
#model.add(Dropout(0.5))
model_visual.add(Dense(1))
model_visual.add(Activation('sigmoid'))

model_visual.compile(optimizer='rmsprop',\
              loss='binary_crossentropy')
model_visual.load_weights('model/epoch9.h5')


# Load model True, False premise

w2v= models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)

loadWeights = 'model/qc_avgw2v.h5'
model_premise=Sequential()
#model.add(Dense(30,input_dim=len(vocab),activation='relu'))
model_premise.add(Dense(200,input_dim=600,activation='relu'))
model_premise.add(Dense(150,activation='relu'))
model_premise.add(Dense(80,activation='relu'))
model_premise.add(Dense(1,activation='sigmoid'))
model_premise.compile(loss='binary_crossentropy',optimizer='adadelta')
model_premise.load_weights(loadWeights)

# Load image caption model


# extract features from each photo in the directory
def extract_features(filename):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # load the photo

    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
#     print(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature

# generate a description for an image
def generate_desc(model, tokenizer, photo, index_word, max_length, beam_size=5):

  captions = [['startseq', 0.0]]
  # seed the generation process
  in_text = 'startseq'
  # iterate over the whole length of the sequence
  for i in range(max_length):
    all_caps = []
    # expand each current candidate
    for cap in captions:
      sentence, score = cap
      # if final word is 'end' token, just add the current caption
      if sentence.split()[-1] == 'endseq':
        all_caps.append(cap)
        continue
      # integer encode input sequence
      sequence = tokenizer.texts_to_sequences([sentence])[0]
      # pad input
      sequence = pad_sequences([sequence], maxlen=max_length)
      # predict next words
      y_pred = model.predict([photo,sequence], verbose=0)[0]
      # convert probability to integer
      yhats = np.argsort(y_pred)[-beam_size:]

      for j in yhats:
        # map integer to word
        word = index_word.get(j)
        # stop if we cannot map the word
        if word is None:
          continue
        # Add word to caption, and generate log prob
        caption = [sentence + ' ' + word, score + np.log(y_pred[j])]
        all_caps.append(caption)

    # order all candidates by score
    ordered = sorted(all_caps, key=lambda tup:tup[1], reverse=True)
    captions = ordered[:beam_size]

  return captions

# load the tokenizer
tokenizer = load(open('model/tokenizer.pkl', 'rb'))
index_word = load(open('model/index_word.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34

filename = 'model/model_weight.h5'
model_caption = load_model(filename)

print("loaded model ...")

def extract_caption(imgpath, model, tokenizer, index_word, max_length):
    photo = extract_features(imgpath)
    # generate description
    captions = generate_desc(model, tokenizer, photo, index_word, max_length)
    caption = captions[0][0].split()[1:-1]
    caption = ' '.join(caption)
    return caption

def readb64(base64_string, rgb=True):
    """
    Đọc ảnh từ dạng base64 -> numpy array\n

    Input
    -------
    **base64_string**: Ảnh ở dạng base64\n
    **rgb**: True nếu ảnh là dạng RBG (đủ 3 channel), False nếu là ảnh xám (1 channel)

    Output:
    -------
    Ảnh dạng numpy array
    """
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    if rgb:
        return cv2.cvtColor(np.array(pimg), cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_BGR2GRAY)



def pred(model_visual, model_premise, model_caption, tokenizer, index_word, max_length):
    if request.method == "POST":
        dataDict = json.loads(request.data.decode('utf-8'))
        img = dataDict.get("img", None)
        question = dataDict.get("question", None)

    start = time.time()
    img = readb64(img, rgb=True)
    # question = "What is this?"
    impath = "tmp/img.jpg"
    cv2.imwrite(impath, img)

    if service.visualornon(question, model_visual, unique_tags) == 0:
        result = "Nonvisual"
    else:
        caption = extract_caption(impath, model_caption, tokenizer, index_word, max_length)
        if service.premise(question, caption, model_premise, w2v) == 0:
            result = "False-premise"
        else:
            result = "True-premise"

    end = time.time() - start
    response = jsonify({"result": result,\
                        "time": end, "status_code": 200})
    response.status_code = 200
    response.status = 'OK'
    return response, 200

def pred2(model_visual, model_premise):
    if request.method == "POST":
        dataDict = json.loads(request.data.decode('utf-8'))
        img = dataDict.get("img", None)
        question = dataDict.get("question", None)
        caption = dataDict.get("caption", None)

    start = time.time()
    img = readb64(img, rgb=True)
    # question = "What is this?"
    impath = "tmp/img.jpg"
    cv2.imwrite(impath, img)

    if service.visualornon(question, model_visual, unique_tags) == 0:
        result = "Nonvisual"
    else:
        # caption = extract_caption(impath, model_caption, tokenizer, index_word, max_length)
        if service.premise(question, caption, model_premise, w2v) == 0:
            result = "False-premise"
        else:
            result = "True-premise"

    end = time.time() - start
    response = jsonify({"result": result,\
                        "time": end, "status_code": 200})
    response.status_code = 200
    response.status = 'OK'
    return response, 200

# pred(model_visual,model_premise , model_caption, tokenizer, index_word, max_length)


@app.route('/st/v1', methods=['POST'])
def pred_():
    return pred(model_visual,model_premise , model_caption, tokenizer, index_word, max_length)

@app.route('/st/v2', methods=['POST'])
def pred2_():
    return pred2(model_visual,model_premise)

if __name__ == "__main__":
    http_server = WSGIServer(('', 4444), app)
    http_server.serve_forever()

