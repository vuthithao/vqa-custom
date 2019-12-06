import nltk
import numpy as np

def visualornon (question, model, unique_tags):
    tokens = nltk.word_tokenize(question)
    tags = []

    tagss = nltk.pos_tag(tokens)
    for i in range(len(tagss)):
        tag = tagss[i][1]
        if tag in unique_tags:
            index = unique_tags.index(tag)
            tags.append(index)

    ques = np.asarray(tags)
    pred = model.predict(np.asarray([ques]), batch_size=1, verbose=0)
    if pred[0][0] > 0.07:
        return 0
    else:
        return 1


def premise(question, caption, model, w2v):
    ques_cap_features = []
    words = question.split(' ')
    count = 0
    ques_features = []
    cap_features = []
    # print "Generating Question_features..."
    for word in words:
        if word in w2v:
            if word.lower() != 'is' or word.lower() != 'a' or word.lower() != 'the' or word.lower() != 'what' or word.lower() != 'that' or word.lower() != 'to' or word.lower() != 'who' or word.lower() != 'why':
                ques_features.append(w2v[word])
    ques_features = sum(ques_features) / float(len(ques_features))

    words = caption.split(' ')
    # count=15
    # print "generating Answer Features..."
    for word in words:
        if word in w2v:
            if word.lower() != 'is' or word.lower() != 'a' or word.lower() != 'the' or word.lower() != 'what' or word.lower() != 'that' or word.lower() != 'to' or word.lower() != 'who' or word.lower() != 'why':
                cap_features.append(w2v[word])
                # ques_cap_features[index][count]=invertvocab[word]
                # count+=1

    cap_features = sum(cap_features) / float(len(cap_features))

    ques_cap_features.append(np.concatenate((ques_features, cap_features), 0))
    ques_cap_features = np.asarray(ques_cap_features)

    pred = model.predict_proba(ques_cap_features)
    thresh = 0.15
    pred_labels = pred > thresh
    print(pred_labels)
    if pred_labels:
        return 1
    else:
        return 0
