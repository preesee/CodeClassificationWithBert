
__author__ = 'unique'

import datetime
# ! -*- coding:utf-8 -*-
import json
import os
import pickle
import random
from time import time

import javalang
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import keras as krs
import tensorflow.keras as krs
import tensorflow.keras.backend as K
from keras import backend as K
from keras.metrics import categorical_accuracy
from keras_bert import load_trained_model_from_checkpoint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizerFast

import utils as utl

TRAIN_PAIRS_JSON_FILE = "/data/TRAIN_txt_PAIRS_JSON_FILE.json"

MODEL_SAVING_DIR = '/data/save_model/'

MODEL_SAVING_DIR = '/data/result/models/'
FIG_SAVING_DIR = '/data/result/figures/'

config_path='/codeclone_data/uncased_L-12_H-768_A-12/bert_config.json'
config_path=os.path.join(os.getcwd(), *config_path.split('/'))

checkpoint_path= '/codeclone_data/uncased_L-12_H-768_A-12/bert_model.ckpt'
checkpoint_path=os.path.join(os.getcwd(), *checkpoint_path.split('/'))

def parse_program(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree


def trans_to_sequences(ast):
    sequence = []
    utl.get_sequence(ast, sequence)
    return sequence

# cfg_embedding_file = os.path.join(os.getcwd(), *CFG_EMBEDDING_FILE.split('/'))
# doc2vec_cfg = Doc2Vec.load(cfg_embedding_file)
# source_code_embedding_file = os.path.join(os.getcwd(), *SOURCE_CODE_EMBEDDING_FILE.split('/'))
# word2vec = KeyedVectors.load(source_code_embedding_file)
train_pairs_json_file = os.path.join(os.getcwd(), *TRAIN_PAIRS_JSON_FILE.split('/'))

label_data_convertor = {}
label_data_convertor['T4'] = 4
label_data_convertor['T2'] = 1
label_data_convertor['T1'] = 0
label_data_convertor['MT3'] = 2
label_data_convertor['ST3'] = 3
data_loaded = json.load(open(train_pairs_json_file))

datarowsT4 = [row for row in data_loaded if row['TYPE'] == 'T4']
datarowsT1 = [row for row in data_loaded if row['TYPE'] == 'T1']
datarowsT2 = [row for row in data_loaded if row['TYPE'] == 'T2']
datarowsMT3 = [row for row in data_loaded if row['TYPE'] == 'MT3']
datarowsST3 = [row for row in data_loaded if row['TYPE'] == 'ST3']

datafortrain = []
datafortrain += datarowsT4[:]
datafortrain += datarowsT1[:]
datafortrain += datarowsT2[:]
datafortrain += datarowsMT3[:]
datafortrain += datarowsST3[:]

# Load training and test set

# test_df = ''  # pd.read_csv(TEST_CSV)


samples_number = 40000000

validation_percent = 0.2
test_percent = 0.2
r_data_loaded = random.sample(datafortrain, len(datafortrain))
r_samples = r_data_loaded[:samples_number]
df = pd.DataFrame(r_samples)
train_df = df

# Prepare embedding
vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding

max_seq_length=510
errorrows=[]

model_name = "google/bert_uncased_L-4_H-512_A-8"
tokenizer = BertTokenizerFast.from_pretrained(model_name)#.from_pretrained(model_name)
code_clones_cols = ['code_clone1', 'code_clone2']

# Iterate over the questions only of both training and test datasets
for dataset in [train_df]:
    for index, row in dataset.iterrows():
        percent = int((index / len(dataset)) * 100)
        if percent % 2 == 0:
            print("loaded dataset rows:" + str(percent) + "%", end="\r")

        # Iterate through the text of both questions of the row
        for code_clone in code_clones_cols:
            try:
                ast = parse_program(row[code_clone])
                ast_seq = trans_to_sequences(ast)
                if len(ast_seq) > max_seq_length:
                    errorrows.append(index)
                    continue
                train_df.at[index, code_clone] = ' '.join(ast_seq)
            except Exception as e:
                #print(e)
                errorrows.append(index)
print(errorrows)
print('errorows: %d' % len(errorrows))

train_df=train_df.drop(errorrows)

inputs_l=train_df["code_clone1"].tolist()
inputs_r=train_df["code_clone2"].tolist()
train_encodings_l = tokenizer(inputs_l,add_special_tokens=True, truncation=True, padding='max_length', max_length=max_seq_length)

train_encodings_r = tokenizer(inputs_r,add_special_tokens=True, truncation=True, padding='max_length', max_length=max_seq_length)

# Split to train validation
validation_size = int(len(train_df) * validation_percent)
training_size = len(train_df) - validation_size

X = train_df[code_clones_cols]
Y = train_df.TYPE.map(lambda x: label_data_convertor[x])



Xl_train, Xl_validation, Xr_train, Xr_validation,  Y_train, Y_validation = train_test_split(
    train_encodings_l.encodings, train_encodings_r.encodings, Y, test_size=validation_size)

Xl_train, Xl_test, Xr_train, Xr_test, Y_train, Y_test = train_test_split(
    train_encodings_l.encodings, train_encodings_r.encodings, Y, test_size=validation_size)

# Split to dicts
X_train = {'left': Xl_train, 'right': Xr_train,}

X_validation = {'left': Xl_validation, 'right': Xr_validation,}

X_test = {'left': Xl_test, 'right': Xr_test,}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values
Y_test = Y_test.values

# Model variables
batch_size = 4
n_epoch = 15
normalized = False




# L1 distiance
def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

# L1 distance
def classification_softmax(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.sum(K.abs(left - right), axis=1, keepdims=True)


bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=max_seq_length)
for l in bert_model.layers:
    l.trainable = True
x11_in = Input(shape=(max_seq_length,), dtype='int32')

x12_in = Input(shape=(None,))

x21_in = Input(shape=(max_seq_length,), dtype='int32')

x22_in = Input(shape=(None,))

x1 = bert_model([x11_in,x12_in])
x2 = bert_model([x21_in,x22_in])


x1 = Lambda(lambda x: x[:])(x1)
x2 = Lambda(lambda x: x[:])(x2)

dist = Lambda(lambda x: classification_softmax(x[0], x[1]))([x1, x2])
dist=krs.layers.Reshape((768,))(dist)   # for keras <2.4 use a fixed reshape value 768 otherwise use  krs.layers.Reshape((-1,))(dist)
#classify = Dense(256, kernel_initializer='uniform')(dist)
classify = Dense(5, kernel_initializer='uniform',activation='softmax')(dist)
# #    classify = bn(classify)
# act = Activation('softmax')
# classify = act(classify)


model = Model([x11_in,x12_in,x21_in,x22_in], classify)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-5),
    metrics=[categorical_accuracy]   #'accuracy'
)
training_start_time = time()


# train_on_batch

l_inp= np.asarray([x.ids for x in X_train['left']])
l_inp_type_ids= np.asarray([x.type_ids for x in X_train['left']])
r_inp= np.asarray([x.ids for x in X_train['right']])
r_inp_type_ids= np.asarray([x.type_ids for x in X_train['right']])

l_vinp= np.asarray(([x.ids for x in X_validation['left']]))
l_vinp_type_ids= np.asarray([x.type_ids for x in X_validation['left']])
r_vinp= np.asarray([x.ids for x in X_validation['right']])
r_vinp_type_ids= np.asarray([x.type_ids for x in X_validation['right']])

l_tinp= np.asarray(([x.ids for x in X_test['left']]))
l_tinp_type_ids= np.asarray([x.type_ids for x in X_test['left']])
r_tinp= np.asarray([x.ids for x in X_test['right']])
r_tinp_type_ids= np.asarray([x.type_ids for x in X_test['right']])
malstm_trained = model.fit(
    [l_inp,l_inp_type_ids,r_inp,r_inp_type_ids],
    krs.utils.to_categorical(Y_train, 5),
    batch_size=batch_size, epochs=n_epoch,

    validation_data=(
    [l_vinp,l_vinp_type_ids,r_vinp,r_vinp_type_ids],
        krs.utils.to_categorical(Y_validation, 5)))

print("Training time finished.\n{} epochs in {}".format(n_epoch,
                                                        datetime.timedelta(seconds=time() - training_start_time)))
scores = model.evaluate( [l_tinp,l_tinp_type_ids,r_tinp,r_tinp_type_ids],
    krs.utils.to_categorical(Y_test, 5))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
for i in range(len(model.metrics_names)):
    print("%s: %.2f%%" % (model.metrics_names[i], scores[i]))

y_pred = model.predict( [l_tinp,l_tinp_type_ids,r_tinp,r_tinp_type_ids])
print(classification_report(Y_test, np.argmax(y_pred, 1)))
model.summary()


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(FIG_SAVING_DIR + str(datetime.datetime.now()) + 'loss_cfg_txt.png')
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(FIG_SAVING_DIR + str(datetime.datetime.now()) + 'acc_cfg_txt.png')
    plt.show()


plot_history(malstm_trained)

hist = malstm_trained.history
modelname = MODEL_SAVING_DIR + str(datetime.datetime.now()) + 'code_clone_bert_split_ast.model.pkl'

with open(modelname, "wb") as file_pi:
    pickle.dump(hist, file_pi)
print("model save done!")
