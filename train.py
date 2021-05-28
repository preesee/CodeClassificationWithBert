#! -*- coding:utf-8 -*-
import datetime
import itertools
import os
import pickle

import keras as krs
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint
from keras_bert.bert import *
from pycparser import c_parser
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizerFast  # BertTokenizer, BertModel, BertForSequenceClassification,

from tree import trans_to_sequences

TRAIN_CSV = '/data/code_classification_data_for_Ccode.csv'

config_path='/data/uncased_L-12_H-768_A-12/bert_config.json'
config_path=os.path.join(os.getcwd(), *config_path.split('/'))

checkpoint_path= '/data/uncased_L-12_H-768_A-12/bert_model.ckpt'
checkpoint_path=os.path.join(os.getcwd(), *checkpoint_path.split('/'))
parser = c_parser.CParser()
# Load training and test set
train_csv = os.path.join(os.getcwd(), *TRAIN_CSV.split('/'))
train_df = pd.read_csv(train_csv)

max_lenth = 511
seq_len=max_lenth
max_seq_length=max_lenth
errorrows=[]
astrows=0
ast512rows=0
model_name = "google/bert_uncased_L-4_H-512_A-8"
token_dict = get_base_dict()  # A dict that contains some special tokens

tokenizer = BertTokenizerFast.from_pretrained(model_name)#.from_pretrained(model_name)
#keras_tokenizer=Tokenizer()
for index, row in train_df.iterrows():
    try:
        ast = parser.parse(row['code'])
        code = trans_to_sequences(ast)
        tokens_for_sents = code
        if len(code) > max_lenth:
            errorrows.append(index)
            ast512rows = ast512rows+1
            continue
        train_df.at[index, 'code'] = ' '.join(tokens_for_sents)
    except Exception as e:
        print(e)
        errorrows.append(index)
print(errorrows)
print('errorows: %d' % len(errorrows))
print('ast rows > 512: %d' % ast512rows)

train_df=train_df.drop(errorrows)
inputs=train_df["code"].tolist()
print(len(inputs))
print('input rows: %d' % len(inputs))

train_encodings = tokenizer(inputs, add_special_tokens=True, truncation=True, padding='max_length', max_length=max_seq_length-2)

validation_percent=0.2
# Split to train validation
validation_size = int(len(train_df) * validation_percent)
training_size = len(train_df) - validation_size

X = train_df["code"]
Y = train_df['label']


X_train, X_validation,X_att_train, X_att_validation, Y_train, Y_validation = train_test_split(
    train_encodings['input_ids'],train_encodings['token_type_ids'], Y, test_size=validation_size)

X_train, X_test,X_att_train, X_att_test, Y_train, Y_test = train_test_split(
    train_encodings['input_ids'],train_encodings['token_type_ids'], Y, test_size=validation_size)


# Split to dicts
X_train = {'data': X_train,
           'token_type_ids':X_att_train
           }
X_validation = {'data': X_validation,
                'token_type_ids':X_att_validation
                }
X_test = {'data': X_test,'token_type_ids':X_att_test }

Y_train = Y_train.values
Y_validation = Y_validation.values
Y_test = Y_test.values

for dataset, side in itertools.product([X_train, X_validation, X_test], ['data','token_type_ids']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=seq_len)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(max_seq_length,), dtype='int32')

x2_in = Input(shape=(None,))

x = bert_model([x1_in,x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(104, activation='softmax')(x)

model = Model([x1_in,x2_in], p)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-5),
    metrics=['accuracy']
)
model.summary()
batch_size=8
n_epoch=10
print('ast texts!')
model_history=model.fit([X_train['data'],X_train['token_type_ids']],  krs.utils.to_categorical(Y_train,104), batch_size=batch_size, epochs=n_epoch,
                            validation_data=([X_validation['data'],X_validation['token_type_ids']],   krs.utils.to_categorical(Y_validation,104)),
                            #callbacks=[checkpointer]
                            )
scores = model.evaluate([X_test['data'],X_test['token_type_ids']],  krs.utils.to_categorical(Y_test,104))
print("estimation report: %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


MODEL_SAVING_DIR = '/codeclone_data/result/models/save_model_his/'
model_saving_dir = os.path.join(os.getcwd(), *MODEL_SAVING_DIR.split('/'))

hist = model_history.history
modelname = model_saving_dir + str(datetime.datetime.now()) + 'ccode_bert_ast.model.pkl'

with open(modelname, "wb") as f:
    pickle.dump(hist, f)
print("model save done!")
