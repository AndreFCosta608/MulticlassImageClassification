import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #run only in CPU no CUDA

import PIL
import random
import pathlib
import datetime
import numpy as np
from PIL import Image
from os import listdir
import tensorflow as tf
from os.path import join
from os.path import isfile
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def salvarLog(nomeArquivo, msg):
  f1 = open(nomeArquivo, 'a')
  f1.write(msg + "\n")
  print(msg)
  f1.close()
  
switch = {
   '01':'LeNet',
   '02':'KarpathyNet',
   '03':'smallVGG',
   '04':'VGG16',
   '05':'VGG19',
   '06':'ResNet50',
   '07':'U-Net',
   '08':'ResNet101',
   '09':'ResNet152',
   '10':'Inception V1',
   '11':'Xception',
   }

tipoModelo = 1
batch_size = 1
epochs = 500 #- Quantidade elevada contando com a automacao de earlystop.

## Dataset para a classificacao de imagens
#data_dir = "/home/pi/projetos/ComparadorTexturas/datasets/Grapevine_Leaves_Image_Dataset"
##data_dir = "/home/pi/Deposito/Projetos/Pessoais/Artigos/ComparativoTexturas/datasets/Grapevine_Leaves_Image_Dataset"
#extOrigem = '.png'
#img_height = 200
#img_width = 200

## Dataset para a classificacao de texturas
data_dir = "/home/pi/projetos/ComparadorTexturas/datasets/Kather_texture_2016_image_tiles_5000"
extOrigem = '.jpg'
img_height = 150
img_width = 150

caminhoSaida = 'logs'
caminhoRedes = 'pesos'



caminhoQuebrado = data_dir.split('/')
nomeDataset = caminhoQuebrado[(len(caminhoQuebrado) - 1)]

if(not(os.path.exists(caminhoSaida))):
  os.mkdir(caminhoSaida)

nomeArquivoRelatorio = caminhoSaida +  '/Treinamento__' + nomeDataset  + '__' + str(datetime.datetime.now()) + '.txt'

mensagem = '\n\n->Modelo rede analisado: ' + str(tipoModelo)
salvarLog(nomeArquivoRelatorio, mensagem)

mensagem = '->Dataset utilizado: ' + data_dir
salvarLog(nomeArquivoRelatorio, mensagem)

dataset = []
indices = []
totalAmostrasClasse = []
nomesClasses = [f.path for f in os.scandir(data_dir) if f.is_dir()]
for nomeClasse in nomesClasses:
  nomeClasseTratado = str(nomeClasse).replace(data_dir, '').replace('/', '')
  indices.append(nomeClasseTratado)
  caminhos = [os.path.join(nomeClasse, nome) for nome in os.listdir(nomeClasse)]
  arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
  jpgs = [arq for arq in arquivos if arq.lower().endswith(extOrigem)]
  for jpg in jpgs:
    nomeImagem = str(jpg)
    nomeImagem = nomeImagem.replace(data_dir, '')
    print('Add ', nomeImagem)
    classeExiste = False
    for iten in totalAmostrasClasse:
      if (iten[0] == nomeClasseTratado):
        iten[1] = iten[1] + 1
        classeExiste = True
    if(classeExiste == False):
      totalAmostrasClasse.append([nomeClasseTratado, 1])    
    
    nrClasse = len(indices) - 1
    imagemCobaia = load_img(jpg)
    imagemCobaia = imagemCobaia.resize((img_height, img_width))
    imagemTemp = img_to_array(imagemCobaia)
    obj = {"classe": nrClasse, "img":imagemTemp}
    dataset.append(obj)
    
    imagemTemp = imagemCobaia.copy()
    imagemTemp = imagemTemp.transpose(Image.FLIP_LEFT_RIGHT)    
    imagemTemp = img_to_array(imagemTemp)
    obj = {"classe": nrClasse, "img":imagemTemp}
    dataset.append(obj)

random.shuffle(dataset)
random.shuffle(dataset)
random.shuffle(dataset)
random.shuffle(dataset)

labels = []
dsTemp = []
for iten in dataset:
  obj = iten
  valor = obj['img']
  classe = obj['classe']
  dsTemp.append(valor)
  labels.append(classe)

dataset = None

qtMaior = 0
qtMenor = 100000000000
for item in totalAmostrasClasse:
  if(item[1] > qtMaior):
    qtMaior = item[1]
  if(item[1] < qtMenor):
    qtMenor = item[1]
qtPercentual = qtMaior * 0.05
qtDiferencaPercentual = qtMaior - qtMenor
if(qtDiferencaPercentual > qtPercentual):
  print('Atencao!!!!.')
  print('O dataset se encontra desequilibrado.')
  print('A rede treinada ficara tendenciosa...')
  x = input('Deseja prosseguir assim mesmo?(s/n)')
  if((x != 's') and (x != 'S')):
    print('Ok. Saindo...')
    quit()

labels = np.array(labels)
dataset = np.array(dsTemp)

(trainX, testX, trainY, testY) = train_test_split(dataset, labels, test_size=0.2)

mensagem = '->Classes no dataset: ' + str(indices)
salvarLog(nomeArquivoRelatorio, mensagem)

AUTOTUNE = tf.data.AUTOTUNE

num_classes = len(indices)
mensagem = '->Numero de classes: ' + str(num_classes)
salvarLog(nomeArquivoRelatorio, mensagem)

model = None
nomeArquivoTreinado = ''

if(tipoModelo == 1):
  from tensorflow.keras.models import Sequential
  
  nomeArquivoTreinado = 'leNet.h5'

  LeNet = Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
  
    layers.Conv2D(20, (5, 5), padding="same"),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  
    layers.Conv2D(50, (5, 5), padding="same"),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  
    layers.Flatten(),
    layers.Dense(500),
    layers.Activation("relu"),
  
    layers.Dense(num_classes),
  
    layers.Activation("softmax")  
  ])  
  LeNet.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics='accuracy')  
  model = LeNet

if(tipoModelo == 2):
  from tensorflow.keras.models import Sequential
  
  nomeArquivoTreinado = 'karpathyNet.h5'

  KarpathyNet = Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
  
    layers.Conv2D(16, (5, 5), padding="same"),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Dropout(0.25),
  
    layers.Conv2D(32, (5, 5), padding="same"),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Dropout(0.25),
  
    layers.Conv2D(64, (5, 5), padding="same"),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Dropout(0.5),
  
    layers.Flatten(),
    layers.Dense(128),
    layers.Activation("relu"),
    layers.Dropout(0.5),
  
    layers.Dense(num_classes),
  
    layers.Activation("softmax")
  ])
  KarpathyNet.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  model = KarpathyNet

if(tipoModelo == 3):
  from tensorflow.keras.models import Sequential
  
  nomeArquivoTreinado = 'smallVGG.h5'
  
  chanDim = 1 #1/-1
  smallVGG = Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
        
    layers.Conv2D(32, (3, 3), padding="same"),
    layers.Activation("relu"),
    layers.BatchNormalization(axis=chanDim),
    layers.MaxPooling2D(pool_size=(3, 3)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), padding="same"),
    layers.Activation("relu"),
    layers.BatchNormalization(axis=chanDim),
    layers.Conv2D(64, (3, 3), padding="same"),
    layers.Activation("relu"),
    layers.BatchNormalization(axis=chanDim),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), padding="same"),
    layers.Activation("relu"),
    layers.BatchNormalization(axis=chanDim),
    layers.Conv2D(128, (3, 3), padding="same"),
    layers.Activation("relu"),
    layers.BatchNormalization(axis=chanDim),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
      
    layers.Flatten(),
    layers.Dense(1024),
    layers.Activation("relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(num_classes),
    layers.Activation("softmax")
  ])
  smallVGG.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  model = smallVGG

if(tipoModelo == 4):
  from tensorflow.keras.models import Sequential
  
  nomeArquivoTreinado = 'vgg16.h5'

  VGG16 = Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
  
    layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"),  
    layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"),  
    layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
    
    layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),  
    layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),  
    layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
    
    layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),  
    layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),  
    layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),  
    layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
    
    layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),  
    layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),  
    layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),  
    layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
    
    layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),  
    layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),  
    layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),  
    layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
    
    layers.Flatten(),
    layers.Dense(units=4096,activation="relu"),  
    layers.Dense(units=4096,activation="relu"),  
    layers.Dense(units=num_classes, activation="softmax")
  ])
  VGG16.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  model = VGG16

if(tipoModelo == 5):
  from tensorflow.keras.models import Sequential
  
  nomeArquivoTreinado = 'vgg19.h5'

  VGG19 = Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
  
    layers.Conv2D(64, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.Conv2D(64, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.MaxPooling2D(pool_size=(2,2), strides= (2,2)),
  
    layers.Conv2D(128, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.Conv2D(128, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.MaxPooling2D(pool_size=(2,2), strides= (2,2)),
  
    layers.Conv2D(256, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.Conv2D(256, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.Conv2D(256, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.Conv2D(256, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.MaxPooling2D(pool_size=(2,2), strides= (2,2)),
  
    layers.Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.MaxPooling2D(pool_size=(2,2), strides= (2,2)), 
    
    layers.Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu'),
    layers.MaxPooling2D(pool_size=(2,2), strides= (2,2)),
  
    layers.Flatten(),
    layers.Dense(4096, activation= 'relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation= 'relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation= 'softmax')
  ])
  VGG19.compile(optimizer= tf.keras.optimizers.Adam(0.003), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  model = VGG19

if(tipoModelo == 6):
  import resnet50
  from tensorflow.keras.models import Model
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.layers import Conv2D
  from tensorflow.keras.layers import Flatten
  from tensorflow.keras.layers import Activation
  from tensorflow.keras.layers import MaxPooling2D
  from tensorflow.keras.layers import ZeroPadding2D
  from tensorflow.keras.layers import AveragePooling2D
  from tensorflow.keras.layers import BatchNormalization
  from tensorflow.keras.initializers import glorot_uniform
    
  nomeArquivoTreinado = 'resNet50.h5'
  
  X_input = layers.Input(shape=(img_height, img_width, 3))
  
  X = ZeroPadding2D((3, 3))(X_input)
  
  X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((3, 3), strides=(2, 2))(X)
  
  X = resnet50.convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
  X = resnet50.identity_block(X, 3, [64, 64, 256], stage=2, block='b')
  X = resnet50.identity_block(X, 3, [64, 64, 256], stage=2, block='c')
  
  X = resnet50.convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
  X = resnet50.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
  X = resnet50.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
  X = resnet50.identity_block(X, 3, [128, 128, 512], stage=3, block='d')
  
  X = resnet50.convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
  X = resnet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
  X = resnet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
  X = resnet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
  X = resnet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
  X = resnet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
  
  X = resnet50.convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
  X = resnet50.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
  X = resnet50.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
  
  X = AveragePooling2D((2, 2), name='avg_pool')(X)
  
  X = Flatten()(X)
  X = Dense(num_classes, activation='softmax', name='fc' + str(num_classes), kernel_initializer = glorot_uniform(seed=0))(X)
  
  ResNet50 = Model(inputs = X_input, outputs = X, name='ResNet50')
  ResNet50.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  
  model = ResNet50
  
if(tipoModelo == 7):
  import uNet
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.layers import Dropout
  from tensorflow.keras.layers import Flatten  
  from tensorflow.keras.layers import BatchNormalization
    
  nomeArquivoTreinado = 'uNet.h5'
  
  unet = uNet.build_unet_model((img_height, img_width, 3))
  unet.summary()
  
  mensagem = '->Topologia da rede: ' + str(unet.summary())
  salvarLog(nomeArquivoRelatorio, mensagem)
  
  layer1 = Flatten()  
  layer2 = Dense(64, activation='relu')
  layer3 = Dense(64, activation='relu')
  layer4 = BatchNormalization()
  layer5 = Dropout(0.5)
  layer6 = Dense(num_classes, activation='softmax')
  
  model = Sequential([unet, layer1, layer2, layer3, layer4, layer5, layer6])
  model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  model.build(input_shape=(None, img_height, img_width, 3))
  

if(tipoModelo == 8):
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.layers import Dropout
  from tensorflow.keras.layers import Flatten
  from tensorflow.keras.layers import BatchNormalization
  from tensorflow.keras.applications.resnet import ResNet101
  
  nomeArquivoTreinado = 'ResNet101.h5'
  
  resNet101 = ResNet101(include_top=False, input_shape=(img_height, img_width, 3))
  resNet101.summary()
  
  mensagem = '->Topologia da rede: ' + str(resNet101.summary())
  salvarLog(nomeArquivoRelatorio, mensagem)
  
  layer1 = Flatten()
  layer2 = Dense(1024, activation='relu')
  layer3 = Dense(1024, activation='relu')
  layer4 = BatchNormalization()
  layer5 = Dropout(0.5)
  layer6 = Dense(num_classes, activation='softmax')
  model = Sequential([resNet101, layer1, layer2, layer3, layer4, layer5, layer6])
  
  model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

  model.build(input_shape=(None, img_height, img_width, 3))
    
if(tipoModelo == 9):
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.layers import Dropout
  from tensorflow.keras.layers import Flatten
  from tensorflow.keras.layers import BatchNormalization
  from tensorflow.keras.applications.resnet import ResNet152
  
  nomeArquivoTreinado = 'resNet152.h5'
  
  resNet152 = ResNet152(include_top=False, input_shape=(img_height, img_width, 3))
  resNet152.summary()
  
  mensagem = '->Topologia da rede: ' + str(resNet152.summary())
  salvarLog(nomeArquivoRelatorio, mensagem)
  
  layer1 = Flatten()
  layer2 = Dense(1024, activation='relu')
  layer3 = Dense(1024, activation='relu')
  layer4 = BatchNormalization()
  layer5 = Dropout(0.5)
  layer6 = Dense(num_classes, activation='softmax')
  model = Sequential([resNet152, layer1, layer2, layer3, layer4, layer5, layer6])
  
  model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  
  model.build(input_shape=(None, img_height, img_width, 3))


if(tipoModelo == 10):
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.layers import Dropout
  from tensorflow.keras.layers import Flatten
  from tensorflow.keras.layers import BatchNormalization
  from tensorflow.keras.applications.inception_v3 import InceptionV3
  
  nomeArquivoTreinado = 'inception.h5'
    
  incepion = InceptionV3(include_top=False, input_shape=(img_height, img_width, 3))
  incepion.summary()
  
  mensagem = '->Topologia da rede: ' + str(incepion.summary())
  salvarLog(nomeArquivoRelatorio, mensagem)
  
  layer1 = Flatten()
  layer2 = Dense(1024, activation='relu')
  layer3 = Dense(1024, activation='relu')
  layer4 = BatchNormalization()
  layer5 = Dropout(0.5)
  layer6 = Dense(num_classes, activation='softmax')
  model = Sequential([incepion, layer1, layer2, layer3, layer4, layer5, layer6])
  
  model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

  model.build(input_shape=(None, img_height, img_width, 3))
    
if(tipoModelo == 11):
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.layers import Dropout
  from tensorflow.keras.layers import Flatten
  from tensorflow.keras.layers import BatchNormalization
  from tensorflow.keras.applications.xception import Xception
  
  nomeArquivoTreinado = 'xception.h5'
    
  xception = Xception(include_top=False, input_shape=(img_height, img_width, 3))
  xception.summary()
  
  mensagem = '->Topologia da rede: ' + str(xception.summary())
  salvarLog(nomeArquivoRelatorio, mensagem)
  
  layer1 = Flatten()
  layer2 = Dense(1024, activation='relu')
  layer3 = Dense(1024, activation='relu')
  layer4 = BatchNormalization()
  layer5 = Dropout(0.5)
  layer6 = Dense(num_classes, activation='softmax')
  model = Sequential([xception, layer1, layer2, layer3, layer4, layer5, layer6])
  
  model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

  model.build(input_shape=(None, img_height, img_width, 3))


if(not(os.path.exists(caminhoRedes))):
  os.mkdir(caminhoRedes)
  
if(not(os.path.exists(caminhoRedes + '/' + nomeDataset))):
  os.mkdir(caminhoRedes + '/' + nomeDataset)  
  
nomeArquivoTreinado = caminhoRedes + '/' + nomeDataset + '/' + nomeArquivoTreinado

mensagem = '->Topologia da rede: ' + str(model.summary())
salvarLog(nomeArquivoRelatorio, mensagem)

mensagem = '->Nome do arquivo salvo com a rede: ' + nomeArquivoTreinado
salvarLog(nomeArquivoRelatorio, mensagem)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(nomeArquivoTreinado, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='accuracy', min_delta=0, patience=10, verbose=1, mode='auto')
dataInicial = datetime.datetime.now()
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, callbacks=[checkpoint, early])
dataFinal = datetime.datetime.now()

print('Treinamento concluido...')

tempoTreinamento = dataFinal - dataInicial
mensagem = '->Tempo de treinamento: ' + str(tempoTreinamento)
salvarLog(nomeArquivoRelatorio, mensagem)

mensagem = '->Quantidade de epocas originais: ' + str(epochs)
salvarLog(nomeArquivoRelatorio, mensagem)

epocasEfetivas = len(history.history['loss'])
mensagem = '->Quantidade de epocas usadas pelo EarlyStopping: ' + str(epocasEfetivas)
salvarLog(nomeArquivoRelatorio, mensagem)

if(epochs > epocasEfetivas):
  epochs = epocasEfetivas

acc = history.history['accuracy']

mensagem = '->history.history[\'accuracy\']: ' + str(acc)
salvarLog(nomeArquivoRelatorio, mensagem)

val_acc = history.history['val_accuracy']
mensagem = '->history.history[\'val_acc\']: ' + str(val_acc)
salvarLog(nomeArquivoRelatorio, mensagem)

loss = history.history['loss']
mensagem = '->history.history[\'loss\']: ' + str(loss)
salvarLog(nomeArquivoRelatorio, mensagem)

val_loss = history.history['val_loss']
mensagem = '->history.history[\'val_loss\']: ' + str(val_loss)
salvarLog(nomeArquivoRelatorio, mensagem)

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

nomeRede = nomeArquivoTreinado.split('/')
nomeRede = nomeRede[len(nomeRede) - 1]
nomeRede = nomeRede.replace('.h5', '')
nomeRede = nomeRede.split('_')
nomeRede = nomeRede[len(nomeRede) - 1]
nomeRede = nomeRede.upper()
nomeArquivoGrafico = caminhoSaida +  '/Treinamento__' + nomeDataset + '___' + nomeRede + '.png'

plt.savefig(nomeArquivoGrafico)
plt.show()

print('Bye...')