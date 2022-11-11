import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #run only in CPU no CUDA

import PIL
import random
import pathlib
import datetime
import numpy as np
from os import listdir
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array 

def salvarLog(nomeArquivo, msg):
  f1 = open(nomeArquivo, 'a')
  f1.write(msg + "\n")
  print(msg)
  f1.close()

switch = {
   '1': 'LeNet',
   '2': 'KarpathyNet',
   '3': 'smallVGG',
   '4': 'VGG16',
   '5': 'VGG19',
   '6': 'ResNet50',
   '7': 'U-Net',
   '8': 'ResNet101',
   '9': 'ResNet152',
   '10': 'Inception V1',
   '11': 'Xception',
   }

tipoModelo = 11

## Dataset para a classificacao de imagens
#data_dir = "/home/pi/Deposito/Projetos/Pessoais/Artigos/ComparativoTexturas/datasets/Grapevine_Leaves_Image_Dataset"
#vetclasses = ['Ala_Idris', 'Buzgulu', 'Nazli', 'Ak', 'Dimnit']
#extOrigem = '.png'
#img_height = 200
#img_width = 200

## Dataset para a classificacao de texturas
data_dir = "/home/pi/Deposito/Projetos/Pessoais/Artigos/ComparativoTexturas/datasets/Kather_texture_2016_image_tiles_5000"
vetclasses = ['06_MUCOSA', '02_STROMA', '05_DEBRIS', '04_LYMPHO', '08_EMPTY', '07_ADIPOSE', '01_TUMOR', '03_COMPLEX']
extOrigem = '.jpg'
img_height = 150
img_width = 150

caminhoSaida = 'logs'
caminhoRedes = 'pesos'

caminhoQuebrado = data_dir.split('/')
nomeDataset = caminhoQuebrado[(len(caminhoQuebrado) - 1)]

nomeArquivoTreinado = ''
if(tipoModelo == 1):
  from tensorflow.keras.models import Sequential
  nomeArquivoTreinado = 'leNet.h5'
  
if(tipoModelo == 2):
  from tensorflow.keras.models import Sequential
  nomeArquivoTreinado = 'karpathyNet.h5'
  
if(tipoModelo == 3):
  from tensorflow.keras.models import Sequential
  nomeArquivoTreinado = 'smallVGG.h5'
  
if(tipoModelo == 4):
  from tensorflow.keras.models import Sequential
  nomeArquivoTreinado = 'vgg16.h5'
  
if(tipoModelo == 5):
  from tensorflow.keras.models import Sequential
  nomeArquivoTreinado = 'vgg19.h5'
  
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
    
if(tipoModelo == 7):
  import uNet
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.layers import Flatten  
  nomeArquivoTreinado = 'uNet.h5'
  
if(tipoModelo == 8):
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.layers import Flatten
  from tensorflow.keras.applications.resnet import ResNet101
  nomeArquivoTreinado = 'ResNet101.h5'
  
if(tipoModelo == 9):
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.layers import Flatten
  from tensorflow.keras.applications.resnet import ResNet152
  nomeArquivoTreinado = 'resNet152.h5'
    
if(tipoModelo == 10):
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.layers import Flatten
  from tensorflow.keras.applications.inception_v3 import InceptionV3
  nomeArquivoTreinado = 'inception.h5'
      
if(tipoModelo == 11):
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.layers import Flatten
  from tensorflow.keras.applications.xception import Xception
  nomeArquivoTreinado = 'xception.h5'

nomeArquivoTreinado = caminhoRedes + '/' + nomeDataset + '/' + nomeArquivoTreinado

if(not(os.path.exists(caminhoSaida))):
  os.mkdir(caminhoSaida)

nomeRedeTreinadaTemp = nomeArquivoTreinado.split('/')
nomeRedeTreinada = nomeRedeTreinadaTemp[len(nomeRedeTreinadaTemp) - 1]
nomeRedeTreinada = nomeRedeTreinada.replace('.h5', '')
nomeRedeTreinadaTemp = nomeRedeTreinada.split('_')
nomeRedeTreinada = nomeRedeTreinadaTemp[len(nomeRedeTreinadaTemp) - 1]
nomeRedeTreinada = nomeRedeTreinada.upper()

nomeArquivoRelatorio = caminhoSaida + '/Testes__' + nomeDataset + '__' + nomeRedeTreinada + ' __' + str(datetime.datetime.now()) + '.txt'

mensagem = '\n\nModelo rede analisado: ' + str(tipoModelo)
salvarLog(nomeArquivoRelatorio, mensagem)

mensagem = 'Dataset utilizado: ' + data_dir
salvarLog(nomeArquivoRelatorio, mensagem)

model = None

mensagem = 'Nome do arquivo com a rede: ' + nomeArquivoTreinado
salvarLog(nomeArquivoRelatorio, mensagem)

model = load_model(nomeArquivoTreinado)
model.summary()

totalAcertosTodasClasses = 0
totalTestesTodasClasses = 0
nomesClasses = [f.path for f in os.scandir(data_dir) if f.is_dir()]
for nomeClasse in nomesClasses:

  nomeClasseTratado = str(nomeClasse).replace(data_dir, '').replace('/', '')
  mensagem = 'Classe em testes: ' + nomeClasseTratado
  salvarLog(nomeArquivoRelatorio, mensagem)

  acertos = 0
  caminhos = [os.path.join(nomeClasse, nome) for nome in os.listdir(nomeClasse)]
  arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
  jpgs = [arq for arq in arquivos if arq.lower().endswith(extOrigem)]

  amostrasPorClasse = 100
  for cobaia in (range(0, (amostrasPorClasse - 1))):
  
    indice = random.randint(0, (len(jpgs) - 1))

    mensagem = '\nCobaia selecionada: ' + str(jpgs[indice]).replace(data_dir, '').replace('/', '')
    salvarLog(nomeArquivoRelatorio, mensagem)
    
    imagemCobaia = load_img(jpgs[indice])
    imagemCobaia = imagemCobaia.resize((img_height, img_width))
    imagemCobaia = img_to_array(imagemCobaia)    
    
    dataset = []
    dataset.append(imagemCobaia)
        
    dataInicial = datetime.datetime.now()
    resultadoPrevisto = model.predict(np.array(dataset))
    dataFinal = datetime.datetime.now()
  
    tempoTreinamento = dataFinal - dataInicial
    mensagem = 'Tempo de analise: ' + str(tempoTreinamento)
    salvarLog(nomeArquivoRelatorio, mensagem)
    
    proba = resultadoPrevisto[0]
    idxs = np.argsort(proba)[::-1][:2]
    idxs = str(idxs)
    idxs = idxs.replace('[', '')
    idxs = idxs.replace(']', '')
    idxs = idxs.split(' ')[0]
    idxs = int(idxs, base=10)
        
    mensagem = 'Resultado previsto: ' + str(idxs) + ' =) ' + vetclasses[idxs]
    salvarLog(nomeArquivoRelatorio, mensagem)
    
    totalTestesTodasClasses = totalTestesTodasClasses + 1
    
    nomeClassePrevista = (vetclasses[idxs]).lower()
    nomeClasseTratado = nomeClasseTratado.lower()
    if(nomeClasseTratado == nomeClassePrevista):
      acertos = acertos + 1
      totalAcertosTodasClasses = totalAcertosTodasClasses + 1


  mensagem = 'Total de acertos: ' + str(acertos)
  salvarLog(nomeArquivoRelatorio, mensagem)
  
  mensagem = 'Percentual de acertos: ' + str(((100 * acertos) / amostrasPorClasse))
  salvarLog(nomeArquivoRelatorio, mensagem)
  
mensagem = '\n\nTotal de acertos em todas as classes: ' + str(totalAcertosTodasClasses)
salvarLog(nomeArquivoRelatorio, mensagem)
  
mensagem = 'Percentual de acertos: ' + str( round(((100 * totalAcertosTodasClasses) / totalTestesTodasClasses), 2))
salvarLog(nomeArquivoRelatorio, mensagem)

print('\n\nBye...')
