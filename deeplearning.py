# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:26:02 2018
@author: Weber (github.com/andweber92 & linkedin.com/in/andersonweber)
"""

# Bibliotecas necessarias
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# importamos os dados que ja estão no keras
fashion_mnist = keras.datasets.fashion_mnist
# Carregamos os dados:
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['Camiseta', 'Calça', 'Pullover', 'Vestido', 'Casaco', 
               'Sandalha', 'Camisa', 'Sneaker', 'Bolsa', 'Ankle boot']
#Explorando os dados
print(train_images.shape)
# (60000, 28, 28)
print(len(train_images))
# 60000
print(train_labels)
# array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
print(test_images.shape)
# (10000, 28, 28)
print(len(test_images))
# 10000
print(test_images)
# array([9, 8, 0, ..., 3, 2, 5], dtype=uint8)
#Pre-processamento dos dados
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
#Dividir por 255 para obter padrão nos dados entre 0 e 1
train_images = train_images / 255.0
test_images = test_images / 255.0
#Plotar as 25 primeiras imagens com legendas
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#Modelo neural
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #achata um array 2d em 1d
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), #diz como modelo é atualizado
              loss='sparse_categorical_crossentropy', #função custo que mede precisao no treino
              metrics=['accuracy']) #acuracia
#Vamos treinar o modelo!
model.fit(train_images, train_labels, epochs=5) #1 mísera linha 
#Acurácia
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Acurácia do teste: {test_acc}')
#Modelo treinado, agora vamos fazer predições!
predictions = model.predict(test_images)
print(f'Predição: {np.argmax(predictions[0])} - {class_names[np.argmax(predictions[0])]}')
print(f'Gabarito: {test_labels[0]} - {class_names[test_labels[0]]}')
#Plotar graficos para cada um dos 10 conjuntos
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  i = 0
#plotamos dados de teste do indice 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
#plotamos dados de teste do indice 12
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
#Plotamos agora os testes e predições onde azul= certo e vermelho = incorreto
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
#Vamos tentar com 1 imagem no dataset:
img = test_images[0]
print(img.shape)
#28x28 pixels de 0 a 255 em cada elemento
#Expand dims pois o tf.keras utiliza analise em lotes e temos apenas 1 imagem
img = (np.expand_dims(img,0))
print(img.shape)
#Agora, predição:
predictions_single = model.predict(img)
print(predictions_single)
#Plotamos uma lista de listas para análise, entretanto, queremos analisar apenas 1, certo?
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
#É preciso mencionar que queremos analisar apenas 1 imagem
np.argmax(predictions_single[0])
#Retorna 9 = Ankle boot