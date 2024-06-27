# PARTE 1
# Permite el procesamiento del lenguaje natural
import nltk
nltk.download("punkt")
# Minimizador de palabras
from nltk.stem.lancaster import LancasterStemmer
# Instanciamos el minimizador
stemmer = LancasterStemmer()
# Permite trabajar con arreglos y realizar manipulaciones, conversiones, etc...
import numpy
# Herramienta de deep learning
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tflearn
import tensorflow
from tensorflow.python.framework import ops

# Permite manipular contenido JSON.
import json
# Permite crear números aleatorios
import random
# Permite guardar los modelos de entrenamiento (Mejora la velocidad, ya que no hay que entrenar desde 0 varias veces)
import pickle

import requests
import os
import matplotlib as mltp
import dload

print("primera parte Correcta!!")

dload.git_clone("https://github.com/orlandojosebv/data_bot.git")  # Estamos clonando el repositorio donde está el JSON

dir_path = os.path.dirname(os.path.realpath(__file__))  # Abrimos el archivo en el directorio
dir_path = dir_path.replace("\\", "//")  # Reemplazamos caracteres
with open(dir_path + '/data_bot/data_bot-main/data.json', 'r') as file:
    database = json.load(file)

words = []
all_words = []
tags = []
aux = []
auxA = []
auxB = []
training = []  # Servirá para el entrenamiento del bot.
exit = []

try:
    with open("Entrenamiento/brain.pickle", "rb") as pickleBrain:  # Abre el archivo usando la librería pickle
        all_words, tags, training, exit = pickle.load(pickleBrain)  # El archivo pickle se guardará en la ruta anteriormente escrita
except:
    for intent in database["intents"]:
        for pattern in intent["patterns"]:
            # Separar la frase en palabras, evitando tener un texto inmenso
            auxWords = nltk.word_tokenize(pattern)
            # Guardamos las palabras
            auxA.append(auxWords)
            auxB.append(auxWords)
            # Guardamos los tags, para saber a quién pertenece
            aux.append(intent["tag"])
    # Símbolos a ignorar
    ignore_words = ['?', '.', '!', ',', '¿', "'", '"', '$', '-', ':', '_', '&', '/', '(', ')', '=', '#', '*']  # Ignoramos los signos de puntuación
    for w in auxB:
        if w not in ignore_words:
            words.append(w)
    import itertools
    words = sorted(set(list(itertools.chain.from_iterable(words))))
    tags = sorted(set(aux))

    # Convertir todo a minúscula, para estandarizar todo
    all_words = [stemmer.stem(w.lower()) for w in words]

    all_words = sorted(list(set(all_words)))
    # Ordenamos los tags
    tags = sorted(tags)
    training = []
    exit = []
    # Creamos una salida falsa
    null_exit = [0 for _ in range(len(tags))]

    for i, document in enumerate(auxA):
        bucket = []
        # Minúscula y quitar signos
        auxWords = [stemmer.stem(w.lower()) for w in document if w != "?"]
        for w in all_words:
            if w in auxWords:
                bucket.append(1)
            else:
                bucket.append(0)
        exit_row = null_exit[:]
        exit_row[tags.index(aux[i])] = 1
        training.append(bucket)
        exit.append(exit_row)
    training = numpy.array(training)  # Se convierte en una matriz
    exit = numpy.array(exit)

    # Creamos el archivo pickle, donde se va a guardar todos los datos.
    with open("Entrenamiento/brain.pickle", "wb") as pickleBrain:
        pickle.dump((all_words, tags, training, exit), pickleBrain)

# Generamos la red neuronal
tensorflow.compat.v1.reset_default_graph()
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)
net = tflearn.input_data(shape=[None, len(training[0])])
# Redes intermedias
net = tflearn.fully_connected(net, 100, activation="Relu")
net = tflearn.fully_connected(net, 50)
net = tflearn.dropout(net, 0.5)
# Neurona de salida
net = tflearn.fully_connected(net, len(exit[0]), activation="softmax")
# Red completada
# Ahora hacemos la regresión de la red neuronal
net = tflearn.regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')
# Generamos el modelo
model = tflearn.DNN(net)
if os.path.isfile(dir_path + "/Entrenamiento/model.tflearn.index"):
    model.load(dir_path + "/Entrenamiento/model.tflearn")
else:
    model.fit(training, exit, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=1000)
    model.save("Entrenamiento/model.tflearn")

# Parte 4
def get_chatbot_response(text):
    bucket = [0 for _ in range(len(all_words))]
    processed_sentence = nltk.word_tokenize(text)
    processed_sentence = [stemmer.stem(word.lower()) for word in processed_sentence]
    
    for word in processed_sentence:
        for i, w in enumerate(all_words):
            if w == word:
                bucket[i] = 1
    
    results = model.predict([numpy.array(bucket)])
    index_results = numpy.argmax(results)
    tag = tags[index_results]
    
    for tagAux in database["intents"]:
        if tagAux['tag'] == tag:
            responses = tagAux['responses']
            return random.choice(responses)
    
    return "Lo siento, no entendí eso. ¿Puedes reformular la pregunta?"

if __name__ == "__main__":
    print("Habla Conmigo!!")
    while True:
        texto = input()
        if texto.lower() == "duerme":
            print("HA DISO UN GUSTO, VUELVE PRONTO AMIGO!")
            break
        else:
            response = get_chatbot_response(texto)
            print(response)
