import gensim
import os
import numpy as np
import nltk

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

physical_devices = tf.config.list_physical_devices('GPU')
# physical_devices = tf.test.gpu_device_name()
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


sentend= None
model = None
embedding_model = None
Emo_text = ['Anger', 'Sadness', 'Disgust', 'Surprise', 'Joy', 'Fear']

def Load_Model():
	global model, sentend, embedding_model
	model = load_model('HashTag_MODEL_TF2')
	sentend = np.ones((300,), dtype=np.float32)
	embedding_model = gensim.models.KeyedVectors.load('/home/tej/Documents/fmSpin/Ani_Working/Docker/Data_Resources/Word2Vec.mem',mmap='r')
	# model._make_predict_function()

def predict_emotion(sentence):
	global model, sentend, embedding_model

	Sentence_Tokenized = nltk.word_tokenize(sentence)

	sentence_vector = [embedding_model[word] for word in Sentence_Tokenized if word in embedding_model.index_to_key]

	if len(sentence_vector)>30:
		sentence_vector = sentence_vector[:30]

	if len(sentence_vector)<30:
		for i in range(30 - len(sentence_vector)):
			sentence_vector.append(sentend)

	vec_input = []
	vec_input.append(sentence_vector)
	x=np.array(vec_input)
	predictions = model.predict(x)
	total = np.sum(predictions[0])
	Anger_per = (predictions[0][0]/total)*100
	Sadness_per = (predictions[0][1]/total)*100
	Disgust_per = (predictions[0][2]/total)*100
	Surprise_per = (predictions[0][3]/total)*100
	Joy_per = (predictions[0][4]/total)*100
	Fear_per = (predictions[0][5]/total)*100

	return ({'ANGER':str(Anger_per), 'DISGUST':str(Disgust_per), 'FEAR':str(Fear_per), 'JOY':str(Joy_per), 'SADNESS':str(Sadness_per), 'SURPRISE':str(Surprise_per)})
