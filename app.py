from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
import librosa
import librosa.display


app = Flask(__name__)

dic = {0 : 'Positif', 1 : 'Negatif'}

model = load_model('model.h5')
# Parameters
input_size = (150,150)

#define input shape
channel = (3,)
input_shape = input_size + channel

#define labels
labels = ['Positif','Negatif']

model.make_predict_function()

def preprocess(img,input_size):
    nimg = img.convert('RGB').resize(input_size, resample= 0)
    img_arr = (np.array(nimg))/255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)

from tensorflow.keras.models import load_model

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(150,150))
	x = preprocess(i,input_size)
	x = reshape([x])
	y = model.predict(x)
	return labels[np.argmax(y)], round(np.max(y*100))

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Web masih dalam pengembangan"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		img.save(img_path)

		f = img_path
		y, _ = librosa.load(f, sr=44100)
		signal = y[0:int(0.9 * _)] 
    	#get mel-spectogram
		S = librosa.feature.melspectrogram(signal)
		S_DB = librosa.power_to_db(S, ref=np.max)
		plt.figure(figsize=(5, 4))
		librosa.display.specshow(S_DB)
		plt.colorbar(format='%+2.0f dB')
		plt.title('Mel spectrogram')
		plt.tight_layout()
		imge_path = "image/" + img.filename + ".png"
		plt.savefig(imge_path)

		p = predict_label(imge_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
