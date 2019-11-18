"""
        TO RUN: python testing.py my_heartbeat.wav
	NOTE: use python3

	To ignore the warnings
		python -W ignore testing.py my_heartbeat.wav
"""

import os
import sys
import librosa
import keras
import numpy as np
from keras.models import load_model

def extract_features(audio_path, offset):
	y, sr = librosa.load(audio_path, offset=offset, duration=3)
	S = librosa.feature.melspectrogram(
	y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
	mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
	# mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
	return mfccs


if __name__ == "__main__":
	# load model
	model = load_model("trained_heartbeat_classifier.h5")
	# File to be classified
	# classify_file = "my_heartbeat.wav"
	classify_file = sys.argv[1]
	x_test = []
	x_test.append(extract_features(classify_file, 0.5))
	x_test = np.asarray(x_test)
	x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
	pred = model.predict(x_test, verbose=1)
	# print(pred)

	pred_class = model.predict_classes(x_test)
	if pred_class[0]:
		print("\nNormal heartbeat")
		print("confidence:", pred[0][1])
	else:
		print("\nAbnormal heartbeat")
		print("confidence:", pred[0][0])