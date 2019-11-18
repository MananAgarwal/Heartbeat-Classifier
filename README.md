# Heart Anomaly Detection by Analysing Stethoscope sounds using Deep Learning

In this work, we take stethoscope sounds (and also heartbeat sounds recorded using the microphone of a mobile phone) as input and apply deep learning to the task of automated cardiac auscultation, i.e. recognizing abnormalities in heart sounds. We describe a novel algorithm which first transforms the one-dimensional time-series inputs into a two-dimensional time-frequency Melspectrograms. It then trains a 4-layer CNN model on the MFCC (Mel-Frequency Cepstral Coefficients) obtained from the Melspectrograms. The trained network automatically distinguish between normal and abnormal heartbeat sound inputs. We did not use any other time sequence based Neural Networks such as RNNs since the temporal behavior of the heartbeat was repeated within the window of observation and different sequential patterns were not needed to be learnt.

Our goal is to provide a reliable, fast and low-cost system that can be used by untrained frontline health workers or anyone with internet access, to help determine whether an individual should be referred for expert diagnosis, particularly in areas where access to clinicians and medical care is limited. This will also help in early diagnosis of CVDs which will drastically decrease the potential risk factors of these deaths.

## To classify a heartbeat sound
The heartbeat audio file must be in .wav format

1. Download the repository
2. Open the terminal/command prompt and cd to the downloaded repository
3. Run the python script "testing.py"
        
        """ python testing.py heartbeat-to-classify.wav"""
        NOTE: Use Python3
        
4. The predicted class and the confidence will be displayed

Download the dataset from [here](http://www.peterjbentley.com/heartchallenge/index.html).
