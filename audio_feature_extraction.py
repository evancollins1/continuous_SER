# Extract 34 acoustic LLDs from IEMOCAP

from pyAudioAnalysis import audioBasicIO 
from pyAudioAnalysis import audioFeatureExtraction
import glob
import os
from keras.preprocessing import sequence
import numpy as np

# .wav files from IEMOCAP dataset saved locally
# change this directory of interest with audio dataset
files = glob.glob(os.path.join('//Users/Evan/Desktop/Click/IEMOCAP_full_release/Session?/sentences/wav/*/', '*.wav'))

files.sort()

feat = []

# Remove dead space in audio
# (Every 1/10 second compares to threshold)

#def envelope(y, sr, threshold):
    #mask = []
    #y_abs = pd.Series(y).apply(np.abs)
    #y_mean = y_abs.rolling(window = int(sr/10), min_periods = 1, center = True).mean()
    #for mean in y_mean:
        #if mean > threshold:
            #mask.append(True)
        #else:
            #mask.append(False)
    #return np.array(y[mask])

#def clean_files(files_list):
    
    #count = 0

    #for file in files_list:
        #y, sr = librosa.load(file)
        #y = envelope(y, sr, 0.0005)
        #save_file = 'clean/' + file
        
        #if not os.path.exists(os.path.dirname(save_file)):
            #try:
                #os.makedirs(os.path.dirname(save_file))
            #except OSError as exc: # Guard against race condition
                #if exc.errno != errno.EEXIST:
                    #raise
        
        #with open(save_file, 'w') as new_file:
            #sf.write(save_file, y, sr)
            #new_file.close()
            
        #count += 1
        #if count % 100 == 0:
            #print('Successfully cleaned and saved 100 files')
    
    #print("Finished")
   
 # Provide input as list of audio file directories
#clean_files(files)

for f in files:
    print("Process..., ", f)
    # if using new version of pyAudioAnalysis
    #[Fs, x] = audioBasicIO.read_audio_file(f)
    # if using version 0.2.5 of pyAudioAnalysis
    [Fs, x] = audioBasicIO.readAudioFile(f)
    
    # Extract audio features
    # 25ms window length; 10ms hop size
    # if using new version of pyAudioAnalysis (includes deltas!)
    #F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.025*Fs, 0.010*Fs)
    # if using version 0.2.5 of pyAudioAnalysis
    F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.025*Fs, 
                                                            0.010*Fs)
    feat.append(F.transpose())

feat_final = sequence.pad_sequences(feat, dtype='float64')
np.save('/Users/Evan/Desktop/Click/feat_34.npy', feat_final)

# Acoustic LLDs of 34 features extracted
f_names

# Extract mean and std HSFs from LLDs of 34 features

import numpy as np
import os
feat_final = np.load('/Users/Evan/Desktop/Click/feat_34.npy')

feat_final_float = feat_final.astype(float)

feat_final_float[feat_final_float==0] =np.nan

# Calculate mean and std
mean = np.nanmean(feat_final_float, axis=1)
std = np.nanstd(feat_final_float, axis=1)

# Output HSFs
feat_hsf = np.hstack([mean, std])
# Final dimensions of concatenated feature vectors of all audio
# Amend dimensions as necessary
feat_hst = feat_hsf.reshape(10039, 1, 68)
np.save('/Users/Evan/Desktop/Click/feat_hsf.npy', feat_hsf)
