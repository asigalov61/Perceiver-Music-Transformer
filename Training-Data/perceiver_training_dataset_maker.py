# -*- coding: utf-8 -*-
"""Perceiver_Training_Dataset_Maker.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/asigalov61/Perceiver-Music-Transformer/blob/main/Training-Data/Perceiver_Training_Dataset_Maker.ipynb

# Perceiver Training Dataset Maker (ver. 1.0)

***

Powered by tegridy-tools: https://github.com/asigalov61/tegridy-tools

***

#### Project Los Angeles

#### Tegridy Code 2023

***

# (SETUP ENVIRONMENT)
"""

#@title Install all dependencies (run only once per session)

!git clone https://github.com/asigalov61/tegridy-tools
!pip install tqdm

#@title Import all needed modules

print('Loading needed modules. Please wait...')
import os

import math
import statistics
import random

from tqdm import tqdm

if not os.path.exists('/content/Dataset'):
    os.makedirs('/content/Dataset')

print('Loading TMIDIX module...')
os.chdir('/content/tegridy-tools/tegridy-tools')

import TMIDIX

print('Done!')

os.chdir('/content/')
print('Enjoy! :)')

"""# (DOWNLOAD SOURCE MIDI DATASET)"""

# Commented out IPython magic to ensure Python compatibility.
#@title Download original LAKH MIDI Dataset

# %cd /content/Dataset/

!wget 'http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz'
!tar -xvf 'lmd_full.tar.gz'
!rm 'lmd_full.tar.gz'

# %cd /content/

#@title Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

"""# (FILE LIST)"""

#@title Save file list
###########

print('Loading MIDI files...')
print('This may take a while on a large dataset in particular.')

dataset_addr = "/content/Dataset"
# os.chdir(dataset_addr)
filez = list()
for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
    filez += [os.path.join(dirpath, file) for file in filenames]
print('=' * 70)

if filez == []:
    print('Could not find any MIDI files. Please check Dataset dir...')
    print('=' * 70)

print('Randomizing file list...')
random.shuffle(filez)

TMIDIX.Tegridy_Any_Pickle_File_Writer(filez, '/content/drive/MyDrive/filez')

#@title Load file list
filez = TMIDIX.Tegridy_Any_Pickle_File_Reader('/content/drive/MyDrive/filez')

"""# (PROCESS)"""

#@title Process MIDIs with TMIDIX MIDI processor

print('=' * 70)
print('TMIDIX MIDI Processor')
print('=' * 70)
print('Starting up...')
print('=' * 70)

###########

START_FILE_NUMBER = 0
LAST_SAVED_BATCH_COUNT = 0

input_files_count = START_FILE_NUMBER
files_count = LAST_SAVED_BATCH_COUNT

melody_chords_f = []
mel_cho_batches = []

stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

print('Processing MIDI files. Please wait...')
print('=' * 70)

for f in tqdm(filez[START_FILE_NUMBER:]):
    try:
        input_files_count += 1

        fn = os.path.basename(f)

        # Filtering out giant MIDIs
        file_size = os.path.getsize(f)

        if file_size < 200000:

          #=======================================================
          # START PROCESSING

          # Convering MIDI to ms score with MIDI.py module
          score = TMIDIX.midi2ms_score(open(f, 'rb').read())

          # INSTRUMENTS CONVERSION CYCLE
          events_matrix = []
          itrack = 1
          patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

          patch_map = [
                      [0, 1, 2, 3, 4, 5, 6, 7], # Piano 
                      [24, 25, 26, 27, 28, 29, 30], # Guitar
                      [32, 33, 34, 35, 36, 37, 38, 39], # Bass
                      [40, 41], # Violin
                      [42, 43], # Cello
                      [46], # Harp
                      [56, 57, 58, 59, 60], # Trumpet
                      [64, 65, 66, 67, 68, 69, 70, 71], # Sax
                      [72, 73, 74, 75, 76, 77, 78], # Flute
                      [-1], # Drums
                      [52, 53], # Choir
                      [16, 17, 18, 19, 20] # Organ
                      ]

          while itrack < len(score):
              for event in score[itrack]:         
                  if event[0] == 'note' or event[0] == 'patch_change':
                      events_matrix.append(event)
              itrack += 1

          events_matrix.sort(key=lambda x: x[1])

          events_matrix1 = []

          for event in events_matrix:
                  if event[0] == 'patch_change':
                      patches[event[2]] = event[3]

                  if event[0] == 'note':
                      event.extend([patches[event[3]]])
                      once = False
                      
                      for p in patch_map:
                          if event[6] in p and event[3] != 9: # Except the drums
                              event[3] = patch_map.index(p)
                              once = True
                              
                      if not once and event[3] != 9: # Except the drums
                          event[3] = 15 # All other instruments/patches channel
                          event[5] = max(80, event[5])
                          
                      if event[3] < 12: # We won't write chans 12-16 for now...
                          events_matrix1.append(event)
                          

          #=======================================================
          # PRE-PROCESSING

          # checking number of instruments in a composition
          instruments_list_without_drums = list(set([y[3] for y in events_matrix1 if y[3] != 9]))

          if len(events_matrix1) > 0 and len(instruments_list_without_drums) > 0:

            # recalculating timings
            for e in events_matrix1:
                e[1] = int(e[1] / 8) # Max 2 seconds for start-times
                e[2] = int(e[2] / 16) # Max 4 seconds for durations

            # Sorting by pitch, then by start-time
            events_matrix1.sort(key=lambda x: x[4], reverse=True)
            events_matrix1.sort(key=lambda x: x[1])

            #=======================================================
            # FINAL PRE-PROCESSING

            melody_chords = []

            pe = events_matrix1[0]
      
            for e in events_matrix1:
              if e[1] >= 0 and e[2] >= 0:

                # Cliping all values...
                time = max(0, min(255, e[1]-pe[1]))             
                dur = max(1, min(255, e[2]))
                cha = max(0, min(11, e[3]))
                ptc = max(1, min(127, e[4]))
                vel = max(8, min(127, e[5]))

                velocity = round(vel / 15)

                # Writing final note 
                melody_chords.append([time, dur, cha, ptc, velocity])

                pe = e

            if len([y for y in melody_chords if y[2] != 9]) > 12: # Filtering out tiny/bad MIDIs...

              times = [y[0] for y in melody_chords[12:]]
              avg_time = sum(times) / len(times)
                
              times_list = list(set(times))
              
              instruments_list = list(set([y[2] for y in melody_chords]))
              num_instr = len(instruments_list)

              if avg_time < 96 and instruments_list != [9]: # Filtering out bad MIDIs...
                if 0 in times_list: # Filtering out (mono) melodies MIDIs
                  if len(melody_chords) > 1536:            

                    #=======================================================
                    # MAIN PROCESSING CYCLE
                    #=======================================================
                    
                    mel_cho = []

                    for m in melody_chords:
                        
                        # WRITING EACH NOTE HERE
                        time = m[0]
                        dur = m[1]
                        cha_vel = (m[2] * 8) + (m[4]-1)
                        cha_ptc = (m[2] * 128) + m[3]


                        mel_cho.extend([time, dur+256, cha_vel+512, cha_ptc+608])

                        stats[m[2]] += 1
                        
                    # TOTAL DICTIONARY SIZE 2144+1 = 2145

                    #=======================================================
                    # FINAL PROCESSING
                    #=======================================================
                    
                    # Assembling overlapping batches (1024 notes prefix/context and 512 notes training sequence)
                    
                    num_batches = (len(mel_cho) // 2048) - 3
                    
                    for i in range(num_batches):
                        
                        mel_cho_batches.append(mel_cho[i*512*4:((i+1)*512*4)])
                        mel_cho_batches.append(mel_cho[(i+1)*512*4:((i+2)*512*4)])
                        mel_cho_batches.append(mel_cho[(i+2)*512*4:((i+3)*512*4)])
                        
                    #=======================================================

                    # Processed files counter
                    files_count += 1

                    # Saving every 5000 processed files
                    if files_count % 5000 == 0:
                      print('SAVING !!!')
                      print('=' * 70)
                      print('Randomizing batches...')
                      print('=' * 70)
                      random.shuffle(mel_cho_batches)
                      print('Prepping final data...')
                      print('=' * 70)
                      for m in tqdm(mel_cho_batches):
                          melody_chords_f.extend(m)
                      mel_cho_batches = []                      
                      print('Saving processed files...')
                      print('=' * 70)
                      print('Data check:', min(melody_chords_f), '===', max(melody_chords_f), '===', len(list(set(melody_chords_f))), '===', len(melody_chords_f))
                      print('=' * 70)
                      print('Processed so far:', files_count, 'out of', input_files_count, '===', files_count / input_files_count, 'good files ratio')
                      print('=' * 70)
                      count = str(files_count)
                      TMIDIX.Tegridy_Any_Pickle_File_Writer(melody_chords_f, '/content/drive/MyDrive/LAKH_INTs_'+count)
                      melody_chords_f = []
                      print('=' * 70)
        
    except KeyboardInterrupt:
        print('Saving current progress and quitting...')
        break  

    except Exception as ex:
        print('WARNING !!!')
        print('=' * 70)
        print('Bad MIDI:', f)
        print('Error detected:', ex)
        print('=' * 70)
        continue

# Saving last processed files...
print('SAVING !!!')
print('=' * 70)
print('Randomizing batches...')
print('=' * 70)
random.shuffle(mel_cho_batches)
print('Prepping final data...')
print('=' * 70)
for m in tqdm(mel_cho_batches):
    melody_chords_f.extend(m)           
print('=' * 70)
print('Saving processed files...')
print('=' * 70)
print('Data check:', min(melody_chords_f), '===', max(melody_chords_f), '===', len(list(set(melody_chords_f))), '===', len(melody_chords_f))
print('=' * 70)
print('Processed so far:', files_count, 'out of', input_files_count, '===', files_count / input_files_count, 'good files ratio')
print('=' * 70)
count = str(files_count)
TMIDIX.Tegridy_Any_Pickle_File_Writer(melody_chords_f, '/content/drive/MyDrive/LAKH_INTs_'+count)

# Displaying resulting processing stats...
print('=' * 70)
print('Done!')   
print('=' * 70)

print('Resulting Stats:')
print('=' * 70)
print('Total good processed MIDI files:', files_count)
print('=' * 70)

print('Instruments stats:')
print('=' * 70)
print('Piano:', stats[0])
print('Guitar:', stats[1])
print('Bass:', stats[2])
print('Violin:', stats[3])
print('Cello:', stats[4])
print('Harp:', stats[5])
print('Trumpet:', stats[6])
print('Sax:', stats[7])
print('Flute:', stats[8])
print('Drums:', stats[9])
print('Choir:', stats[10])
print('Organ:', stats[11])
print('=' * 70)

"""# (TEST INTS)"""

#@title Test INTs

train_data1 = melody_chords_f

print('Sample INTs', train_data1[:15])

out = train_data1[:200000]

if len(out) != 0:
    
    song = out
    song_f = []
    time = 0
    dur = 0
    vel = 0
    pitch = 0
    channel = 0
                    
    for ss in song:
      
      if ss > 0 and ss < 256:

          time += ss * 8
        
      if ss >= 256 and ss < 512:

          dur = (ss-256) * 16

      if ss >= 512 and ss < 608:

          channel = (ss-512) // 8
          vel = (((ss-512) % 8)+1) * 15
              
      if ss >= 608 and ss < 608+(12*128):
          
          pitch = (ss-608) % 128

          song_f.append(['note', time, dur, channel, pitch, vel ])

    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                        output_signature = 'Perceiver',  
                                                        output_file_name = '/content/Perceiver-Music-Composition', 
                                                        track_name='Project Los Angeles',
                                                        list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 65, 73, 0, 53, 19, 0, 0, 0, 0],
                                                        number_of_ticks_per_quarter=500)

    print('Done!')

"""# Congrats! You did it! :)"""