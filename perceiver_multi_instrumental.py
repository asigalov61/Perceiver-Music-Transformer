# -*- coding: utf-8 -*-
"""Perceiver_Multi_Instrumental.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/asigalov61/Perceiver-Music-Transformer/blob/main/Perceiver_Multi_Instrumental.ipynb

# Perceiver Multi-Instrumental (ver. 1.0)

***

Powered by tegridy-tools: https://github.com/asigalov61/tegridy-tools

***

WARNING: This complete implementation is a functioning model of the Artificial Intelligence. Please excercise great humility, care, and respect. https://www.nscai.gov/

***

#### Project Los Angeles

#### Tegridy Code 2022

***

# (Setup Environment)
"""

#@title nvidia-smi gpu check
!nvidia-smi

#@title Install all dependencies (run only once per session)

!git clone https://github.com/asigalov61/Perceiver-Music-Transformer
!pip install einops
!pip install torch
!pip install torch-summary

!pip install tqdm
!pip install matplotlib

!apt install fluidsynth #Pip does not work for some reason. Only apt works
!pip install midi2audio

#@title Import all needed modules

print('Loading needed modules. Please wait...')
import os
import random
import copy

from collections import OrderedDict

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import torch
from torchsummary import summary

print('Loading core modules...')
os.chdir('/content/Perceiver-Music-Transformer')

import TMIDIX

from perceiver_ar_pytorch import PerceiverAR
from autoregressive_wrapper import AutoregressiveWrapper

from midi2audio import FluidSynth
from IPython.display import Audio, display

os.chdir('/content/')
print('Done!')

"""# (DOWNLOAD MODEL)"""

#@title Download Perceiver Pre-Trained Multi-Instrumental Model
!wget --no-check-certificate -O 'Perceiver-Multi-Instrumental-Model.pth' "https://onedrive.live.com/download?cid=8A0D502FC99C608F&resid=8A0D502FC99C608F%2118754&authkey=AKJDJYLlRnEpBuU"

"""# (LOAD)"""

#@title Load/Reload the model

full_path_to_model_checkpoint = "/content/Perceiver-Multi-Instrumental-Model.pth" #@param {type:"string"}

print('Loading the model...')
# Load model

# constants

SEQ_LEN = 8192 * 4 # 32k
PREFIX_SEQ_LEN = (8192 * 4) - 1024

model = PerceiverAR(
    num_tokens = 512,
    dim = 1024,
    depth = 24,
    heads = 16,
    dim_head = 64,
    cross_attn_dropout = 0.5,
    max_seq_len = SEQ_LEN,
    cross_attn_seq_len = PREFIX_SEQ_LEN
)
model = AutoregressiveWrapper(model)
model.cuda()

state_dict = torch.load(full_path_to_model_checkpoint)

model.load_state_dict(state_dict)

model.eval()

print('Done!')

# Model stats
summary(model)

"""# (GENERATE)

# Continuation
"""

#@title Load Seed/Custom MIDI
full_path_to_custom_MIDI_file = "/content/Perceiver-Music-Transformer/Perceiver-MI-Seed-1.mid" #@param {type:"string"}

print('Loading custom MIDI file...')
score = TMIDIX.midi2ms_score(open(full_path_to_custom_MIDI_file, 'rb').read())

events_matrix = []

itrack = 1

patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

patch_map = [[0, 1, 2, 3, 4, 5, 6, 7], # Piano 
              [24, 25, 26, 27, 28, 29, 30], # Guitar
              [32, 33, 34, 35, 36, 37, 38, 39], # Bass
              [40, 41], # Violin
              [42, 43], # Cello
              [46], # Harp
              [56, 57, 58, 59, 60], # Trumpet
              [71, 72], # Clarinet
              [73, 74, 75], # Flute
              [-1], # Fake Drums
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
                event[3] = 0 # All other instruments/patches channel
                event[5] = max(80, event[5])
                
            if event[3] < 12: # We won't write chans 11-16 for now...
                events_matrix1.append(event)

# Sorting...
events_matrix1.sort(key=lambda x: (x[1], x[3]))

# recalculating timings
for e in events_matrix1:
    e[1] = int(e[1] / 16)
    e[2] = int(e[2] / 32)

# final processing...

inputs = []

melody = []

melody_chords = []

pe = events_matrix1[0]
for e in events_matrix1:

    time = max(0, min(127, e[1]-pe[1]))
    dur = max(1, min(127, e[2]))
    cha = max(0, min(11, e[3]))
    ptc = max(1, min(127, e[4]))
    vel = max(19, min(127, e[5]))

    div_vel = int(vel / 19)

    chan_vel = (cha * 11) + div_vel

    # Continuation / Inpainting
    inputs.extend([chan_vel, time+128, dur+256, ptc+384])

    # Melody Orchestration
    if time != 0:
      if ptc < 60:
        ptc = (ptc % 12) + 60  

      # Converted to Piano
      melody.extend([div_vel, time+128, dur+256, ptc+384])

    # For future development
    melody_chords.append([time, dur, cha, ptc, vel])

    pe = e

# =================================

out1 = inputs

if len(out1) != 0:
    
    song = out1
    song_f = []
    time = 0
    dur = 0
    vel = 0
    pitch = 0
    channel = 0
    son = []
    song1 = []

    for s in song:
      if s > 127:
        son.append(s)

      else:
        if len(son) == 4:
          song1.append(son)
        son = []
        son.append(s)
    
    for s in song1:
      if s[0] > 0 and s[1] >= 128:
        if s[2] > 256 and s[3] > 384:

          channel = s[0] // 11

          vel = (s[0] % 11) * 19

          time += (s[1]-128) * 16
      
          dur = (s[2] - 256) * 32
          
          pitch = (s[3] - 384)
                                    
          song_f.append(['note', time, dur, channel, pitch, vel ])

    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                        output_signature = 'Perceiver',  
                                                        output_file_name = '/content/Perceiver-Music-Composition', 
                                                        track_name='Project Los Angeles',
                                                        list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                        number_of_ticks_per_quarter=500)

    print('Done!')

print('Displaying resulting composition...')
fname = '/content/Perceiver-Music-Composition'

x = []
y =[]
c = []

colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']

for s in song_f:
  x.append(s[1] / 1000)
  y.append(s[4])
  c.append(colors[s[3]])

FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
display(Audio(str(fname + '.wav'), rate=16000))

plt.figure(figsize=(14,5))
ax=plt.axes(title=fname)
ax.set_facecolor('black')

plt.scatter(x,y, c=c)
plt.xlabel("Time")
plt.ylabel("Pitch")
plt.show()

"""# Continuation"""

#@title Single Continuation Block Generator

#@markdown NOTE: Play with the settings to get different results
number_of_prime_tokens = 256 #@param {type:"slider", min:128, max:512, step:16}
number_of_tokens_to_generate = 512 #@param {type:"slider", min:64, max:512, step:32}
temperature = 0.8 #@param {type:"slider", min:0.1, max:1, step:0.1}

#===================================================================
print('=' * 70)
print('Perceiver Music Model Continuation Generator')
print('=' * 70)

print('Generation settings:')
print('=' * 70)
print('Number of prime tokens:', number_of_prime_tokens)
print('Number of tokens to generate:', number_of_tokens_to_generate)
print('Model temperature:', temperature)

print('=' * 70)
print('Generating...')

inp = inputs[:4] * 8192

inp = inp[(1024-number_of_prime_tokens)+len(inputs[:number_of_prime_tokens]):] + inputs[:number_of_prime_tokens]

inp = torch.LongTensor(inp).cuda()

out = model.generate(inp[None, ...], 
                     number_of_tokens_to_generate, 
                     temperature=temperature)  

out1 = out.cpu().tolist()[0]

if len(out1) != 0:
    
    song = inputs[:number_of_prime_tokens] + out1
    song_f = []
    time = 0
    dur = 0
    vel = 0
    pitch = 0
    channel = 0
    son = []
    song1 = []

    for s in song:
      if s > 127:
        son.append(s)

      else:
        if len(son) == 4:
          song1.append(son)
        son = []
        son.append(s)
    
    for s in song1:
      if s[0] > 0 and s[1] >= 128:
        if s[2] > 256 and s[3] > 384:

          channel = s[0] // 11

          vel = (s[0] % 11) * 19

          time += (s[1]-128) * 16
      
          dur = (s[2] - 256) * 32
          
          pitch = (s[3] - 384)
                                    
          song_f.append(['note', time, dur, channel, pitch, vel ])

    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                        output_signature = 'Perceiver',  
                                                        output_file_name = '/content/Perceiver-Music-Composition', 
                                                        track_name='Project Los Angeles',
                                                        list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                        number_of_ticks_per_quarter=500)

    print('Done!')

print('Displaying resulting composition...')
fname = '/content/Perceiver-Music-Composition'

x = []
y =[]
c = []

colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']

for s in song_f:
  x.append(s[1] / 1000)
  y.append(s[4])
  c.append(colors[s[3]])

FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
display(Audio(str(fname + '.wav'), rate=16000))

plt.figure(figsize=(14,5))
ax=plt.axes(title=fname)
ax.set_facecolor('black')

plt.scatter(x,y, c=c)
plt.xlabel("Time")
plt.ylabel("Pitch")
plt.show()

#@title Auto-Continue Custom MIDI

number_of_continuation_notes = 400 #@param {type:"slider", min:10, max:2000, step:10}
number_of_prime_tokens = 256 #@param {type:"slider", min:128, max:512, step:16}
temperature = 0.8 #@param {type:"slider", min:0.1, max:1, step:0.1}

#===================================================================
print('=' * 70)
print('Perceiver Music Model Auto-Continuation Generator')
print('=' * 70)

print('Generation settings:')
print('=' * 70)
print('Number of continuation notes:', number_of_continuation_notes)
print('Number of prime tokens:', number_of_prime_tokens)
print('Model temperature:', temperature)

print('=' * 70)
print('Generating...')

out2 = copy.deepcopy(inputs[:number_of_prime_tokens])

for i in tqdm(range(number_of_continuation_notes)):

  inp = inputs[:4] * 8192

  inp = inp[(1024-number_of_prime_tokens)+len(out2):] + out2

  inp = torch.LongTensor(inp).cuda()

  out = model.generate(inp[None, ...], 
                      4, 
                      temperature=temperature)  

  out1 = out.cpu().tolist()[0]
  out2.extend(out1)

if len(out2) != 0:
    
    song = out2
    song_f = []
    time = 0
    dur = 0
    vel = 0
    pitch = 0
    channel = 0
    son = []
    song1 = []

    for s in song:
      if s > 127:
        son.append(s)

      else:
        if len(son) == 4:
          song1.append(son)
        son = []
        son.append(s)
    
    for s in song1:
      if s[0] > 0 and s[1] >= 128:
        if s[2] > 256 and s[3] > 384:

          channel = s[0] // 11

          vel = (s[0] % 11) * 19

          time += (s[1]-128) * 16
      
          dur = (s[2] - 256) * 32
          
          pitch = (s[3] - 384)
                                    
          song_f.append(['note', time, dur, channel, pitch, vel ])

    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                        output_signature = 'Perceiver',  
                                                        output_file_name = '/content/Perceiver-Music-Composition', 
                                                        track_name='Project Los Angeles',
                                                        list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                        number_of_ticks_per_quarter=500)

    print('Done!')

print('Displaying resulting composition...')
fname = '/content/Perceiver-Music-Composition'

x = []
y =[]
c = []

colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']

for s in song_f:
  x.append(s[1] / 1000)
  y.append(s[4])
  c.append(colors[s[3]])

FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
display(Audio(str(fname + '.wav'), rate=16000))

plt.figure(figsize=(14,5))
ax=plt.axes(title=fname)
ax.set_facecolor('black')

plt.scatter(x,y, c=c)
plt.xlabel("Time")
plt.ylabel("Pitch")
plt.show()

"""# Congrats! You did it! :)"""