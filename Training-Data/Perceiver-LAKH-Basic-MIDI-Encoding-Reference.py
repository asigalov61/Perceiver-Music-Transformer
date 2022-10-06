# MIDI encoder

full_path_to_custom_MIDI_file = "/notebooks/tegridy-tools/tegridy-tools/seed2.mid" #@param {type:"string"}

print('Loading custom MIDI file...')
#print('Loading MIDI file...')
score = TMIDIX.midi2ms_score(open(f, 'rb').read())

events_matrix = []

itrack = 1

while itrack < len(score):
    for event in score[itrack]:         
      events_matrix.append(event)
    itrack += 1

# Sorting...
events_matrix.sort(key=lambda x: x[1])

# recalculating timings
for e in events_matrix:
    e[1] = int(e[1] / 10)
    if e[0] == 'note':
      e[2] = int(e[2] / 20)

# final processing...

data_tags = ['note', 
             'patch_change'
            ] # SOS/EOS 256

melody_chords = []

patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

pe = events_matrix[0]
for e in events_matrix:
  if e[0] in data_tags:

    if e[0] == 'note':
      # ['note', start_time, duration, channel, note, velocity]
      time = max(0, min(255, e[1]-pe[1]))
      duration = max(1, min(255, e[2]))

      if e[3] != 9:
        channel = max(0, min(127, patches[e[3]]))
      else:
        channel = max(128, min(255, patches[e[3]]+128))

      if e[3] != 9:

        pitch = max(1, min(127, e[4]))
      else:
        pitch = max(129, min(255, e[4]+128))

      if e[3] != 9:
        velocity = max(1, min(127, e[5]))
      else:
        velocity = max(129, min(255, e[5]+128))

      melody_chords.extend([256, time, duration, channel, pitch, velocity])

    if e[0] == 'patch_change':
      # ['patch_change', dtime, channel, patch]
      time = max(0, min(127, e[1]-pe[1]))
      channel = max(0, min(15, e[2]))
      patch = max(0, min(127, e[3]))

      patches[channel] = patch

    pe = e # Previous event

# Break between compositions
melody_chords_f.extend([256, 255, 255, 0, 0, 0])
melody_chords_f.extend(melody_chords)

#=================================================================================================

# Partial MIDI decoder [ WIP]

out1 = out.cpu().tolist()[0]

if len(out) != 0:
    
    song = out
    song_f = []
    time = 0
    dur = 0
    vel = 0
    pitch = 0
    channel = 0

    son = []

    song1 = []

    for s in song:
      if s < 256:
        son.append(s)

      else:
        if len(son) == 5:
          song1.append(son)
        son = []
        #son.append(s)
    
    for s in song1:

      if s[0] < 256: # note event
        
        time += s[0] * 10
        dur = s[1] * 20
        if s[2] < 128:
          channel = s[2] // 16

        else:
          channel = 9

        if s[3] < 128:

          pitch = s[3]
        else:
          pitch = s[3] - 128

        if s[4] < 128:
          vel = s[4]
        else:
          vel = s[4] - 128
                                  
        song_f.append(['note', time, dur, channel, pitch, vel ])


    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                        output_signature = 'Perceiver',  
                                                        output_file_name = '/content/Perceiver-Music-Composition', 
                                                        track_name='Project Los Angeles',
                                                        list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                        number_of_ticks_per_quarter=500)

    print('Done!')