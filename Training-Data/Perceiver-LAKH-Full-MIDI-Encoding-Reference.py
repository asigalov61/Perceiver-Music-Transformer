# MIDI encoder

full_path_to_custom_MIDI_file = "/notebooks/tegridy-tools/tegridy-tools/seed2.mid" 

print('Loading custom MIDI file...')

score = TMIDIX.midi2ms_score(open(full_path_to_custom_MIDI_file, 'rb').read())

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
             'control_change', # 128
             'patch_change', # 129
             'pitch_wheel_change', # 130
            ] # SOS/EOS 131

melody_chords = []

pe = events_matrix[0]
for e in events_matrix:
  if e[0] in data_tags:

    if e[0] == 'note':
      # ['note', start_time, duration, channel, note, velocity]
      time = max(0, min(127, e[1]-pe[1]))
      duration = max(1, min(127, e[2]))
      channel = max(0, min(15, e[3]))
      pitch = max(1, min(127, e[4]))
      velocity = max(1, min(127, e[5]))

      melody_chords.extend([131, time, duration, channel, pitch, velocity])

    if e[0] == 'control_change':
      # ['control_change', dtime, channel, controller(0-127), value(0-127)]
      time = max(0, min(127, e[1]-pe[1]))
      channel = max(0, min(15, e[2]))
      controller = max(0, min(127, e[3]))
      value = max(0, min(127, e[4]))

      melody_chords.extend([131, 128, time, channel, controller, value])

    if e[0] == 'patch_change':
      # ['patch_change', dtime, channel, patch]
      time = max(0, min(127, e[1]-pe[1]))
      channel = max(0, min(15, e[2]))
      patch = max(0, min(127, e[3]))

      melody_chords.extend([131, 129, time, channel, patch, 0])

    if e[0] == 'pitch_wheel_change':
      # ['pitch_wheel_change', dtime, channel, pitch_wheel]
      time = max(0, min(127, e[1]-pe[1]))
      channel = max(0, min(15, e[2]))
      pitch_wheel = max(0, min(127, ((e[3] // 128)+64)))

      melody_chords.extend([131, 130, time, channel, pitch_wheel, 0])

    pe = e # Previous event

# Break between compositions
melody_chords_f.extend([131, 127, 127, 0, 0, 0])
melody_chords_f.extend(melody_chords)

#=================================================================================================

# MIDI decoder

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
      if s < 131:
        son.append(s)

      else:
        if len(son) == 5:
          song1.append(son)
        son = []
        #son.append(s)
    
    for s in song1:

      if s[0] < 128: # note event
        
        time += s[0] * 10
        dur = s[1] * 20
        channel = s[2]
        pitch = s[3]
        vel = s[4]
                                  
        song_f.append(['note', time, dur, channel, pitch, vel ])

      if s[0] == 128: # control_change event
        time += s[1] * 10
        channel = s[2]
        controller = s[3]
        value = s[4]

        song_f.append(['control_change', time, channel, controller, value])

      if s[0] == 129: # patch_change event
        time += s[1] * 10
        channel = s[2]
        patch = s[3]

        song_f.append(['patch_change', time, channel, patch])

      if s[0] == 130: # pitch_wheel_change event
        time += s[1] * 10
        channel = s[2]
        pitch_wheel = (s[3] - 64) * 128

        song_f.append(['pitch_wheel_change', time, channel, pitch_wheel])

    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                        output_signature = 'Perceiver',  
                                                        output_file_name = '/content/Perceiver-Music-Composition', 
                                                        track_name='Project Los Angeles',
                                                        list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                        number_of_ticks_per_quarter=500)

    print('Done!')