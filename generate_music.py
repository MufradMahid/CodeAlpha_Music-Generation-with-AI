from music21 import instrument, note, stream, chord
import numpy as np
from tensorflow.keras.models import load_model
import random
import pickle

model = load_model("music_model.h5")

# Sample notes manually (simple)
notes = ['C4', 'E4', 'G4', 'A4', 'F4']
generated_notes = []

for i in range(50):
    generated_notes.append(random.choice(notes))

output_notes = []

for pattern in generated_notes:
    if '.' in pattern:
        notes_in_chord = pattern.split('.')
        chord_notes = [note.Note(int(n)) for n in notes_in_chord]
        output_notes.append(chord.Chord(chord_notes))
    else:
        output_notes.append(note.Note(pattern))

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='generated_music.mid')
