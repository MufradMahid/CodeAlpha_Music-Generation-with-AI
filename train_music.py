from music21 import converter, instrument, note, chord
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
import os

notes = []

for file in os.listdir("midi_songs"):
    midi = converter.parse("midi_songs/" + file)
    parts = instrument.partitionByInstrument(midi)
    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

sequence_length = 50
unique_notes = sorted(set(notes))
note_to_int = {note: number for number, note in enumerate(unique_notes)}

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    network_input.append([note_to_int[n] for n in notes[i:i+sequence_length]])
    network_output.append(note_to_int[notes[i+sequence_length]])

X = np.reshape(network_input, (len(network_input), sequence_length, 1))
X = X / float(len(unique_notes))
y = to_categorical(network_output)

model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=10, batch_size=64)

model.save("music_model.h5")
