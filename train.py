import numpy as np
import tensorflow as tf
from music21 import converter, instrument, note, chord
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "data"
SEQUENCE_LENGTH = 100

notes = []

# Load MIDI files
for file in os.listdir(DATASET_PATH):
    if file.endswith(".mid"):
        midi = converter.parse(os.path.join(DATASET_PATH, file))
        notes_to_parse = midi.recurse()

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

# Save notes
with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)

print("Total notes:", len(notes))

# Prepare sequences
unique_notes = sorted(set(notes))
note_to_int = {note: num for num, note in enumerate(unique_notes)}

network_input = []
network_output = []

for i in range(len(notes) - SEQUENCE_LENGTH):
    seq_in = notes[i:i + SEQUENCE_LENGTH]
    seq_out = notes[i + SEQUENCE_LENGTH]
    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)

network_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
network_input = network_input / float(len(unique_notes))

network_output = to_categorical(network_output)

# Build model
model = Sequential([
    LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dropout(0.3),
    Dense(256),
    Dense(len(unique_notes), activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()

# Train model
model.fit(network_input, network_output, epochs=20, batch_size=64)

# Save model
model.save("model/music_model.h5")
