import numpy as np
import tensorflow as tf
from music21 import instrument, note, stream, chord
import pickle
import random

SEQUENCE_LENGTH = 100

# Load notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

unique_notes = sorted(set(notes))
note_to_int = {note: num for num, note in enumerate(unique_notes)}
int_to_note = {num: note for note, num in note_to_int.items()}

# Load model
model = tf.keras.models.load_model("model/music_model.h5")

# Prepare seed
start = random.randint(0, len(notes) - SEQUENCE_LENGTH - 1)
pattern = [note_to_int[n] for n in notes[start:start + SEQUENCE_LENGTH]]

prediction_output = []

# Generate notes
for _ in range(200):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(len(unique_notes))

    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)

    pattern.append(index)
    pattern = pattern[1:]

# Convert to MIDI
offset = 0
output_notes = []

for pattern in prediction_output:
    if '.' in pattern or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes_list = []

        for n in notes_in_chord:
            new_note = note.Note(int(n))
            new_note.storedInstrument = instrument.Piano()
            notes_list.append(new_note)

        new_chord = chord.Chord(notes_list)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write("midi", fp="output/ai_music.mid")

print("🎵 Music generated! Check output/ai_music.mid")
