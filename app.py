import pickle
import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io.wavfile as wav
import pretty_midi
import gradio as gr
from music21 import stream, note, chord, instrument as m21instrument

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=2):
        super(MusicLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        rnn_out, hidden = self.lstm(x, hidden)
        output = self.fc(rnn_out[:, -1, :])
        return output, hidden


with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

vocab_size = len(label_encoder.classes_)
model = MusicLSTM(vocab_size=vocab_size, hidden_size=512).to(device)
model.load_state_dict(torch.load('music_lstm.pth', map_location=device))
model.eval()


def midi_to_wav(midi_path, wav_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    audio = pm.fluidsynth(fs=22050)
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
    wav.write(wav_path, 22050, audio)


INSTRUMENTS = {
    "Piano": m21instrument.Piano(),
    "Violin": m21instrument.Violin(),
    "Guitar": m21instrument.Guitar(),
    "Flute": m21instrument.Flute(),
    "Trumpet": m21instrument.Trumpet(),
    "Cello": m21instrument.Violoncello(),
    "Saxophone": m21instrument.Saxophone(),
    "Clarinet": m21instrument.Clarinet(),
}

def generate_music(filename, instrument_name, length, temperature):
    if not filename.strip():
        filename = "generated_music"
    filename = filename.strip().replace(" ", "_")

    seed = torch.randint(0, vocab_size, (50,)).tolist()
    input_seq = torch.tensor([seed], dtype=torch.long).to(device)
    generated = list(seed)
    hidden = None

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            probs = torch.softmax(output / temperature, dim=1)
            next_idx = torch.multinomial(probs, 1).item()
            generated.append(next_idx)
            input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    note_names = label_encoder.inverse_transform(generated)
    midi_stream = stream.Stream()
    midi_stream.append(INSTRUMENTS.get(instrument_name, m21instrument.Piano()))

    for token in note_names:
        if '_' in token:
            pattern, dur_str = token.rsplit('_', 1)
            dur = float(dur_str)
        else:
            pattern, dur = token, 1.0

        try:
            if '.' in pattern:
                chord_notes = [note.Note(midi=int(n) + 60) for n in pattern.split('.')]
                c = chord.Chord(chord_notes)
                c.duration.quarterLength = dur
                midi_stream.append(c)
            elif pattern.lstrip('-').isdigit():
                n = note.Note(midi=int(pattern) + 60)
                n.duration.quarterLength = dur
                midi_stream.append(n)
            else:
                n = note.Note(pattern)
                n.duration.quarterLength = dur
                midi_stream.append(n)
        except Exception:
            continue

    midi_path = f"{filename}.mid"
    wav_path = f"{filename}.wav"
    midi_stream.write('midi', fp=midi_path)

    try:
        midi_to_wav(midi_path, wav_path)
        return midi_path, wav_path
    except Exception:
        return midi_path, None


demo = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(label="Song Name", placeholder="e.g. my_melody"),
        gr.Dropdown(choices=list(INSTRUMENTS.keys()), value="Piano", label="Instrument"),
        gr.Slider(50, 500, value=200, step=50, label="Number of Notes"),
        gr.Slider(0.5, 1.5, value=0.8, step=0.1, label="Temperature (creativity)"),
    ],
    outputs=[
        gr.File(label="Download MIDI"),
        gr.Audio(label="Play Music", type="filepath"),
    ],
    title="Music Generation with LSTM",
    description="Generate original music using an LSTM trained on classical MIDI files. Choose your instrument, name your song, adjust the settings, and play or download the result.",
)

if __name__ == "__main__":
    demo.launch()
