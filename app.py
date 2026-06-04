import pickle
import torch
import torch.nn as nn
from music21 import stream, note, chord
import gradio as gr

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


def generate_music(length, temperature):
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

    output_path = 'generated_music.mid'
    midi_stream.write('midi', fp=output_path)
    return output_path


demo = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Slider(50, 500, value=200, step=50, label="Number of Notes"),
        gr.Slider(0.5, 1.5, value=0.8, step=0.1, label="Temperature (creativity)"),
    ],
    outputs=gr.File(label="Download Generated MIDI"),
    title="Music Generation with LSTM",
    description="Generate original piano music using an LSTM trained on classical MIDI files. Download the MIDI and open it in any media player.",
)

if __name__ == "__main__":
    demo.launch()
