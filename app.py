import pickle
import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io.wavfile as wav
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

# Default generation settings
current_settings = {
    "length": 200,
    "temperature": 0.8,
    "instrument": "Piano",
    "tempo_scale": 1.0,
}


def parse_chat(message, history):
    msg = message.lower()
    response_parts = []

    if any(w in msg for w in ["slower", "slow down", "slow"]):
        current_settings["tempo_scale"] = min(current_settings["tempo_scale"] * 1.5, 4.0)
        response_parts.append("Slowed tempo down.")

    if any(w in msg for w in ["faster", "speed up", "fast"]):
        current_settings["tempo_scale"] = max(current_settings["tempo_scale"] * 0.7, 0.25)
        response_parts.append("Sped tempo up.")

    if any(w in msg for w in ["more notes", "longer", "more music"]):
        current_settings["length"] = min(current_settings["length"] + 100, 500)
        response_parts.append(f"Increased notes to {current_settings['length']}.")

    if any(w in msg for w in ["fewer notes", "shorter", "less"]):
        current_settings["length"] = max(current_settings["length"] - 100, 50)
        response_parts.append(f"Decreased notes to {current_settings['length']}.")

    if any(w in msg for w in ["creative", "random", "wild", "experimental"]):
        current_settings["temperature"] = min(current_settings["temperature"] + 0.2, 1.5)
        response_parts.append(f"Increased creativity to {current_settings['temperature']:.1f}.")

    if any(w in msg for w in ["structured", "calm", "predictable", "less random"]):
        current_settings["temperature"] = max(current_settings["temperature"] - 0.2, 0.5)
        response_parts.append(f"Decreased creativity to {current_settings['temperature']:.1f}.")

    for inst in INSTRUMENTS:
        if inst.lower() in msg:
            current_settings["instrument"] = inst
            response_parts.append(f"Switched instrument to {inst}.")
            break

    if any(w in msg for w in ["reset", "default", "start over"]):
        current_settings.update({"length": 200, "temperature": 0.8, "instrument": "Piano", "tempo_scale": 1.0})
        response_parts.append("Reset all settings to default.")

    if not response_parts:
        reply = "I didn't understand that. Try: 'slower', 'faster', 'more notes', 'use violin', 'more creative', or 'reset'."
    else:
        reply = " ".join(response_parts) + f"\n\nCurrent settings: {current_settings['instrument']} | {current_settings['length']} notes | creativity {current_settings['temperature']:.1f} | tempo x{1/current_settings['tempo_scale']:.1f}. Hit Generate to apply!"

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history


def midi_to_wav(midi_path, wav_path):
    from mido import MidiFile
    mid = MidiFile(midi_path)
    sample_rate = 22050
    duration = mid.length
    samples = int(sample_rate * duration)
    audio = np.zeros(samples, dtype=np.float32)

    t = 0
    for msg in mid.play():
        if msg.type == 'note_on' and msg.velocity > 0:
            freq = 440.0 * (2.0 ** ((msg.note - 69) / 12.0))
            start = int(t * sample_rate)
            end = min(start + int(0.5 * sample_rate), samples)
            ts = np.linspace(0, end - start, end - start, endpoint=False) / sample_rate
            audio[start:end] += 0.3 * np.sin(2 * np.pi * freq * ts) * np.exp(-3 * ts)

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    wav.write(wav_path, sample_rate, np.int16(audio * 32767))


def generate_music(filename, instrument_name, length, temperature):
    if not filename.strip():
        filename = "generated_music"
    filename = filename.strip().replace(" ", "_")

    length = current_settings["length"]
    temperature = current_settings["temperature"]
    instrument_name = current_settings["instrument"]
    tempo_scale = current_settings["tempo_scale"]

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
            dur = float(dur_str) * tempo_scale
        else:
            pattern, dur = token, 1.0 * tempo_scale

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


with gr.Blocks(title="Music Generation with LSTM") as demo:
    gr.Markdown("# Music Generation with LSTM")
    gr.Markdown("""Generate original music using an LSTM trained on classical MIDI files. Use the chatbot to adjust settings, then hit **Generate**.

### What the chatbot understands:
| Command | Example |
|---|---|
| Tempo | *"make it slower"*, *"speed it up"* |
| Length | *"more notes"*, *"make it shorter"* |
| Creativity | *"more creative"*, *"more structured"* |
| Instrument | *"use violin"*, *"switch to flute"* |
| Reset | *"reset"*, *"start over"* |
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Chat to adjust settings")
            chatbot = gr.Chatbot(height=300, type="messages")
            chat_input = gr.Textbox(placeholder="e.g. make it slower, use violin, more creative...")
            chat_input.submit(parse_chat, [chat_input, chatbot], [chatbot])
            chat_input.submit(lambda: "", None, chat_input)

        with gr.Column(scale=1):
            gr.Markdown("### Generate Music")
            filename_input = gr.Textbox(label="Song Name", placeholder="e.g. my_melody")
            instrument_input = gr.Dropdown(choices=list(INSTRUMENTS.keys()), value="Piano", label="Instrument")
            length_input = gr.Slider(50, 500, value=200, step=50, label="Number of Notes")
            temperature_input = gr.Slider(0.5, 1.5, value=0.8, step=0.1, label="Temperature (creativity)")
            generate_btn = gr.Button("Generate", variant="primary")
            midi_output = gr.File(label="Download MIDI")
            audio_output = gr.Audio(label="Play Music", type="filepath")

    generate_btn.click(
        generate_music,
        inputs=[filename_input, instrument_input, length_input, temperature_input],
        outputs=[midi_output, audio_output]
    )

if __name__ == "__main__":
    demo.launch()
