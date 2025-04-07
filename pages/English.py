import os
import numpy as np
import torch
import tempfile
import streamlit as st
from pathlib import Path
import soundfile as sf
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import noisereduce as nr
import base64
import time
import gc  # NEW

# Set page configuration
st.set_page_config(
    page_title="Audio Stem Separator",
    page_icon="🎵",
    layout="wide"
)

# Helper function to create a download link for audio files
def get_binary_file_downloader_html(bin_file, file_label):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:audio/wav;base64,{bin_str}" download="{os.path.basename(file_label)}">Download {os.path.basename(file_label)}</a>'
    return href

class MultiStemExtractor:
    def __init__(self, output_folder):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        with st.spinner("Loading Spleeter 5-stem model..."):
            self.separator = Separator("spleeter:5stems")
        st.success("Instrument separation is ready!")

    def extract_stems(self, audio_path: Path, prediction: dict, sr: int) -> dict:
        try:
            stems = ["vocals", "drums", "bass", "piano", "other"]
            result_files = {}
            for stem_name in stems:
                if stem_name not in prediction:
                    st.warning(f"No '{stem_name}' stem found in the audio")
                    continue

                stem_audio = prediction[stem_name]

                if len(stem_audio.shape) == 2:
                    if stem_name == "vocals":
                        vocals_mono = stem_audio.mean(axis=1)
                        with st.spinner("Applying noise reduction to vocals..."):
                            processed_audio = nr.reduce_noise(
                                y=vocals_mono,
                                sr=sr,
                                stationary=False,
                                prop_decrease=0.4,
                                freq_mask_smooth_hz=150,
                                time_constant_s=1.0
                            )
                    else:
                        processed_audio = stem_audio.mean(axis=1)
                else:
                    processed_audio = stem_audio

                output_path = self.output_folder / f"{audio_path.stem}_{stem_name}.wav"
                sf.write(str(output_path), processed_audio, sr)
                result_files[stem_name] = str(output_path)

            return result_files

        except Exception as e:
            st.error(f"Error extracting stems: {e}")
            return {}

    def process_file(self, audio_file, file_name):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
                tmp.write(audio_file.getvalue())
                tmp_path = Path(tmp.name)

            st.info(f"Processing: {file_name}")

            audio_loader = AudioAdapter.default()
            with st.spinner("Loading audio file..."):
                waveform, sr = audio_loader.load(str(tmp_path), sample_rate=44100)
            st.success(f"Audio loaded successfully at {sr} Hz sample rate")

            with st.spinner("Separating stems..."):
                prediction = self.separator.separate(waveform)

            result_files = self.extract_stems(tmp_path, prediction, sr)

            os.unlink(str(tmp_path))
            del waveform
            del prediction
            gc.collect()  # 💡 force memory cleanup

            return result_files

        except Exception as e:
            st.error(f"Error processing {file_name}: {e}")
            return {}

def main():
    st.title("🎵 Audio Stem Separator")
    st.markdown("""
    Upload your audio file to separate it into individual stems:
    - Vocals
    - Drums
    - Bass
    - Piano
    - Other (background instruments/effects)
    """)

    max_file_size = 150 * 1024 * 1024  # 150MB

    temp_output_dir = tempfile.mkdtemp()
    extractor = MultiStemExtractor(temp_output_dir)

    uploaded_file = st.file_uploader(
        "Choose an audio file (MP3, MP4, WAV, etc.)",
        type=["mp3", "mp4", "wav", "flac", "m4a", "ogg"],
        help=f"Maximum file size: {max_file_size / (1024 * 1024):.1f}MB"
    )

    if uploaded_file is not None:
        if uploaded_file.size > max_file_size:
            st.error(f"File too large! Maximum size is {max_file_size / (1024 * 1024):.1f}MB")
        else:
            start_time = time.time()
            result_files = extractor.process_file(uploaded_file, uploaded_file.name)
            processing_time = time.time() - start_time
            st.success(f"Processing completed in {processing_time:.2f} seconds!")

            if result_files:
                st.subheader("Download Separated Stems")
                col1, col2 = st.columns(2)
                with col1:
                    for stem in ["vocals", "drums"]:
                        if stem in result_files:
                            st.markdown(get_binary_file_downloader_html(
                                result_files[stem],
                                f"{Path(uploaded_file.name).stem}_{stem}.wav"
                            ), unsafe_allow_html=True)
                with col2:
                    for stem in ["bass", "piano", "other"]:
                        if stem in result_files:
                            st.markdown(get_binary_file_downloader_html(
                                result_files[stem],
                                f"{Path(uploaded_file.name).stem}_{stem}.wav"
                            ), unsafe_allow_html=True)

                st.info("⚠️ Temporary files are only available during this session. Refreshing the page will remove all files. Please download what you need!")

    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Upload your audio file (MP3, MP4, WAV, etc.)
    2. Wait for the stems to be separated
    3. Download each stem individually

    Your file is not saved — everything is processed temporarily in memory.
    """)

if __name__ == "__main__":
    main()
