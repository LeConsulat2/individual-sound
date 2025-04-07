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

# 페이지 설정
st.set_page_config(
    page_title="오디오 스템 분리기",
    page_icon="🎵",
    layout="wide"
)

# 오디오 파일 다운로드 링크 생성 도우미 함수
def get_binary_file_downloader_html(bin_file, file_label):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:audio/wav;base64,{bin_str}" download="{os.path.basename(file_label)}">다운로드 {os.path.basename(file_label)}</a>'
    return href

class MultiStemExtractor:
    def __init__(self, output_folder):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # 스피너를 사용하여 모델 로딩 진행 상황 표시
        with st.spinner("Spleeter 5 스템 모델을 불러오는 중..."):
            self.separator = Separator("spleeter:5stems")
        st.success("악기 분리가 준비되었습니다!")

    def extract_stems(self, audio_path: Path, prediction: dict, sr: int) -> dict:
        try:
            # 처리할 스템 리스트
            stems = ["vocals", "drums", "bass", "piano", "other"]
            result_files = {}
            
            for stem_name in stems:
                if stem_name not in prediction:
                    st.warning(f"오디오에서 '{stem_name}' 스템을 찾을 수 없습니다.")
                    continue

                stem_audio = prediction[stem_name]

                # 스테레오인 경우 모노로 변환
                if len(stem_audio.shape) == 2:
                    if stem_name == "vocals":
                        # 보컬에만 노이즈 제거 적용
                        vocals_mono = stem_audio.mean(axis=1)
                        with st.spinner("보컬에 노이즈 제거 적용 중..."):
                            processed_audio = nr.reduce_noise(
                                y=vocals_mono,
                                sr=sr,
                                stationary=False,
                                prop_decrease=0.4,
                                freq_mask_smooth_hz=150,
                                time_constant_s=1.0
                            )
                    else:
                        # 다른 스템은 단순히 모노로 변환
                        processed_audio = stem_audio.mean(axis=1)
                else:
                    processed_audio = stem_audio
                
                # 원본 파일명과 스템 이름을 기반으로 출력 파일명 생성
                output_path = self.output_folder / f"{audio_path.stem}_{stem_name}.wav"
                sf.write(str(output_path), processed_audio, sr)
                
                # 각 스템의 출력 경로 저장
                result_files[stem_name] = str(output_path)
                
            return result_files

        except Exception as e:
            st.error(f"스템 추출 중 오류 발생: {e}")
            return {}

    def process_file(self, audio_file, file_name):
        try:
            # 업로드된 파일을 저장할 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
                tmp.write(audio_file.getvalue())
                tmp_path = Path(tmp.name)
            
            st.info(f"처리 중: {file_name}")
            
            # 오디오 파일 로드
            audio_loader = AudioAdapter.default()
            with st.spinner("오디오 파일 불러오는 중..."):
                waveform, sr = audio_loader.load(str(tmp_path), sample_rate=44100)
            st.success(f"오디오 파일이 {sr} Hz 샘플레이트로 성공적으로 로드되었습니다.")

            # 오디오 파일 처리
            with st.spinner("스템 분리 중..."):
                prediction = self.separator.separate(waveform)
            
            # 스템 추출
            result_files = self.extract_stems(tmp_path, prediction, sr)
            
            # 임시 파일 삭제
            os.unlink(str(tmp_path))
            
            return result_files
            
        except Exception as e:
            st.error(f"{file_name} 처리 중 오류 발생: {e}")
            return {}

def main():
    st.title("🎵 오디오 스템 분리기")
    st.markdown("""
    오디오 파일을 업로드하여 다음 스템으로 분리합니다:
    - 보컬 (vocals)
    - 드럼 (drums)
    - 베이스 (bass)
    - 피아노 (piano)
    - 기타 (other: 배경 악기/효과음)
    """)
    
    # 파일 크기 제한 (44.1kHz 스테레오 오디오 15분 ≈ 15MB)
    # 계산: 44100 samples/sec * 2 채널 * 2 bytes/sample * 60 sec/min * 15 min ≈ 158MB
    max_file_size = 500  * 1024 * 1024  # 158MB
    
    # 출력 파일을 위한 임시 디렉토리 생성
    temp_output_dir = tempfile.mkdtemp()
    
    # 스템 추출기 초기화
    extractor = MultiStemExtractor(temp_output_dir)
    
    # 파일 업로더
    uploaded_file = st.file_uploader(
        "오디오 파일 선택 (MP3, MP4, WAV 등)",
        type=["mp3", "mp4", "wav", "flac", "m4a", "ogg"],
        help=f"최대 파일 크기: {max_file_size / (1024 * 1024):.1f}MB (약 15분 분량)"
    )
    
    if uploaded_file is not None:
        # 파일 크기 확인
        if uploaded_file.size > max_file_size:
            st.error(f"파일이 너무 큽니다! 최대 크기는 {max_file_size / (1024 * 1024):.1f}MB 입니다.")
        else:
            start_time = time.time()
            
            # 파일 처리
            result_files = extractor.process_file(uploaded_file, uploaded_file.name)
            
            # 처리 시간 표시
            processing_time = time.time() - start_time
            st.success(f"처리가 {processing_time:.2f}초에 완료되었습니다!")
            
            # 다운로드 링크 표시
            if result_files:
                st.subheader("분리된 스템 다운로드")
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
                
                # 임시 파일에 대한 안내
                st.info("참고: 임시 파일은 현재 세션 동안만 사용 가능합니다. 페이지를 새로고침하면 모든 파일이 삭제됩니다. 필요한 스템은 반드시 다운로드해주세요.")
    
    # 사용 방법 및 안내
    st.markdown("---")
    st.markdown("""
    ### 사용 방법:
    1. 오디오 파일 업로드 (MP3, MP4, WAV 등)
    2. 스템이 분리될 때까지 기다리세요
    3. 각 스템을 개별적으로 다운로드하세요
    
    업로드된 파일은 영구적으로 저장되지 않으며, 모든 처리는 메모리와 임시 파일을 통해 이루어집니다.
    """)

if __name__ == "__main__":
    main()
