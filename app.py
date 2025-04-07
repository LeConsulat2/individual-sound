import os
import numpy as np
# import torch # 사용하지 않으므로 제거
import tempfile
import streamlit as st
from pathlib import Path
import soundfile as sf
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import noisereduce as nr
import base64
import time
import logging

# --- 기본 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Streamlit 페이지 설정 ---
st.set_page_config(
    page_title="Audio Stem Separator",
    page_icon="🎵",
    layout="wide",
    # initial_sidebar_state="expanded" # 필요하면 사이드바를 기본으로 열어둘 수 있음
)

# --- 도우미 함수 ---

@st.cache_data # 다운로드 링크 생성 캐싱
def get_binary_file_downloader_html(bin_file, file_label):
    """바이너리 파일 다운로드 HTML 링크 생성"""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        download_filename = os.path.basename(file_label) # 파일명 직접 사용
        href = f'<a href="data:audio/wav;base64,{bin_str}" download="{download_filename}">Download {download_filename}</a>'
        return href
    except FileNotFoundError:
        logging.error(f"다운로드 링크 생성 실패 (파일 없음): {bin_file}")
        return f"<span style='color: red;'>오류: {os.path.basename(file_label)} 파일을 찾을 수 없습니다.</span>"
    except Exception as e:
        logging.error(f"다운로드 링크 생성 중 오류 ({bin_file}): {e}")
        return f"<span style='color: red;'>오류: {os.path.basename(file_label)} 다운로드 링크 생성 실패.</span>"

# --- 메인 처리 클래스 ---

class MultiStemExtractor:
    """오디오 로딩, 스템 분리, 후처리 담당"""

    @st.cache_resource # 모델 로딩 캐싱 (세션당 한 번)
    def get_separator(_self):
        """Spleeter Separator 모델 로드 및 캐싱"""
        logging.info("Spleeter Separator (spleeter:5stems) 초기화 중...")
        try:
            # Spleeter는 처음 실행 시 모델 파일을 다운로드할 수 있음
            separator = Separator("spleeter:5stems")
            logging.info("Spleeter Separator 초기화 완료.")
            return separator
        except Exception as e:
            logging.error(f"Spleeter Separator 초기화 실패: {e}", exc_info=True)
            st.error(f"치명적 오류: 분리 모델을 로드할 수 없습니다. 로그를 확인하세요. 오류: {e}")
            st.stop() # 모델 로드 실패 시 앱 중지

    def __init__(self, output_folder):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"출력 디렉토리: {self.output_folder}")

        # 캐시된 메서드를 사용하여 separator 로드
        if "separator_loaded" not in st.session_state:
             with st.spinner("분리 엔진 초기화 중 (모델 다운로드 가능)..."):
                 self.separator = self.get_separator()
             st.session_state.separator_loaded = True
             st.success("✅ 분리 엔진 준비 완료!")
        else:
             self.separator = self.get_separator() # 이미 로드됨


    def extract_stems(self, audio_path_stem: str, prediction: dict, sr: int) -> dict:
        """스템 추출, 처리(모노 변환, 노이즈 제거), 저장"""
        result_files = {}
        stems_to_process = ["vocals", "drums", "bass", "piano", "other"]
        total_stems = len(stems_to_process)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, stem_name in enumerate(stems_to_process):
            status_text.text(f"스템 처리 중: {stem_name}...")
            if stem_name not in prediction:
                logging.warning(f"Spleeter 출력에서 '{stem_name}' 스템 찾을 수 없음 ({audio_path_stem}).")
                st.warning(f"⚠️ '{stem_name}' 스템이 없거나 조용한 구간입니다.")
                continue

            stem_audio = prediction[stem_name] # NumPy 배열

            if not isinstance(stem_audio, np.ndarray) or stem_audio.size == 0:
                 logging.warning(f"'{stem_name}' 스템이 비어있거나 유효하지 않음 ({audio_path_stem}).")
                 st.warning(f"⚠️ '{stem_name}' 스템이 비어있거나 유효하지 않습니다.")
                 continue

            try:
                # --- 모노 변환 및 노이즈 제거 (보컬만) ---
                processed_audio = None
                if stem_audio.ndim == 2 and stem_audio.shape[1] > 1: # 스테레오 확인
                    mono_audio = stem_audio.mean(axis=1)
                    if stem_name == "vocals":
                        logging.info(f"보컬 노이즈 제거 적용 중 ({audio_path_stem})...")
                        status_text.text("보컬 노이즈 제거 적용 중...")
                        # 파라미터는 소스에 따라 조절 필요
                        processed_audio = nr.reduce_noise(
                            y=mono_audio,
                            sr=sr,
                            stationary=False, # 음악의 배경 노이즈는 non-stationary일 수 있음
                            prop_decrease=0.6, # 필요시 조절 (0.4 ~ 0.8)
                            # freq_mask_smooth_hz=150,
                            # time_constant_s=1.0
                        )
                        logging.info(f"보컬 노이즈 제거 완료 ({audio_path_stem}).")
                    else:
                        processed_audio = mono_audio # 다른 스템은 모노 변환만
                else: # 이미 모노
                    processed_audio = stem_audio.squeeze() # 1D 배열로 만듦

                # --- 처리된 스템 저장 ---
                if processed_audio is not None and processed_audio.size > 0:
                    output_filename = f"{audio_path_stem}_{stem_name}.wav"
                    output_path = self.output_folder / output_filename
                    logging.info(f"{stem_name} 스템 저장 중: {output_path}...")
                    # WAV 포맷, 16-bit PCM으로 명시적 저장
                    sf.write(str(output_path), processed_audio, sr, subtype='PCM_16')
                    result_files[stem_name] = str(output_path)
                    logging.info(f"{stem_name} 스템 저장 완료 ({audio_path_stem}).")
                else:
                    logging.warning(f"처리된 '{stem_name}' 스템 오디오가 비어있음 ({audio_path_stem}). 저장 건너뜀.")
                    st.warning(f"⚠️ 처리된 '{stem_name}' 스템이 비어있어 저장하지 않았습니다.")

            except Exception as e:
                logging.error(f"스템 처리 중 오류 ('{stem_name}', {audio_path_stem}): {e}", exc_info=True)
                st.error(f"❌ '{stem_name}' 스템 처리 중 오류 발생. 로그 확인 후 건너뜁니다.")

            # 진행률 업데이트
            progress_bar.progress((i + 1) / total_stems)

        status_text.text("✅ 스템 처리 완료.")
        progress_bar.empty() # 완료 후 진행률 바 숨기기
        return result_files

    def process_file(self, uploaded_file):
        """업로드된 단일 파일 처리 전체 과정 핸들링"""
        file_name = uploaded_file.name
        file_stem = Path(file_name).stem # 확장자 제외한 파일명
        temp_dir_input = tempfile.TemporaryDirectory() # 입력용 임시 디렉토리 관리
        tmp_path = Path(temp_dir_input.name) / file_name

        try:
            logging.info(f"업로드 파일 '{file_name}' 임시 저장: {tmp_path}")
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            logging.info(f"'{file_name}' 임시 저장 완료: {tmp_path}")

            st.info(f"파일 처리 중: {file_name}")

            # --- 오디오 로드 ---
            audio_loader = AudioAdapter.default()
            target_sr = 44100 # Spleeter 모델은 보통 44.1kHz로 학습됨
            waveform = None
            sr = None
            with st.spinner(f"오디오 파일 로딩 중 '{file_name}'..."):
                try:
                    logging.info(f"오디오 로딩: {tmp_path} (Target SR: {target_sr}Hz)...")
                    waveform, sr = audio_loader.load(str(tmp_path), sample_rate=target_sr)
                    logging.info(f"오디오 로드 완료: shape={waveform.shape}, sample_rate={sr}Hz")
                    duration_seconds = waveform.shape[0] / sr if sr > 0 else 0
                    st.success(f"🎧 오디오 로드 완료 ({sr} Hz, 길이: {duration_seconds:.2f}초)")
                    if sr != target_sr:
                        st.warning(f"참고: 오디오가 원본 샘플링 레이트에서 {sr}Hz로 리샘플링되었습니다.")
                except Exception as e:
                    logging.error(f"오디오 파일 로드 실패 {tmp_path}: {e}", exc_info=True)
                    st.error(f"❌ 오디오 파일 로딩 오류: {e}. 유효한 오디오 파일인지 확인해주세요.")
                    temp_dir_input.cleanup()
                    return {}

            # --- 스템 분리 ---
            prediction = None
            with st.spinner("스템 분리 중 (시간이 소요될 수 있습니다)..."):
                try:
                    logging.info(f"스템 분리 시작: {file_name}...")
                    prediction = self.separator.separate(waveform) # 분리 실행
                    logging.info(f"스템 분리 완료: {file_name}. 분리된 스템: {list(prediction.keys())}")
                    st.success("🎶 스템 분리 완료!")
                except Exception as e:
                    logging.error(f"Spleeter 분리 중 오류 ({file_name}): {e}", exc_info=True)
                    st.error(f"❌ 분리 실패: {e}. 파일이 너무 길거나 복잡할 수 있습니다.")
                    temp_dir_input.cleanup()
                    return {}

            # --- 스템 추출, 처리, 저장 ---
            with st.spinner("스템 후처리 중 (노이즈 제거, 저장)..."):
                 logging.info(f"스템 추출 및 후처리 시작: {file_name}...")
                 result_files = self.extract_stems(file_stem, prediction, sr)
                 logging.info(f"스템 추출 완료: {file_name}. 결과 파일: {result_files.keys()}")


            # --- 임시 입력 파일 정리 ---
            logging.info(f"임시 입력 디렉토리 정리: {temp_dir_input.name}")
            temp_dir_input.cleanup()

            return result_files

        except Exception as e:
            logging.error(f"파일 처리 중 예외 발생 {file_name}: {e}", exc_info=True)
            st.error(f"❌ 처리 중 예기치 않은 오류 발생: {e}")
            if 'temp_dir_input' in locals() and os.path.exists(temp_dir_input.name):
                 temp_dir_input.cleanup()
            return {}

# --- 메인 Streamlit 앱 로직 ---

def main():
    st.title("🎵 오디오 스템 분리기 (5 Stems)")
    st.markdown("""
    오디오 파일을 업로드하면 5개의 스템(보컬, 드럼, 베이스, 피아노, 그 외 악기)으로 분리합니다.
    *   **보컬** 스템에는 자동으로 노이즈 제거가 적용됩니다.
    *   모든 결과 스템은 **모노 WAV** 파일로 제공됩니다.
    """)
    st.divider() # 구분선 추가

    # --- 사이드바 설정 ---
    with st.sidebar:
        st.header("ℹ️ 사용 방법 및 참고사항")
        st.markdown("""
        1.  **파일 선택:** 'Browse files' 버튼을 눌러 MP3, WAV 등의 오디오 파일을 선택합니다.
        2.  **처리 대기:** 앱이 오디오를 로드하고, Spleeter 모델을 사용해 스템을 분리하고, 보컬 노이즈를 제거합니다. 파일 길이에 따라 시간이 걸릴 수 있습니다.
        3.  **다운로드:** 처리가 완료되면 각 스템의 다운로드 링크와 미리듣기 플레이어가 나타납니다. 링크를 클릭하여 WAV 파일을 저장하세요.

        **참고사항:**
        *   **처리 시간:** 스템 분리는 계산량이 많습니다. 파일이 길수록 더 오래 걸립니다.
        *   **분리 품질:** 결과물의 품질은 원본 믹싱 상태와 모델 성능에 따라 달라집니다. 스템 간에 소리가 약간 섞일 수 있습니다.
        *   **모노 출력:** 모든 스템은 모노 WAV 파일로 변환됩니다.
        *   **노이즈 제거:** 보컬의 배경 잡음 감소를 위해 적용되며, 보컬 톤이 약간 변할 수 있습니다.
        *   **임시 파일:** 업로드 및 생성된 파일은 현재 세션 동안만 유지되며, 창을 닫거나 새로고침하면 **삭제됩니다.** 필요한 파일은 반드시 다운로드하세요.
        *   **모델:** `spleeter:5stems` 사전 훈련 모델을 사용합니다.
        """)
        st.divider()
        # st.info("Powered by Spleeter & Streamlit") # 필요시 추가

    # --- 메인 화면 설정 ---
    # 파일 크기 제한 설정 (예: 100MB) - 너무 크면 메모리/시간 문제 발생 가능
    max_file_size_mb = 100
    max_file_size_bytes = max_file_size_mb * 1024 * 1024

    # 세션 상태에 출력 디렉토리 유지 (새로고침해도 유지되도록)
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = tempfile.mkdtemp(prefix="stem_outputs_")
        logging.info(f"세션 출력 디렉토리 생성: {st.session_state.output_dir}")

    # 파일 업로더
    uploaded_file = st.file_uploader(
        f"오디오 파일 선택",
        type=["mp3", "wav", "flac", "m4a", "ogg", "aac"], # 지원 포맷 명시
        help=f"최대 파일 크기: {max_file_size_mb}MB. 긴 파일은 처리 시간이 오래 걸립니다."
    )

    if uploaded_file is not None:
        # 파일 크기 확인
        if uploaded_file.size > max_file_size_bytes:
            st.error(f"파일 '{uploaded_file.name}'의 크기가 너무 큽니다 ({uploaded_file.size / (1024*1024):.1f} MB). 허용 최대 크기: {max_file_size_mb} MB.")
            st.warning("더 작은 파일을 업로드하거나 긴 트랙을 나눠서 시도해보세요.")
        else:
            # 파일 체크 후 Extractor 초기화 (모델 로드/캐시 확인)
            try:
                # 출력 디렉토리는 세션 상태에서 가져옴
                extractor = MultiStemExtractor(st.session_state.output_dir)
            except Exception as e:
                 st.error("오디오 처리 엔진 초기화 실패. 진행할 수 없습니다.")
                 logging.critical("MultiStemExtractor 초기화 중 main 루프에서 오류 발생.", exc_info=True)
                 st.stop()

            start_time = time.time()

            # 파일 처리 실행
            result_files = extractor.process_file(uploaded_file)

            processing_time = time.time() - start_time
            logging.info(f"{uploaded_file.name} 처리 총 소요 시간: {processing_time:.2f} 초")

            if result_files:
                st.success(f"🎉 처리 완료! (소요 시간: {processing_time:.2f} 초)")
                st.subheader("분리된 스템 다운로드 (Mono WAV)")

                # 생성된 스템을 기준으로 동적 컬럼 생성 (최대 3개)
                available_stems = sorted(result_files.keys())
                num_columns = min(len(available_stems), 3)
                cols = st.columns(num_columns)

                col_idx = 0
                for stem in available_stems:
                    with cols[col_idx]:
                        file_path = result_files[stem]
                        # 다운로드 링크 생성
                        download_link = get_binary_file_downloader_html(file_path, os.path.basename(file_path))
                        st.markdown(download_link, unsafe_allow_html=True)
                        # 오디오 플레이어 추가 (미리듣기)
                        try:
                            with open(file_path, 'rb') as audio_file:
                                audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/wav')
                        except Exception as e:
                            logging.warning(f"{stem} 오디오 플레이어 생성 실패: {e}")
                            st.caption(f"({stem} 미리듣기 불가)")

                    col_idx = (col_idx + 1) % num_columns

                st.info("ℹ️ 다운로드 링크는 모노 WAV 파일을 제공합니다. 이 파일들은 임시 파일이며, 브라우저 탭을 닫거나 새로고침하면 사라집니다.")
            else:
                # result_files가 비어있으면 처리 중 어딘가에서 실패함
                st.error("처리 실패. 생성된 스템이 없습니다. 이전 메시지나 로그를 확인해주세요.")

    # ffmpeg 설치 여부 기본 확인 (Spleeter 백엔드에서 필요할 수 있음)
    # if os.system("ffmpeg -version > nul 2>&1") != 0 and os.system("ffmpeg -version > /dev/null 2>&1") != 0:
    #      st.sidebar.warning("⚠️ 경고: 시스템 경로에서 `ffmpeg`를 찾을 수 없습니다. 일부 오디오 포맷 처리 시 문제가 발생할 수 있습니다.")


if __name__ == "__main__":
    main()