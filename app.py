import streamlit as st

def main():
    # Configure the page
    st.set_page_config(page_title="Welcome Home", page_icon="🎵", layout="wide")

    # Main welcome title and introduction
    st.title("🎵 Audio Stem Separator")
    st.markdown("Separate vocals, instruments, and more — with just a few clicks.")

    st.markdown("---")

    # ❗ IMPORTANT NOTICE
    st.markdown("### ⚠️ Please Note")
    st.markdown("""
    **📌 No audios or stems are saved on the server.**  
    You must **download the separated files** after processing.  
    If you refresh or leave the page, the files will be **deleted** permanently.
    """, unsafe_allow_html=True)

    st.markdown("""
    **📌 업로드한 오디오나 분리된 파일은 서버에 저장되지 않습니다.**  
    **처리 완료 후 반드시 다운로드** 해주세요.  
    새로고침하거나 페이지를 벗어나면 **모든 파일이 삭제됩니다.**
    """, unsafe_allow_html=True)

    st.markdown("---")

    # English Description
    st.subheader("🔹 What is this app?")
    st.markdown("""
    This application uses state-of-the-art machine learning to extract individual **stems** from an audio file.  
    Upload a song, and you’ll get separate tracks for **vocals**, **drums**, **bass**, **piano**, and more.

    - 🎧 Clean separation  
    - 🚀 Fast processing  
    - 💾 Easy downloads  
    """)

    # Korean Description
    st.subheader("🔹 이 앱은 무엇인가요?")
    st.markdown("""
    이 앱은 최신 머신러닝 기술을 활용하여 오디오 파일에서 **보컬**, **드럼**, **베이스**, **피아노** 등을 분리해줍니다.  
    노래를 업로드하면 각 악기별로 나눠진 트랙을 받을 수 있어요.

    - 🎧 깨끗한 분리  
    - 🚀 빠른 처리 속도  
    - 💾 간편한 다운로드  
    """)

    st.markdown("---")

    # Navigation note
    st.markdown("👈 Choose your working language from the **sidebar menu** (English / Korean) to get started.")
    st.markdown("👈 사이드바 메뉴(English / Korean)에서 원하는 언어를 선택해 시작하세요.")

    st.markdown("---")
    st.caption("Made with ❤️ by Jonathan. Happy music processing!")

if __name__ == "__main__":
    main()
