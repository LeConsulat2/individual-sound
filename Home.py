import streamlit as st

def main():
    # Configure the page
    st.set_page_config(page_title="Welcome Home", page_icon="🎵", layout="wide")

    # Main welcome title and introduction
    st.title("Welcome to the Audio Stem Separator App")
    st.write("""
    This innovative application leverages state-of-the-art machine learning techniques to separate your audio files into individual stems.
    Whether you want to isolate vocals, drums, bass, piano, or other instruments, our app makes it effortless.
    """)

    # Display key features of the app
    st.markdown("---")
    st.subheader("Key Features")
    st.markdown("""
    - **Advanced Stem Separation:** Extract individual components (vocals, drums, bass, piano, other) from any audio file.
    - **Noise Reduction:** Enhance the quality of your vocals with effective noise reduction.
    - **Easy Download:** Download each separated stem with a simple click.
    - **Multi-language Support:** Choose your preferred language (English or Korean) from the sidebar.
    """)

    # Sidebar language selection
    st.sidebar.header("Select Your Language")
    language = st.sidebar.selectbox("Language", ["English", "Korean"])

    # Display language-specific info and navigation link
    if language == "English":
        st.info("You have selected the **English** version. Click the link below to start using the app in English.")
        st.markdown("[Go to English Version](./English.py)")
    else:
        st.info("You have selected the **Korean** version. Click the link below to start using the app in Korean.")
        st.markdown("[Go to Korean Version](./Korean.py)")

    # Footer message
    st.markdown("---")
    st.write("Enjoy your experience and happy music processing!")

if __name__ == "__main__":
    main()
