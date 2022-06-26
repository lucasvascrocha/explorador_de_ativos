import streamlit as st
from PIL import Image

def home():
    st.title('')               
    st.title('OCR - Character reading from images')
    st.subheader('Read characters from images or videos!')
    image = Image.open('images/home.png')
    st.image(image, use_column_width=True)

    st.subheader('What applications can I use for the character reader?')
    st.write('For any situation where you want to read characters through images or videos!')
    st.write('Examples of use are companies with a production line that need to read product codes, containers, parts, etc.')

    st.title('')
    st.subheader('How does it work?')
    st.write('The algorithm analyzes the video or image and can deliver photos with the reading of the characters and a file containing the texts found.')
    st.write('Files with character reading can be integrated with third-party tools or added to a database.')

    st.subheader("Try it. It's free!")
    st.write('Go to the "Character reading" tab and test with our data, or enter your video and test the application right now!')

        