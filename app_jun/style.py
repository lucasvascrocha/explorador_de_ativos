import base64
import streamlit as st

from streamlit_option_menu import option_menu

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

#esconder botões do streamlit
def hidden_menu_and_footer():
    hide_menu = '''
    <style>
    #MainMenu {
        visibility:hidden;
    }

    footer{
        visibility:hidden;
    }

    </style>
    '''
    st.markdown(hide_menu, unsafe_allow_html=True)

#linha no cabeçalho branca desing
def headerstyle():
    st.markdown(
    f"""
    <nav class="navbar fixed-top navbar-light bg-white" style="color: #ffffff; padding: 0.8rem 1rem;">
        <span class="navbar-brand mb-0 h1" " >  </span>
    </nav>
    """, unsafe_allow_html=True
    )


#espaço entre plots
def space(tamanho):
    if tamanho == 1:
        st.title('')
    if tamanho == 2:
        st.header('') 
    if tamanho == 3:
        st.write('') 

def sidebarwidth():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 250px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 250px;
            margin-left: -500px;
        }
        </style>
        """,
        unsafe_allow_html=True,
        )    