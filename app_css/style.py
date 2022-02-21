import base64
import streamlit as st

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

def set_css():
    css_settings ='''
    <style>

    p{
        color:rgb(0, 0, 0);
        text-align: center;
        font-size: 14px;
        font-family: 'Bree Serif', serif;
    }

    .css-ocqkz7 {
        display: flex;
        flex-wrap: wrap;
        -webkit-box-flex: 1;
        flex-grow: 1;
        -webkit-box-align: stretch;
        align-items: stretch;
        gap: 1rem;
        border-radius: 30px;
        background: rgb(240, 238, 238);
    }

    .icon {  
    float: right;
    font-size:500%;
    position: absolute;
    top:0rem;
    right:-0.3rem;
    opacity: .16;
    }

    {
    width: 800px;
    display: flex;
    }


    .kpi-card
    {
    overflow: hidden;
    position: relative;
    box-shadow: 1px 1px 3px rgba(0,0,0,0.75);;
    display: inline-block;
    float: left;
    padding: 1em;
    border-radius: 0.3em;
    font-family: sans-serif;  
    width: 240px;
    min-width: 180px;
    margin-left: 0.5em;
    margin-top: 0.5em;
    }

    .card-value {
    display: block;
    font-size: 200%;  
    font-weight: bolder;
    }

    .card-text {
    display:block;
    font-size: 70%;
    padding-left: 0.2em;
    }

    .card-deck {
        -webkit-box-orient: horizontal;
        -webkit-box-direction: normal;
        -ms-flex-flow: row wrap;
        flex-flow: row wrap;
        margin-right: 0px;
        margin-left: -15px;
        width: 830px;

    }

    </style>
    '''
    st.markdown(css_settings, unsafe_allow_html=True)