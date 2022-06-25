mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"lucas.vasconcelos3@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[theme]\n\
primaryColor = '#FF4B4B'\n\
backgroundColor = '#FFFFFF'\n\
secondaryBackgroundColor = '#FFFFFF'\n\
textColor = '#31333F'\n\
font = 'serif'\n\

[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml


