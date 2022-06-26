import pandas as pd
import numpy as np
import streamlit as st
import cv2
import easyocr
import tempfile
import streamlit as st
from PIL import Image

def ocr_reader():
    
    st.title('OCR - Character reading from images')
    st.subheader('Read characters from images or videos!')

    # adiciona caixa de seleção à Sidebox
    formato = st.sidebar.selectbox("Choose format", ['Image','Video'])

    # carrega modelo ocr
    reader = easyocr.Reader(['en'], gpu=True)

    st_empty = st.empty()

    st_empty2 = st.empty()

    st.title(' ')

    if formato == 'Video':
            
        video_file = st.file_uploader("Attach a video",type = ['mp4'])

        #video teste
        if st.button('Try with our video'):
            #video_file = cv2.VideoCapture("images/video_test.mp4")

            # manipulação para deixar em formato legível para opencv
            #tfile = tempfile.NamedTemporaryFile(delete=False)
            #tfile.write(video_file.read())

            #vid_capture = cv2.VideoCapture(tfile.name)

            vid_capture = cv2.VideoCapture("images/video_test.mp4")

            #with st.spinner("Vídeo em análise"):

            list = []

            df_previsoes = pd.DataFrame()
            i = 0
            while True:
                print(i)
                i += 1

                conectado, frame = vid_capture.read()
                                    
                if not conectado:
                    break

                frame_cp = frame.copy()

                # aplica modelo ao frame
                resultados = reader.readtext(frame_cp,paragraph=False,rotation_info=[0,0,0])

                if resultados != []:
                    for (bbox, text, prob) in resultados:

                        print("{:.4f}: {}".format(prob, text))

                        # coordenadas da bounding box do OCR
                        (tl, tr, br, bl) = bbox
                        tl = (int(tl[0]), int(tl[1]))
                        tr = (int(tr[0]), int(tr[1]))
                        br = (int(br[0]), int(br[1]))
                        bl = (int(bl[0]), int(bl[1]))

                        # desenha retangulo na imagem
                        cv2.rectangle(frame_cp, tl, br, (0, 255, 0), 2)
                        # escreve previsão na imagem
                        cv2.putText(frame_cp, text, (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        list.append(str(text))

                    

                    df_previsoes = pd.DataFrame()
                    df_previsoes[f'frame: {i}'] = list
                    st.dataframe(df_previsoes)
                    st_empty.image(frame_cp)

            st_empty.image(frame_cp)

        #video usuário
        i = 0
        if video_file is not None:
           
            # manipulação para deixar em formato legível para opencv
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())

            vid_capture = cv2.VideoCapture(tfile.name)

            #with st.spinner("Vídeo em análise"):

            list = []

            df_previsoes = pd.DataFrame()
            while True:
                print(i)
                i += 1

                conectado, frame = vid_capture.read()
                                    
                if not conectado:
                    break

                frame_cp = frame.copy()

                # aplica modelo ao frame
                resultados = reader.readtext(frame_cp,paragraph=False,rotation_info=[0,0,0])

                if resultados != []:
                    for (bbox, text, prob) in resultados:

                        print("{:.4f}: {}".format(prob, text))

                        # coordenadas da bounding box do OCR
                        (tl, tr, br, bl) = bbox
                        tl = (int(tl[0]), int(tl[1]))
                        tr = (int(tr[0]), int(tr[1]))
                        br = (int(br[0]), int(br[1]))
                        bl = (int(bl[0]), int(bl[1]))

                        # desenha retangulo na imagem
                        cv2.rectangle(frame_cp, tl, br, (0, 255, 0), 2)
                        # escreve previsão na imagem
                        cv2.putText(frame_cp, text, (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        list.append(str(text))

                    

                    df_previsoes = pd.DataFrame()
                    df_previsoes[f'frame: {i}'] = list
                    st.dataframe(df_previsoes)
                    st_empty.image(frame_cp)

            st_empty.image(frame_cp)
            
    else:
        image = st.file_uploader(label = "Attach an image",type=['png','jpg','jpeg'])
        
        # botão para usar imagem de teste
        if st.button('Try with our image'):
            image_exemple = Image.open("images/container_exemplo_02.jpg")
            st.image(image_exemple) 

            input_img_arr = np.array(image_exemple)
            frame_cp = input_img_arr.copy()

            with st.spinner("Analyzing image"):
                # aplica modelo
                resultados = reader.readtext(frame_cp,paragraph=False,rotation_info=[0,0,0])

                list = []
                
                for (bbox, text, prob) in resultados:

                    print("{:.4f}: {}".format(prob, text))

                    # coordenadas da bounding box do OCR
                    (tl, tr, br, bl) = bbox
                    tl = (int(tl[0]), int(tl[1]))
                    tr = (int(tr[0]), int(tr[1]))
                    br = (int(br[0]), int(br[1]))
                    bl = (int(bl[0]), int(bl[1]))

                    # desenha retangulo e escreve texto do OCR no frame
                    cv2.rectangle(frame_cp, tl, br, (0, 255, 0), 2)
                    cv2.putText(frame_cp, text, (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    list.append(text)

                st.image(frame_cp)
                st.dataframe(list)

        #imagem do usuário
        if image is not None:
            
            # le e mostra imagem
            input_image = Image.open(image) 
            st.image(input_image) 

            input_img_arr = np.array(input_image)
            frame_cp = input_img_arr.copy()

            with st.spinner("Analyzing image"):
                # aplica modelo
                resultados = reader.readtext(frame_cp,paragraph=False,rotation_info=[0,0,0])

                list = []
                
                for (bbox, text, prob) in resultados:

                    print("{:.4f}: {}".format(prob, text))

                    # coordenadas da bounding box do OCR
                    (tl, tr, br, bl) = bbox
                    tl = (int(tl[0]), int(tl[1]))
                    tr = (int(tr[0]), int(tr[1]))
                    br = (int(br[0]), int(br[1]))
                    bl = (int(bl[0]), int(bl[1]))

                    # desenha retangulo e escreve texto do OCR no frame
                    cv2.rectangle(frame_cp, tl, br, (0, 255, 0), 2)
                    cv2.putText(frame_cp, text, (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    list.append(text)

                st.image(frame_cp)
                st.dataframe(list)