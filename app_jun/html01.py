import streamlit as st

#card 
def card_info(title, subtitle1, subtitle2, text, link):

    st.markdown(
        f"""
    <div class="card-main" style="width: 100%;" "border-radius: 30px;">
    <div class="card-body">
        <h5 class="card-title" style="text-align: center;">{title}</h5>
        <h6 class="card-subtitle mb-2 text-muted" style="text-align: center;">{subtitle1}</h6>
        <h6 class="card-subtitle mb-2 text-muted" style="text-align: center;">{subtitle2}</h6>
        <p class="card-text" style="text-align: left;">{text}</p>
        <a href="#" class="card-link" style="text-align: center;">{link}</a>
    </div>
    </div>
        """, unsafe_allow_html=True
        )      












# KPI com cards bootstrap
# st.markdown(
#         f"""
#         <div class="card-deck" style= "-webkit-box-orient: horizontal;  width: 830px;" >

#         <div class="card">
#             <div class="card-body text-center">
#             <p class="card-text" style="font-size: 20px; color: rgb(167, 174, 177); font-weight: bold;">P/L</p>
#             <p class="card-text" style=" font-size: 40px">{fundamentus['P/L'][0]}</p>
#             </div>
#         </div>
#         <div class="card">
#             <div class="card-body text-center">
#             <p class="card-text" style="font-size: 20px; color: rgb(167, 174, 177); font-weight: bold;">P/VP</p>
#             <p class="card-text" style=" font-size: 40px;">{fundamentus['P/VP'][0]}</p>
#             </div>
#         </div>
#         <div class="card">
#             <div class="card-body text-center">
#             <p class="card-text" style="font-size: 20px;color: rgb(167, 174, 177); font-weight: bold; ">Recomendação</p>
#             <p class="card-text" style=" font-size: 40px; ">{info['recommendationKey']}</p>
#             </div>
#         </div>
#         <div class="card">
#             <div class="card-body text-center">
#             <p class="card-text" style="font-size: 20px;color: rgb(167, 174, 177); font-weight: bold;">Próximo dividendo</p>
#             <p class="card-text" style=" font-size: 20px;">{pfizer.calendar.transpose()['Earnings Date'].dt.strftime('%d/%m/%Y')[0]}</p>
#             </div>
#         </div>
#         </div>
#         """,unsafe_allow_html=True
# )
