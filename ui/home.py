from __future__ import annotations

import os

import streamlit as st

from ui.layout import centered_local_img, get_img_as_base64
from ui.navigation import set_page

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LOGO_DIR = ROOT_DIR / "assets" / "logos"

def show_home_page():
    # Sidebar'ı gizle
    st.markdown("""<style>[data-testid="stSidebar"] {display: none;}</style>""", unsafe_allow_html=True)
    
    # --- LOGO ALANI ---
    img_b64 = get_img_as_base64("Lab_Logo.png")
    
    st.markdown("<div style='text-align: center; padding-top: 30px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    
    if img_b64:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{img_b64}" width="440" 
                     style="border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.15);">
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/1046/1046857.png", width=150)
        
    st.markdown("</div>", unsafe_allow_html=True)

    # --- BAŞLIK ALANI ---
    st.markdown(
        """
        <h1 style='text-align: center; color: #2C3E50; font-family: sans-serif; font-size: 3rem; margin-bottom: 10px;'>
            Pişirme Laboratuvarı
        </h1>
        <h3 style='text-align: center; color: #7F8C8D; font-weight: 300; margin-top: 0;'>
            Performans Analiz Paneli
        </h3>
        <br>
        <div style="text-align: center; max-width: 700px; margin: 0 auto; background-color: #f8f9fa; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef;">
            <p style='color: #555; font-size: 1.1rem; margin: 0; line-height: 1.6;'>
                 Aşağıdaki kategorilerden ürün seçimi yaparak 
                <b>pişme oranı, renk dağılımı ve kalite standartlarını</b> yapay zeka ile analiz edebilirsiniz.
            </p>
        </div>
        <br><br>
        """, 
        unsafe_allow_html=True
    )

    home_cards = [
    {"page": "Patates", "label": "Patates", "logo": "Patates_Logo.png", "emoji": "🍟"},
    {"page": "Pizza", "label": "Pizza", "logo": "Pizza_Logo.png", "emoji": "🍕"},
    {"page": "Börek", "label": "Börek", "logo": "Borek_Logo.png", "emoji": "🥐"},
    {"page": "Small Cake", "label": "Small Cake", "logo": "Smallcake_Logo.png", "emoji": "🧁"},
    {"page": "Pyro Cam", "label": "Pyro Cam", "logo": "PyroCam_Logo.png", "emoji": "🔥"},
    {"page": "Ekmek", "label": "Ekmek", "logo": "Bread_Logo.png", "emoji": "🍞"},
    {"page": "Data Merger", "label": "Data Merger", "logo": "DataMerger_Logo.png", "emoji": "📈"},
    {"page": "Teflon Blok","label": "Teflon Blok","logo": "TeflonBlock_Logo.png","emoji": "🌡️"},
    {"page": "Kurabiye","label": "Kurabiye","logo": "Cookie_Logo.png","emoji": "🍪"},
    {"page": "Unlu Disk","label": "Unlu Disk","logo": "FlourDisk_Logo.png","emoji": "⚪"},
]

    # Kart sayısı arttıkça satırları otomatik olarak 4'erli gruplara böl.
    cards_per_row = 4
    rows = [
        home_cards[i:i + cards_per_row]
        for i in range(0, len(home_cards), cards_per_row)
    ]

    for row_idx, row_cards in enumerate(rows):
        cols = st.columns(len(row_cards), gap="medium")

        for col, card in zip(cols, row_cards):
            with col:
                if os.path.exists(card["logo"]):
                    centered_local_img(card["logo"], width=150, height=100)
                else:
                    st.markdown(
                        f"""
                        <div style="
                            height:100px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            font-size:46px;
                        ">
                            {card["emoji"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.button(
                    card["label"],
                    use_container_width=True,
                    key=f"home_btn_{card['page']}",
                    on_click=set_page,
                    args=(card["page"],)
                )

        if row_idx < len(rows) - 1:
            st.write("")
