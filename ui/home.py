from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.layout import centered_local_img, get_img_as_base64
from ui.navigation import set_page


# ---------------------------------------------------------
# DOSYA YOLLARI
# ---------------------------------------------------------

# ui/home.py -> ui -> repo ana klasörü
ROOT_DIR = Path(__file__).resolve().parents[1]

# repo/assets/logos
LOGO_DIR = ROOT_DIR / "assets" / "logos"


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

def show_home_page():
    # Ana sayfada sidebar gizlenir
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # ANA LOGO
    # -----------------------------------------------------

    lab_logo_path = LOGO_DIR / "Lab_Logo.png"

    img_b64 = get_img_as_base64(
        str(lab_logo_path),
        thumb_width=700,
        thumb_height=350
    )

    st.markdown(
        """
        <div style="
            text-align:center;
            padding-top:20px;
            margin-bottom:20px;
        ">
        """,
        unsafe_allow_html=True
    )

    if img_b64:
        st.markdown(
            f"""
            <div style="
                display:flex;
                justify-content:center;
                align-items:center;
            ">
                <img
                    src="data:image/png;base64,{img_b64}"
                    width="440"
                    style="
                        object-fit:contain;
                        border-radius:20px;
                    "
                >
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                height:180px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:90px;
            ">
                🧪
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # BAŞLIK
    # -----------------------------------------------------

    st.markdown(
        """
        <h1 style="
            text-align:center;
            color:#2C3E50;
            font-family:sans-serif;
            font-size:3rem;
            margin-bottom:10px;
        ">
            Pişirme Laboratuvarı
        </h1>

        <h3 style="
            text-align:center;
            color:#7F8C8D;
            font-weight:300;
            margin-top:0;
        ">
            Performans Analiz Paneli
        </h3>

        <br>

        <div style="
            text-align:center;
            max-width:700px;
            margin:0 auto;
            background-color:#f8f9fa;
            padding:20px;
            border-radius:12px;
            border:1px solid #e9ecef;
        ">
            <p style="
                color:#555;
                font-size:1.1rem;
                margin:0;
                line-height:1.6;
            ">
                Aşağıdaki kategorilerden birini seçerek
                <b>pişme oranı, renk dağılımı ve performans sonuçlarını</b>
                analiz edebilirsiniz.
            </p>
        </div>

        <br><br>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # KARTLAR
    # -----------------------------------------------------

    home_cards = [
        {
            "page": "Patates",
            "label": "Patates",
            "logo": "Patates_Logo.png",
            "emoji": "🍟"
        },
        {
            "page": "Pizza",
            "label": "Pizza",
            "logo": "Pizza_Logo.png",
            "emoji": "🍕"
        },
        {
            "page": "Börek",
            "label": "Börek",
            "logo": "Borek_Logo.png",
            "emoji": "🥐"
        },
        {
            "page": "Small Cake",
            "label": "Small Cake",
            "logo": "Smallcake_Logo.png",
            "emoji": "🧁"
        },
        {
            "page": "Pyro Cam",
            "label": "Pyro Cam",
            "logo": "PyroCam_Logo.png",
            "emoji": "🔥"
        },
        {
            "page": "Ekmek",
            "label": "Ekmek",
            "logo": "Bread_Logo.png",
            "emoji": "🍞"
        },
        {
            "page": "Data Merger",
            "label": "Data Merger",
            "logo": "DataMerger_Logo.png",
            "emoji": "📈"
        },
        {
            "page": "Teflon Blok",
            "label": "Teflon Blok",
            "logo": "TeflonBlock_Logo.png",
            "emoji": "🌡️"
        },
        {
            "page": "Kurabiye",
            "label": "Kurabiye",
            "logo": "Cookie_Logo.png",
            "emoji": "🍪"
        },
        {
            "page": "Unlu Disk",
            "label": "Unlu Disk",
            "logo": "FlourDisk_Logo.png",
            "emoji": "⚪"
        }
    ]

    # Her satırda en fazla 4 kart
    cards_per_row = 4

    for start_index in range(
        0,
        len(home_cards),
        cards_per_row
    ):
        row_cards = home_cards[
            start_index:start_index + cards_per_row
        ]

        columns = st.columns(
            cards_per_row,
            gap="medium"
        )

        for column_index, column in enumerate(columns):
            with column:
                # Son satırdaki boş kolonlar
                if column_index >= len(row_cards):
                    st.empty()
                    continue

                card = row_cards[column_index]

                logo_path = LOGO_DIR / card["logo"]

                if logo_path.exists():
                    centered_local_img(
                        str(logo_path),
                        width=150,
                        height=100
                    )
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

        st.write("")
