from __future__ import annotations

import base64

import streamlit as st

def add_global_watermark():
    st.markdown(
        """
        <style>
        .global-watermark {
            position: fixed;
            left: 18px;
            bottom: 14px;
            right: auto;
            top: auto;

            z-index: 999999;
            padding: 8px 12px;
            border-radius: 10px;

            background: rgba(255, 255, 255, 0.72);
            color: rgba(40, 40, 40, 0.72);

            font-size: 12px;
            font-weight: 500;
            line-height: 1.35;
            text-align: left;

            pointer-events: none;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            backdrop-filter: blur(4px);
        }

        .global-watermark-line {
            display: block;
            white-space: nowrap;
        }

        @media (prefers-color-scheme: dark) {
            .global-watermark {
                background: rgba(30, 30, 30, 0.62);
                color: rgba(240, 240, 240, 0.72);
            }
        }
        </style>

        <div class="global-watermark">
            <span class="global-watermark-line">Bolu PCİ - Sistem Tasarım - Pişirme Laboratuvarı</span>
            <span class="global-watermark-line">Y&uuml;cel Can Aksu</span>
        </div>
        """,
        unsafe_allow_html=True
    )

def get_img_as_base64(file_path, thumb_width=300, thumb_height=200):
    try:
        from PIL import Image
        import io

        with Image.open(file_path) as img:
            img = img.convert("RGBA")
            img.thumbnail((thumb_width, thumb_height))

            canvas = Image.new("RGBA", (thumb_width, thumb_height), (255, 255, 255, 0))

            x = (thumb_width - img.width) // 2
            y = (thumb_height - img.height) // 2
            canvas.paste(img, (x, y), img)

            buffer = io.BytesIO()
            canvas.save(buffer, format="PNG", optimize=True)

        return base64.b64encode(buffer.getvalue()).decode()

    except FileNotFoundError:
        return None

# HTML Resim Ortala Fonksiyonu
def centered_local_img(file_path, width=150, height=100):
    img_b64 = get_img_as_base64(file_path, width * 2, height * 2)
    if img_b64:
        img_tag = f'<img src="data:image/png;base64,{img_b64}" width="{width}" height="{height}" style="border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); object-fit: cover;">'
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 15px;">
                {img_tag}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Görsel Yok: {file_path}")
