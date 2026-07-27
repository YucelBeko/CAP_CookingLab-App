from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import zoom as ndimage_zoom

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    streamlit_image_coordinates = None

def run_potato():
    #st.title("Patates Kızartması Analizi")
    #up_files = st.file_uploader("Görsel yükle", type=["jpg","jpeg","png"], accept_multiple_files=True)
    #if not up_files:
        #st.info("Başlamak için görsel yükleyin."); return
    #for up in up_files:
        # ---------- Renkler ----------
        def hex_to_bgr(hex_code):
            h = hex_code.lstrip("#")
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return (b, g, r)

        CLASS_COLORS = {"dough": "#FF99FF", "cooked": "#FFFF66", "burnt": "#FF0000"}
        COLORS_BGR = {k: hex_to_bgr(v) for k, v in CLASS_COLORS.items()}

        # ---------- Maske ----------
        def chroma_mask_simple(bgr):
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
            A0, B0 = A.astype(np.float32) - 128, B.astype(np.float32) - 128
            C = np.sqrt(A0*A0 + B0*B0)
            chroma_mask = (C > 12)
            not_gray = (B > 125)
            not_dark = (L > 110)
            mask = (chroma_mask & not_gray & not_dark).astype(np.uint8) * 255
            return mask, L, A, B

        # ---------- KMeans Sınıflandırma ----------
        def classify_simple(L, A, B, mask, bgr):
            m = (mask > 0)
            cls = np.full(mask.shape, -1, np.int8)
            if not np.any(m):
                return cls

            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            H = hsv[...,0].astype(np.float32)
            S = hsv[...,1].astype(np.float32)
            V = hsv[...,2].astype(np.float32)
            Lf, Af, Bf = L.astype(np.float32), A.astype(np.float32), B.astype(np.float32)

            Hr = (H/180.0) * (2*np.pi)
            H_sin = np.sin(Hr); H_cos = np.cos(Hr)

            X = np.stack([Lf/255.0, Af/255.0, Bf/255.0,
                        S/255.0, V/255.0, H_sin, H_cos], axis=-1)[m].astype(np.float32)

            mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-6
            Xn = (X - mu) / sigma

            K = 4
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-3)
            _, labels, centers = cv2.kmeans(Xn, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
            C = centers * sigma + mu  # denormalize

            def metrics(c):
                Lc = c[0]*255; Ac = c[1]*255; Bc = c[2]*255
                Sc = c[3]*255; Vc = c[4]*255
                hs, hc = c[5], c[6]
                Hc = (np.degrees(np.arctan2(hs, hc)) % 360.0) / 2.0  # 0..180
                in_yellow = (15 <= Hc <= 35)
                return Lc, Ac, Bc, Sc, Vc, Hc, in_yellow

            km_to_class, chosen = {}, set()

            # ---- BURNT (daha net koyuluk + sarı dışı) ----
            burnt_scores = []
            for i, c in enumerate(C):
                Lc, Ac, Bc, Sc, Vc, Hc, in_y = metrics(c)
                s = 1.6*(255-Vc) + 1.2*Ac + 0.9*max(0, 160-Bc) + (80 if not in_y else -40)
                burnt_scores.append((s, i))
            burnt_idx = max(burnt_scores)[1]
            km_to_class[burnt_idx] = 2; chosen.add(burnt_idx)

            # ---- RAW (çok parlak + düşük S + sarı bant şart) ----
            raw_scores = []
            for i, c in enumerate(C):
                if i in chosen: continue
                Lc, Ac, Bc, Sc, Vc, Hc, in_y = metrics(c)
                s = 2.1*Vc - 1.8*Sc + (90 if in_y else -120)
                # kapı: çiğ ancak gerçekten parlak ve düşük S ve sarıdaysa
                if (Vc < 220) or (Sc > 90) or (not in_y):
                    s -= 9_999
                raw_scores.append((s, i))
            raw_idx = max(raw_scores)[1]
            km_to_class[raw_idx] = 0; chosen.add(raw_idx)

            # ---- COOKED (sarı bant + orta V,S) ----
            cooked_scores = []
            for i, c in enumerate(C):
                if i in chosen: continue
                Lc, Ac, Bc, Sc, Vc, Hc, in_y = metrics(c)
                s = (100 if in_y else -60) - 1.0*abs(Vc-185) - 0.8*abs(Sc-130) + 0.4*Bc
                cooked_scores.append((s, i))
            if cooked_scores:
                cooked_idx = max(cooked_scores)[1]
                km_to_class[cooked_idx] = 1; chosen.add(cooked_idx)

            # ---- Kalan kümeler için daha sıkı fallback ----
            for i, c in enumerate(C):
                if i in km_to_class: continue
                Lc, Ac, Bc, Sc, Vc, Hc, in_y = metrics(c)
                if (Vc <= 120) or ((Ac >= 155) and (Bc <= 140) and (Vc <= 190)) or ((not in_y) and (Sc >= 100) and (Vc <= 190)):
                    km_to_class[i] = 2  # burnt
                elif (Vc >= 210) and (Sc <= 80) and in_y:
                    km_to_class[i] = 0  # raw
                else:
                    km_to_class[i] = 1  # cooked

            # ---- Piksel etiketleri ----
            cls_vals = labels.flatten()
            rrcc = np.argwhere(m)
            for (r, c), lab in zip(rrcc, cls_vals):
                cls[r, c] = km_to_class[int(lab)]

            # =========================
            #  SON KAPI (pixel-level)
            # =========================
            hsv2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            H2 = hsv2[...,0].astype(np.float32)
            S2 = hsv2[...,1].astype(np.float32)
            V2 = hsv2[...,2].astype(np.float32)
            Hr = (H2/180.0) * (2*np.pi)
            in_yellow = ( (np.degrees(np.arctan2(np.sin(Hr), np.cos(Hr))) % 360.0)/2.0 >= 15 ) & \
                        ( (np.degrees(np.arctan2(np.sin(Hr), np.cos(Hr))) % 360.0)/2.0 <= 35 )

            # ÇİĞ için sıkı kapı: çok parlak + düşük S + sarı bant ZORUNLU
            bad_raw = (cls == 0) & ( (V2 < 215) | (S2 > 90) | (~in_yellow) )
            cls[bad_raw] = 1  # pişmişe çevir

            # YANIK için minimum koyuluk / (sarı dışı + kızarmış) şartı
            #bad_burnt = (cls == 2) & ~( (V2 <= 100) | ((S2 >= 100) & (~in_yellow) & (V2 <= 190)) )
            #cls[bad_burnt] = 1  # pişmişe çevir

            return cls
        # ---------- Heatmap ----------
        def create_heatmap(bgr, image_path):
            mask, L, A, B = chroma_mask_simple(bgr)
            cls = classify_simple(L, A, B, mask, bgr)
            heat_bgr = bgr.copy()
            mo = (mask > 0)
            heat_bgr[(cls == 0) & mo] = COLORS_BGR["dough"]
            heat_bgr[(cls == 1) & mo] = COLORS_BGR["cooked"]
            heat_bgr[(cls == 2) & mo] = COLORS_BGR["burnt"]
            return mask, cls, heat_bgr
        st.markdown(
            """
            <style>
            .block-container h1 {
                margin-top: -80px;   /* başlığın üst boşluğunu azaltır */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # ---------- Streamlit ----------
        st.title("Patates Kızartması Analizi")

        uploads = st.file_uploader("Görselleri yükle (tek/çoklu)", type=["jpg","jpeg","png"], accept_multiple_files=True)

        if uploads:
            for up in uploads:
                file_bytes = np.frombuffer(up.getvalue(), np.uint8)
                bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                assert bgr is not None, f"Görsel okunamadı: {up.name}"

                mask, cls, heat_bgr = create_heatmap(bgr, up.name)

                c1, c2 = st.columns(2, gap="small")
                with c1:
                    st.subheader("Orijinal")
                    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                with c2:
                    st.subheader("Isı Haritası")
                    st.image(cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

                # ---- Pie chart bu noktada ----
                counts = [
                    int(np.count_nonzero(cls == 0)),
                    int(np.count_nonzero(cls == 1)),
                    int(np.count_nonzero(cls == 2)),
                ]
                total = max(sum(counts), 1)
                perc = [100.0 * c / total for c in counts]

                labels = ["Undercooked", "Cooked", "Overcooked"]
                colors = [CLASS_COLORS["dough"], CLASS_COLORS["cooked"], CLASS_COLORS["burnt"]]

                fig, ax = plt.subplots(figsize=(2, 2), dpi=110)
                ax.pie(counts, labels=labels, colors=colors, startangle=90,
                    counterclock=False, wedgeprops=dict(edgecolor="white", linewidth=1),
                    autopct=lambda p: f"{p:.1f}%" if p > 0 else "")
                ax.axis("equal")
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

                st.markdown(
                    f"**Undercooked:** {perc[0]:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Cooked:** {perc[1]:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Overcooked:** {perc[2]:.1f}%"
                )

                st.divider()
        else:
            st.info("Başlamak için görsel/ler yükle.")
