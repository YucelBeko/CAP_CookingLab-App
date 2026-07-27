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

PYRO_DARK_V_MAX = 50
PYRO_MEDIUM_V_MAX = 130
PYRO_HAZE_S_MAX = 65
PYRO_OVERLAY_ALPHA = 0.45


@dataclass
class PyroStainReport:
    total_pixels: int
    dark_pixels: int
    medium_pixels: int
    haze_pixels: int
    clean_pixels: int

    @property
    def dark_pct(self) -> float:
        return 100.0 * self.dark_pixels / self.total_pixels if self.total_pixels else 0.0

    @property
    def medium_pct(self) -> float:
        return 100.0 * self.medium_pixels / self.total_pixels if self.total_pixels else 0.0

    @property
    def haze_pct(self) -> float:
        return 100.0 * self.haze_pixels / self.total_pixels if self.total_pixels else 0.0

    @property
    def soiling_pct(self) -> float:
        return self.dark_pct + self.medium_pct + self.haze_pct

    @property
    def clean_pct(self) -> float:
        return 100.0 * self.clean_pixels / self.total_pixels if self.total_pixels else 0.0


def pyro_build_polygon_mask(image: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], color=255)
    return mask


def pyro_order_polygon_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """4 noktayı merkez etrafında sıralar; self-crossing polygon riskini azaltır."""
    pts = np.array(points, dtype=np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    return [(int(pts[i, 0]), int(pts[i, 1])) for i in order]


def pyro_analyze_stains(image: np.ndarray, mask: np.ndarray) -> tuple[PyroStainReport, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)
    roi = mask > 0

    dark_mask = (v_ch < PYRO_DARK_V_MAX) & roi

    medium_mask = (
        (v_ch >= PYRO_DARK_V_MAX) &
        (v_ch < PYRO_MEDIUM_V_MAX) &
        roi & ~dark_mask
    )

    haze_mask = (
        (v_ch >= PYRO_MEDIUM_V_MAX) &
        (s_ch < PYRO_HAZE_S_MAX) &
        roi & ~dark_mask & ~medium_mask
    )

    clean_mask = roi & ~dark_mask & ~medium_mask & ~haze_mask

    report = PyroStainReport(
        total_pixels=int(np.sum(roi)),
        dark_pixels=int(np.sum(dark_mask)),
        medium_pixels=int(np.sum(medium_mask)),
        haze_pixels=int(np.sum(haze_mask)),
        clean_pixels=int(np.sum(clean_mask)),
    )

    color_layer = image.copy()
    color_layer[dark_mask] = (0, 0, 220)       # red: charring
    color_layer[medium_mask] = (0, 128, 255)   # orange: medium soiling
    color_layer[haze_mask] = (0, 210, 220)     # yellow: haze

    blended = cv2.addWeighted(image, 1.0 - PYRO_OVERLAY_ALPHA, color_layer, PYRO_OVERLAY_ALPHA, 0)
    output = image.copy()
    output[roi] = blended[roi]
    return report, output



@st.cache_data(show_spinner=False)
def _pyro_prepare_images(file_bytes, max_display_width=900):
    """Decode once and cache the full/display-resolution Pyro Cam images."""
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr is None:
        return None, None, None

    H, W = bgr.shape[:2]
    scale = min(max_display_width / W, 1.0)
    display_w = max(1, int(W * scale))
    display_h = max(1, int(H * scale))

    display_bgr = cv2.resize(
        bgr,
        (display_w, display_h),
        interpolation=cv2.INTER_AREA
    )

    return bgr, display_bgr, float(scale)


@st.fragment
def run_pyrocam():
    from PIL import Image

    st.markdown(
        """
        <style>
        .block-container h1 { margin-top: -60px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Pyro Cam Analizi")
    st.info(
        "Cam üzerindeki analiz bölgesini 4 nokta ile seç. "
        "Seçilen polygon içinde charring, medium soiling, haze ve clean oranları hesaplanır."
    )

    upload = st.file_uploader(
        "Pyro cam görseli yükle",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="pyrocam_upload"
    )

    if upload is None:
        st.info("Başlamak için pyro cam görseli yükle.")
        return

    file_bytes = upload.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    bgr, base_display_bgr, scale = _pyro_prepare_images(
        file_bytes,
        max_display_width=900
    )

    if bgr is None or base_display_bgr is None or scale is None:
        st.warning("Görsel okunamadı.")
        return

    H, W = bgr.shape[:2]
    display_h, display_w = base_display_bgr.shape[:2]

    st.subheader("1. Analiz Bölgesi Seçimi")
    st.caption(
        "Canvas üzerinde cam bölgesinin 4 köşesini işaretle. "
        "Köşeleri saat yönünde veya saat yönünün tersinde seçmek en temiz sonucu verir."
    )

    if streamlit_image_coordinates is None:
        st.error(
            "Pyro Cam bölge seçimi için `streamlit-image-coordinates` paketi gerekli. "
            "requirements.txt içine `streamlit-image-coordinates==0.1.9` ekleyip app'i yeniden deploy et."
        )
        return

    points_key = f"pyrocam_points_{file_hash}"

    if points_key not in st.session_state:
        st.session_state[points_key] = []
        
    last_click_key = f"pyrocam_last_click_{file_hash}"
    
    if last_click_key not in st.session_state:
        st.session_state[last_click_key] = None
        
    canvas_col, info_col = st.columns([3, 1], gap="medium")

    with info_col:
        st.markdown("### Seçim")
        st.write("Görselin üstüne 4 kere tıkla.")
        st.write("Köşeleri mümkünse saat yönünde seç.")
        st.write("Analiz sadece seçilen polygon içinde yapılır.")
        st.write(f"Seçilen nokta: **{len(st.session_state[points_key])} / 4**")

        if st.button("Noktaları Sıfırla", key=f"pyrocam_reset_{file_hash}"):
            st.session_state[points_key] = []
            st.session_state[last_click_key] = None
            st.rerun(scope="fragment")

    # Her tıklamada orijinal görseli tekrar decode/resize etmek yerine
    # cache'teki display kopyası üzerinde işaretleri çiz.
    display_bgr = base_display_bgr.copy()

    for idx, (x_orig, y_orig) in enumerate(st.session_state[points_key], start=1):
        x_disp = int(x_orig * scale)
        y_disp = int(y_orig * scale)

        cv2.circle(display_bgr, (x_disp, y_disp), 8, (0, 255, 0), -1)
        cv2.putText(
            display_bgr,
            str(idx),
            (x_disp + 10, y_disp - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    if len(st.session_state[points_key]) == 4:
        pts_disp = np.array(
            [[int(x * scale), int(y * scale)] for x, y in st.session_state[points_key]],
            dtype=np.int32
        ).reshape((-1, 1, 2))

        cv2.polylines(
            display_bgr,
            [pts_disp],
            isClosed=True,
            color=(0, 255, 0),
            thickness=3
        )

    display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
    display_pil = Image.fromarray(display_rgb).convert("RGB")

    with canvas_col:
        clicked = streamlit_image_coordinates(
            display_pil,
            key=f"pyrocam_click_{file_hash}"
        )

    if clicked is not None and len(st.session_state[points_key]) < 4:
        x_display = int(clicked["x"])
        y_display = int(clicked["y"])
    
        click_signature = (x_display, y_display)
    
        if st.session_state[last_click_key] != click_signature:
            st.session_state[last_click_key] = click_signature
    
            x_original = int(x_display / scale)
            y_original = int(y_display / scale)
    
            x_original = max(0, min(W - 1, x_original))
            y_original = max(0, min(H - 1, y_original))
    
            st.session_state[points_key].append((x_original, y_original))
            st.rerun(scope="fragment")

    points = st.session_state[points_key]

    st.write(f"Seçilen nokta sayısı: **{len(points)} / 4**")

    if len(points) != 4:
        st.warning("Analiz için tam 4 nokta seçmelisin.")
        return

    ordered_points = pyro_order_polygon_points(points)
    preview = bgr.copy()
    pts = np.array(ordered_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(preview, [pts], isClosed=True, color=(0, 255, 0), thickness=5)

    st.subheader("2. Seçilen Bölge Önizleme")
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

    if not st.button("Pyro Cam Analizini Çalıştır", use_container_width=True):
        return

    mask = pyro_build_polygon_mask(bgr, ordered_points)
    report, overlay = pyro_analyze_stains(bgr, mask)

    st.subheader("3. Analiz Sonucu")

    row1_col1, row1_col2 = st.columns(2, gap="medium")

    with row1_col1:
        st.subheader("Seçilen Bölge")
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

    with row1_col2:
        st.subheader("Kir Analizi Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

    row2_col1, row2_col2 = st.columns(2, gap="medium")

    with row2_col1:
        st.subheader("Kir Dağılımı")

        labels = ["Charring", "Medium Soiling", "Haze", "Clean"]
        counts = [report.dark_pixels, report.medium_pixels, report.haze_pixels, report.clean_pixels]
        colors = ["#DC0000", "#FF8000", "#FFD200", "#D9D9D9"]

        fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=100)
        total_count = max(sum(counts), 1)

        filtered_counts = []
        filtered_colors = []
        legend_labels = []

        for label, count, color in zip(labels, counts, colors):
            if count > 0:
                pct = 100.0 * count / total_count
                filtered_counts.append(count)
                filtered_colors.append(color)
                legend_labels.append(f"{label}  %{pct:.1f}")

        def autopct_func(pct):
            return f"%{pct:.1f}" if pct >= 5 else ""

        wedges, texts, autotexts = ax.pie(
            filtered_counts,
            labels=None,
            colors=filtered_colors,
            startangle=90,
            counterclock=False,
            autopct=autopct_func,
            pctdistance=0.72,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
            textprops=dict(fontsize=9, color="black")
        )

        for t in autotexts:
            t.set_fontsize(9)
            t.set_fontweight("bold")

        ax.set_title("Pyro Cam Kir Dağılımı", fontsize=11, pad=8)
        ax.axis("equal")
        ax.legend(wedges, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, fontsize=8)
        fig.subplots_adjust(bottom=0.28, top=0.88)

        st.pyplot(fig, clear_figure=True, use_container_width=True)
        plt.close(fig)

    with row2_col2:
        st.subheader("Özet")
        summary_df = pd.DataFrame([{
            "Analiz Alanı (px)": report.total_pixels,
            "Charring (%)": round(report.dark_pct, 2),
            "Medium Soiling (%)": round(report.medium_pct, 2),
            "Haze (%)": round(report.haze_pct, 2),
            "Toplam Kir (%)": round(report.soiling_pct, 2),
            "Clean (%)": round(report.clean_pct, 2)
        }])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
