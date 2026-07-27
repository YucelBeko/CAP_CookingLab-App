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

BREAD_ANALYSIS_VERSION = "bread_local_contrast_v2"

# -------------------------------------------------
# 1. EKMEK MASKELEME EŞİKLERİ - SABİT
# -------------------------------------------------
# Amaç:
# Siyah/gri zemin, glare/parlama ve arka planı dışarıda bırakıp
# sadece ekmek gövdesini yakalamak.

BREAD_MASK_MIN_AREA_RATIO = 0.015

# HSV
BREAD_MASK_V_MIN = 45
BREAD_MASK_V_MAX = 250
BREAD_MASK_S_MIN = 18

# LAB, OpenCV LAB:
# L: 0-255
# A0 = A - 128
# B0 = B - 128
BREAD_MASK_A_MIN = -8
BREAD_MASK_A_MAX = 52
BREAD_MASK_B_MIN = 5
BREAD_MASK_B_MAX = 85

# Parlama / glare dışlama
BREAD_GLARE_V_MIN = 225
BREAD_GLARE_S_MAX = 42
BREAD_GLARE_B_MAX = 14

# Siyah / gri arka plan dışlama
BREAD_BG_V_MAX = 55
BREAD_BG_S_MAX = 22
BREAD_BG_B_MAX = 8

# Morfoloji
BREAD_MASK_OPEN_KERNEL = 5
BREAD_MASK_CLOSE_KERNEL = 21
BREAD_MASK_CLOSE_ITER = 2


# -------------------------------------------------
# 2. ÇATLAK TESPİT EŞİKLERİ - SABİT
# -------------------------------------------------
# Ana mantık:
# Pikselin kendi çevresine göre daha açık olup olmadığına bakıyoruz.

BREAD_INNER_ERODE_KERNEL = 13

# Lokal ortalama için blur.
# Büyük olursa genel kabuk rengine göre kıyaslar.
# Küçük olursa küçük texture/pütürleri de çatlak sanabilir.
BREAD_LOCAL_BLUR_KERNEL = 71

# Çatlak için minimum şartlar
BREAD_CRACK_ABSOLUTE_L_MIN = 95
BREAD_CRACK_ABSOLUTE_V_MIN = 85

# Kritik eşik:
# Piksel, lokal çevresinden en az bu kadar açık olmalı.
BREAD_CRACK_DELTA_L_MIN = 8.0

# Kahverengilik farkı:
# Çatlak içi genelde çevresine göre daha az kızarmış/kahverengi olur.
BREAD_CRACK_DELTA_BROWN_MIN = 2.5

# Çok beyaz glare bölgelerini çatlak saymamak için.
BREAD_CRACK_GLARE_V_MIN = 232
BREAD_CRACK_GLARE_S_MAX = 32
BREAD_CRACK_GLARE_B_MAX = 13

# Speckle / pütür temizliği
BREAD_CRACK_MIN_AREA = 55
BREAD_CRACK_MIN_LONG_SIDE = 18

# Overlay
BREAD_CRACK_OVERLAY_ALPHA = 0.55
BREAD_CRACK_COLOR_BGR = (255, 255, 0)   # cyan
BREAD_OUTLINE_COLOR_BGR = (0, 255, 0)   # green


@dataclass
class BreadCrackReport:
    bread_pixels: int
    crack_pixels: int
    crack_pct: float
    mean_delta_l: float
    mean_delta_brown: float
    crack_score: float


def bread_keep_best_component(mask_u8):
    """
    En büyük komponenti körü körüne seçmek yerine,
    alan + merkeze yakınlık kombinasyonu ile ekmek gövdesini seçer.
    Siyah zemin/glare büyük alan oluşturursa bunu azaltır.
    """
    m = (mask_u8 > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)

    if n <= 1:
        return np.zeros_like(mask_u8)

    H, W = mask_u8.shape[:2]
    image_area = H * W
    min_area = image_area * BREAD_MASK_MIN_AREA_RATIO

    img_center = np.array([W / 2.0, H / 2.0], dtype=np.float32)
    max_dist = np.sqrt((W / 2.0) ** 2 + (H / 2.0) ** 2)

    best_label = None
    best_score = -1.0

    for lab_id in range(1, n):
        area = stats[lab_id, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        cx, cy = centroids[lab_id]
        comp_center = np.array([cx, cy], dtype=np.float32)

        dist = np.linalg.norm(comp_center - img_center)
        center_score = 1.0 - min(dist / max_dist, 1.0)

        # Alan yine önemli ama merkezdeki objeye öncelik veriyoruz.
        score = area * (0.45 + 0.55 * center_score)

        if score > best_score:
            best_score = score
            best_label = lab_id

    if best_label is None:
        return np.zeros_like(mask_u8)

    out = ((labels == best_label).astype(np.uint8) * 255)
    return out


def bread_fill_external_contour(mask_u8):
    """
    Komponent içindeki küçük boşlukları doldurur.
    Çatlaklar maskede delik gibi kalmasın diye dış konturu dolduruyoruz.
    """
    cnts, _ = cv2.findContours(
        (mask_u8 > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(mask_u8)

    if not cnts:
        return filled

    biggest = max(cnts, key=cv2.contourArea)
    cv2.drawContours(filled, [biggest], -1, 255, -1)

    return filled


def bread_build_surface_mask(bgr):
    """
    Ekmek yüzey maskesi.
    Bu maske çatlak analizi için ROI belirler.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H_ch, S_ch, V_ch = cv2.split(hsv)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L_ch = lab[..., 0]
    A0 = lab[..., 1] - 128
    B0 = lab[..., 2] - 128

    # Ekmek kabuğu: sıcak sarı/turuncu/kahverengi alan
    warm_bread = (
        (V_ch >= BREAD_MASK_V_MIN) &
        (V_ch <= BREAD_MASK_V_MAX) &
        (S_ch >= BREAD_MASK_S_MIN) &
        (A0 >= BREAD_MASK_A_MIN) &
        (A0 <= BREAD_MASK_A_MAX) &
        (B0 >= BREAD_MASK_B_MIN) &
        (B0 <= BREAD_MASK_B_MAX)
    )

    # Çatlak içi daha açık/pale olabilir; onu da ekmek maskesine dahil ediyoruz.
    pale_bread_crack = (
        (L_ch >= 95) &
        (V_ch >= 80) &
        (S_ch >= 10) &
        (A0 >= -12) &
        (A0 <= 40) &
        (B0 >= 4) &
        (B0 <= 65)
    )

    # Parlama/glare dışla.
    glare = (
        (V_ch >= BREAD_GLARE_V_MIN) &
        (S_ch <= BREAD_GLARE_S_MAX) &
        (B0 <= BREAD_GLARE_B_MAX)
    )

    # Siyah/gri arka plan dışla.
    background_gray_dark = (
        (V_ch <= BREAD_BG_V_MAX) |
        (
            (S_ch <= BREAD_BG_S_MAX) &
            (B0 <= BREAD_BG_B_MAX)
        )
    )

    mask = (warm_bread | pale_bread_crack) & (~glare) & (~background_gray_dark)
    mask_u8 = mask.astype(np.uint8) * 255

    # Morfoloji: ufak gürültüleri temizle, ekmek gövdesini kapat.
    open_kernel = np.ones(
        (BREAD_MASK_OPEN_KERNEL, BREAD_MASK_OPEN_KERNEL),
        np.uint8
    )

    close_kernel = np.ones(
        (BREAD_MASK_CLOSE_KERNEL, BREAD_MASK_CLOSE_KERNEL),
        np.uint8
    )

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel, iterations=BREAD_MASK_CLOSE_ITER)

    # En iyi komponenti seç.
    mask_u8 = bread_keep_best_component(mask_u8)

    # Dış konturu doldur.
    mask_u8 = bread_fill_external_contour(mask_u8)

    # Son küçük smoothing
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return mask_u8


def bread_masked_blur(channel_float, mask_bool, kernel_size):
    """
    Maskeli lokal ortalama.
    Normal GaussianBlur yaparsak arka plan sıfırları ortalamaya karışır.
    Bu yüzden numerator / denominator yapıyoruz.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    mask_f = mask_bool.astype(np.float32)

    numerator = cv2.GaussianBlur(
        channel_float * mask_f,
        (kernel_size, kernel_size),
        0
    )

    denominator = cv2.GaussianBlur(
        mask_f,
        (kernel_size, kernel_size),
        0
    )

    denominator = np.maximum(denominator, 1e-6)

    return numerator / denominator


def bread_filter_crack_components(crack_mask_u8):
    """
    Küçük pütür/noise alanlarını temizler.
    Çatlak genelde bağlı, şerit veya leke formunda olmalı.
    """
    m = (crack_mask_u8 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    if n <= 1:
        return np.zeros_like(crack_mask_u8)

    clean = np.zeros_like(crack_mask_u8)

    for lab_id in range(1, n):
        area = stats[lab_id, cv2.CC_STAT_AREA]
        w = stats[lab_id, cv2.CC_STAT_WIDTH]
        h = stats[lab_id, cv2.CC_STAT_HEIGHT]
        long_side = max(w, h)

        keep = (
            (area >= BREAD_CRACK_MIN_AREA) and
            (long_side >= BREAD_CRACK_MIN_LONG_SIDE)
        )

        if keep:
            clean[labels == lab_id] = 255

    return clean


def bread_detect_cracks(bgr, bread_mask_u8):
    """
    Lokal kontrast tabanlı çatlak tespiti.
    Mutlak açık renge değil, çevreye göre açıklık farkına bakar.
    """
    if bread_mask_u8 is None or np.count_nonzero(bread_mask_u8) == 0:
        empty = np.zeros(bgr.shape[:2], dtype=np.uint8)
        report = BreadCrackReport(
            bread_pixels=0,
            crack_pixels=0,
            crack_pct=0.0,
            mean_delta_l=0.0,
            mean_delta_brown=0.0,
            crack_score=0.0
        )
        return empty, report, empty, empty

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, S_ch, V_ch = cv2.split(hsv)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L_ch = lab[..., 0].astype(np.float32)
    A0 = (lab[..., 1] - 128).astype(np.float32)
    B0 = (lab[..., 2] - 128).astype(np.float32)

    bread_bool = bread_mask_u8 > 0

    # Kenarları biraz içeri alıyoruz.
    # Çünkü ekmek sınırı/parlama/keskin kontur çatlak gibi algılanabiliyor.
    erode_kernel = np.ones(
        (BREAD_INNER_ERODE_KERNEL, BREAD_INNER_ERODE_KERNEL),
        np.uint8
    )

    inner_mask_u8 = cv2.erode(
        bread_mask_u8,
        erode_kernel,
        iterations=1
    )

    inner_bool = inner_mask_u8 > 0

    if np.count_nonzero(inner_bool) == 0:
        inner_bool = bread_bool.copy()
        inner_mask_u8 = bread_mask_u8.copy()

    # Kahverengilik metriği.
    # A0 kırmızılık, B0 sarılık/kahverengilik gibi davranıyor.
    brownness = (0.70 * A0 + 0.55 * B0).astype(np.float32)

    local_L = bread_masked_blur(
        L_ch,
        bread_bool,
        BREAD_LOCAL_BLUR_KERNEL
    )

    local_brownness = bread_masked_blur(
        brownness,
        bread_bool,
        BREAD_LOCAL_BLUR_KERNEL
    )

    delta_L = L_ch - local_L
    delta_brown = local_brownness - brownness

    glare_inside = (
        (V_ch >= BREAD_CRACK_GLARE_V_MIN) &
        (S_ch <= BREAD_CRACK_GLARE_S_MAX) &
        (B0 <= BREAD_CRACK_GLARE_B_MAX)
    )

    # Ana çatlak adayı
    crack_candidate = (
        inner_bool &
        (~glare_inside) &
        (L_ch >= BREAD_CRACK_ABSOLUTE_L_MIN) &
        (V_ch >= BREAD_CRACK_ABSOLUTE_V_MIN) &
        (delta_L >= BREAD_CRACK_DELTA_L_MIN) &
        (delta_brown >= BREAD_CRACK_DELTA_BROWN_MIN)
    )

    crack_mask_u8 = crack_candidate.astype(np.uint8) * 255

    # Çok küçük boşlukları birleştir.
    close_kernel = np.ones((3, 3), np.uint8)
    crack_mask_u8 = cv2.morphologyEx(
        crack_mask_u8,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )

    # Speckle temizliği
    crack_mask_u8 = bread_filter_crack_components(crack_mask_u8)

    bread_pixels = int(np.count_nonzero(bread_bool))
    crack_pixels = int(np.count_nonzero(crack_mask_u8 > 0))

    crack_pct = 100.0 * crack_pixels / bread_pixels if bread_pixels else 0.0

    if crack_pixels > 0:
        crack_region = crack_mask_u8 > 0
        mean_delta_l = float(np.mean(delta_L[crack_region]))
        mean_delta_brown = float(np.mean(delta_brown[crack_region]))
    else:
        mean_delta_l = 0.0
        mean_delta_brown = 0.0

    # Basit birleşik skor.
    # Alan + belirginlik beraber değerlendiriliyor.
    # Bu skor şimdilik kalite karşılaştırma amaçlı.
    crack_score = (
        crack_pct * 4.5 +
        mean_delta_l * 1.4 +
        mean_delta_brown * 1.0
    )

    crack_score = float(np.clip(crack_score, 0, 100))

    report = BreadCrackReport(
        bread_pixels=bread_pixels,
        crack_pixels=crack_pixels,
        crack_pct=crack_pct,
        mean_delta_l=mean_delta_l,
        mean_delta_brown=mean_delta_brown,
        crack_score=crack_score
    )

    # Debug için normalize delta L görseli
    delta_l_vis = np.zeros_like(bread_mask_u8)
    if np.count_nonzero(bread_bool) > 0:
        dl = delta_L.copy()
        dl[~bread_bool] = 0
        dl_clip = np.clip((dl + 20) * (255 / 60), 0, 255)
        delta_l_vis = dl_clip.astype(np.uint8)

    return crack_mask_u8, report, inner_mask_u8, delta_l_vis


def bread_make_overlay(bgr, bread_mask_u8, crack_mask_u8):
    """
    Çatlakları cyan overlay, ekmek dış konturunu yeşil gösterir.
    """
    overlay = bgr.copy()

    crack_bool = crack_mask_u8 > 0

    color_layer = bgr.copy()
    color_layer[crack_bool] = BREAD_CRACK_COLOR_BGR

    blended = cv2.addWeighted(
        bgr,
        1.0 - BREAD_CRACK_OVERLAY_ALPHA,
        color_layer,
        BREAD_CRACK_OVERLAY_ALPHA,
        0
    )

    overlay[crack_bool] = blended[crack_bool]

    # Ekmek konturu
    cnts, _ = cv2.findContours(
        (bread_mask_u8 > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if cnts:
        cv2.drawContours(
            overlay,
            cnts,
            -1,
            BREAD_OUTLINE_COLOR_BGR,
            4,
            lineType=cv2.LINE_AA
        )

    return overlay


def bread_make_mask_preview(bgr, bread_mask_u8):
    """
    Maskenin doğru çalışıp çalışmadığını kontrol etmek için.
    Dışarıyı karartır, ekmek konturunu yeşil çizer.
    """
    preview = bgr.copy()

    mask_bool = bread_mask_u8 > 0

    darkened = (preview * 0.35).astype(np.uint8)
    preview[~mask_bool] = darkened[~mask_bool]

    cnts, _ = cv2.findContours(
        (bread_mask_u8 > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if cnts:
        cv2.drawContours(
            preview,
            cnts,
            -1,
            BREAD_OUTLINE_COLOR_BGR,
            4,
            lineType=cv2.LINE_AA
        )

    return preview


def bread_make_crack_only_preview(bgr, bread_mask_u8, crack_mask_u8):
    """
    Sadece çatlak maskesini beyaz zemin üzerinde gösterir.
    """
    H, W = bread_mask_u8.shape[:2]
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    bread_bool = bread_mask_u8 > 0
    crack_bool = crack_mask_u8 > 0

    # Ekmek alanını açık gri göster
    canvas[bread_bool] = (235, 235, 235)

    # Çatlakları cyan göster
    canvas[crack_bool] = BREAD_CRACK_COLOR_BGR

    cnts, _ = cv2.findContours(
        (bread_mask_u8 > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if cnts:
        cv2.drawContours(
            canvas,
            cnts,
            -1,
            BREAD_OUTLINE_COLOR_BGR,
            3,
            lineType=cv2.LINE_AA
        )

    return canvas

def analyze_bread_image(file_bytes, version=BREAD_ANALYSIS_VERSION):
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr is None:
        return None

    bread_mask_u8 = bread_build_surface_mask(bgr)

    if np.count_nonzero(bread_mask_u8) == 0:
        return None

    crack_mask_u8, report, inner_mask_u8, delta_l_vis = bread_detect_cracks(
        bgr,
        bread_mask_u8
    )

    overlay = bread_make_overlay(
        bgr,
        bread_mask_u8,
        crack_mask_u8
    )

    mask_preview = bread_make_mask_preview(
        bgr,
        bread_mask_u8
    )

    crack_only_preview = bread_make_crack_only_preview(
        bgr,
        bread_mask_u8,
        crack_mask_u8
    )

    return {
        "bgr": bgr,
        "bread_mask_u8": bread_mask_u8,
        "inner_mask_u8": inner_mask_u8,
        "crack_mask_u8": crack_mask_u8,
        "overlay": overlay,
        "mask_preview": mask_preview,
        "crack_only_preview": crack_only_preview,
        "delta_l_vis": delta_l_vis,
        "report": report
    }


def run_bread_surface():
    st.markdown(
        """
        <style>
        .block-container h1 { margin-top: -60px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Ekmek Yüzey Analizi")

    st.info(
        "Bu analizde önce ekmek yüzeyi maskelenir. "
        "Sonra çatlaklar, mutlak açık renge göre değil; kendi çevresine göre daha açık ve daha az kahverengi kalan bölgeler olarak hesaplanır."
    )

    uploads = st.file_uploader(
        "Ekmek görsellerini yükle",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="bread_surface_uploads"
    )

    if not uploads:
        st.info("Başlamak için ekmek görsellerini yükle.")
        return

    for idx, up in enumerate(uploads, start=1):
        file_bytes = up.getvalue()
        result = analyze_bread_image(file_bytes)

        st.markdown(f"## {idx}. Analiz: `{up.name}`")

        if result is None:
            st.warning("Görsel okunamadı veya ekmek yüzeyi maskelenemedi.")
            st.divider()
            continue

        bgr = result["bgr"]
        overlay = result["overlay"]
        mask_preview = result["mask_preview"]
        crack_only_preview = result["crack_only_preview"]
        report = result["report"]

        # -------------------------------------------------
        # 1. Satır: Orijinal / Çatlak Overlay
        # -------------------------------------------------
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.subheader("Orijinal")
            st.image(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        with col2:
            st.subheader("Çatlak Overlay")
            st.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        # -------------------------------------------------
        # 2. Satır: Maske debug
        # -------------------------------------------------
        debug_col1, debug_col2 = st.columns(2, gap="medium")

        with debug_col1:
            st.subheader("Ekmek Yüzey Maskesi")
            st.image(
                cv2.cvtColor(mask_preview, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        with debug_col2:
            st.subheader("Sadece Çatlak Maskesi")
            st.image(
                cv2.cvtColor(crack_only_preview, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        # -------------------------------------------------
        # 3. Satır: Özet
        # -------------------------------------------------
        st.subheader("Özet")

        summary_df = pd.DataFrame([{
            "Ekmek Alanı (px)": report.bread_pixels,
            "Çatlak Alanı (px)": report.crack_pixels,
            "Çatlak Yüzey (%)": round(report.crack_pct, 2),
            "Ortalama Lokal Açıklık Farkı": round(report.mean_delta_l, 2),
            "Ortalama Kahverengilik Farkı": round(report.mean_delta_brown, 2),
            "Çatlak Skoru / 100": round(report.crack_score, 1)
        }])

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Not: Çatlak Yüzey (%) alan bazlıdır. Çatlak Skoru ise alan + lokal açıklık farkı + kahverengilik farkını birlikte değerlendirir."
        )

        st.divider()
