from __future__ import annotations

from dataclasses import dataclass

import cv2
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

TOAST_ANALYSIS_VERSION = "toast_coverage_homogeneity_v1"

# -------------------------------------------------
# 1. TOAST MASKING THRESHOLDS
# -------------------------------------------------
# Goal: separate toast bread pixels from a bright, near-gray background.
# The background stays low-saturation even where vignetting darkens it,
# while toast (raw, toasted, or burnt) keeps a warm, saturated color.
# This makes saturation a more reliable separator here than brightness alone.

TOAST_MASK_S_MIN = 30
TOAST_MASK_V_MAX = 225

TOAST_MASK_OPEN_KERNEL = 5
TOAST_MASK_CLOSE_KERNEL = 15
TOAST_MASK_CLOSE_ITER = 2

# Any connected component smaller than this fraction of the image area is
# treated as noise and dropped. Multiple toast slices in one tray are fine;
# only isolated speckles are removed.
TOAST_MASK_MIN_AREA_RATIO = 0.001

# Small cleanup kernel applied to the cooked/uncooked split to suppress
# speckle noise from dark seeds inside the crumb.
TOAST_CLASS_OPEN_KERNEL = 3

# -------------------------------------------------
# 2. COOKED / UNCOOKED CLASSIFICATION
# -------------------------------------------------
# Ry = (L* / 255) * 100, matching the convention already used elsewhere
# in this codebase (see sections/smallcake.py). Lower Ry means darker,
# more toasted; higher Ry means paler, less toasted. A pixel is
# classified as cooked when its Ry falls at or below the threshold.

TOAST_DEFAULT_RY_THRESHOLD = 52.0
TOAST_RY_THRESHOLD_MIN = 20.0
TOAST_RY_THRESHOLD_MAX = 80.0

# -------------------------------------------------
# 3. HOMOGENEITY SCORING
# -------------------------------------------------
# Homogeneity is scored from the 90th percentile absolute deviation of Ry
# from its mean, then normalized against a reference spread. A P90
# deviation at or beyond the reference spread scores 0; a perfectly even
# region scores 100. These reference values were calibrated against a
# sample toast tray image and are meant to be re-tuned once production
# data is available.
TOAST_HOMOGENEITY_RY_LIMIT_COOKED = 18.0
TOAST_HOMOGENEITY_RY_LIMIT_OVERALL = 35.0

# Overlay colors (BGR)
TOAST_COOKED_COLOR_BGR = (0, 140, 255)      # orange
TOAST_UNCOOKED_COLOR_BGR = (255, 220, 130)  # pale blue
TOAST_OVERLAY_ALPHA = 0.65
TOAST_OUTLINE_COLOR_BGR = (0, 255, 0)       # green


@dataclass
class ToastBreadReport:
    toast_pixels: int
    cooked_pixels: int
    uncooked_pixels: int
    cooked_pct: float
    ry_threshold: float
    mean_ry_overall: float
    mean_ry_cooked: float
    cooked_region_homogeneity: float
    overall_homogeneity: float


def toast_build_mask(bgr):
    """
    Builds a binary mask of all toast bread pixels against the white/gray
    background. Every sufficiently large connected component is kept,
    since a tray can hold any number of slices and slice boundaries are
    not relevant to this analysis.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[..., 1].astype(np.float32)
    v_channel = hsv[..., 2].astype(np.float32)

    raw_mask = (
        (s_channel >= TOAST_MASK_S_MIN) &
        (v_channel <= TOAST_MASK_V_MAX)
    ).astype(np.uint8) * 255

    open_kernel = np.ones((TOAST_MASK_OPEN_KERNEL, TOAST_MASK_OPEN_KERNEL), np.uint8)
    close_kernel = np.ones((TOAST_MASK_CLOSE_KERNEL, TOAST_MASK_CLOSE_KERNEL), np.uint8)

    mask_u8 = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(
        mask_u8, cv2.MORPH_CLOSE, close_kernel, iterations=TOAST_MASK_CLOSE_ITER
    )

    h, w = mask_u8.shape[:2]
    min_area = TOAST_MASK_MIN_AREA_RATIO * h * w

    n_components, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8), connectivity=8
    )

    clean_mask = np.zeros_like(mask_u8)
    for component_id in range(1, n_components):
        if stats[component_id, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == component_id] = 255

    return clean_mask


def toast_compute_ry_map(bgr):
    """
    Returns the per-pixel Ry map (0-100), derived from the OpenCV LAB L
    channel using the same scaling convention used elsewhere in this
    codebase.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[..., 0]
    return (l_channel / 255.0) * 100.0


def toast_classify_cooked(ry_map, mask_bool, ry_threshold):
    """
    Splits the toast mask into cooked (Ry <= threshold) and uncooked
    (Ry > threshold) pixels, with a light morphological cleanup to
    suppress speckle noise from dark seeds in the crumb.
    """
    cooked_raw = mask_bool & (ry_map <= ry_threshold)
    uncooked_raw = mask_bool & (ry_map > ry_threshold)

    open_kernel = np.ones(
        (TOAST_CLASS_OPEN_KERNEL, TOAST_CLASS_OPEN_KERNEL), np.uint8
    )

    cooked_u8 = cv2.morphologyEx(
        cooked_raw.astype(np.uint8) * 255, cv2.MORPH_OPEN, open_kernel
    )
    uncooked_u8 = cv2.morphologyEx(
        uncooked_raw.astype(np.uint8) * 255, cv2.MORPH_OPEN, open_kernel
    )

    return cooked_u8 > 0, uncooked_u8 > 0


def toast_homogeneity_score(values, spread_limit):
    """
    Homogeneity score in [0, 100] based on the 90th percentile absolute
    deviation from the mean. A tighter distribution scores higher.
    Returns (score, mean, p90_deviation, std).
    """
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    mean_value = float(np.mean(values))
    p90_deviation = float(np.percentile(np.abs(values - mean_value), 90))
    std_value = float(np.std(values))

    score = 100.0 * (1.0 - p90_deviation / spread_limit)
    score = float(np.clip(score, 0, 100))

    return score, mean_value, p90_deviation, std_value


def toast_make_overlay(bgr, mask_bool, cooked_bool, uncooked_bool):
    """
    Cooked pixels are tinted warm (orange), uncooked pixels are tinted
    pale blue. The outer toast contour is drawn in green for a quick
    visual QC of the mask.
    """
    color_layer = bgr.copy()
    color_layer[cooked_bool] = TOAST_COOKED_COLOR_BGR
    color_layer[uncooked_bool] = TOAST_UNCOOKED_COLOR_BGR

    blended = cv2.addWeighted(
        bgr, 1.0 - TOAST_OVERLAY_ALPHA, color_layer, TOAST_OVERLAY_ALPHA, 0
    )

    overlay = bgr.copy()
    overlay[mask_bool] = blended[mask_bool]

    mask_u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cv2.drawContours(overlay, contours, -1, TOAST_OUTLINE_COLOR_BGR, 3, lineType=cv2.LINE_AA)

    return overlay


def toast_make_ry_heatmap(ry_map, mask_bool):
    """
    Continuous Ry heatmap for visual inspection of homogeneity, on a
    white canvas outside the toast mask. Darker (more toasted) areas map
    to hotter colors.
    """
    h, w = ry_map.shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    ry_norm = np.clip((ry_map - 20.0) / (80.0 - 20.0), 0.0, 1.0)
    ry_u8 = (ry_norm * 255).astype(np.uint8)

    # Invert before colormapping so darker/more-toasted pixels read as
    # hotter colors under COLORMAP_JET.
    heat = cv2.applyColorMap(255 - ry_u8, cv2.COLORMAP_JET)
    canvas[mask_bool] = heat[mask_bool]

    return canvas


def analyze_toast_image(file_bytes, ry_threshold=TOAST_DEFAULT_RY_THRESHOLD):
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr is None:
        return None

    mask_u8 = toast_build_mask(bgr)
    mask_bool = mask_u8 > 0

    if np.count_nonzero(mask_bool) == 0:
        return None

    ry_map = toast_compute_ry_map(bgr)

    cooked_bool, uncooked_bool = toast_classify_cooked(ry_map, mask_bool, ry_threshold)

    toast_pixels = int(np.count_nonzero(mask_bool))
    cooked_pixels = int(np.count_nonzero(cooked_bool))
    uncooked_pixels = int(np.count_nonzero(uncooked_bool))
    cooked_pct = 100.0 * cooked_pixels / toast_pixels if toast_pixels else 0.0

    ry_overall_values = ry_map[mask_bool]
    ry_cooked_values = ry_map[cooked_bool]

    cooked_homogeneity, mean_ry_cooked, _, _ = toast_homogeneity_score(
        ry_cooked_values, TOAST_HOMOGENEITY_RY_LIMIT_COOKED
    )
    overall_homogeneity, mean_ry_overall, _, _ = toast_homogeneity_score(
        ry_overall_values, TOAST_HOMOGENEITY_RY_LIMIT_OVERALL
    )

    report = ToastBreadReport(
        toast_pixels=toast_pixels,
        cooked_pixels=cooked_pixels,
        uncooked_pixels=uncooked_pixels,
        cooked_pct=cooked_pct,
        ry_threshold=float(ry_threshold),
        mean_ry_overall=mean_ry_overall,
        mean_ry_cooked=mean_ry_cooked,
        cooked_region_homogeneity=cooked_homogeneity,
        overall_homogeneity=overall_homogeneity,
    )

    overlay = toast_make_overlay(bgr, mask_bool, cooked_bool, uncooked_bool)
    ry_heatmap = toast_make_ry_heatmap(ry_map, mask_bool)

    return {
        "bgr": bgr,
        "mask_u8": mask_u8,
        "ry_map": ry_map,
        "cooked_bool": cooked_bool,
        "uncooked_bool": uncooked_bool,
        "overlay": overlay,
        "ry_heatmap": ry_heatmap,
        "report": report,
    }


def run_toast():
    st.markdown(
        """
        <style>
        .block-container h1 { margin-top: -60px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Tost Ekmeği Analizi")

    st.info(
        "Tost, beyaz/gri arkaplandan ayrılır. Her piksel Ry (L*) değerine göre "
        "pişmiş veya pişmemiş olarak sınıflandırılır. Ayrıca hem pişen bölgenin "
        "kendi içindeki, hem de tüm tost yüzeyindeki homojenlik ayrı ayrı raporlanır."
    )

    ry_threshold = st.slider(
        "Pişmiş / Pişmemiş Ry eşiği (bu değerin altı pişmiş kabul edilir)",
        min_value=TOAST_RY_THRESHOLD_MIN,
        max_value=TOAST_RY_THRESHOLD_MAX,
        value=TOAST_DEFAULT_RY_THRESHOLD,
        step=1.0,
        key="toast_ry_threshold",
    )

    uploads = st.file_uploader(
        "Tost görsellerini yükle",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="toast_uploads",
    )

    if not uploads:
        st.info("Başlamak için tost görsellerini yükle.")
        return

    summary_rows = []

    for idx, up in enumerate(uploads, start=1):
        file_bytes = up.getvalue()
        result = analyze_toast_image(file_bytes, ry_threshold=ry_threshold)

        st.markdown(f"## {idx}. Analiz: `{up.name}`")

        if result is None:
            st.warning("Görsel okunamadı veya tost yüzeyi maskelenemedi.")
            st.divider()
            continue

        bgr = result["bgr"]
        overlay = result["overlay"]
        ry_heatmap = result["ry_heatmap"]
        report = result["report"]

        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.subheader("Orijinal")
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Pişmiş / Pişmemiş")
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col3:
            st.subheader("Ry Isı Haritası")
            st.image(cv2.cvtColor(ry_heatmap, cv2.COLOR_BGR2RGB), use_container_width=True)

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)

        with metric_1:
            st.metric("Pişen Alan (%)", f"{report.cooked_pct:.1f}")

        with metric_2:
            st.metric("Ort. Ry (Genel)", f"{report.mean_ry_overall:.1f}")

        with metric_3:
            st.metric(
                "Pişen Bölge Homojenliği", f"{report.cooked_region_homogeneity:.1f} / 100"
            )

        with metric_4:
            st.metric("Genel Homojenlik", f"{report.overall_homogeneity:.1f} / 100")

        summary_rows.append({
            "Dosya": up.name,
            "Tost Alanı (px)": report.toast_pixels,
            "Pişen Alan (px)": report.cooked_pixels,
            "Pişen Alan (%)": round(report.cooked_pct, 2),
            "Ry Eşiği": report.ry_threshold,
            "Ort. Ry (Pişen)": round(report.mean_ry_cooked, 2),
            "Ort. Ry (Genel)": round(report.mean_ry_overall, 2),
            "Pişen Bölge Homojenlik": round(report.cooked_region_homogeneity, 1),
            "Genel Homojenlik": round(report.overall_homogeneity, 1),
        })

        st.caption(
            "Not: Pişen Bölge Homojenliği yalnızca pişmiş piksellerin Ry dağılımını, "
            "Genel Homojenlik ise pişmiş + pişmemiş tüm tost yüzeyinin Ry dağılımını "
            "değerlendirir. Her ikisi de düşükse ısıtma dengesiz demektir; sadece "
            "genel homojenlik düşükse asıl sorun pişen/pişmeyen bölge arasındaki "
            "kontrasttır."
        )

        st.divider()

    if len(summary_rows) > 1:
        st.subheader("Toplu Özet")

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(
            summary_df["Dosya"],
            summary_df["Pişen Alan (%)"],
            color="#d97706",
        )
        ax.set_ylabel("Pişen Alan (%)")
        ax.set_title("Görsel Bazında Pişme Oranı")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
