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

FLOUR_DISK_MAX_ANALYSIS_WIDTH = 1600

FLOUR_DISK_MIN_AREA_RATIO = 0.03
FLOUR_DISK_MAX_AREA_RATIO = 0.45
FLOUR_DISK_MIN_CIRCULARITY = 0.60
FLOUR_DISK_MIN_ASPECT = 0.70
FLOUR_DISK_MAX_ASPECT = 1.30

# Disk kenarındaki gölge/sınır etkisini dışarıda bırakmak için
FLOUR_DISK_EDGE_MARGIN_RATIO = 0.045

# Geçici browning score referansları
FLOUR_DISK_LIGHT_L_REF = 70.0
FLOUR_DISK_DARK_L_REF = 35.0

# Homojenlik skorunda P90 ΔE'nin 20 olması yaklaşık 0 puan kabul edilir.
# Bu değer gerçek testlerle kalibre edilecek.
FLOUR_DISK_HOMOGENEITY_DE_LIMIT = 20.0


def _flour_disk_resize_for_analysis(
    bgr,
    max_width=FLOUR_DISK_MAX_ANALYSIS_WIDTH
):
    H, W = bgr.shape[:2]

    if W <= max_width:
        return bgr

    scale = max_width / W

    return cv2.resize(
        bgr,
        (int(W * scale), int(H * scale)),
        interpolation=cv2.INTER_AREA
    )


def _flour_disk_delta_e76(lab_1, lab_2):
    lab_1 = np.asarray(lab_1, dtype=np.float32)
    lab_2 = np.asarray(lab_2, dtype=np.float32)

    return float(np.linalg.norm(lab_1 - lab_2))


def _flour_disk_find_mask(bgr):
    """
    Beyaz zemin üzerindeki en büyük, yaklaşık dairesel koyu diski bulur.
    """

    H, W = bgr.shape[:2]
    image_area = H * W

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)

    _, rough_mask = cv2.threshold(
        gray_blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (21, 21)
    )

    rough_mask = cv2.morphologyEx(
        rough_mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    rough_mask = cv2.morphologyEx(
        rough_mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    contours, _ = cv2.findContours(
        rough_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < image_area * FLOUR_DISK_MIN_AREA_RATIO:
            continue

        if area > image_area * FLOUR_DISK_MAX_AREA_RATIO:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        circularity = (
            4.0 * np.pi * area /
            (perimeter * perimeter)
        )

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)

        if circularity < FLOUR_DISK_MIN_CIRCULARITY:
            continue

        if not (
            FLOUR_DISK_MIN_ASPECT
            <= aspect_ratio
            <= FLOUR_DISK_MAX_ASPECT
        ):
            continue

        # Büyük ve dairesel contour'lara öncelik ver
        candidate_score = area * circularity

        candidates.append(
            {
                "contour": contour,
                "score": candidate_score,
                "area": area,
                "circularity": circularity
            }
        )

    if not candidates:
        raise ValueError(
            "Unlu disk bulunamadı. Disk/zemin kontrastı veya "
            "segmentasyon limitleri kontrol edilmeli."
        )

    best = max(
        candidates,
        key=lambda item: item["score"]
    )

    contour = best["contour"]

    disk_mask = np.zeros((H, W), dtype=np.uint8)

    cv2.drawContours(
        disk_mask,
        [contour],
        -1,
        255,
        -1
    )

    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)

    edge_margin = max(
        5,
        int(radius * FLOUR_DISK_EDGE_MARGIN_RATIO)
    )

    erode_size = edge_margin * 2 + 1

    analysis_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (erode_size, erode_size)
    )

    analysis_mask = cv2.erode(
        disk_mask,
        analysis_kernel,
        iterations=1
    )

    if np.count_nonzero(analysis_mask) == 0:
        raise ValueError(
            "Disk maskesi kenar daraltma sonrasında boş kaldı."
        )

    return {
        "disk_mask": disk_mask,
        "analysis_mask": analysis_mask,
        "contour": contour,
        "center_x": float(center_x),
        "center_y": float(center_y),
        "radius": float(radius),
        "circularity": float(best["circularity"])
    }


def _flour_disk_normalize_white_background(
    bgr,
    disk_mask,
    target_white=235.0
):
    """
    Disk dışındaki parlak beyaz zemini referans alarak
    basit exposure / white balance düzeltmesi yapar.
    """

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    background_pixels = (
        (disk_mask == 0) &
        (gray >= 170)
    )

    if np.count_nonzero(background_pixels) < 1000:
        return bgr

    background_bgr = bgr[background_pixels].astype(np.float32)

    channel_medians = np.median(
        background_bgr,
        axis=0
    )

    gains = (
        target_white /
        np.maximum(channel_medians, 1.0)
    )

    # Aşırı düzeltmeyi önle
    gains = np.clip(gains, 0.80, 1.20)

    normalized = (
        bgr.astype(np.float32) *
        gains.reshape(1, 1, 3)
    )

    return np.clip(
        normalized,
        0,
        255
    ).astype(np.uint8)


def _flour_disk_make_region_masks(
    analysis_mask,
    center_x,
    center_y,
    radius
):
    """
    Merkez, iç halka, dış halka ve yönsel bölgeleri oluşturur.
    """

    H, W = analysis_mask.shape

    yy, xx = np.indices((H, W))

    distance = np.sqrt(
        (xx - center_x) ** 2 +
        (yy - center_y) ** 2
    )

    normalized_radius = (
        distance /
        max(radius, 1.0)
    )

    valid = analysis_mask > 0

    center_mask = (
        valid &
        (normalized_radius <= 0.33)
    )

    inner_ring_mask = (
        valid &
        (normalized_radius > 0.33) &
        (normalized_radius <= 0.66)
    )

    outer_ring_mask = (
        valid &
        (normalized_radius > 0.66)
    )

    top_mask = valid & (yy < center_y)
    bottom_mask = valid & (yy >= center_y)

    left_mask = valid & (xx < center_x)
    right_mask = valid & (xx >= center_x)

    return {
        "Merkez": center_mask,
        "İç Halka": inner_ring_mask,
        "Dış Halka": outer_ring_mask,
        "Üst": top_mask,
        "Alt": bottom_mask,
        "Sol": left_mask,
        "Sağ": right_mask
    }


def _flour_disk_region_statistics(
    region_name,
    region_mask,
    L_star,
    a_star,
    b_star,
    overall_lab
):
    pixel_count = int(np.count_nonzero(region_mask))

    if pixel_count == 0:
        return None

    mean_L = float(np.mean(L_star[region_mask]))
    mean_a = float(np.mean(a_star[region_mask]))
    mean_b = float(np.mean(b_star[region_mask]))

    region_lab = np.array(
        [mean_L, mean_a, mean_b],
        dtype=np.float32
    )

    return {
        "Bölge": region_name,
        "L*": round(mean_L, 2),
        "a*": round(mean_a, 2),
        "b*": round(mean_b, 2),
        "Genel Ortalamaya ΔE": round(
            _flour_disk_delta_e76(
                region_lab,
                overall_lab
            ),
            2
        ),
        "Piksel": pixel_count
    }


def _flour_disk_create_overlay(
    bgr,
    contour,
    analysis_mask,
    center_x,
    center_y,
    radius
):
    overlay = bgr.copy()

    cv2.drawContours(
        overlay,
        [contour],
        -1,
        (0, 255, 0),
        3
    )

    analysis_contours, _ = cv2.findContours(
        analysis_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        analysis_contours,
        -1,
        (255, 255, 0),
        2
    )

    center = (
        int(round(center_x)),
        int(round(center_y))
    )

    cv2.circle(
        overlay,
        center,
        int(radius * 0.33),
        (255, 0, 255),
        2
    )

    cv2.circle(
        overlay,
        center,
        int(radius * 0.66),
        (255, 0, 255),
        2
    )

    cv2.line(
        overlay,
        (
            int(center_x - radius),
            int(center_y)
        ),
        (
            int(center_x + radius),
            int(center_y)
        ),
        (0, 255, 255),
        2
    )

    cv2.line(
        overlay,
        (
            int(center_x),
            int(center_y - radius)
        ),
        (
            int(center_x),
            int(center_y + radius)
        ),
        (0, 255, 255),
        2
    )

    return overlay


def _flour_disk_create_heatmap_figure(
    bgr,
    L_smooth,
    analysis_mask,
    contour
):
    x, y, w, h = cv2.boundingRect(contour)

    padding = max(
        10,
        int(0.08 * max(w, h))
    )

    H, W = analysis_mask.shape

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)

    crop_rgb = cv2.cvtColor(
        bgr[y1:y2, x1:x2],
        cv2.COLOR_BGR2RGB
    )

    crop_L = L_smooth[y1:y2, x1:x2]
    crop_mask = analysis_mask[y1:y2, x1:x2] > 0

    masked_L = np.ma.masked_where(
        ~crop_mask,
        crop_L
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    ax.imshow(
        crop_rgb,
        alpha=0.35
    )

    heatmap = ax.imshow(
        masked_L,
        cmap="turbo",
        alpha=0.80
    )

    colorbar = fig.colorbar(
        heatmap,
        ax=ax,
        fraction=0.045,
        pad=0.03
    )

    colorbar.set_label("L*")

    ax.set_title("Unlu Disk L* Dağılımı")
    ax.axis("off")

    fig.tight_layout()

    return fig


def _flour_disk_analyze_image(
    bgr,
    normalize_background=True
):
    bgr = _flour_disk_resize_for_analysis(bgr)

    segmentation = _flour_disk_find_mask(bgr)

    if normalize_background:
        normalized_bgr = _flour_disk_normalize_white_background(
            bgr,
            segmentation["disk_mask"]
        )
    else:
        normalized_bgr = bgr.copy()

    lab = cv2.cvtColor(
        normalized_bgr,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    L_star = lab[:, :, 0] * 100.0 / 255.0
    a_star = lab[:, :, 1] - 128.0
    b_star = lab[:, :, 2] - 128.0

    analysis_bool = segmentation["analysis_mask"] > 0

    overall_L = float(np.mean(L_star[analysis_bool]))
    overall_a = float(np.mean(a_star[analysis_bool]))
    overall_b = float(np.mean(b_star[analysis_bool]))

    overall_lab = np.array(
        [overall_L, overall_a, overall_b],
        dtype=np.float32
    )

    # Lokal kamera noise/tekstür yerine geniş renk farklarını ölç
    L_smooth = cv2.GaussianBlur(
        L_star,
        (0, 0),
        9
    )

    a_smooth = cv2.GaussianBlur(
        a_star,
        (0, 0),
        9
    )

    b_smooth = cv2.GaussianBlur(
        b_star,
        (0, 0),
        9
    )

    smooth_mean_lab = np.array([
        np.mean(L_smooth[analysis_bool]),
        np.mean(a_smooth[analysis_bool]),
        np.mean(b_smooth[analysis_bool])
    ])

    delta_e_map = np.sqrt(
        (L_smooth - smooth_mean_lab[0]) ** 2 +
        (a_smooth - smooth_mean_lab[1]) ** 2 +
        (b_smooth - smooth_mean_lab[2]) ** 2
    )

    delta_e_pixels = delta_e_map[analysis_bool]

    mean_delta_e = float(np.mean(delta_e_pixels))
    p90_delta_e = float(
        np.percentile(delta_e_pixels, 90)
    )

    homogeneity_score = float(np.clip(
        100.0 * (
            1.0 -
            p90_delta_e /
            FLOUR_DISK_HOMOGENEITY_DE_LIMIT
        ),
        0,
        100
    ))

    browning_score = float(np.clip(
        100.0 * (
            FLOUR_DISK_LIGHT_L_REF -
            overall_L
        ) / (
            FLOUR_DISK_LIGHT_L_REF -
            FLOUR_DISK_DARK_L_REF
        ),
        0,
        100
    ))

    region_masks = _flour_disk_make_region_masks(
        segmentation["analysis_mask"],
        segmentation["center_x"],
        segmentation["center_y"],
        segmentation["radius"]
    )

    region_rows = []

    for region_name, region_mask in region_masks.items():
        row = _flour_disk_region_statistics(
            region_name,
            region_mask,
            L_star,
            a_star,
            b_star,
            overall_lab
        )

        if row is not None:
            region_rows.append(row)

    region_df = pd.DataFrame(region_rows)

    region_lookup = {
        row["Bölge"]: row
        for _, row in region_df.iterrows()
    }

    center_lab = np.array([
        region_lookup["Merkez"]["L*"],
        region_lookup["Merkez"]["a*"],
        region_lookup["Merkez"]["b*"]
    ])

    edge_lab = np.array([
        region_lookup["Dış Halka"]["L*"],
        region_lookup["Dış Halka"]["a*"],
        region_lookup["Dış Halka"]["b*"]
    ])

    center_edge_delta_e = _flour_disk_delta_e76(
        center_lab,
        edge_lab
    )

    overlay = _flour_disk_create_overlay(
        normalized_bgr,
        segmentation["contour"],
        segmentation["analysis_mask"],
        segmentation["center_x"],
        segmentation["center_y"],
        segmentation["radius"]
    )

    summary = {
        "L*": overall_L,
        "a*": overall_a,
        "b*": overall_b,
        "browning_score": browning_score,
        "homogeneity_score": homogeneity_score,
        "mean_delta_e": mean_delta_e,
        "p90_delta_e": p90_delta_e,
        "center_edge_delta_e": center_edge_delta_e,
        "circularity": segmentation["circularity"]
    }

    return {
        "normalized_bgr": normalized_bgr,
        "overlay_bgr": overlay,
        "analysis_mask": segmentation["analysis_mask"],
        "contour": segmentation["contour"],
        "L_smooth": L_smooth,
        "region_df": region_df,
        "summary": summary
    }


def run_flour_disk():
    st.title("Unlu Disk Renk Analizi")

    st.caption(
        "Unlu disk yüzeyinde ortalama renk, kızarıklık "
        "ve renk homojenliği analizi."
    )

    normalize_background = st.checkbox(
        "Beyaz zemine göre ışık / renk düzeltmesi uygula",
        value=True,
        key="flour_disk_normalize"
    )

    uploads = st.file_uploader(
        "Unlu disk görsel(ler)i yükle",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="flour_disk_uploads"
    )

    if not uploads:
        st.info(
            "Başlamak için bir veya daha fazla unlu disk görseli yükle."
        )
        return

    general_summary_rows = []

    for image_index, upload in enumerate(uploads, start=1):
        file_bytes = upload.getvalue()

        file_hash = hashlib.md5(
            file_bytes
        ).hexdigest()[:10]

        st.markdown(
            f"## {image_index}. Görsel: `{upload.name}`"
        )

        image_array = np.frombuffer(
            file_bytes,
            np.uint8
        )

        bgr = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if bgr is None:
            st.error("Görsel okunamadı.")
            continue

        try:
            with st.spinner(
                "Disk segmentasyonu ve renk analizi yapılıyor..."
            ):
                result = _flour_disk_analyze_image(
                    bgr,
                    normalize_background=normalize_background
                )

        except Exception as exc:
            st.error(
                f"Unlu disk analiz hatası: {exc}"
            )
            continue

        summary = result["summary"]

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)

        with metric_1:
            st.metric(
                "Browning Score",
                f"{summary['browning_score']:.1f} / 100"
            )

        with metric_2:
            st.metric(
                "Homojenlik",
                f"{summary['homogeneity_score']:.1f} / 100"
            )

        with metric_3:
            st.metric(
                "Ortalama L*",
                f"{summary['L*']:.2f}"
            )

        with metric_4:
            st.metric(
                "Merkez–Kenar ΔE",
                f"{summary['center_edge_delta_e']:.2f}"
            )

        detail_1, detail_2, detail_3, detail_4 = st.columns(4)

        with detail_1:
            st.metric(
                "Ortalama a*",
                f"{summary['a*']:.2f}"
            )

        with detail_2:
            st.metric(
                "Ortalama b*",
                f"{summary['b*']:.2f}"
            )

        with detail_3:
            st.metric(
                "Ortalama ΔE",
                f"{summary['mean_delta_e']:.2f}"
            )

        with detail_4:
            st.metric(
                "P90 ΔE",
                f"{summary['p90_delta_e']:.2f}"
            )

        image_col_1, image_col_2 = st.columns(
            2,
            gap="large"
        )

        with image_col_1:
            st.subheader("Normalize Edilmiş Görsel")

            st.image(
                cv2.cvtColor(
                    result["normalized_bgr"],
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

        with image_col_2:
            st.subheader("Analiz Bölgeleri")

            st.image(
                cv2.cvtColor(
                    result["overlay_bgr"],
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

        st.subheader("L* Heatmap")

        heatmap_fig = _flour_disk_create_heatmap_figure(
            result["normalized_bgr"],
            result["L_smooth"],
            result["analysis_mask"],
            result["contour"]
        )

        st.pyplot(
            heatmap_fig,
            use_container_width=False
        )

        plt.close(heatmap_fig)

        st.subheader("Bölgesel Renk Sonuçları")

        st.dataframe(
            result["region_df"].style.format({
                "L*": "{:.2f}",
                "a*": "{:.2f}",
                "b*": "{:.2f}",
                "Genel Ortalamaya ΔE": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        region_csv = (
            result["region_df"]
            .to_csv(index=False)
            .encode("utf-8-sig")
        )

        st.download_button(
            "Bölgesel Sonuçları CSV İndir",
            data=region_csv,
            file_name=(
                f"unlu_disk_{file_hash}_bolgesel_sonuclar.csv"
            ),
            mime="text/csv",
            key=f"flour_disk_csv_{file_hash}"
        )

        general_summary_rows.append({
            "Dosya": upload.name,
            "L*": round(summary["L*"], 2),
            "a*": round(summary["a*"], 2),
            "b*": round(summary["b*"], 2),
            "Browning Score": round(
                summary["browning_score"],
                1
            ),
            "Homojenlik Score": round(
                summary["homogeneity_score"],
                1
            ),
            "Ortalama ΔE": round(
                summary["mean_delta_e"],
                2
            ),
            "P90 ΔE": round(
                summary["p90_delta_e"],
                2
            ),
            "Merkez-Kenar ΔE": round(
                summary["center_edge_delta_e"],
                2
            )
        })

        st.divider()

    if general_summary_rows:
        st.header("Görseller Arası Karşılaştırma")

        summary_df = pd.DataFrame(
            general_summary_rows
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        summary_csv = (
            summary_df
            .to_csv(index=False)
            .encode("utf-8-sig")
        )

        st.download_button(
            "Genel Karşılaştırmayı CSV İndir",
            data=summary_csv,
            file_name="unlu_disk_genel_karsilastirma.csv",
            mime="text/csv",
            key="flour_disk_summary_csv"
        )
