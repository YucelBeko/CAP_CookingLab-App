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

COOKIE_MIN_AREA_RATIO = 0.0030
COOKIE_MAX_AREA_RATIO = 0.08

COOKIE_MASK_L_MIN = 70
COOKIE_MASK_S_MIN = 35
COOKIE_MASK_A0_MIN = -8
COOKIE_MASK_B0_MIN = 2

COOKIE_MIN_CIRCULARITY = 0.35
COOKIE_MIN_ASPECT = 0.55
COOKIE_MAX_ASPECT = 1.50

# Kurabiyeleri satır satır sıralamak için tolerans
COOKIE_ROW_GROUP_FACTOR = 0.65

def _cookie_class_from_score(score):
    if score < 30:
        return "Açık"
    elif score < 55:
        return "Altın"
    elif score < 75:
        return "Koyu Altın"
    else:
        return "Koyu"


def _cookie_class_ascii(class_name):
    """
    cv2.putText Türkçe karakterleri düzgün göstermediği için
    yalnızca overlay etiketini ASCII'ye çevirir.
    """
    ascii_map = {
        "Açık": "Acik",
        "Altın": "Altin",
        "Koyu Altın": "Koyu Altin",
        "Koyu": "Koyu",
        "Çok Koyu": "Cok Koyu",
        "İdeal": "Ideal",
    }

    return ascii_map.get(class_name, class_name)


def _cookie_build_product_mask(bgr):
    """
    Koyu tepsi üzerindeki sarı/kahverengi kurabiyeleri ayırır.

    LAB B kanalını doğrudan kullanmıyoruz; OpenCV LAB'de
    nötr gri yaklaşık 128 olduğu için B0 = B - 128 kullanıyoruz.
    """

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)

    A0 = A.astype(np.int16) - 128
    B0 = B.astype(np.int16) - 128

    # Açık + doygun + sarı/kahverengi ürün alanları
    mask = (
        (L > COOKIE_MASK_L_MIN) &
        (S > COOKIE_MASK_S_MIN) &
        (A0 > COOKIE_MASK_A0_MIN) &
        (B0 > COOKIE_MASK_B0_MIN)
    ).astype(np.uint8) * 255

    # Kırıntı ve ince bağlantıları kopar.
    # Burada CLOSE kullanmıyoruz; kurabiyeleri birbirine bağlıyordu.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )

    return mask


def _cookie_extract_contours(product_mask):
    """
    Kurabiye contour'larını bulur;
    küçük kırıntıları ve gerçek dışı büyük bölgeleri eler.
    """

    H, W = product_mask.shape[:2]
    image_area = H * W

    min_area = image_area * COOKIE_MIN_AREA_RATIO
    max_area = image_area * COOKIE_MAX_AREA_RATIO

    contours, _ = cv2.findContours(
        product_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
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

        # Isırılmış/deforme kurabiyeyi de kaybetmemek için
        # önceki 0.55 sınırını biraz gevşettik.
        if circularity < COOKIE_MIN_CIRCULARITY:
            continue

        if not (
            COOKIE_MIN_ASPECT
            <= aspect_ratio
            <= COOKIE_MAX_ASPECT
        ):
            continue

        valid_contours.append(contour)

    return valid_contours


def _cookie_sort_contours(contours):
    """
    Contour'ları satır-satır ve soldan sağa sıralar.
    """
    if not contours:
        return []

    items = []
    heights = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w / 2
        cy = y + h / 2
        heights.append(h)
        items.append({
            "contour": c,
            "x": x, "y": y, "w": w, "h": h,
            "cx": cx, "cy": cy
        })

    median_h = np.median(heights) if heights else 50
    row_tol = median_h * COOKIE_ROW_GROUP_FACTOR

    items = sorted(items, key=lambda d: d["cy"])

    rows = []
    current_row = []

    for item in items:
        if not current_row:
            current_row.append(item)
            continue

        row_y = np.mean([r["cy"] for r in current_row])

        if abs(item["cy"] - row_y) <= row_tol:
            current_row.append(item)
        else:
            rows.append(sorted(current_row, key=lambda d: d["cx"]))
            current_row = [item]

    if current_row:
        rows.append(sorted(current_row, key=lambda d: d["cx"]))

    ordered = []
    for row in rows:
        ordered.extend([r["contour"] for r in row])

    return ordered


def _cookie_score_from_lab_pixels(L_pixels):
    """
    Browning score:
    Düşük L -> daha koyu/kızarmış.
    Şimdilik absolute-tunable bir skor.
    """
    if len(L_pixels) == 0:
        return 0.0, 0.0, 0.0

    mean_L = float(np.mean(L_pixels))
    p25_L = float(np.percentile(L_pixels, 25))

    # mean ve p25 karışımı; kenar koyuluğunu biraz hissettirsin
    weighted_L = 0.7 * mean_L + 0.3 * p25_L

    # Bu değerler gerektiğinde tune edilir.
    light_ref = 210.0
    dark_ref = 135.0

    score = 100.0 * (light_ref - weighted_L) / (light_ref - dark_ref)
    score = float(np.clip(score, 0, 100))

    return score, mean_L, p25_L


def _cookie_analyze_image(bgr):
    """
    Tek görseldeki tüm kurabiyeleri analiz eder.
    """
    H, W = bgr.shape[:2]

    # Büyük görseller için hafif küçültme
    max_w = 1800
    if W > max_w:
        scale = max_w / W
        new_w = int(W * scale)
        new_h = int(H * scale)
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        H, W = bgr.shape[:2]

    product_mask = _cookie_build_product_mask(bgr)
    contours = _cookie_extract_contours(product_mask)
    contours = _cookie_sort_contours(contours)

    if len(contours) == 0:
        raise ValueError("Kurabiye bulunamadı. Işık/arka plan/threshold kontrol edilmeli.")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    overlay = bgr.copy()
    all_rows = []

    for idx, contour in enumerate(contours, start=1):
        single_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(single_mask, [contour], -1, 255, -1)

        # çok küçük delikleri kapat
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        single_mask = cv2.morphologyEx(single_mask, cv2.MORPH_CLOSE, kernel)

        L_pixels = L[single_mask > 0]

        score, mean_L, p25_L = _cookie_score_from_lab_pixels(L_pixels)
        cookie_class = _cookie_class_from_score(score)

        area_px = int(np.count_nonzero(single_mask))

        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2

        # contour çiz
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)

        # etiket
        label1 = f"{idx}"
        cookie_class_overlay = _cookie_class_ascii(cookie_class)
        label2 = f"{score:.0f} - {cookie_class_overlay}"

        cv2.putText(
            overlay, label1,
            (x, max(20, y - 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2, cv2.LINE_AA
        )

        cv2.putText(
            overlay, label2,
            (x, max(40, y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 0), 2, cv2.LINE_AA
        )

        all_rows.append({
            "Kurabiye No": idx,
            "Score": round(score, 1),
            "Sınıf": cookie_class,
            "Mean L": round(mean_L, 1),
            "P25 L": round(p25_L, 1),
            "Alan (px)": area_px,
            "BBox X": x,
            "BBox Y": y,
            "BBox W": w,
            "BBox H": h,
        })

    result_df = pd.DataFrame(all_rows)

    mean_score = float(result_df["Score"].mean())
    min_score = float(result_df["Score"].min())
    max_score = float(result_df["Score"].max())
    std_score = float(result_df["Score"].std()) if len(result_df) > 1 else 0.0

    # Homojenlik: std ne kadar düşükse o kadar iyi
    homogeneity_score = float(np.clip(100 - std_score * 4, 0, 100))

    summary = {
        "count": len(result_df),
        "mean_score": mean_score,
        "min_score": min_score,
        "max_score": max_score,
        "std_score": std_score,
        "homogeneity_score": homogeneity_score,
        "lightest_cookie": int(result_df.loc[result_df["Score"].idxmin(), "Kurabiye No"]),
        "darkest_cookie": int(result_df.loc[result_df["Score"].idxmax(), "Kurabiye No"]),
    }

    return {
        "original_bgr": bgr,
        "overlay_bgr": overlay,
        "mask": product_mask,
        "result_df": result_df,
        "summary": summary,
    }


def run_cookie():
    st.title("Kurabiye Alt Yüzey Analizi")
    st.caption("Çoklu kurabiye görselinde browning / kızarıklık analizi.")

    uploads = st.file_uploader(
        "Kurabiye görsel(ler)i yükle",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="cookie_uploads"
    )

    if not uploads:
        st.info("Başlamak için bir veya daha fazla kurabiye görseli yükle.")
        return

    for img_idx, up in enumerate(uploads, start=1):
        st.markdown(f"## {img_idx}. Analiz: `{up.name}`")

        file_bytes = np.frombuffer(up.read(), np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if bgr is None:
            st.error("Görsel okunamadı.")
            continue

        try:
            result = _cookie_analyze_image(bgr)
        except Exception as exc:
            st.error(f"Kurabiye analiz hatası: {exc}")
            st.divider()
            continue

        summary = result["summary"]
        df = result["result_df"]

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Kurabiye Sayısı", summary["count"])
        with m2:
            st.metric("Ort. Score", f"{summary['mean_score']:.1f}")
        with m3:
            st.metric("Homojenlik", f"{summary['homogeneity_score']:.1f} / 100")
        with m4:
            st.metric("Score Std", f"{summary['std_score']:.2f}")

        m5, m6, m7, m8 = st.columns(4)
        with m5:
            st.metric("En Açık", f"No {summary['lightest_cookie']}")
        with m6:
            st.metric("En Koyu", f"No {summary['darkest_cookie']}")
        with m7:
            st.metric("Min Score", f"{summary['min_score']:.1f}")
        with m8:
            st.metric("Max Score", f"{summary['max_score']:.1f}")

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.subheader("Orijinal")
            st.image(
                cv2.cvtColor(result["original_bgr"], cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        with c2:
            st.subheader("Analiz Overlay")
            st.image(
                cv2.cvtColor(result["overlay_bgr"], cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        st.subheader("Kurabiye Sonuç Tablosu")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # sınıf dağılımı
        class_counts = df["Sınıf"].value_counts().reindex(
            ["Açık", "Altın", "Koyu Altın", "Koyu"],
            fill_value=0
        )

        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(class_counts.index, class_counts.values)
        ax.set_title("Kurabiye Sınıf Dağılımı")
        ax.set_ylabel("Adet")
        ax.set_xlabel("Sınıf")
        st.pyplot(fig, clear_figure=True, use_container_width=False)
        plt.close(fig)

        with st.expander("Segmentasyon Maskesini Göster", expanded=False):
            st.image(
                result["mask"],
                caption="Beyaz: kurabiye olarak bulunan alan",
                use_container_width=True
            )

        st.divider()
