from __future__ import annotations

import cv2
import numpy as np
import streamlit as st


# =========================================================
# TOST EKMEĞİ / GRILL PERFORMANCE
# =========================================================

DEFAULT_BROWNING_THRESHOLD = 65.0

# Arka planın beyaz olduğunu varsayarak oldukça geniş tutuldu.
# Daha sonra gerçek test görsellerine göre kalibre ederiz.
BACKGROUND_L_MIN = 215
BACKGROUND_CHROMA_MAX = 18

# Çok küçük gürültü komponentlerini temizlemek için
MIN_COMPONENT_RATIO = 0.0005

# Görselleştirme renkleri - BGR
GRILLED_COLOR = (0, 180, 0)       # green
NOT_GRILLED_COLOR = (180, 180, 180)
CONTOUR_COLOR = (0, 255, 255)


# =========================================================
# IMAGE / MASK HELPERS
# =========================================================

def _decode_image(upload) -> np.ndarray | None:
    data = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _build_bread_mask(bgr: np.ndarray) -> np.ndarray:
    """
    White background üzerindeki grilled toast bread yüzeyini bulur.

    Ekmekler tek tek ayrılmaz.
    Her anlamlı foreground component analiz maskesine dahil edilir.
    """

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32) - 128.0
    B = lab[:, :, 2].astype(np.float32) - 128.0

    chroma = np.sqrt(A * A + B * B)

    # Beyaz arka plan:
    # yüksek L + düşük chroma
    background = (
        (L >= BACKGROUND_L_MIN)
        & (chroma <= BACKGROUND_CHROMA_MAX)
    )

    mask = (~background).astype(np.uint8) * 255

    # Küçük gürültüleri temizle
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_small
    )

    # Küçük komponentleri kaldır.
    # Ancak ekmekleri tek component haline getirmiyoruz.
    h, w = mask.shape
    min_area = max(
        100,
        int(h * w * MIN_COMPONENT_RATIO)
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean_mask = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            clean_mask[labels == label] = 255

    # Hafif closing, kenarlardaki küçük boşlukları toparlar
    clean_mask = cv2.morphologyEx(
        clean_mask,
        cv2.MORPH_CLOSE,
        kernel_small
    )

    return clean_mask


def _calculate_ry(bgr: np.ndarray) -> np.ndarray:
    """
    Mevcut uygulamadaki Ry yaklaşımına paralel olarak
    OpenCV L kanalını 0-100 aralığına taşır.

    Düşük Ry = daha koyu / daha fazla kızarmış.
    """

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0].astype(np.float32)

    return (L / 255.0) * 100.0


def _build_browning_heatmap(
    bgr: np.ndarray,
    bread_mask: np.ndarray,
) -> np.ndarray:
    """
    Browning'i görsel olarak göstermek için sürekli renk haritası.

    Burada sınıflandırma yapılmaz.
    Sadece gerçek Ry değerleri görselleştirilir.
    """

    ry = _calculate_ry(bgr)

    heat = np.full(
                        (bgr.shape[0], bgr.shape[1], 3),
                        255,
                        dtype=np.uint8
                    )

    valid = bread_mask > 0

    if np.any(valid):
        values = ry[valid]

        # Daha koyu = daha yüksek browning
        normalized = np.clip(
            (100.0 - values) / 100.0,
            0.0,
            1.0
        )

        normalized_u8 = (
            normalized * 255.0
        ).astype(np.uint8)

        colored = cv2.applyColorMap(
            normalized_u8,
            cv2.COLORMAP_TURBO
        )

    heat[valid] = colored[valid]

    # Arka plan beyaz kalsın
    heat[~valid] = (255, 255, 255)

    return heat


def _build_grill_mask(
    ry: np.ndarray,
    bread_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Ry threshold'un altında kalan alanı grilled kabul eder.
    """

    valid = bread_mask > 0

    grilled = (
        valid
        & (ry <= threshold)
    )

    return grilled


def _calculate_grill_percentage(
    grill_mask: np.ndarray,
    bread_mask: np.ndarray,
) -> float:

    total_pixels = int(np.count_nonzero(bread_mask))
    grilled_pixels = int(np.count_nonzero(grill_mask))

    if total_pixels == 0:
        return 0.0

    return (
        grilled_pixels
        / total_pixels
        * 100.0
    )


def _build_binary_overlay(
    bgr: np.ndarray,
    bread_mask: np.ndarray,
    grill_mask: np.ndarray,
) -> np.ndarray:

    output = np.full_like(
        bgr,
        255
    )

    bread = bread_mask > 0

    # Önce bütün bread alanını gri yap
    output[bread] = NOT_GRILLED_COLOR

    # Sonra grilled alanını yeşil yap
    output[grill_mask] = GRILLED_COLOR

    # Ekmek alanının sınırlarını çiz
    contours, _ = cv2.findContours(
        bread_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        output,
        contours,
        -1,
        CONTOUR_COLOR,
        2
    )

    return output


def _build_browning_overlay(
    bgr: np.ndarray,
    bread_mask: np.ndarray,
    grill_mask: np.ndarray,
) -> np.ndarray:

    heat = _build_browning_heatmap(
        bgr,
        bread_mask
    )

    contours, _ = cv2.findContours(
        bread_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        heat,
        contours,
        -1,
        CONTOUR_COLOR,
        2
    )

    return heat


# =========================================================
# MAIN SECTION
# =========================================================

def run_toast_bread():

    st.markdown(
        """
        <style>
        .block-container h1 {
            margin-top: -60px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Tost Ekmeği / Grill Performansı")

    st.info(
        "Tost ekmekleri tek tek analiz edilmez. "
        "Fotoğraftaki tüm grilled bread yüzeyi tek bir test alanı "
        "olarak değerlendirilir."
    )

    # -----------------------------------------------------
    # SETTINGS
    # -----------------------------------------------------

    threshold = st.slider(
        "Browning Threshold",
        min_value=20.0,
        max_value=90.0,
        value=DEFAULT_BROWNING_THRESHOLD,
        step=0.5,
        help=(
            "Ry değeri bu eşikten düşük olan alanlar "
            "grilled kabul edilir. Daha düşük Ry daha koyu "
            "ve daha fazla kızarmış yüzeyi ifade eder."
        ),
    )

    st.caption(
        f"Mevcut threshold: **Ry ≤ {threshold:.1f} → Grilled**"
    )

    # -----------------------------------------------------
    # UPLOAD
    # -----------------------------------------------------

    uploads = st.file_uploader(
        "Tost ekmeği görseli yükle",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="toast_bread_upload",
    )

    if not uploads:
        st.info(
            "Başlamak için bir veya daha fazla tost ekmeği "
            "görseli yükle."
        )
        return

    # -----------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------

    summary_rows = []

    for idx, upload in enumerate(uploads):

        bgr = _decode_image(upload)

        if bgr is None:
            st.error(
                f"{upload.name}: Görsel okunamadı."
            )
            continue

        bread_mask = _build_bread_mask(bgr)

        bread_pixels = int(
            np.count_nonzero(bread_mask)
        )

        if bread_pixels == 0:
            st.warning(
                f"{upload.name}: Bread yüzeyi bulunamadı. "
                "Arka plan threshold'larını kontrol etmek gerekebilir."
            )
            continue

        ry = _calculate_ry(bgr)

        grill_mask = _build_grill_mask(
            ry,
            bread_mask,
            threshold,
        )

        grill_percentage = _calculate_grill_percentage(
            grill_mask,
            bread_mask,
        )

        non_grill_percentage = (
            100.0 - grill_percentage
        )

        browning_heatmap = _build_browning_overlay(
            bgr,
            bread_mask,
            grill_mask,
        )

        binary_overlay = _build_binary_overlay(
            bgr,
            bread_mask,
            grill_mask,
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.subheader(
            f"{idx + 1}. Analiz: {upload.name}"
        )

        # KPI
        k1, k2, k3 = st.columns(3)

        with k1:
            st.metric(
                "Grill Area",
                f"{grill_percentage:.1f}%",
            )

        with k2:
            st.metric(
                "Non-Grilled Area",
                f"{non_grill_percentage:.1f}%",
            )

        with k3:
            valid_ry = ry[bread_mask > 0]

            st.metric(
                "Ortalama Ry",
                f"{np.mean(valid_ry):.1f}",
            )

        # -------------------------------------------------
        # VISUALS
        # -------------------------------------------------

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Orijinal**")
            st.image(
                cv2.cvtColor(
                    bgr,
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

        with c2:
            st.markdown("**Browning Heatmap**")
            st.image(
                cv2.cvtColor(
                    browning_heatmap,
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

        with c3:
            st.markdown("**Grill Area Mask**")
            st.image(
                cv2.cvtColor(
                    binary_overlay,
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

        # -------------------------------------------------
        # SIMPLE RESULT BAR
        # -------------------------------------------------

        st.markdown(
            f"""
            <div style="
                margin-top:10px;
                margin-bottom:10px;
                padding:14px;
                border-radius:8px;
                background:#f5f7fa;
                text-align:center;
                font-size:18px;
            ">
                <b>Grill Performance:</b>
                <span style="font-size:24px;">
                    {grill_percentage:.1f}%
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        summary_rows.append(
            {
                "Dosya": upload.name,
                "Grill Area (%)": round(
                    grill_percentage,
                    2,
                ),
                "Non-Grilled Area (%)": round(
                    non_grill_percentage,
                    2,
                ),
                "Ortalama Ry": round(
                    float(np.mean(
                        ry[bread_mask > 0]
                    )),
                    2,
                ),
                "Threshold": round(
                    threshold,
                    2,
                ),
            }
        )

        st.divider()

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    if summary_rows:

        st.subheader("Genel Karşılaştırma")

        import pandas as pd

        summary_df = pd.DataFrame(
            summary_rows
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = (
            summary_df
            .to_csv(index=False)
            .encode("utf-8-sig")
        )

        st.download_button(
            "Sonuçları CSV İndir",
            data=csv_data,
            file_name="tost_ekmegi_grill_sonuclari.csv",
            mime="text/csv",
            key="toast_bread_summary_csv",
        )
