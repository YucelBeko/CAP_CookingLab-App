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

from ui.navigation import set_page

TEFLON_N_ROWS = 5
TEFLON_N_COLS = 6
TEFLON_ZOOM_FACTOR = 60


def _teflon_analyze_dataframe(
    df,
    threshold=100.0,
    n_rows=TEFLON_N_ROWS,
    n_cols=TEFLON_N_COLS
):
    """
    İlk kez bütün Teflon blokların threshold değerine ulaştığı satırı bulur.
    O satırdaki 30 blok sıcaklığını ve özet metrikleri döndürür.
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    n_blocks = n_rows * n_cols
    sensor_cols = [f"blok_{i}" for i in range(1, n_blocks + 1)]

    required_cols = ["Step", "Minute"] + sensor_cols
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        preview = ", ".join(missing_cols[:10])

        if len(missing_cols) > 10:
            preview += f" ... (+{len(missing_cols) - 10})"

        raise ValueError(
            "Excel dosyasında gerekli kolonlar bulunamadı:\n"
            f"{preview}"
        )

    # Blok sıcaklıklarını numeric hale getir
    numeric_blocks = df[sensor_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Bütün blokların threshold üzerine çıktığı satırlar
    all_reached_mask = numeric_blocks.ge(threshold).all(axis=1)
    reached_positions = np.flatnonzero(all_reached_mask.to_numpy())

    if len(reached_positions) == 0:
        max_min_temperature = numeric_blocks.min(axis=1).max()

        if pd.isna(max_min_temperature):
            max_min_text = "hesaplanamadı"
        else:
            max_min_text = f"{max_min_temperature:.1f} °C"

        raise ValueError(
            f"Test boyunca bütün bloklar {threshold:.1f} °C değerine ulaşmadı.\n"
            f"Bloklar arasındaki en yüksek minimum sıcaklık: {max_min_text}"
        )

    end_position = int(reached_positions[0])

    end_row = df.iloc[end_position]
    end_values = numeric_blocks.iloc[end_position].to_numpy(dtype=float)

    end_step = pd.to_numeric(
        pd.Series([end_row["Step"]]),
        errors="coerce"
    ).iloc[0]

    end_minute = pd.to_numeric(
        pd.Series([end_row["Minute"]]),
        errors="coerce"
    ).iloc[0]

    last_block_index = int(np.argmin(end_values))
    last_block_num = last_block_index + 1

    result = {
        "values": end_values,
        "temp_grid": end_values.reshape(n_rows, n_cols),
        "sensor_cols": sensor_cols,
        "threshold": float(threshold),
        "end_position": end_position,
        "end_step": float(end_step) if pd.notna(end_step) else None,
        "end_minute": float(end_minute) if pd.notna(end_minute) else None,
        "last_block_num": last_block_num,
        "last_block_temp": float(end_values[last_block_index]),
        "minimum_temp": float(np.min(end_values)),
        "maximum_temp": float(np.max(end_values)),
        "average_temp": float(np.mean(end_values)),
        "std_temp": float(np.std(end_values)),
        "delta_t": float(np.max(end_values) - np.min(end_values)),
        "n_rows": n_rows,
        "n_cols": n_cols,
    }

    return result


def _teflon_create_figure(result):
    """
    Teflon sıcaklık heatmap figürünü ve PNG bytes çıktısını oluşturur.
    """
    values = result["values"]
    temp_grid = result["temp_grid"]

    n_rows = result["n_rows"]
    n_cols = result["n_cols"]

    threshold = result["threshold"]
    end_minute = result["end_minute"]
    last_block_num = result["last_block_num"]
    delta_t = result["delta_t"]

    smooth_grid = ndimage_zoom(
        temp_grid,
        TEFLON_ZOOM_FACTOR,
        order=3
    )

    col_centers = np.arange(0.5, n_cols, 1.0)
    row_centers = np.arange(n_rows - 0.5, 0.0, -1.0)

    x_grid, y_grid = np.meshgrid(
        col_centers,
        row_centers
    )

    t_min = float(np.min(values) - 1.0)
    t_max = float(np.max(values) + 1.0)

    # Bütün sıcaklıklar aynıysa color scale çökmesin
    if abs(t_max - t_min) < 0.001:
        t_min -= 1
        t_max += 1

    fig, ax = plt.subplots(figsize=(11, 9.5))

    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    cmap = plt.get_cmap("RdYlGn_r")

    heatmap = ax.imshow(
        smooth_grid,
        origin="upper",
        extent=[0, n_cols, 0, n_rows],
        cmap=cmap,
        vmin=t_min,
        vmax=t_max,
        aspect="equal"
    )

    contour_levels = np.arange(
        np.floor(t_min),
        np.ceil(t_max) + 1,
        1
    )

    if len(contour_levels) >= 2:
        ax.contour(
            x_grid,
            y_grid,
            temp_grid,
            levels=contour_levels,
            colors="white",
            linewidths=0.4,
            alpha=0.35
        )

    # Blok sıcaklık noktaları
    for i, temperature in enumerate(values):
        grid_row = i // n_cols
        grid_col = i % n_cols

        center_x = col_centers[grid_col]
        center_y = row_centers[grid_row]

        ax.scatter(
            center_x,
            center_y,
            s=45,
            color="white",
            zorder=5,
            edgecolors="#333333",
            linewidths=0.8
        )

        ax.text(
            center_x,
            center_y,
            f"{i + 1}\n{temperature:.1f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "#000000",
                "alpha": 0.5,
                "edgecolor": "none"
            }
        )

    # En yavaş bloğu halka ile işaretle
    last_index = last_block_num - 1

    last_center_x = col_centers[last_index % n_cols]
    last_center_y = row_centers[last_index // n_cols]

    ax.scatter(
        last_center_x,
        last_center_y,
        s=240,
        facecolors="none",
        edgecolors="white",
        linewidths=2.5,
        zorder=6
    )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)

    ax.set_xticks(col_centers)
    ax.set_xticklabels(
        [f"C{i + 1}" for i in range(n_cols)],
        color="white",
        fontsize=8
    )

    middle_row_labels = [
        f"R{i}" for i in range(2, n_rows)
    ]

    y_tick_labels = (
        ["Arka"] +
        middle_row_labels +
        ["Ön"]
    )

    ax.set_yticks(row_centers)
    ax.set_yticklabels(
        y_tick_labels,
        color="white",
        fontsize=8
    )

    ax.set_xlabel(
        "← Sol (Left)          Sağ (Right) →",
        color="white",
        fontsize=9
    )

    ax.set_ylabel(
        "← Ön (Front)          Arka (Back) →",
        color="white",
        fontsize=9,
        labelpad=10
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    ax.tick_params(colors="white")

    colorbar = fig.colorbar(
        heatmap,
        ax=ax,
        fraction=0.03,
        pad=0.02
    )

    colorbar.set_label(
        "Sıcaklık (°C)",
        color="white",
        fontsize=10
    )

    colorbar.ax.yaxis.set_tick_params(color="white")

    plt.setp(
        colorbar.ax.yaxis.get_ticklabels(),
        color="white"
    )

    minute_text = (
        f"{end_minute:.2f} dk"
        if end_minute is not None
        else "-"
    )

    ax.text(
        0.5,
        -0.07,
        (
            f"ΔT = {delta_t:.1f} °C"
            f"  |  Test sonu: {minute_text}"
            f"  |  En yavaş blok: blok_{last_block_num}"
        ),
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#aaaaaa"
    )

    ax.set_title(
        (
            f"Yüzey Isıtma Homojenliği – Teflon Blok Testi "
            f"({n_rows}×{n_cols})\n"
            f"Tüm blokların ≥ {threshold:.0f} °C olduğu ilk an"
        ),
        color="white",
        fontsize=11,
        pad=12
    )

    fig.tight_layout()

    png_buffer = io.BytesIO()

    fig.savefig(
        png_buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    png_buffer.seek(0)

    return fig, png_buffer.getvalue()

def run_teflon_block():
    st.title("Teflon Blok Analizi")

    st.caption(
        "30 bloklu Teflon yüzey sıcaklık testinde test sonu, "
        "ΔT ve sıcaklık homojenliği analizi."
    )

    st.button(
        "← Ana Sayfa",
        key="teflon_back_home",
        on_click=set_page,
        args=("Home",)
    )

    upload_col, setting_col = st.columns([2, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Teflon blok test Excel dosyasını yükle",
            type=["xlsx", "xls"],
            key="teflon_file"
        )

    with setting_col:
        threshold = st.number_input(
            "Test sonu sıcaklık eşiği (°C)",
            min_value=0.0,
            max_value=500.0,
            value=100.0,
            step=1.0,
            key="teflon_threshold"
        )

    if "teflon_result" not in st.session_state:
        st.session_state.teflon_result = None

    if uploaded_file is None:
        st.info(
            "Analizi başlatmak için blok_1–blok_30 kolonlarını "
            "içeren Excel dosyasını yükle."
        )
        return

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    analysis_signature = (
        file_hash,
        round(float(threshold), 4)
    )

    if (
        st.session_state.get("teflon_analysis_signature")
        != analysis_signature
    ):
        st.session_state.teflon_analysis_signature = analysis_signature
        st.session_state.teflon_result = None

    if st.button(
        "Teflon Analizini Çalıştır",
        type="primary",
        use_container_width=True,
        key="teflon_run_analysis"
    ):
        try:
            with st.spinner(
                "Excel okunuyor ve sıcaklık haritası oluşturuluyor..."
            ):
                df = pd.read_excel(
                    io.BytesIO(file_bytes)
                )

                result = _teflon_analyze_dataframe(
                    df=df,
                    threshold=threshold
                )

                st.session_state.teflon_result = result

            st.success("Teflon blok analizi tamamlandı.")

        except Exception as exc:
            st.session_state.teflon_result = None
            st.error(f"Teflon analiz hatası: {exc}")

    result = st.session_state.get("teflon_result")

    if result is None:
        return

    st.subheader("Test Sonu Özeti")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    with metric_1:
        end_minute = result["end_minute"]

        st.metric(
            "Test Sonu",
            (
                f"{end_minute:.2f} dk"
                if end_minute is not None
                else "-"
            )
        )

    with metric_2:
        st.metric(
            "ΔT",
            f"{result['delta_t']:.1f} °C"
        )

    with metric_3:
        st.metric(
            "En Yavaş Blok",
            f"blok_{result['last_block_num']}"
        )

    with metric_4:
        st.metric(
            "Ortalama Sıcaklık",
            f"{result['average_temp']:.1f} °C"
        )

    detail_1, detail_2, detail_3, detail_4 = st.columns(4)

    with detail_1:
        st.metric(
            "Minimum",
            f"{result['minimum_temp']:.1f} °C"
        )

    with detail_2:
        st.metric(
            "Maximum",
            f"{result['maximum_temp']:.1f} °C"
        )

    with detail_3:
        st.metric(
            "Standart Sapma",
            f"{result['std_temp']:.2f} °C"
        )

    with detail_4:
        end_step = result["end_step"]

        st.metric(
            "Test Sonu Step",
            (
                f"{end_step:.0f}"
                if end_step is not None
                else "-"
            )
        )

    fig, png_bytes = _teflon_create_figure(result)

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.download_button(
        "Heatmap PNG İndir",
        data=png_bytes,
        file_name="teflon_block_heatmap.png",
        mime="image/png",
        use_container_width=True,
        key="teflon_download_heatmap"
    )

    st.subheader("Blok Sıcaklık Tablosu")

    grid_df = pd.DataFrame(
        result["temp_grid"],
        index=["Arka", "R2", "R3", "R4", "Ön"],
        columns=[
            f"C{i + 1}"
            for i in range(result["n_cols"])
        ]
    )

    st.dataframe(
        grid_df.style.format("{:.1f} °C"),
        use_container_width=True
    )

    block_df = pd.DataFrame({
        "Blok": result["sensor_cols"],
        "Sıcaklık (°C)": result["values"]
    })

    block_df["Fark / Minimum (°C)"] = (
        block_df["Sıcaklık (°C)"]
        - result["minimum_temp"]
    )

    with st.expander(
        "30 blok detay tablosu",
        expanded=False
    ):
        st.dataframe(
            block_df.style.format({
                "Sıcaklık (°C)": "{:.1f}",
                "Fark / Minimum (°C)": "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True
        )
