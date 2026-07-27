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

def run_borek():
    st.markdown(
        """
        <style>
        .block-container h1 { margin-top: -60px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # =========================================================
    # Yardımcılar
    # =========================================================
    def hex_to_bgr(hex_code):
        h = hex_code.lstrip("#")
        return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

    def simple_mask_white_bg(bgr, V_thresh=230, min_area=20000):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Beyaz arka plan dışarıda kalacak şekilde threshold
        m = (hsv[..., 2] < V_thresh).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]

        mask = np.zeros_like(m)

        if cnts:
            biggest = max(cnts, key=cv2.contourArea)
            cv2.drawContours(mask, [biggest], -1, 255, -1)

        return mask

    def cnt_color_group(flat_bgr_pixels, target_colors):
        total_count = 0
        for c in target_colors:
            total_count += np.count_nonzero(np.all(flat_bgr_pixels == c, axis=1))
        return total_count

    # =========================================================
    # Renk Eşleme
    # =========================================================
    COLOR_MAP = {
        "#FFFEB5": "#818100",
        "#FEFF94": "#666732",
        "#FEFE7A": "#01FEFF",
        "#FFD15C": "#32CDFF",

        "#EDB256": "#FF99FF",
        "#E4A741": "#FF00FF",

        "#C38F49": "#FFFE67",
        "#B89057": "#CDCC01",
        "#996F3C": "#0167CC",
        "#916533": "#0101CC",
        "#845A37": "#01FF01",
        "#6C5033": "#01CC01",

        "#68533E": "#FE0001",
        "#404032": "#C10100"
    }

    SRC_BGR = np.array([hex_to_bgr(h) for h in COLOR_MAP.keys()], dtype=np.uint8)
    DST_BGR = np.array([hex_to_bgr(h) for h in COLOR_MAP.values()], dtype=np.uint8)
    SRC_LAB = cv2.cvtColor(SRC_BGR[np.newaxis, :, :], cv2.COLOR_BGR2LAB)[0]

    def make_exclusion_mask_with_canvas(bgr, key_prefix):
        """
        streamlit-image-coordinates ile çoklu polygon tabanlı exclusion mask.
        Web app'te canvas'tan daha stabil çalışır.
        """
        from PIL import Image

        H, W = bgr.shape[:2]

        with st.expander("Ispanak / taşan bölge hariç tutma", expanded=False):
            enabled = st.checkbox(
                "Bu görselde analizden çıkarılacak bölgeleri seç",
                key=f"{key_prefix}_exclude_enabled"
            )

            if not enabled:
                return None

            if streamlit_image_coordinates is None:
                st.error(
                    "Bu seçim için `streamlit-image-coordinates` paketi gerekli. "
                    "requirements.txt içine `streamlit-image-coordinates==0.1.9` ekle."
                )
                return None

            st.caption(
                "Hariç tutulacak bölge için görsel üzerinde noktalar seç. "
                "Her bölge için en az 3 nokta seçip 'Bölgeyi Tamamla' butonuna bas. "
                "Birden fazla bölge ekleyebilirsin."
            )

            current_points_key = f"{key_prefix}_exclude_current_points"
            polygons_key = f"{key_prefix}_exclude_polygons"
            last_click_key = f"{key_prefix}_exclude_last_click"
            click_gen_key = f"{key_prefix}_exclude_click_gen"

            if current_points_key not in st.session_state:
                st.session_state[current_points_key] = []

            if polygons_key not in st.session_state:
                st.session_state[polygons_key] = []

            if last_click_key not in st.session_state:
                st.session_state[last_click_key] = None

            if click_gen_key not in st.session_state:
                st.session_state[click_gen_key] = 0

            max_display_width = 700
            scale = min(max_display_width / W, 1.0)

            display_w = int(W * scale)
            display_h = int(H * scale)

            display_bgr = cv2.resize(
                bgr,
                (display_w, display_h),
                interpolation=cv2.INTER_AREA
            )

            # Daha önce tamamlanan polygonları çiz
            for poly in st.session_state[polygons_key]:
                pts_disp = np.array(
                    [[int(x * scale), int(y * scale)] for x, y in poly],
                    dtype=np.int32
                ).reshape((-1, 1, 2))

                overlay = display_bgr.copy()
                cv2.fillPoly(overlay, [pts_disp], (0, 0, 255))
                display_bgr = cv2.addWeighted(display_bgr, 0.8, overlay, 0.2, 0)
                cv2.polylines(display_bgr, [pts_disp], isClosed=True, color=(0, 0, 255), thickness=2)

            # O an çizilmekte olan polygonu çiz
            current_points = st.session_state[current_points_key]

            for idx, (x_orig, y_orig) in enumerate(current_points, start=1):
                x_disp = int(x_orig * scale)
                y_disp = int(y_orig * scale)

                cv2.circle(display_bgr, (x_disp, y_disp), 6, (255, 0, 0), -1)
                cv2.putText(
                    display_bgr,
                    str(idx),
                    (x_disp + 8, y_disp - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )

            if len(current_points) >= 2:
                pts_disp = np.array(
                    [[int(x * scale), int(y * scale)] for x, y in current_points],
                    dtype=np.int32
                ).reshape((-1, 1, 2))

                cv2.polylines(display_bgr, [pts_disp], isClosed=False, color=(255, 0, 0), thickness=2)

            display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            display_pil = Image.fromarray(display_rgb)

            clicked = streamlit_image_coordinates(
                display_pil,
                key=f"{key_prefix}_exclude_click_{st.session_state[click_gen_key]}"
            )

            if clicked is not None:
                x_display = int(clicked["x"])
                y_display = int(clicked["y"])

                click_signature = (x_display, y_display)

                if st.session_state[last_click_key] != click_signature:
                    st.session_state[last_click_key] = click_signature

                    x_original = int(x_display / scale)
                    y_original = int(y_display / scale)

                    x_original = max(0, min(W - 1, x_original))
                    y_original = max(0, min(H - 1, y_original))

                    st.session_state[current_points_key].append((x_original, y_original))
                    st.rerun()

            c1, c2, c3 = st.columns(3)

            with c1:
                if st.button("Son Noktayı Sil", key=f"{key_prefix}_exclude_undo"):
                    if st.session_state[current_points_key]:
                        st.session_state[current_points_key].pop()
                    st.rerun()

            with c2:
                if st.button("Bölgeyi Tamamla", key=f"{key_prefix}_exclude_finalize"):
                    if len(st.session_state[current_points_key]) >= 3:
                        st.session_state[polygons_key].append(
                            st.session_state[current_points_key].copy()
                        )
                        st.session_state[current_points_key] = []
                        st.session_state[last_click_key] = None
                        st.session_state[click_gen_key] += 1
                        st.rerun()
                    else:
                        st.warning("Bir bölgeyi tamamlamak için en az 3 nokta seç.")

            with c3:
                if st.button("Tüm Bölgeleri Sıfırla", key=f"{key_prefix}_exclude_reset"):
                    st.session_state[current_points_key] = []
                    st.session_state[polygons_key] = []
                    st.session_state[last_click_key] = None
                    st.session_state[click_gen_key] += 1
                    st.rerun()

            st.write(f"Tamamlanan bölge sayısı: **{len(st.session_state[polygons_key])}**")
            st.write(f"Aktif polygon nokta sayısı: **{len(st.session_state[current_points_key])}**")

            if len(st.session_state[polygons_key]) == 0:
                return None

            exclusion_mask = np.zeros((H, W), dtype=np.uint8)

            for poly in st.session_state[polygons_key]:
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(exclusion_mask, [pts], 255)

            if np.count_nonzero(exclusion_mask) == 0:
                return None

            ok, encoded = cv2.imencode(".png", exclusion_mask)

            if not ok:
                return None

            st.success("Seçilen polygon bölgeler analizden çıkarılacak.")
            return encoded.tobytes()

    
    def recolor_by_lab(img_bgr, mask):
        H, W = img_bgr.shape[:2]

        lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab_flat = lab_img.reshape(-1, 3).astype(np.int32)

        src32 = SRC_LAB.astype(np.int32)

        recolored = np.full((lab_flat.shape[0], 3), 255, dtype=np.uint8)

        mask_flat = mask.reshape(-1) > 0
        selected_pixels = lab_flat[mask_flat]

        if selected_pixels.size == 0:
            return recolored.reshape(H, W, 3)

        # Her pikselin tüm referans renklere uzaklığını aynı anda hesapla
        diff = selected_pixels[:, None, :] - src32[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)

        nearest_idx = np.argmin(dist2, axis=1)

        recolored[mask_flat] = DST_BGR[nearest_idx]

        return recolored.reshape(H, W, 3)
    
    # =========================================================
    # Ana Pişmişlik Sınıfları
    # =========================================================
    RAW_BGR = np.array(
        [hex_to_bgr(c) for c in [
            "#818100",
            "#666732",
            "#01FEFF",
            "#32CDFF"
        ]],
        dtype=np.uint8
    )

    COOKED_BGR = np.array(
        [hex_to_bgr(c) for c in [
            "#FF99FF",
            "#FF00FF",
            "#FFFE67",
            "#CDCC01",
            "#0167CC",
            "#0101CC",
            "#01FF01",
            "#01CC01"
        ]],
        dtype=np.uint8
    )

    BURNT_BGR = np.array(
        [hex_to_bgr(c) for c in [
            "#FE0001",
            "#C10100"
        ]],
        dtype=np.uint8
    )

    MAIN_LABELS = ["Undercooked", "Cooked", "Overcooked"]
    MAIN_COLORS = ["#FFFF66", "#FF9900", "#C10100"]

    # =========================================================
    # Kızarma Grupları (3 ana grup)
    # =========================================================
    BROWNING_GROUPS = [
    {
        "name": "Açık Kızarma",
        "dst_hex_list": ["#FFFE67", "#CDCC01"],
        "plot_color": "#E5D33F"
    },
    {
        "name": "Orta Kızarma",
        "dst_hex_list": ["#0167CC", "#0101CC"],
        "plot_color": "#2D7BD8"
    },
    {
        "name": "Çok Kızarma",
        "dst_hex_list": ["#01FF01", "#01CC01"],
        "plot_color": "#31C85A"
    }
]

    BROWNING_WEIGHTS_BY_SURFACE = {
        # Sıra: Açık Kızarma, Orta Kızarma, Çok Kızarma
        # Üst yüzeyde açık kızarma daha düşük puanlıdır.
        "ust": [20, 55, 90],

        # Alt yüzeyden daha az kızarma beklendiği için daha toleranslıdır.
        "alt": [45, 75, 95]
    }

    BROWNING_GROUP_BGR = [
        np.array([hex_to_bgr(c) for c in group["dst_hex_list"]], dtype=np.uint8)
        for group in BROWNING_GROUPS
    ]

    BROWNING_GROUP_COLORS = [
        group["plot_color"] for group in BROWNING_GROUPS
    ]

    # =========================================================
    # Session State Başlat
    # =========================================================
    for key in [
        "ust_borek_files", "alt_borek_files",
        "ust_borek_uploader_key", "alt_borek_uploader_key"
    ]:
        if key not in st.session_state:
            if "files" in key:
                st.session_state[key] = {}
            else:
                st.session_state[key] = 0

    # =========================================================
    # Grafik Fonksiyonları
    # =========================================================
    def draw_waffle_chart(main_perc):
        total_squares = 100

        exact = np.array(main_perc) / 100.0 * total_squares
        base = np.floor(exact).astype(int)

        remainder = total_squares - base.sum()
        fractional_order = np.argsort(exact - base)[::-1]

        for i in fractional_order[:remainder]:
            base[i] += 1

        waffle_grid = []
        for class_id, square_count in enumerate(base):
            waffle_grid.extend([class_id] * square_count)

        waffle_grid = waffle_grid[:100]
        while len(waffle_grid) < 100:
            waffle_grid.append(int(np.argmax(base)))

        waffle_arr = np.array(waffle_grid).reshape((10, 10))
        cmap_waffle = plt.cm.colors.ListedColormap(MAIN_COLORS)

        fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=100)
        ax.matshow(waffle_arr, cmap=cmap_waffle, vmin=0, vmax=2)

        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)

        ax.set_xticks([])
        ax.set_yticks([])

        legend_elements = [
            Patch(
                facecolor=MAIN_COLORS[i],
                edgecolor="w",
                label=f"{MAIN_LABELS[i]} (%{main_perc[i]:.1f})"
            )
            for i in range(3)
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=3,
            frameon=False,
            fontsize=9
        )

        ax.set_title("Pişmişlik Dağılımı", fontsize=11, pad=10)

        return fig

    def draw_browning_pie_chart(main_counts, browning_counts):
        fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=100)

        pie_labels = (
            ["Undercooked"] +
            [group["name"] for group in BROWNING_GROUPS] +
            ["Overcooked"]
        )

        pie_counts = (
            [main_counts[0]] +
            browning_counts +
            [main_counts[2]]
        )

        # Undercooked pembe: açık kızarma sarısıyla karışmasın.
        pie_colors = (
            ["#FF99FF"] +
            BROWNING_GROUP_COLORS +
            [MAIN_COLORS[2]]
        )

        total = max(sum(pie_counts), 1)

        filtered_counts = []
        filtered_colors = []
        legend_labels = []

        for label, count, color in zip(pie_labels, pie_counts, pie_colors):
            if count > 0:
                pct = 100.0 * count / total
                filtered_counts.append(count)
                filtered_colors.append(color)
                legend_labels.append(f"{label}  %{pct:.1f}")

        if len(filtered_counts) == 0:
            ax.text(0.5, 0.5, "Veri yok", ha="center", va="center", fontsize=11)
            ax.axis("off")
            return fig

        def autopct_func(pct):
            # Küçük dilimler üst üste binmesin diye yüzdeyi sadece büyük dilimlerde göster.
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

        ax.set_title("Kızarma + Pişmişlik Dağılımı", fontsize=11, pad=8)
        ax.axis("equal")

        ax.legend(
            wedges,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
            frameon=False,
            fontsize=8
        )

        fig.subplots_adjust(bottom=0.28, top=0.88)
        return fig

    # =========================================================
    # Tek Görsel Analizi
    # =========================================================
    @st.cache_data(show_spinner=False)
    def analyze_image(file_bytes, surface_prefix, exclusion_mask_png_bytes=None):
        arr = np.frombuffer(file_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if bgr is None:
            return None

        
        mask = simple_mask_white_bg(bgr)

        if exclusion_mask_png_bytes is not None:
            exclusion_arr = np.frombuffer(exclusion_mask_png_bytes, np.uint8)
            exclusion_mask = cv2.imdecode(exclusion_arr, cv2.IMREAD_GRAYSCALE)

            if exclusion_mask is not None:
                if exclusion_mask.shape != mask.shape:
                    exclusion_mask = cv2.resize(
                        exclusion_mask,
                        (mask.shape[1], mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                mask[exclusion_mask > 0] = 0

        heat_bgr = recolor_by_lab(bgr, mask)

        flat = heat_bgr[mask > 0]

        if flat.size == 0:
            return None

        raw_count = cnt_color_group(flat, RAW_BGR)
        cooked_count = cnt_color_group(flat, COOKED_BGR)
        burnt_count = cnt_color_group(flat, BURNT_BGR)

        main_counts = [raw_count, cooked_count, burnt_count]
        total = max(sum(main_counts), 1)

        main_perc = [100.0 * c / total for c in main_counts]

        # Kızarma grupları
        browning_counts = []
        for group_colors in BROWNING_GROUP_BGR:
            group_count = cnt_color_group(flat, group_colors)
            browning_counts.append(group_count)

        # Cooked içinde olup browning gruplarına girmeyen erken/açık pişmiş tonları
        # Açık Kızarma grubuna ekliyoruz.
        other_cooked_count = max(cooked_count - sum(browning_counts), 0)
        if len(browning_counts) > 0:
            browning_counts[0] += other_cooked_count

        browning_total = sum(browning_counts)

        if browning_total > 0:
            browning_perc_inside_cooked = [
                100.0 * c / browning_total for c in browning_counts
            ]

            surface_weights = BROWNING_WEIGHTS_BY_SURFACE.get(
                surface_prefix,
                BROWNING_WEIGHTS_BY_SURFACE["ust"]
            )

            weighted_score = sum(
                c * surface_weights[i]
                for i, c in enumerate(browning_counts)
            )

            browning_score_0_to_100 = weighted_score / browning_total
            dominant_browning_index = int(np.argmax(browning_counts))
            dominant_browning_name = BROWNING_GROUPS[dominant_browning_index]["name"]
        else:
            browning_perc_inside_cooked = [0.0 for _ in BROWNING_GROUPS]
            browning_score_0_to_100 = 0.0
            dominant_browning_name = "Yok"

        return {
            "bgr": bgr,
            "heat_bgr": heat_bgr,
            "main_counts": main_counts,
            "main_perc": main_perc,
            "browning_counts": browning_counts,
            "browning_perc_inside_cooked": browning_perc_inside_cooked,
            "browning_score_0_to_100": browning_score_0_to_100,
            "dominant_browning_name": dominant_browning_name
        }

    # =========================================================
    # Bölüm Render Fonksiyonu
    # =========================================================
    def render_surface_section(section_title, prefix):
        files_key = f"{prefix}_borek_files"
        uploader_key_name = f"{prefix}_borek_uploader_key"

        st.header(section_title)

        upload_col1, upload_col2 = st.columns([4, 1])

        with upload_col1:
            uploads = st.file_uploader(
                f"{section_title} için görsel yükle",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"{prefix}_uploader_{st.session_state[uploader_key_name]}"
            )

        with upload_col2:
            st.write("")
            st.write("")
            if st.button("Listeyi Temizle", key=f"{prefix}_clear_all"):
                st.session_state[files_key] = {}
                st.session_state[uploader_key_name] += 1
                st.rerun()

        if uploads:
            for up in uploads:
                file_bytes = up.getvalue()
                file_hash = hashlib.md5(file_bytes).hexdigest()
                file_id = f"{up.name}_{len(file_bytes)}_{file_hash}"

                if file_id not in st.session_state[files_key]:
                    st.session_state[files_key][file_id] = {
                        "name": up.name,
                        "bytes": file_bytes
                    }

        files = list(st.session_state[files_key].items())

        if not files:
            st.info(f"{section_title} için henüz görsel yüklenmedi.")
            return

        st.success(f"{len(files)} adet görsel analiz listesinde.")

        analyzed_items = []

        for file_id, item in files:

            arr_preview = np.frombuffer(item["bytes"], np.uint8)
            bgr_preview = cv2.imdecode(arr_preview, cv2.IMREAD_COLOR)

            exclusion_mask_png_bytes = None

            if bgr_preview is not None:
                exclusion_mask_png_bytes = make_exclusion_mask_with_canvas(
                        bgr_preview,
                        key_prefix=f"{prefix}_{file_id}"
                    )

            result = analyze_image(
                item["bytes"],
                prefix,
                exclusion_mask_png_bytes
            )

            if result is None:
                st.warning(f"{item['name']} okunamadı veya maske oluşturulamadı, atlandı.")
                continue

            analyzed_items.append({
                "file_id": file_id,
                "name": item["name"],
                "result": result
            })

        if not analyzed_items:
            st.warning("Analiz edilebilir görsel bulunamadı.")
            return

        for idx, item in enumerate(analyzed_items, start=1):
            file_id = item["file_id"]
            name = item["name"]
            r = item["result"]

            bgr = r["bgr"]
            heat_bgr = r["heat_bgr"]
            main_perc = r["main_perc"]
            main_counts = r["main_counts"]
            browning_counts = r["browning_counts"]
            browning_perc_inside_cooked = r["browning_perc_inside_cooked"]
            browning_score_0_to_100 = r["browning_score_0_to_100"]
            dominant_browning_name = r["dominant_browning_name"]

            title_col, remove_col = st.columns([5, 1])

            with title_col:
                st.markdown(f"## {idx}. Analiz: `{name}`")

            with remove_col:
                if st.button("Kaldır", key=f"{prefix}_remove_{file_id}"):
                    if file_id in st.session_state[files_key]:
                        del st.session_state[files_key][file_id]
                    st.rerun()

            # =====================================================
            # 1. SATIR -> Orijinal | Analiz
            # =====================================================
            row1_col1, row1_col2 = st.columns(2, gap="medium")

            with row1_col1:
                st.subheader("Orijinal")
                st.image(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

            with row1_col2:
                st.subheader("Analiz")
                st.image(
                    cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

            # =====================================================
            # 2. SATIR -> Waffle | Browning Pie
            # =====================================================
            row2_col1, row2_col2 = st.columns(2, gap="medium")

            with row2_col1:
                st.subheader("Pişmişlik Dağılımı")
                fig_waffle = draw_waffle_chart(main_perc)
                st.pyplot(fig_waffle, clear_figure=True, use_container_width=True)
                plt.close(fig_waffle)

            with row2_col2:
                st.subheader("Kızarma + Pişmişlik Dağılımı")
                fig_browning = draw_browning_pie_chart(
                    main_counts,
                    browning_counts
                )
                st.pyplot(fig_browning, clear_figure=True, use_container_width=True)
                plt.close(fig_browning)

            # =====================================================
            # 3. SATIR -> Özet Tablo
            # =====================================================
            st.subheader("Özet")

            summary_df = pd.DataFrame([{
                "Undercooked (%)": round(main_perc[0], 2),
                "Cooked (%)": round(main_perc[1], 2),
                "Overcooked (%)": round(main_perc[2], 2),
                "Kızarma Skoru / 100": round(browning_score_0_to_100, 1),
                "Baskın Kızarma Tipi": dominant_browning_name
            }])

            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.divider()

    # =========================================================
    # Sayfa
    # =========================================================
    st.title("Börek Analizi")

    render_surface_section("Üst Yüzey Analizi", "ust")
    st.markdown("---")
    render_surface_section("Alt Yüzey Analizi", "alt")
