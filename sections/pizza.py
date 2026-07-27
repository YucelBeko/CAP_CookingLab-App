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

def run_pizza():
    # CSS to reduce top margin
    st.markdown(
        """
        <style>
        .block-container h1 { margin-top: -80px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Pizza Analysis")

    # ==========================================
    # 1. UNIVERSAL COLOR CENTERS (Fixed Model)
    # ==========================================
    # These coordinates represent the ideal "center" for each class in LAB color space.
    # Format: [L (Lightness), A (Green-Red), B (Blue-Yellow)]
    UNIVERSAL_CENTERS = np.array([
        [ 30.0,   2.0,   2.0],  # BURNT: Very dark, low saturation.
        [ 65.0,  18.0,  25.0],  # DARK BROWN: Dark but has reddish/orange tint.
        [145.0,  25.0,  50.0],  # BROWN: Golden/Cooked cheese color.
        [215.0,  10.0,  40.0],  # LIGHT BROWN: Pale yellow/white transition.
        [245.0,   1.0,   5.0]   # DOUGH: Raw dough, nearly white/grey.
    ], dtype=np.float32)

    CLASS_NAMES = ["Burnt", "Dark Brown", "Brown", "Light Brown", "Dough"]

    # ==========================================
    # 2. MASKING THRESHOLDS
    # ==========================================
    # -- HSV Thresholds (For Background/Tray Removal) --
    # Pixels darker than this V value are considered background/tray.
    # Increase if the tray is being detected as burnt pizza.
    HSV_V_BLACK_MAX = 40 
    
    # Glare detection: High Value (V) but Low Saturation (S) = White Glare.
    HSV_V_GLARE_MIN = 235
    HSV_S_GLARE_MAX = 30

    # -- LAB Thresholds (For Pizza Inclusion) --
    # L_MIN is kept low (1) to ensure dark burnt edges are included in the initial mask.
    LAB_L_MIN, LAB_L_MAX =   1, 255
    LAB_A_MIN, LAB_A_MAX =   0, 100
    LAB_B_MIN, LAB_B_MAX =   0, 130

    # -- Dough Specific Thresholds --
    DOUGH_L_MIN, DOUGH_L_MAX = 100, 300
    DOUGH_A_MIN, DOUGH_A_MAX = -5,  10
    DOUGH_B_MIN, DOUGH_B_MAX =  0,  60

    # ==========================================
    # COLOR PALETTE
    # ==========================================
    # Hex codes for visualization: Dough -> Burnt
    CUSTOM_HEX = ["#FF99FF", "#FFFFB5", "#FFFF66", "#CCCC00", "#FF0000"] 
    
    def make_custom_cmap():
        cmap = ListedColormap(CUSTOM_HEX, name="pizza5")
        bins = np.linspace(0.0, 1.0, 6)
        norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)
        return cmap, norm, bins

    CMAP5, NORM5, BINS5 = make_custom_cmap()

    # ==========================================
    # 3. MASKING LOGIC (PRUNING METHOD)
    # ==========================================
    def hsv_exclusion_mask(img):
        """Creates a mask of pixels to EXCLUDE (Background & Glare)."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        S, V = hsv[...,1], hsv[...,2]
        return (V <= HSV_V_BLACK_MAX) | ((V >= HSV_V_GLARE_MIN) & (S <= HSV_S_GLARE_MAX))

    def lab_inclusion_mask(img):
        """Creates a mask of pixels to INCLUDE based on broad color ranges."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
        L, A0, B0 = lab[...,0], lab[...,1]-128, lab[...,2]-128
        brown = (L >= LAB_L_MIN) & (L <= LAB_L_MAX) & \
                (A0 >= LAB_A_MIN) & (A0 <= LAB_A_MAX) & \
                (B0 >= LAB_B_MIN) & (B0 <= LAB_B_MAX)
        dough = (L >= DOUGH_L_MIN) & (L <= DOUGH_L_MAX) & \
                (A0 >= DOUGH_A_MIN) & (A0 <= DOUGH_A_MAX) & \
                (B0 >= DOUGH_B_MIN) & (B0 <= DOUGH_B_MAX)
        return (brown | dough)

    def keep_largest_component(mask_u8):
        """Removes small noise blobs, keeping only the largest object (the pizza)."""
        m = (mask_u8 > 0).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1: return m * 255
        # Index 0 is background, start from 1
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return ((lab == k).astype(np.uint8) * 255)

    def build_pizza_mask_pruned(img, keep_only_largest=True):
        """
        Generates the final binary mask using 'Pruning' logic.
        1. Creates a loose mask.
        2. Aggressively erodes (opens) it to sever connections between the pizza and background noise.
        3. Keeps the largest component.
        4. Dilates (closes) it back to restore the original shape.
        """
        # 1. Base Mask
        inc = lab_inclusion_mask(img)
        exc = hsv_exclusion_mask(img)
        m0  = (inc & (~exc)).astype(np.uint8) * 255
        
        # 2. Pruning (Morphological Open)
        # Using a large kernel (9x9) and high iterations (5) to strip away edge noise.
        # Decreasing iterations preserves more edge detail but may keep background noise.
        pruning_kernel = np.ones((1,1), np.uint8) 
        m_pruned = cv2.morphologyEx(m0, cv2.MORPH_OPEN, pruning_kernel, iterations=1)

        # 3. Component Selection
        if keep_only_largest:
            m_main = keep_largest_component(m_pruned)
        else:
            m_main = m_pruned

        # 4. Restoration (Morphological Close)
        # Fills in holes created by the pruning process.
        closing_kernel = np.ones((1,1), np.uint8)
        m_final = cv2.morphologyEx(m_main, cv2.MORPH_CLOSE, closing_kernel, iterations=20)
        
        return m_final
    
    def outline_on(img, mask_u8):
        """Draws a green contour around the detected mask."""
        out = img.copy()
        if (mask_u8 > 0).any():
            cnts, _ = cv2.findContours((mask_u8 > 0).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Thickness 20 for visibility
            cv2.drawContours(out, cnts, -1, (0, 255, 0), 20, lineType=cv2.LINE_AA)
        return out

    # ==========================================
    # 4. ANALYSIS ENGINE (CLASSIFICATION)
    # ==========================================
    def predict_universal_map(img_bgr, mask_u8, centers, class_order):
        H, W = mask_u8.shape
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
        L, A0, B0 = lab[...,0], lab[...,1]-128, lab[...,2]-128
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        S, V = hsv[...,1].astype(np.int16), hsv[...,2].astype(np.int16)
        m = (mask_u8 > 0)

        class_idx = np.full((H,W), -1, dtype=np.int16)
        if not np.any(m):
            counts = {c:0 for c in class_order}; perc = {c:0.0 for c in class_order}; dom = None
            return class_idx, counts, perc, dom

        # --- Step 1: KNN Classification (Euclidean Distance) ---
        X = np.stack([L[m], A0[m], B0[m]], axis=1).astype(np.float32)
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        class_idx[m] = labels

        burnt_i = 0 
        dark_i  = 1 
        brown_i = 2 
        light_i = 3 
        dough_i = 4 

        # --- Step 2: Hard Rules & Logic Corrections ---

        # Rule A: Charcoal/Ash Detection
        # L < 85: Pixels this dark are burnt regardless of color.
        is_charcoal = (L < 85) & m 
        
        # Matte Black Detection: Dark pixels (up to L=115) with very low color (A&B < 20).
        is_ash = (L >= 85) & (L < 115) & (A0 < 20) & (B0 < 20) & m

        # Shiny Burnt Detection: High Value (V) but low Saturation (S).
        is_shiny_burnt = (V < 190) & (S < 40) & (L < 180) & m

        # Açık/gri-bej hamur koruması
        # Sadece ash/shiny_burnt gibi kararsız alanlardan hamuru korur.
        # Çok koyu gerçek yanığı korumaz.
        dough_protection = (
            (L >= 105) &
            (V >= 120) &
            (S <= 75) &
            (A0 >= -8) & (A0 <= 22) &
            (B0 >= -5) & (B0 <= 38) &
            m
        )
        
        true_burnt_mask = (
            is_charcoal |
            ((is_ash | is_shiny_burnt) & (~dough_protection))
        )

        class_idx[true_burnt_mask] = burnt_i

        # Rule B: Shadow/Crack Protection
        # If labeled Burnt, but has significant Red/Yellow (A or B > 20), it's just a dark shadow.
        shadow_in_burnt = (class_idx == burnt_i) & ((A0 > 20) | (B0 > 20)) & m
        class_idx[shadow_in_burnt] = dark_i

        # Rule C: Brown Expansion
        # If labeled Light Brown but is darker than L=185, force it to Brown.
        should_be_brown = (class_idx == light_i) & (L < 185) & m
        class_idx[should_be_brown] = brown_i

        # Rule D: Dough Tolerance
        # If labeled Dough but has some color (A>5 or B>25), it's likely Light Brown.
        fake_dough = (
            (class_idx == dough_i) &
            (S > 95) &
            ((A0 > 10) | (B0 > 42)) &
            m
        )
        class_idx[fake_dough] = light_i

        
        # Rule E: Pale Dough Override
        # Açık, düşük doygunluklu, gri-bej hamur alanları Brown'a kaçmasın.
        # Bu kural özellikle alt yüzeydeki / iyi kızarmamış hamur bölgelerini korur.
        
        pale_dough_strong = (
            (L >= 130) &
            (V >= 125) &
            (S <= 85) &
            (A0 >= -10) & (A0 <= 22) &
            (B0 >= -5) & (B0 <= 36) &
            m
        )
        
        pale_dough_soft = (
            (L >= 115) &
            (V >= 115) &
            (S <= 105) &
            (A0 >= -12) & (A0 <= 28) &
            (B0 >= -5) & (B0 <= 46) &
            m
        )
        
        # Strong alan direkt Dough.
        class_idx[pale_dough_strong] = dough_i
        
        # Soft alan Brown/Dark/Burnt'a kaçtıysa en azından Light Brown'a çek.
        class_idx[
            pale_dough_soft &
            (
                (class_idx == burnt_i) |
                (class_idx == dark_i) |
                (class_idx == brown_i)
            )
        ] = light_i

        # --- Statistics ---
        counts = {c: int(np.count_nonzero(class_idx == i)) for i,c in enumerate(class_order)}
        tot = sum(counts.values())
        perc = {c: (counts[c]/tot if tot>0 else 0.0) for c in class_order}
        dominant = max(class_order, key=lambda c: counts[c]) if tot>0 else None
        
        return class_idx, counts, perc, dominant

    # ==========================================
    # 5. VISUALIZATION HELPERS
    # ==========================================
    def pct_line(perc: dict, order: list[str]) -> str:
        """Returns a single line string of percentages."""
        return " | ".join(f"{k}: {perc[k]*100:.1f}%" for k in order)

    def heatmap_overlay(img_bgr, class_idx, class_order, alpha=0.6, cmap=CMAP5, norm=NORM5):
        """Overlays the analyzed colors onto the original image."""
        n = len(class_order)
        score = np.zeros(class_idx.shape, dtype=np.float32)
        valid = (class_idx >= 0)
        # Map indices to score (0..1) for colormap
        score[valid] = 1.0 - (class_idx[valid].astype(np.float32) / (n-1))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        colored = (sm.to_rgba(score)[...,:3] * 255).astype(np.uint8)
        colored_bgr = colored[..., ::-1]
        out = img_bgr.copy().astype(np.float32)
        out[valid] = (1-alpha)*out[valid] + alpha*colored_bgr[valid]
        return np.clip(out,0,255).astype(np.uint8)

    def show_heatmap_figure(img_bgr, overlay_bgr, cmap=CMAP5, norm=NORM5, bins=BINS5):
        """Creates the Matplotlib figure with original image, heatmap, and colorbar."""
        fig, axes = plt.subplots(1,3, figsize=(14,5), gridspec_kw={"width_ratios":[1,1,0.06]}, dpi=100)
        axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); axes[0].set_title(""); axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)); axes[1].set_title(""); axes[1].axis("off")
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=axes[2])
        tick_pos = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        cbar.set_ticks(tick_pos); cbar.set_ticklabels(["Dough","Light Brown","Brown","Dark Brown","Burnt"])
        plt.tight_layout()
        return fig

    # ==========================================
    # 6. MAIN UI EXECUTION
    # ==========================================
    
    # Sidebar for upload only
    with st.sidebar:
        st.markdown("""<style>section[data-testid="stSidebar"] > div:first-child {padding-top: 0rem; margin-top: -3rem;}</style>""", unsafe_allow_html=True)
        st.sidebar.title("Settings")
        # Regional analysis sliders removed as requested.

    files = st.file_uploader("Upload Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if not files:
        st.info("Please upload images to begin analysis.")
        return

    decoded = []
    for uf in files:
        data = np.frombuffer(uf.getvalue(), np.uint8)
        decoded.append((uf.name, cv2.imdecode(data, cv2.IMREAD_COLOR)))

    grid = st.columns(2)
    
    for i, (name, img) in enumerate(decoded):
        # 1. Generate Mask
        mask = build_pizza_mask_pruned(img)
        
        # 2. Analyze Colors
        class_idx, counts, perc, dominant = predict_universal_map(
            img, mask, UNIVERSAL_CENTERS, CLASS_NAMES
        )
        
        # 3. Create Heatmap
        heat_over = heatmap_overlay(img, class_idx, CLASS_NAMES, alpha=0.6, cmap=CMAP5, norm=NORM5)

        col = grid[i % 2]
        with col:
            st.subheader(name)
            
            # Show original with green contour outline
            base = outline_on(img, mask)
            fig = show_heatmap_figure(base, heat_over, cmap=CMAP5, norm=NORM5, bins=BINS5)
            st.pyplot(fig, clear_figure=True); plt.close(fig)

            # Percentage Line
            st.markdown(
                f"<div style='text-align:center; margin-top:-8px; margin-bottom:10px'>"
                f"<b>Yüzdeler:</b> {pct_line(perc, CLASS_NAMES)}</div>",
                unsafe_allow_html=True
            )

            # Data prep for Pie Chart
            display_order = ["Dough", "Light Brown", "Brown", "Dark Brown", "Burnt"]
            display_colors = ["#FF99FF", "#FFFFB5", "#FFFF66", "#CCCC00", "#FF0000"]
            counts_list = [counts.get(k, 0) for k in display_order]
            total = sum(counts_list) if sum(counts_list) > 0 else 1
            perc_list = [100.0 * c / total for c in counts_list]

            # ==================================================
            # SMART PIE CHART (ÇAKIŞMA ÖNLEYİCİ + TAŞMA KORUMALI)
            # ==================================================
            # 1. Figür oluştur (DPI artırıldı, boyut ayarlandı)
            fig_pie, ax_pie = plt.subplots(figsize=(6, 4), dpi=120)
            
            # Pasta dilimlerini çiz
            wedges, _ = ax_pie.pie(perc_list, colors=display_colors, startangle=90, counterclock=False)
            
            # Etiket kutusu stili
            bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.5)
            kw = dict(arrowprops=dict(arrowstyle="-", lw=0.5), bbox=bbox_props, zorder=0, va="center")
            
            # 2. Etiket verilerini topla
            labels_to_draw = []
            for j, p in enumerate(wedges):
                val = perc_list[j]
                
                # Eğer %0 olanları da görmek istiyorsan alttaki satırı sil veya yorum yap:
                if val <= 0.5: continue 
                
                # Açıyı hesapla (Dilimin tam ortası)
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                
                # Koordinatları bul
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                
                # Etiket hangi tarafta? (1: Sağ, -1: Sol)
                side = 1 if x >= 0 else -1
                
                labels_to_draw.append({
                    "text": f"{display_order[j]}\n%{val:.1f}",
                    "x": x,
                    "y": y,
                    "ang": ang,
                    "side": side,
                    "val": val
                })

            # 3. Sağ ve Sol taraftaki etiketleri ayır ve Yüksekliklerine (Y) göre sırala
            # Bu sıralama, yukarıdan aşağıya doğru yerleştirme yapmamızı sağlar.
            right_labels = sorted([l for l in labels_to_draw if l["side"] == 1], key=lambda k: k["y"], reverse=True)
            left_labels  = sorted([l for l in labels_to_draw if l["side"] == -1], key=lambda k: k["y"], reverse=True)

            # 4. Akıllı Yerleştirme Fonksiyonu
            def draw_side_labels(label_group, is_left=False):
                # Başlangıç tavan noktası (Grafiğin en tepesinden biraz yukarı)
                last_y = 1.5 
                min_dist = 0.30 # Etiketler arası minimum dikey mesafe (Çakışmayı önler)

                for lbl in label_group:
                    # İdeal Y pozisyonu (Dilimin kendi hizası)
                    ideal_y = lbl["y"] * 1.15
                    
                    # Eğer ideal pozisyon, bir önceki etikete çok yakınsa, onu aşağı it.
                    if last_y - ideal_y < min_dist:
                        target_y = last_y - min_dist
                    else:
                        target_y = ideal_y

                    # Çok aşağı gitmemesi için taban sınırı koy (-1.5'in altına inmesin)
                    target_y = max(target_y, -1.5)
                    
                    # Bir sonraki etiket için referans noktasını güncelle
                    last_y = target_y

                    # Çizim ayarları
                    align = "right" if is_left else "left"
                    connection_style = f"angle,angleA=0,angleB={lbl['ang']}"
                    kw["arrowprops"].update({"connectionstyle": connection_style})
                    
                    # Etiketi bas
                    ax_pie.annotate(lbl["text"], 
                                    xy=(lbl["x"], lbl["y"]), 
                                    # X ekseninde biraz dışarı aç (1.4 katı), Y ekseninde hesaplanan yere koy
                                    xytext=(1.4 * lbl["side"], target_y),
                                    horizontalalignment=align, 
                                    fontsize=9, **kw)

            # Fonksiyonu her iki taraf için çalıştır
            draw_side_labels(right_labels, is_left=False)
            draw_side_labels(left_labels, is_left=True)

            # Grafiği çiz (bbox_inches='tight' ile kesilmeyi önle)
            st.pyplot(fig_pie, clear_figure=True, use_container_width=False, bbox_inches='tight', pad_inches=0.2)
            plt.close(fig_pie)
            # ==================================================

            
            # Text Summary
            burnt_pct = perc["Burnt"] * 100
            dough_pct = perc["Dough"] * 100
            cooked_pct = (perc["Dark Brown"] + perc["Brown"] + perc["Light Brown"]) * 100
            
            st.markdown(
                f"<div style='text-align:center; margin-top:-10px; margin-bottom:15px'>"
                f" <b>Undercooked:</b> {dough_pct:.1f}%   |   "
                f" <b>Cooked:</b> {cooked_pct:.1f}%   |   "
                f" <b>Overcooked:</b> {burnt_pct:.1f}%"
                f"</div>",
                unsafe_allow_html=True
            )
            st.divider()
