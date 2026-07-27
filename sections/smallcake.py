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

SHADE_THRESHOLDS = [
    (7.2, 17), (9.3, 16), (12.2, 15), (16.4, 14), (20.1, 13),
    (22.9, 12), (26.5, 11), (31.7, 10), (38.5, 9), (46.9, 8),
    (54.2, 7), (64.3, 6), (75.2, 5)
]

# Renk Kodu -> BGR
SHADE_COLOR_MAP_BGR = {
    4:  (73,  74,  38),   # Çok Açık
    5:  (45,  64,  54),
    6:  (0,  255,  255),  # Sarı
    7:  (0,  192,  255),
    8:  (128, 128, 255),  # Kırmızımsı
    9:  (255,   0, 255),  # Mor
    10: (128, 255, 255),
    11: (0, 128, 128),
    12: (255, 128, 128),
    13: (255,   0, 128),
    14: (0, 255,   0),    # Yeşil
    15: (0, 128,   0),
    16: (255,   0,   0),  # Mavi 
    17: (128,   0,   0)   # Koyu
}

SHADE_COLOR_MAP_RGB = {k: (v[2]/255.0, v[1]/255.0, v[0]/255.0) for k, v in SHADE_COLOR_MAP_BGR.items()}

# ==========================================
# 2. YARDIMCI FONKSİYONLAR
# ==========================================

def apply_clahe_lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)

def get_shade_number(ry_val):
    for limit, shade in SHADE_THRESHOLDS:
        if ry_val < limit:
            return shade
    return 4

def robust_cake_mask(img_bgr):
    alpha = 1.2
    beta = 30
    image_for_masking = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    gray = cv2.cvtColor(image_for_masking, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    cake_mask = cv2.bitwise_not(white_mask)
    return cake_mask

def get_inscribed_circle(mask_u8):
    """
    Bir maskenin içine sığabilecek EN BÜYÜK daireyi (Inscribed Circle) bulur.
    Bunun için Distance Transform kullanır.
    """
    # Her pikselin en yakın sıfıra (siyaha) olan uzaklığını hesapla
    dist_transform = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    
    # En büyük uzaklık değeri = Yarıçap
    # En büyük uzaklığın olduğu yer = Merkez
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    
    radius = max_val
    center = max_loc # (x, y)
    
    return center, radius

def get_13_zones(mask_shape, center, r_max):
    H, W = mask_shape[:2]
    cx, cy = int(center[0]), int(center[1])
    r1 = int(r_max)
    r2 = int(0.6 * r1)
    r3 = int(0.3 * r1)
    
    zones = []
    
    # 1. Merkez (C)
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r3, 255, -1)
    zones.append(("C", m))

    # 2. İç Halka (M1-M4)
    for i in range(4):
        m = np.zeros((H, W), dtype=np.uint8)
        angle_start = i * 90
        angle_end = (i + 1) * 90
        cv2.ellipse(m, (cx, cy), (r2, r2), 0, angle_start, angle_end, 255, -1)
        cv2.circle(m, (cx, cy), r3, 0, -1)
        zones.append((f"M{i+1}", m))

    # 3. Dış Halka (O1-O8)
    for i in range(8):
        m = np.zeros((H, W), dtype=np.uint8)
        angle_start = i * 45
        angle_end = (i + 1) * 45
        cv2.ellipse(m, (cx, cy), (r1, r1), 0, angle_start, angle_end, 255, -1)
        cv2.circle(m, (cx, cy), r2, 0, -1)
        zones.append((f"O{i+1}", m))
        
    return zones, r1, r2, r3

def create_pixel_heatmap(img_bgr, circle_mask_u8):
    """
    Bölge ortalaması almadan, her pikseli kendi Ry değerine göre boyar.
    Sadece circle_mask_u8 içindeki alanı işler.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L_channel = lab[:, :, 0]
    
    # Ry haritası hesapla (Vektörize işlem - Hızlı)
    # Ry = (L / 255) * 100
    ry_map = (L_channel.astype(np.float32) / 255.0) * 100.0
    
    # Çıktı resmi (Boş)
    heatmap = np.zeros_like(img_bgr)
    
    # Maskenin dolu olduğu yerlerdeki koordinatlar
    y_idxs, x_idxs = np.where(circle_mask_u8 > 0)
    
    if len(y_idxs) == 0:
        return heatmap

    # İlgili piksellerin Ry değerleri
    target_rys = ry_map[y_idxs, x_idxs]
    
    # Her piksel için tek tek renk bulmak yavaş olabilir ama en doğrusu bu.
    # Hızlandırmak için np.digitize kullanılabilir ama senin eşikler non-linear.
    # Basit bir map ile yapalım:
    
    # Piksel piksel boyama (Görselleştirme amacıyla)
    for y, x, ry in zip(y_idxs, x_idxs, target_rys):
        shade = get_shade_number(ry)
        color = SHADE_COLOR_MAP_BGR.get(shade, (128,128,128))
        heatmap[y, x] = color
        
    return heatmap

def analyze_single_cake(img_bgr, mask_bool):
    processed = apply_clahe_lab(img_bgr)
    mask_u8 = mask_bool.astype(np.uint8) * 255
    
    # 1. GEOMETRİ: İçeri sığan en büyük daire (Inscribed Circle)
    (cx, cy), radius = get_inscribed_circle(mask_u8)
    
    # Eğer radius çok küçükse (gürültü) atla
    if radius < 5: return None, None, None, None
    
    # 13 Bölgeyi oluştur
    zones, r1, r2, r3 = get_13_zones(img_bgr.shape, (cx, cy), radius)
    
    vis_layer_zones = np.zeros_like(img_bgr)
    line_layer = np.zeros_like(mask_u8)
    
    zone_results = []
    
    # Temiz Daire Maskesi (Analizin sınırları artık bu mükemmel daire)
    clean_circle_mask = np.zeros_like(mask_u8)
    cv2.circle(clean_circle_mask, (int(cx), int(cy)), int(radius), 255, -1)

    # --- A) BÖLGESEL ANALİZ (13 ZONE) ---
    for z_name, z_mask in zones:
        valid_zone = (z_mask > 0)
        
        # Çizgiler
        z_cnts, _ = cv2.findContours(valid_zone.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(line_layer, z_cnts, -1, 255, 2)
        
        # Renk Hesabı
        pixels = processed[valid_zone]
        if len(pixels) == 0: continue

        brightness = np.mean(pixels, axis=1)
        if len(brightness) > 0:
            p5, p95 = np.percentile(brightness, [5, 95])
            filtered_pixels = pixels[(brightness >= p5) & (brightness <= p95)]
            if len(filtered_pixels) == 0: filtered_pixels = pixels
        else:
            filtered_pixels = pixels
        
        avg_bgr = np.mean(filtered_pixels, axis=0)
        lab_px = cv2.cvtColor(np.uint8([[avg_bgr]]), cv2.COLOR_BGR2Lab)[0][0]
        L_val = lab_px[0]
        ry = (L_val / 255.0) * 100.0
        
        shade = get_shade_number(ry)
        color = SHADE_COLOR_MAP_BGR.get(shade, (128, 128, 128))
        zone_results.append(shade)
        
        # Boyama (Solid)
        vis_layer_zones[valid_zone] = color
        
        # Yazı
        M = cv2.moments(z_mask)
        if M["m00"] > 0:
            tx, ty = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(vis_layer_zones, str(shade), (tx-6, ty+4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Siyah çizgileri bas
    vis_layer_zones[line_layer > 0] = (0, 0, 0)
    
    # --- B) PİKSEL ANALİZİ (HEATMAP) ---
    # Sadece o temiz dairenin içindeki her pikseli analiz et
    vis_layer_pixel = create_pixel_heatmap(processed, clean_circle_mask)
    # Piksel analizinde de dış çerçeveyi siyah çizelim ki net dursun
    vis_layer_pixel[line_layer > 0] = (0,0,0)

    return vis_layer_zones, vis_layer_pixel, zone_results, clean_circle_mask

# =========================
# 3. ARAYÜZ VE AKIŞ
# =========================

def run_smallcake():
    
    st.markdown("<style>.block-container h1{margin-top:-80px}</style>", unsafe_allow_html=True)
    st.title("Small Cake Analizi")
                
    uploads = st.file_uploader("Kek Görseli Yükle", type=["jpg","jpeg","png"], accept_multiple_files=True, key="uploads_sc")

    if not uploads:
        st.info("Lütfen analiz edilecek kek görsellerini yükleyin.")
        return

    for up in uploads:
        st.divider()
        st.subheader(f"Dosya: {up.name}")
        
        file_bytes = np.frombuffer(up.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None: continue
        
        mask_u8 = robust_cake_mask(img_bgr)
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sonuç Tuvalleri (Beyaz Zemin)
        # EN standardı gibi temiz görünmesi için beyaz zemin kullanıyoruz.
        h, w = img_bgr.shape[:2]
        canvas_zones = np.ones((h, w, 3), dtype=np.uint8) * 255
        canvas_pixels = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        all_file_shades = []

        found_any = False
        for c in cnts:
            if cv2.contourArea(c) < 1000: continue
            
            single_mask = np.zeros_like(mask_u8)
            cv2.drawContours(single_mask, [c], -1, 255, -1)
            
            # Analiz Fonksiyonu (Hem Zone hem Pixel döndürür)
            v_zones, v_pixel, shades, clean_mask = analyze_single_cake(img_bgr, single_mask > 0)
            
            if v_zones is not None:
                found_any = True
                roi = clean_mask > 0
                
                # Zone Canvas'a işle
                canvas_zones[roi] = v_zones[roi]
                # Siyah çizgileri netleştir (Zone için)
                black_px_z = np.all(v_zones == [0,0,0], axis=-1) & roi
                canvas_zones[black_px_z] = [0,0,0]
                
                # Pixel Canvas'a işle
                canvas_pixels[roi] = v_pixel[roi]
                # Siyah çizgileri netleştir (Pixel için - opsiyonel, çerçeve görünsün diye)
                black_px_p = np.all(v_pixel == [0,0,0], axis=-1) & roi
                canvas_pixels[black_px_p] = [0,0,0]

                all_file_shades.extend(shades)

        if not found_any:
            st.warning("Kek tespit edilemedi.")
            continue

        # --- GÖRSELLEŞTİRME (3 KOLON) ---
        c1, c2, c3 = st.columns([1, 1, 1])
        
        with c1:
            st.markdown("##### 1. Orijinal Görüntü")
            st.markdown("*Ham Görüntü*")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            if all_file_shades:
                avg_s = sum(all_file_shades) / len(all_file_shades)
                st.info(f"Ortalama Shade: **{avg_s:.2f}**")
                
        with c2:
            st.markdown("##### 2. Bölgesel (Zone) Analiz")
            st.markdown("*EN Standardı Stili (Solid)*")
            st.image(cv2.cvtColor(canvas_zones, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with c3:
            st.markdown("##### 3. Piksel Bazlı Analiz")
            st.markdown("*Bölge ortalaması yok, ham doku*")
            st.image(cv2.cvtColor(canvas_pixels, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Grafik Alanı
        if all_file_shades:
            shade_counts = {s: all_file_shades.count(s) for s in set(all_file_shades)}
            sorted_shades = sorted(shade_counts.keys())
            counts = [shade_counts[s] for s in sorted_shades]
            bar_colors = [SHADE_COLOR_MAP_RGB.get(s, (0.5,0.5,0.5)) for s in sorted_shades]
            
            fig, ax = plt.subplots(figsize=(10, 3))
            bars = ax.bar(range(len(counts)), counts, color=bar_colors, tick_label=[str(s) for s in sorted_shades])
            ax.set_title("Bölgesel Renk Dağılımı")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 1), textcoords="offset points", ha='center', va='bottom')
            st.pyplot(fig)
            plt.close(fig)
