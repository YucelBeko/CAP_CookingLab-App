from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data_merger_core as dm_core

DM_MAX_ANALOG_PLOT_POINTS = 2000


def _dm_save_uploaded_file_to_temp(uploaded_file):
    """
    Streamlit UploadedFile objesini geçici dosya path'ine çevirir.
    Mevcut data_merger_core fonksiyonları path beklediği için gerekli.
    """
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _dm_excel_bytes(report, gm10_raw=None, vts_raw=None, udaq_raw=None):
    """
    data_merger_core.save_excel path istiyor.
    Streamlit download_button için Excel'i memory içinde üretir.
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="Ozet_Rapor", index=False)

        if gm10_raw is not None:
            gm10_raw.to_excel(writer, sheet_name="GM10_Ham", index=False, header=False)

        if vts_raw is not None:
            vts_raw.to_excel(writer, sheet_name="VTS_Ham", index=False)

        if udaq_raw is not None:
            udaq_raw.to_excel(writer, sheet_name="UDAQ_Ham", index=False)

    output.seek(0)
    return output.getvalue()


def _dm_compute_region_stats(report, selected_cols, col_types, x_min, x_max):
    """
    Plotly seçimi sonrası x_min - x_max zaman aralığındaki kolon istatistiklerini hesaplar.
    """
    if report is None or report.empty:
        return None

    if "Minute" not in report.columns:
        return None

    minute = pd.to_numeric(report["Minute"], errors="coerce")

    mask = (minute >= x_min) & (minute <= x_max)

    if not mask.any():
        return None

    rows = []

    for col in selected_cols:
        if col not in report.columns:
            continue

        y = pd.to_numeric(report.loc[mask, col], errors="coerce").dropna()

        if len(y) == 0:
            continue

        ctype = col_types.get(col, "analog")

        if ctype == "digital" and set(y.unique()).issubset({0, 1, 0.0, 1.0}):
            on_count = int((y == 1).sum())
            off_count = int((y == 0).sum())
            total = len(y)
            on_pct = 100 * on_count / total if total > 0 else 0

            rows.append({
                "Parametre": col,
                "Tip": "Digital",
                "Ortalama": round(float(y.mean()), 3),
                "Min": round(float(y.min()), 3),
                "Max": round(float(y.max()), 3),
                "Range": round(float(y.max() - y.min()), 3),
                "Std": round(float(y.std()), 3),
                "ON %": round(on_pct, 1),
                "ON Count": on_count,
                "OFF Count": off_count,
                "N": total
            })

        else:
            rows.append({
                "Parametre": col,
                "Tip": "Analog",
                "Ortalama": round(float(y.mean()), 3),
                "Min": round(float(y.min()), 3),
                "Max": round(float(y.max()), 3),
                "Range": round(float(y.max() - y.min()), 3),
                "Std": round(float(y.std()), 3),
                "ON %": None,
                "ON Count": None,
                "OFF Count": None,
                "N": len(y)
            })

    if not rows:
        return None

    return pd.DataFrame(rows)

def _dm_extract_selected_x_range(plotly_event):
    """
    st.plotly_chart selection event içinden seçilen X aralığını alır.
    Önce points dener, olmazsa box selection bilgisini dener.
    """
    if plotly_event is None:
        return None, None

    # Streamlit event objesini dict'e çevirmeyi dene
    try:
        event_dict = plotly_event.to_dict()
    except Exception:
        try:
            event_dict = dict(plotly_event)
        except Exception:
            event_dict = {}

    selection = event_dict.get("selection", {})

    # 1) Önce selected points üzerinden dene
    points = selection.get("points", []) or []

    x_values = []
    for p in points:
        try:
            x_values.append(float(p.get("x")))
        except Exception:
            pass

    if x_values:
        return min(x_values), max(x_values)

    # 2) Box selection üzerinden dene
    boxes = selection.get("box", []) or []

    for box in boxes:
        # Streamlit/Plotly versiyonuna göre farklı gelebiliyor.
        # O yüzden birkaç olası yapıyı destekliyoruz.
        try:
            if "x" in box and isinstance(box["x"], list) and len(box["x"]) >= 2:
                xs = [float(v) for v in box["x"]]
                return min(xs), max(xs)
        except Exception:
            pass

        try:
            if "x0" in box and "x1" in box:
                return float(min(box["x0"], box["x1"])), float(max(box["x0"], box["x1"]))
        except Exception:
            pass

        try:
            if "range" in box and "x" in box["range"]:
                xs = box["range"]["x"]
                return float(min(xs)), float(max(xs))
        except Exception:
            pass

    return None, None


def _dm_render_chart(report, sampling_hz=1.0):
    st.subheader("Grafik Önizleme")

    if report is None or report.empty:
        st.info("Grafik için önce verileri birleştir.")
        return

    columns = [c for c in report.columns if c not in ("Step", "Minute")]

    if not columns:
        st.warning("Grafikte gösterilecek kolon bulunamadı.")
        return

    col_types, _ = dm_core.classify_columns(report, columns)

    selected_cols = st.multiselect(
        "Grafikte gösterilecek parametreler",
        options=columns,
        default=columns[: min(4, len(columns))],
        key="dm_chart_selected_cols"
    )

    if not selected_cols:
        st.info("Grafik için en az bir parametre seç.")
        return

    with st.expander("Axis Ayarları", expanded=False):
        st.caption("Boş bırakırsan otomatik ölçek kullanılır. Step değeri grid/tick aralığıdır.")
        st.markdown("**Grafik / Axis Başlıkları**")

        title_c1, title_c2, title_c3, title_c4 = st.columns(4)

        with title_c1:
            chart_title = st.text_input(
                "Grafik Başlığı",
                value="Data Merger Grafiği",
                key="dm_chart_title"
            )

        with title_c2:
            x_axis_title = st.text_input(
                "X Axis Başlığı",
                value="Time (minutes)",
                key="dm_x_axis_title"
            )

        with title_c3:
            y1_axis_title = st.text_input(
                "Y1 Axis Başlığı",
                value="Analog values",
                key="dm_y1_axis_title"
            )

        with title_c4:
            y2_axis_title = st.text_input(
                "Y2 Axis Başlığı",
                value="Digital values",
                key="dm_y2_axis_title"
            )

        st.divider()

        ax1, ax2, ax3 = st.columns(3)

        with ax1:
            st.markdown("**X Axis / Minute**")
            x_min_txt = st.text_input("X Min", value="", key="dm_x_min")
            x_max_txt = st.text_input("X Max", value="", key="dm_x_max")
            x_step_txt = st.text_input("X Step", value="", key="dm_x_step")

        with ax2:
            st.markdown("**Y1 Axis / Analog**")
            y1_min_txt = st.text_input("Y1 Min", value="", key="dm_y1_min")
            y1_max_txt = st.text_input("Y1 Max", value="", key="dm_y1_max")
            y1_step_txt = st.text_input("Y1 Step", value="", key="dm_y1_step")

        with ax3:
            st.markdown("**Y2 Axis / Digital**")
            y2_min_txt = st.text_input("Y2 Min", value="", key="dm_y2_min")
            y2_max_txt = st.text_input("Y2 Max", value="", key="dm_y2_max")
            y2_step_txt = st.text_input("Y2 Step", value="", key="dm_y2_step")

    x_min = _dm_parse_float(x_min_txt)
    x_max = _dm_parse_float(x_max_txt)
    x_step = _dm_parse_float(x_step_txt)

    y1_min = _dm_parse_float(y1_min_txt)
    y1_max = _dm_parse_float(y1_max_txt)
    y1_step = _dm_parse_float(y1_step_txt)

    y2_min = _dm_parse_float(y2_min_txt)
    y2_max = _dm_parse_float(y2_max_txt)
    y2_step = _dm_parse_float(y2_step_txt)

    fig = go.Figure()

    if "Minute" in report.columns:
        x_full = pd.to_numeric(report["Minute"], errors="coerce").to_numpy()
    else:
        x_full = np.arange(1, len(report) + 1, dtype=float) / (sampling_hz * 60.0)

    analog_downsampled = False

    for col in selected_cols:
        y_full = pd.to_numeric(report[col], errors="coerce").to_numpy()
        is_digital = col_types.get(col) == "digital"

        # Analog kanallarda çok uzun testleri tarayıcıya tamamen göndermek yerine
        # eşit aralıklı örnekle. Digital kanalları geçiş kaybı olmaması için koru.
        if (not is_digital) and len(x_full) > DM_MAX_ANALOG_PLOT_POINTS:
            plot_idx = np.linspace(
                0,
                len(x_full) - 1,
                DM_MAX_ANALOG_PLOT_POINTS,
                dtype=int
            )
            x_plot = x_full[plot_idx]
            y_plot = y_full[plot_idx]
            analog_downsampled = True
        else:
            x_plot = x_full
            y_plot = y_full

        if is_digital:
            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    mode="lines",
                    name=col,
                    yaxis="y2",
                    line_shape="hv"
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    mode="lines",
                    name=col,
                    yaxis="y"
                )
            )

    if analog_downsampled:
        st.caption(
            f"Grafik performansı için analog kanallar en fazla "
            f"{DM_MAX_ANALOG_PLOT_POINTS:,} nokta ile gösteriliyor. "
            "İstatistikler tam veri üzerinden hesaplanır."
        )

    fig.update_layout(
        dragmode="select",
        selectdirection="h",
        title=dict(
            text=chart_title,
            x=0.01,
            xanchor="left"
        ),
        height=620,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=70, b=40),
        xaxis=dict(
            title=x_axis_title,
            showgrid=True
        ),
        yaxis=dict(
            title=y1_axis_title,
            showgrid=True
        ),
        yaxis2=dict(
            title=y2_axis_title,
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

        # Axis ranges
    if x_min is not None or x_max is not None:
        current_x = pd.to_numeric(pd.Series(x), errors="coerce")
        auto_x_min = float(np.nanmin(current_x)) if np.isfinite(np.nanmin(current_x)) else 0
        auto_x_max = float(np.nanmax(current_x)) if np.isfinite(np.nanmax(current_x)) else 1

        fig.update_xaxes(
            range=[
                x_min if x_min is not None else auto_x_min,
                x_max if x_max is not None else auto_x_max
            ]
        )

    if y1_min is not None or y1_max is not None:
        analog_values = []

        for col in selected_cols:
            if col_types.get(col) != "digital":
                vals = pd.to_numeric(report[col], errors="coerce").dropna()
                if len(vals) > 0:
                    analog_values.append(vals)

        if analog_values:
            analog_all = pd.concat(analog_values)
            auto_y1_min = float(analog_all.min())
            auto_y1_max = float(analog_all.max())

            fig.update_yaxes(
                range=[
                    y1_min if y1_min is not None else auto_y1_min,
                    y1_max if y1_max is not None else auto_y1_max
                ],
                secondary_y=False
            )

    if y2_min is not None or y2_max is not None:
        digital_values = []

        for col in selected_cols:
            if col_types.get(col) == "digital":
                vals = pd.to_numeric(report[col], errors="coerce").dropna()
                if len(vals) > 0:
                    digital_values.append(vals)

        if digital_values:
            digital_all = pd.concat(digital_values)
            auto_y2_min = float(digital_all.min())
            auto_y2_max = float(digital_all.max())

            fig.update_layout(
                yaxis2=dict(
                    title=y2_axis_title,
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    range=[
                        y2_min if y2_min is not None else auto_y2_min,
                        y2_max if y2_max is not None else auto_y2_max
                    ]
                )
            )

    # Axis steps / tick intervals
    if x_step is not None and x_step > 0:
        fig.update_xaxes(dtick=x_step)

    if y1_step is not None and y1_step > 0:
        fig.update_yaxes(dtick=y1_step)

    if y2_step is not None and y2_step > 0:
        fig.update_layout(
            yaxis2=dict(
                title=y2_axis_title,
                overlaying="y",
                side="right",
                showgrid=False,
                dtick=y2_step,
                range=fig.layout.yaxis2.range if fig.layout.yaxis2.range else None
            )
        )

    with st.expander("Referans / Alarm çizgisi ekle", expanded=False):
        c1, c2, c3 = st.columns([1, 1, 2])

        with c1:
            alarm_value = st.number_input(
                "Değer",
                value=100.0,
                step=1.0,
                key="dm_alarm_value"
            )

        with c2:
            alarm_direction = st.radio(
                "Yön",
                ["Yatay", "Dikey"],
                horizontal=True,
                key="dm_alarm_direction"
            )

        with c3:
            alarm_label = st.text_input(
                "Etiket",
                value="Alarm",
                key="dm_alarm_label"
            )

        if "dm_alarm_lines" not in st.session_state:
            st.session_state.dm_alarm_lines = []

        c_add, c_clear = st.columns(2)

        with c_add:
            if st.button("Alarm çizgisi ekle", key="dm_add_alarm"):
                st.session_state.dm_alarm_lines.append(
                    {
                        "value": alarm_value,
                        "direction": alarm_direction,
                        "label": alarm_label
                    }
                )
                st.rerun()

        with c_clear:
            if st.button("Alarm çizgilerini temizle", key="dm_clear_alarm"):
                st.session_state.dm_alarm_lines = []
                st.rerun()

    for alarm in st.session_state.get("dm_alarm_lines", []):
        if alarm["direction"] == "Yatay":
            fig.add_hline(
                y=alarm["value"],
                line_dash="dash",
                annotation_text=alarm["label"],
                annotation_position="top left"
            )
        else:
            fig.add_vline(
                x=alarm["value"],
                line_dash="dash",
                annotation_text=alarm["label"],
                annotation_position="top left"
            )

    st.caption(
    "Bölge istatistiği için grafiğin sağ üst toolbar kısmından Box Select veya Lasso Select seçip "
    "grafik üzerinde zaman aralığı seçebilirsin."
    )

    plotly_event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode=("box", "lasso"),
        key="dm_plotly_chart"
    )

    sel_x_min, sel_x_max = _dm_extract_selected_x_range(plotly_event)

    if sel_x_min is not None and sel_x_max is not None:
        st.subheader("Seçili Bölge İstatistikleri")
        st.caption(f"Seçilen zaman aralığı: **{sel_x_min:.2f} - {sel_x_max:.2f} dakika**")

        stats_df = _dm_compute_region_stats(
            report=report,
            selected_cols=selected_cols,
            col_types=col_types,
            x_min=sel_x_min,
            x_max=sel_x_max
        )

        if stats_df is not None:
            st.dataframe(
                stats_df,
                use_container_width=True,
                height=300
            )

            csv_bytes = stats_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "Seçili Bölge İstatistiklerini CSV İndir",
                data=csv_bytes,
                file_name="secili_bolge_istatistikleri.csv",
                mime="text/csv",
                key="dm_download_region_stats"
            )
        else:
            st.warning("Seçili aralıkta hesaplanabilecek veri bulunamadı.")
    else:
        st.info("Henüz grafik üzerinde bir bölge seçilmedi.")

def _dm_parse_float(text):
    """
    Axis textbox değerini float'a çevirir.
    Boşsa None döner.
    Virgül/nokta ikisini de destekler.
    """
    try:
        text = str(text).strip()
        if text == "":
            return None
        return float(text.replace(",", "."))
    except Exception:
        return None

    
def run_data_merger():
    st.title("Data Merger")
    st.caption("GM10, VTS ve UDAQ datalogger çıktılarının tek raporda birleştirilmesi.")

    with st.expander("Dosya Seçimi", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            skip_gm10 = st.checkbox("GM10 verim yok", key="dm_skip_gm10")
            gm10_file = st.file_uploader(
                "GM10 File",
                type=["xlsx", "xls"],
                disabled=skip_gm10,
                key="dm_gm10_file"
            )

        with c2:
            skip_vts = st.checkbox("VTS verim yok", key="dm_skip_vts")
            vts_file = st.file_uploader(
                "VTS File",
                type=["xlsx", "xls", "csv"],
                disabled=skip_vts,
                key="dm_vts_file"
            )

        with c3:
            skip_udaq = st.checkbox("UDAQ verim yok", key="dm_skip_udaq")
            udaq_file = st.file_uploader(
                "UDAQ File",
                type=["log", "txt"],
                disabled=skip_udaq,
                key="dm_udaq_file"
            )

        sampling_hz = st.number_input(
            "Sampling rate / Hz",
            min_value=0.01,
            value=1.0,
            step=0.1,
            key="dm_sampling_hz"
        )

    if "dm_report" not in st.session_state:
        st.session_state.dm_report = None

    if "dm_excel_bytes" not in st.session_state:
        st.session_state.dm_excel_bytes = None

    if st.button("Verileri Birleştir", type="primary", key="dm_process"):
        if skip_gm10 and skip_vts and skip_udaq:
            st.error("Tüm veri kaynakları kapalı. Birleştirilecek veri yok.")
            return

        if not skip_gm10 and gm10_file is None:
            st.error("GM10 dosyası seç veya 'GM10 verim yok' seçeneğini işaretle.")
            return

        if not skip_vts and vts_file is None:
            st.error("VTS dosyası seç veya 'VTS verim yok' seçeneğini işaretle.")
            return

        if not skip_udaq and udaq_file is None:
            st.error("UDAQ dosyası seç veya 'UDAQ verim yok' seçeneğini işaretle.")
            return

        temp_paths = []

        try:
            with st.spinner("Dosyalar okunuyor ve rapor oluşturuluyor..."):
                gm10_raw = gm10_clean = None
                vts_raw = vts_clean = None
                udaq_raw = udaq_clean = None

                if not skip_gm10:
                    gm10_path = _dm_save_uploaded_file_to_temp(gm10_file)
                    temp_paths.append(gm10_path)
                    gm10_raw, gm10_clean = dm_core.read_gm10(gm10_path)

                if not skip_vts:
                    vts_path = _dm_save_uploaded_file_to_temp(vts_file)
                    temp_paths.append(vts_path)
                    vts_raw = dm_core.read_vts(vts_path)
                    vts_clean = vts_raw.copy()

                if not skip_udaq:
                    udaq_path = _dm_save_uploaded_file_to_temp(udaq_file)
                    temp_paths.append(udaq_path)
                    udaq_raw = dm_core.read_udaq(udaq_path)
                    udaq_clean = udaq_raw.reset_index(drop=True)

                report = dm_core.build_report(gm10_clean, vts_clean, udaq_clean)

                # core.build_report Minute kolonunu Step / 60 olarak oluşturuyor.
                # Burada sampling rate'i dikkate alarak düzeltiyoruz.
                if "Step" in report.columns:
                    report["Minute"] = pd.to_numeric(report["Step"], errors="coerce") / (sampling_hz * 60.0)
                else:
                    report.insert(0, "Step", range(1, len(report) + 1))
                    report.insert(1, "Minute", np.arange(1, len(report) + 1) / (sampling_hz * 60.0))

                excel_bytes = _dm_excel_bytes(
                    report,
                    gm10_raw=gm10_raw,
                    vts_raw=vts_raw,
                    udaq_raw=udaq_raw
                )

                st.session_state.dm_report = report
                st.session_state.dm_excel_bytes = excel_bytes

            st.success("Rapor başarıyla oluşturuldu.")

        except Exception as exc:
            st.session_state.dm_report = None
            st.session_state.dm_excel_bytes = None
            st.error(f"Data Merger hatası: {exc}")

        finally:
            for path in temp_paths:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

    report = st.session_state.get("dm_report")
    excel_bytes = st.session_state.get("dm_excel_bytes")

    if report is not None:
        st.subheader("Rapor Özeti")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Satır", f"{len(report):,}")
        with m2:
            st.metric("Kolon", f"{len(report.columns):,}")
        with m3:
            st.metric("Süre / dk", f"{report['Minute'].max():.1f}" if "Minute" in report.columns else "-")

        st.dataframe(
            report.head(300),
            use_container_width=True,
            height=360
        )

        if excel_bytes is not None:
            st.download_button(
                "Excel Raporunu İndir",
                data=excel_bytes,
                file_name="Firin_Performans_Raporu.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dm_download_excel"
            )

        st.divider()
        _dm_render_chart(report, sampling_hz=sampling_hz)
