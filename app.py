from __future__ import annotations

import streamlit as st

# This must remain the first Streamlit command in the entry script.
st.set_page_config(
    page_title="Pişirme Laboratuvarı",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import after set_page_config because some section modules use Streamlit decorators.
from sections import (  # noqa: E402
    run_borek,
    run_bread_surface,
    run_cookie,
    run_data_merger,
    run_flour_disk,
    run_pizza,
    run_potato,
    run_pyrocam,
    run_smallcake,
    run_teflon_block,
    run_toast_bread,
)
from ui.home import show_home_page  # noqa: E402
from ui.layout import add_global_watermark  # noqa: E402
from ui.navigation import ensure_navigation_state, set_page  # noqa: E402


PAGE_HANDLERS = {
    "Patates": run_potato,
    "Pizza": run_pizza,
    "Börek": run_borek,
    "Small Cake": run_smallcake,
    "Pyro Cam": run_pyrocam,
    "Ekmek": run_bread_surface,
    "Data Merger": run_data_merger,
    "Teflon Blok": run_teflon_block,
    "Kurabiye": run_cookie,
    "Unlu Disk": run_flour_disk,
    "Tost Ekmeği": run_toast_bread,
}


def render_sidebar(current_page: str) -> None:
    with st.sidebar:
        st.markdown(
            '<style>[data-testid="stSidebar"] {display: block;}</style>',
            unsafe_allow_html=True,
        )
        st.button(
            "🏠 Ana Sayfa",
            use_container_width=True,
            on_click=set_page,
            args=("Home",),
            key="global_home_button",
        )
        st.divider()
        st.caption(f"Mod: {current_page}")


def main() -> None:
    ensure_navigation_state()
    current_page = st.session_state.current_page

    add_global_watermark()

    if current_page == "Home":
        show_home_page()
        return

    render_sidebar(current_page)

    handler = PAGE_HANDLERS.get(current_page)
    if handler is None:
        st.error(f"Bilinmeyen sayfa: {current_page}")
        st.button(
            "Ana Sayfaya Dön",
            on_click=set_page,
            args=("Home",),
            key="unknown_page_home_button",
        )
        return

    handler()


if __name__ == "__main__":
    main()
