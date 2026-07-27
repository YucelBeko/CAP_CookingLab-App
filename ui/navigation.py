from __future__ import annotations

import streamlit as st


def ensure_navigation_state() -> None:
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"


def set_page(page_name: str) -> None:
    """Streamlit-safe page navigation callback."""
    st.session_state.current_page = page_name
