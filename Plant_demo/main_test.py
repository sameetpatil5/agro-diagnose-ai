from streamlit_option_menu import option_menu
import streamlit as st

# Sidebar
with st.sidebar:
    st.image("Plant_demo/home_page.jpeg", use_container_width=True)
    st.markdown("### 🌿 Welcome to PlantCare AI!")
    st.markdown("Effortlessly diagnose plant diseases and get actionable insights.")

    # Navigation Menu
    app_mode = option_menu(
        "Navigation",
        ["Home", "About", "Disease Recognition"],
        icons=["house", "info-circle", "camera"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f8f9fa"},
            "icon": {"color": "green", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#e9ecef",
            },
            "nav-link-selected": {"background-color": "#28a745"},
        },
    )

    # Collapsible Section
    with st.expander("Learn More"):
        st.write("Explore how this platform can help farmers protect crops.")
        st.write("Visit the About page for details.")
