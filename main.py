import os
from datetime import datetime

import streamlit as st

from pages import gps_exploration, physical_capabilities, recovery_status

# Configuration
st.set_page_config(
    page_title="FC Performance Insights",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #001489;
        color: white;
    }
    /* Player card styling */
    .player-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Header styling */
    .main-header {
        color: #001489;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Logo container */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }
    /* Metric container */
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Helper functions
def format_sidebar_image(image_path, caption=None, width=None):
    """Display a properly formatted image in the sidebar with optional caption"""
    if os.path.exists(image_path):
        if caption:
            st.sidebar.markdown(f"### {caption}")

        # Calculate image dimensions to fit sidebar perfectly
        img_html = f'<img src="data:image/png;base64,{get_base64_encoded_image(image_path)}" style="width: 100%; border-radius: 8px; margin-bottom: 15px;">'
        st.sidebar.markdown(img_html, unsafe_allow_html=True)
    else:
        st.sidebar.warning(f"Image not found: {image_path}")


def get_base64_encoded_image(image_path):
    """Return base64 encoded image for HTML embedding"""
    import base64

    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def create_metric_card(title, value, delta=None, description=None):
    """Create a formatted metric card with title, value, and optional delta"""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### {title}")
    with col2:
        if delta:
            st.metric(label="", value=value, delta=delta)
        else:
            st.markdown(f"## {value}")
        if description:
            st.markdown(
                f"<small>{description}</small>", unsafe_allow_html=True
            )


# Sidebar navigation
with st.sidebar:
    st.title("FC Performance Insights")

    # Team logos with perfect sizing
    format_sidebar_image("images/chelsea.png")
    format_sidebar_image("images/eleven.jpg")

    st.divider()

    # Player profile with enhanced styling
    st.markdown("### Selected Player")
    format_sidebar_image("images/cucurella.jpeg", width="100%")

    st.markdown("**Marc Cucurella Saseta**")
    st.markdown("Position: Left Back | Age: 26")

    st.divider()

    # Navigation menu with icons
    st.subheader("📊 Navigation")
    page = st.radio(
        "",
        options=[
            "🏠 Home",
            "📍 GPS Tracking",
            "💪 Physical Capabilities",
            "🔄 Recovery Status",
        ],
        key="nav",
    )

    # Current date display
    current_date = datetime.now().strftime("%d %B %Y")
    st.markdown(
        f"<div style='text-align: center; margin-top: 20px;'>{current_date}</div>",
        unsafe_allow_html=True,
    )

# Main content area
if "🏠 Home" in page:
    # Main header
    st.markdown(
        "<h1 class='main-header'>⚽ FC Performance Insights</h1>",
        unsafe_allow_html=True,
    )

    # Welcome message
    st.markdown(
        """
        <div class="player-card">
            <h2>Welcome to the FC Performance Insights Platform!</h2>
            <p style="font-size: 18px;">This platform combines data analytics with sports science to optimize player performance and reduce injury risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key performance indicators
    st.subheader("Player Overview: Marc Cucurella")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        create_metric_card("Match Fitness", "92%", "+5%", "Last 7 days")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        create_metric_card(
            "Injury Risk", "Low", "-12%", "Compared to last month"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        create_metric_card(
            "Season Availability", "94%", None, "Matches played: 32/34"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Dashboard sections
    st.markdown("## Performance Insights")

    tab1, tab2, tab3 = st.tabs(
        ["Load Management", "Physical Development", "Recovery Status"]
    )

    with tab1:
        st.markdown("### Weekly Load Profile")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Training Load (Last 7 Days)")
            # Placeholder for a chart
            st.bar_chart(
                {
                    "Monday": 420,
                    "Tuesday": 380,
                    "Wednesday": 520,
                    "Thursday": 450,
                    "Friday": 350,
                    "Saturday": 200,
                    "Sunday": 100,
                }
            )

        with col2:
            st.markdown("#### Seasonal Load Progression")
            # Placeholder for a line chart
            st.line_chart(
                {
                    "Week 1": 320,
                    "Week 2": 340,
                    "Week 3": 360,
                    "Week 4": 380,
                    "Week 5": 400,
                    "Week 6": 410,
                }
            )

    with tab2:
        st.markdown("### Physical Testing Results")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Speed & Power")
            data = {
                "10m Sprint (s)": 1.68,
                "30m Sprint (s)": 4.05,
                "Max Velocity (km/h)": 34.2,
                "Countermovement Jump (cm)": 38.5,
            }
            for key, value in data.items():
                st.markdown(f"**{key}:** {value}")

        with col2:
            st.markdown("#### Endurance & Strength")
            data = {
                "Yo-Yo Test (m)": 2720,
                "Squat 1RM (kg)": 120,
                "Bench Press 1RM (kg)": 85,
                "Core Endurance (s)": 178,
            }
            for key, value in data.items():
                st.markdown(f"**{key}:** {value}")

    with tab3:
        st.markdown("### Recovery Metrics")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Sleep Quality")
            # Placeholder for sleep quality data
            st.line_chart(
                {
                    "Mon": 85,
                    "Tue": 82,
                    "Wed": 88,
                    "Thu": 90,
                    "Fri": 86,
                    "Sat": 92,
                    "Sun": 89,
                }
            )

        with col2:
            st.markdown("#### Muscle Soreness")
            # Placeholder for muscle soreness data
            st.bar_chart(
                {
                    "Quads": 2.1,
                    "Hamstrings": 1.8,
                    "Calves": 2.4,
                    "Lower Back": 1.2,
                    "Upper Back": 0.8,
                }
            )

    # FAQ section with expandable sections
    st.markdown("## 📚 FAQ")

    with st.expander("What is FC Performance Insights?"):
        st.markdown(
            """
            FC Performance Insights is a comprehensive platform designed to track and analyze the performance of football players 
            through various metrics like load management, injury prevention, physical capabilities, and recovery status.
            
            Our platform helps coaches and sports scientists make data-driven decisions to optimize player performance and reduce injury risk.
        """
        )

    with st.expander("How can I use the platform?"):
        st.markdown(
            """
            Use the navigation menu on the left to explore different insights and analytics for players and teams:
            
            - **Home**: Overview of key metrics and recent performance
            - **GPS Tracking**: Detailed movement and intensity analysis
            - **Physical Capabilities**: Test results and physical development tracking
            - **Recovery Status**: Monitoring of recovery metrics and readiness to perform
        """
        )

    with st.expander("Can I customize the data for my team?"):
        st.markdown(
            """
            Yes, the platform allows fully personalized data input and analysis:
            
            - Upload your team's GPS tracking data
            - Input custom physical test results
            - Set individual recovery protocols
            - Create tailored dashboards for specific players or positions
        """
        )

elif "GPS" in page:
    gps_exploration.show()

elif "Physical" in page:
    physical_capabilities.show()

elif "Recovery" in page:
    recovery_status.show()
