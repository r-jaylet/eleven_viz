import streamlit as st

from pages import gps_exploration, physical_capabilities, recovery_status

# Set page title and layout (MUST be the first Streamlit command)
st.set_page_config(page_title="FC Performance Insights", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["Home", "GPS", "Physical Capabilities", "Recovery Status"]
)

# Home Page
if page == "Home":
    st.title("⚽ FC Performance Insights Vizathon")

    # Sidebar content
    st.sidebar.image("images/chelsea.png")  # First image to fill width
    st.sidebar.image("images/eleven.jpg")  # Second image to fill width

    # Text and third image
    st.sidebar.markdown("### Selected Player")
    st.sidebar.image("images/cucurella.jpeg")  # Third image

    # FAQ Section
    st.sidebar.markdown("### 📚 FAQ")
    st.sidebar.markdown(
        """
        - **What is FC Performance Insights?**
          - A platform to track and analyze the performance of football players through various metrics like load demand, injury history, physical capabilities, and recovery status.
          
        - **How can I use the platform?**
          - Use the navigation menu on the left to explore different insights and analytics for players and teams.

        - **Can I customize the data for my team?**
          - Yes, the platform allows personalized data input and analysis to optimize player performance.
        """
    )

    st.markdown(
        """
        ## Welcome to the FC Performance Insights Platform!  
        ### CFCInsights
        
        This platform is designed to create the most compelling **Physical Performance Interface** for elite football players and their coaches. 
        
        The goal? To showcase innovative design skills, technical mastery, and user-focused thinking while making a real impact in professional football.
        
        ### 📌 Key Performance Insights:
        - **Load Demand**: Track games, matches played, training sessions, and season availability.
        - **Injury History**: Monitor injury status, risk categories, and recent injuries.
        - **Physical Development**: Assess test capabilities, strength & conditioning plans, and priorities.
        - **Biography**: Access player details such as nationality, position, team, and league.
        - **Recovery**: Analyze nutrition, sleep, wellness, and performance adherence.
        - **External Factors**: Evaluate external influences like environment, team dynamics, and motivation.
        
        🔍 Use the navigation panel to explore different insights and analytics.
        """,
        unsafe_allow_html=True,
    )

elif page == "GPS":
    gps_exploration.show()

elif page == "Physical Capabilities":
    physical_capabilities.show()

elif page == "Recovery Status":
    recovery_status.show()
