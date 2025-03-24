import streamlit as st

from pages import gps_exploration, physical_capabilities, recovery_status

# Set page configuration
st.set_page_config(
    page_title="FC Performance Insights",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define available players
PLAYERS = {
    "Marc Cucurella": "images/cucurella.jpeg",
}

# Initialize session state for selected player
if "selected_player" not in st.session_state:
    st.session_state.selected_player = "Marc Cucurella"


# Define sidebar function
def create_sidebar():
    with st.sidebar:
        st.title("FC Performance Insights")
        st.sidebar.image("images/chelsea.png", width=300)
        st.sidebar.image("images/logo-eleven-vert.png", width=300)

        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Go to:",
            ["Home", "GPS", "Physical Capabilities", "Recovery Status"],
        )

        # Player selection
        st.subheader("Player Selection")
        selected_player = st.selectbox(
            "Choose a player:",
            options=list(PLAYERS.keys()),
            index=list(PLAYERS.keys()).index(st.session_state.selected_player),
        )

        # Update session state
        st.session_state.selected_player = selected_player
        st.image(PLAYERS[st.session_state.selected_player], width=300)

        # FAQ Section (collapsed by default)
        with st.expander("📚 FAQ"):
            st.markdown(
                """
                - **What is FC Performance Insights?**
                  - A platform to track and analyze football players' performance through metrics like injury history, physical capabilities, and recovery status.
                
                - **How can I use the platform?**
                  - Use the navigation menu above to explore different insights and analytics for players and teams.
                
                - **Can I customize the data for my team?**
                  - Yes, the platform allows personalized data input and analysis to optimize player performance.
                """
            )

        # Footer
        st.markdown("---")
        st.caption("© 2025 FC Performance Insights")

    return page


# Main app content
def main():
    # Apply custom CSS
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f7f9;
        }
        .stApp {
            max-width: 100%;
            padding: 0;
            margin: 0;
        }
        h1, h2, h3 {
            color: #0e1c36;
        }
        .stSidebar {
            background-color: #0e1c36;
            color: white;
        }
        .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Create sidebar and get selected page
    page = create_sidebar()

    # Display content based on selected page
    if page == "Home":
        st.title("⚽ FC Performance Insights")

        st.markdown(
            """
            ## Welcome to the FC Performance Insights Platform!
            
            ### CFCInsights
            
            This comprehensive platform offers elite football players and coaches a cutting-edge **Physical Performance Interface** that transforms how teams monitor and enhance player development.
            
            Designed with both clarity and depth, CFCInsights combines technical sophistication with intuitive visualization to deliver actionable insights directly to your fingertips.
            """
        )

        st.markdown("---")

        st.subheader("📊 Performance Metrics Dashboard")

        metrics_col1, metrics_col2 = st.columns(2)

        with metrics_col1:
            st.markdown(
                """
                - **Load Management** - Track match minutes, training load, and season availability
                - **Injury Analytics** - Visualize injury patterns, risk factors, and rehabilitation progress
                - **Physical Profiling** - Monitor key performance indicators and benchmarks
                """
            )

        with metrics_col2:
            st.markdown(
                """
                - **Player Biography** - Access comprehensive player information and statistics
                - **Recovery Tracking** - Analyze sleep quality, nutrition, and wellness indicators
                - **External Factors** - Evaluate environmental conditions and performance impact
                """
            )

        st.info(
            "🔍 Navigate using the sidebar to explore specialized analytics modules for your players."
        )

    elif page == "GPS":
        gps_exploration.show()

    elif page == "Physical Capabilities":
        physical_capabilities.show()

    elif page == "Recovery Status":
        recovery_status.show()


# Run the app
if __name__ == "__main__":
    main()
