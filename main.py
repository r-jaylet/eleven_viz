import streamlit as st

from pages import gps_exploration, physical_capabilities, recovery_status


# Define sidebar function
def create_sidebar():

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "GPS", "Physical Capabilities", "Recovery Status"],
    )

    # Sidebar content for Home page
    if page == "Home":
        st.sidebar.image("images/chelsea.png")
        st.sidebar.image("images/eleven.jpg")
        st.sidebar.markdown("### Selected Player")
        st.sidebar.image("images/cucurella.jpeg")

        # FAQ Section
        st.sidebar.markdown("### 📚 FAQ")
        st.sidebar.markdown(
            """
            - **What is FC Performance Insights?**
              - A platform to track and analyze football players' performance through metrics like injury history, physical capabilities, and recovery status.
            
            - **How can I use the platform?**
              - Use the navigation menu above to explore different insights and analytics for players and teams.
            
            - **Can I customize the data for my team?**
              - Yes, the platform allows personalized data input and analysis to optimize player performance.
            """
        )

    return page


# Main app content
def main():
    # Create sidebar and get selected page
    page = create_sidebar()

    # Display content based on selected page
    if page == "Home":
        st.title("⚽ FC Performance Insights Vizathon")

        st.markdown(
            """
            ## Welcome to the FC Performance Insights Platform!
            
            ### CFCInsights
            
            This comprehensive platform offers elite football players and coaches a cutting-edge **Physical Performance Interface** that transforms how teams monitor and enhance player development.
            
            Designed with both clarity and depth, CFCInsights combines technical sophistication with intuitive visualization to deliver actionable insights directly to your fingertips.
            
            ### 📊 Performance Metrics Dashboard:
            
            - **Load Management** - Track match minutes, training load, and season availability with precision
            - **Injury Analytics** - Visualize injury patterns, risk factors, and rehabilitation progress
            - **Physical Profiling** - Monitor key performance indicators and benchmark against positional standards
            - **Player Biography** - Access comprehensive player information including career statistics
            - **Recovery Tracking** - Analyze sleep quality, nutrition intake, and wellness indicators
            - **External Factors** - Evaluate environmental conditions and their impact on performance
            
            🔍 **Navigate using the sidebar to explore specialized analytics modules for your players.**
            """,
            unsafe_allow_html=True,
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
