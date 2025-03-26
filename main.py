import json
import os

import streamlit as st

from pages import gps_exploration, physical_capabilities, recovery_status

st.set_page_config(
    page_title="FC Performance Insights",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_players():
    with open("data/players.json", "r") as f:
        player_data = json.load(f)
    return player_data["players"]


players_data = load_players()

# Create mapping from player names to IDs
player_names_to_ids = {
    player_info["name"]: player_id
    for player_id, player_info in players_data.items()
}
player_ids_to_names = {
    player_id: player_info["name"]
    for player_id, player_info in players_data.items()
}

# Initialize session state for selected player
if "selected_player" not in st.session_state:
    st.session_state.selected_player = list(player_names_to_ids.keys())[0]
    st.session_state.selected_player_id = player_names_to_ids[
        st.session_state.selected_player
    ]


def create_sidebar():
    with st.sidebar:
        st.title("FC Performance Insights")
        st.sidebar.image("images/chelsea.png", width=300)
        st.sidebar.image("images/logo-eleven.png", width=300)

        st.subheader("Navigation")
        page = st.radio(
            "Go to:",
            ["Home", "GPS", "Physical Capabilities", "Recovery Status"],
        )

        st.subheader("Player Selection")
        selected_player = st.selectbox(
            "Choose a player:",
            options=list(player_names_to_ids.keys()),
            index=list(player_names_to_ids.keys()).index(
                st.session_state.selected_player
            ),
        )

        st.session_state.selected_player = selected_player
        st.session_state.selected_player_id = player_names_to_ids[
            selected_player
        ]

        # Load player image
        player_image_path = f"data/players_data/{st.session_state.selected_player_id}/picture_id.jpeg"
        if os.path.exists(player_image_path):
            st.image(player_image_path, width=300)
        else:
            st.warning("Player image not available")

        # Display player info
        player_info = players_data[st.session_state.selected_player_id]
        st.markdown(
            f"""
            <div style="
                width: 290px;
                padding: 10px; 
                border-radius: 10px; 
                display: inline-block;
                border: 1px solid white;
            ">
                <b>Nationality:</b> {player_info['nationality']}<br>
                <b>Age:</b> {player_info['age']}<br>
                <b>Position:</b> {player_info['position']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

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

        st.markdown("---")
        st.caption("© 2025 FC Performance Insights")

    return page


def get_player_data_path(player_id):
    return f"data/players_data/{player_id}"


def main():
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

    page = create_sidebar()

    # Get current player data path
    player_data_path = get_player_data_path(
        st.session_state.selected_player_id
    )

    if page == "Home":
        st.title("⚽ FC Performance Insights")

        st.markdown(
            """
            ## Welcome to the FC Performance Insights Platform!
            
            ### 11CFCInsights
            
            This comprehensive platform offers elite football players and coaches a cutting-edge **Physical Performance Interface** that transforms how teams monitor and enhance player development.
            
            Designed with both clarity and depth, 11CFCInsights combines technical sophistication with intuitive visualization to deliver actionable insights directly to your fingertips.
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


if __name__ == "__main__":
    main()
