
import streamlit as st
import json
import random

from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IoT Wearable AI Brain",
    page_icon="⌚",
    layout="wide"
)


# ============================================================
# 2. TITLE
# ============================================================

st.title("⌚ IoT Wearable AI Brain")

st.write(
    "An edge-simulated multi-agent system that analyzes "
    "wearable telemetry and decides when to interrupt the user."
)


# ============================================================
# 3. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    user_api_key = st.text_input(
        "Groq API Key:",
        type="password"
    )

    st.divider()

    st.subheader("Telemetry Controls")

    heart_rate = st.slider(
        "❤️ Heart Rate",
        min_value=40,
        max_value=180,
        value=75
    )

    movement = st.slider(
        "🏃 Movement Level",
        min_value=0,
        max_value=100,
        value=50
    )

    battery = st.slider(
        "🔋 Battery %",
        min_value=1,
        max_value=100,
        value=50
    )

    unfamiliar_city = st.checkbox(
        "🗺️ Navigating unfamiliar city"
    )

    active_app = st.selectbox(
        "📱 Active App",
        [
            "None",
            "Beads Counter",
            "Maps",
            "Music",
            "Messages",
            "Fitness"
        ]
    )

    sleeping = st.checkbox(
        "😴 User may be sleeping"
    )

    st.divider()

    if st.button("🔄 Reset"):

        st.session_state.telemetry_history = []

        st.session_state.last_profile = ""

        st.session_state.last_action = ""

        st.rerun()


# ============================================================
# 4. SESSION MEMORY
# ============================================================

if "telemetry_history" not in st.session_state:

    st.session_state.telemetry_history = []


if "last_profile" not in st.session_state:

    st.session_state.last_profile = ""


if "last_action" not in st.session_state:

    st.session_state.last_action = ""


# ============================================================
# 5. CREATE TELEMETRY
# ============================================================

telemetry = {

    "heart_rate": heart_rate,

    "movement_level": movement,

    "battery_level": battery,

    "gps": {
        "unfamiliar_city": unfamiliar_city
    },

    "active_app": active_app,

    "sleeping": sleeping
}


# ============================================================
# 6. DISPLAY CURRENT TELEMETRY
# ============================================================

st.subheader("📡 Current Wearable Telemetry")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "❤️ Heart Rate",
        f"{heart_rate} BPM"
    )

with col2:

    st.metric(
        "🏃 Movement",
        movement
    )

with col3:

    st.metric(
        "🔋 Battery",
        f"{battery}%"
    )

with col4:

    if unfamiliar_city:
        st.metric("🗺️ Navigation", "Unfamiliar")
    else:
        st.metric("🗺️ Navigation", "Normal")


st.json(telemetry)


# ============================================================
# 7. TELEMETRY HISTORY
# ============================================================

if len(st.session_state.telemetry_history) > 0:

    st.subheader("📊 Rolling Telemetry Window")

    st.dataframe(
        st.session_state.telemetry_history,
        use_container_width=True
    )


# ============================================================
# 8. CHECK API KEY
# ============================================================

if not user_api_key:

    st.info(
        "Enter your Groq API key in the sidebar to activate "
        "the Profiler and Action agents."
    )

    st.stop()


# ============================================================
# 9. CREATE LLM
# ============================================================

llm = ChatGroq(

    model="openai/gpt-oss-20b",

    temperature=0,

    api_key=user_api_key
)


# ============================================================
# 10. DEFINE STATE
# ============================================================

class State(TypedDict):

    telemetry: dict

    profile: str

    action: str

    messages: Annotated[list, add_messages]


# ============================================================
# 11. PROFILER AGENT
# ============================================================

def profiler_agent(state: State):

    telemetry_data = state["telemetry"]

    prompt = f"""
You are the Profiler Agent for a smartwatch.

Analyze the following wearable telemetry:

{json.dumps(telemetry_data, indent=2)}

Create a concise description of the user's current state.

Consider:

- Heart rate
- Movement
- Sleeping status
- Battery
- GPS/navigation
- Active application

Examples:

"User is sleeping."

"User is highly active."

"User is commuting through an unfamiliar city."

"User is sitting still but has an unusually high heart rate."

Return only the current user-state description.
"""

    response = llm.invoke(prompt)

    profile = response.content

    return {
        "profile": profile
    }


# ============================================================
# 12. ACTION AGENT
# ============================================================

def action_agent(state: State):

    telemetry_data = state["telemetry"]

    profile = state["profile"]

    prompt = f"""
You are the Action Agent for a smartwatch.

Current telemetry:

{json.dumps(telemetry_data, indent=2)}

Profiler Agent says:

{profile}

Your job is to decide whether the smartwatch should
interrupt the user.

IMPORTANT RULES:

1. If heart rate is very high while movement is very low,
   consider this a possible emergency.

2. If battery is very low AND the user is navigating
   an unfamiliar city, DO NOT interrupt them with
   unnecessary notifications.

3. Low battery by itself should normally result in a
   low-priority battery warning.

4. Normal activity should result in no interruption.

5. Sleeping users should generally not be interrupted
   unless there is a potentially serious emergency.

Return exactly:

ACTION: <one of EMERGENCY, WARNING, QUIET>
REASON: <short explanation>

Do not provide medical diagnosis.
"""

    response = llm.invoke(prompt)

    action = response.content

    return {
        "action": action
    }


# ============================================================
# 13. BUILD MULTI-AGENT GRAPH
# ============================================================

builder = StateGraph(State)


# Add agents

builder.add_node(
    "profiler",
    profiler_agent
)

builder.add_node(
    "action",
    action_agent
)


# START → Profiler

builder.add_edge(
    START,
    "profiler"
)


# Profiler → Action

builder.add_edge(
    "profiler",
    "action"
)


# Action → END

builder.add_edge(
    "action",
    END
)


# Compile graph

graph = builder.compile()


# ============================================================
# 14. RUN SIMULATION
# ============================================================

if st.button(
    "🚀 Analyze Wearable Context",
    type="primary"
):

    # -----------------------------------------------
    # Add telemetry to rolling window
    # -----------------------------------------------

    st.session_state.telemetry_history.append(
        {
            "Heart Rate": heart_rate,
            "Movement": movement,
            "Battery": battery,
            "Unfamiliar City": unfamiliar_city,
            "Active App": active_app,
            "Sleeping": sleeping
        }
    )


    # Keep only last 5 readings

    st.session_state.telemetry_history = (
        st.session_state.telemetry_history[-5:]
    )


    # -----------------------------------------------
    # Initial state
    # -----------------------------------------------

    initial_state = {

        "telemetry": telemetry,

        "profile": "",

        "action": "",

        "messages": []
    }


    # -----------------------------------------------
    # Run graph
    # -----------------------------------------------

    with st.spinner(
        "🧠 Multi-agent system is analyzing..."
    ):

        result = graph.invoke(
            initial_state
        )


    # Save results

    st.session_state.last_profile = result["profile"]

    st.session_state.last_action = result["action"]


    # ==================================================
    # DISPLAY AGENT RESULTS
    # ==================================================

    st.divider()

    st.subheader("🧠 Profiler Agent")

    st.info(
        result["profile"]
    )


    st.subheader("⚡ Action Agent")

    action_text = result["action"]


    # Determine UI based on action

    if "EMERGENCY" in action_text.upper():

        st.error(
            f"🚨 {action_text}"
        )

    elif "WARNING" in action_text.upper():

        st.warning(
            f"⚠️ {action_text}"
        )

    else:

        st.success(
            f"🔕 {action_text}"
        )


# ============================================================
# 15. SHOW LAST RESULT
# ============================================================

if st.session_state.last_profile:

    st.divider()

    st.subheader("📋 Latest AI Decision")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🧠 User Context")

        st.write(
            st.session_state.last_profile
        )

    with col2:

        st.markdown("### ⚡ Action")

        st.write(
            st.session_state.last_action
        )
