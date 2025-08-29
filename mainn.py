import streamlit as st
import asyncio
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import time

st.set_page_config(page_title="Destinova Research Assistant", layout="wide")
st.title("🔍 Destinova Research Assistant")

# Sidebar for API keys
with st.sidebar:
    st.header("🔐 API Keys")
    gemini_api = st.text_input("Gemini API Key", type="password")
    serper_api = st.text_input("Serper API Key", type="password")
    run_btn = st.button("Run Research")

# User topic input
user_input = st.text_area(
    "Enter the topic for 2025 research:", "What are the new Business mens tactics"
)

# Initialize session state to keep history
if "history" not in st.session_state:
    st.session_state.history = []


# Async logic to run Crew
async def run_crew(user_input, progress_callback):
    os.environ["GEMINI_API_KEY"] = gemini_api
    os.environ["SERPER_API_KEY"] = serper_api

    llm = LLM(model="gemini/gemini-2.5-pro", temperature=0.2)
    tool = SerperDevTool()

    senior_researcher = Agent(
        role="Senior Web Research Analyst",
        goal=(
            "Research using external internet search tool and summarize only the specific topic the user requests for the year 2025 "
            "with accuracy, clarity, and trusted context. "
            "You are strictly instructed to provide a source link at the end of each key point or paragraph "
            "based on the information retrieved via the search tool."
            "You have to process all information You got From search tool"
            "make sure to write right and accurate url link which you get from external search tool response."
            "No need to use LLM Search feature, always use External tools AND ONLY ONE TIME "
            "If any error occurs in the program, immediately stop further processing and terminate the execution"
        ),
        backstory=(
            "You are a seasoned researcher skilled at delivering high-quality insights on any topic the user provides—"
            "from technology and education to finance, science, or public policy. "
            "You strictly avoid speculation, outdated facts, or promotional fluff. "
            "You focus only on well-reasoned, fact-based developments related to 2025 and always include clear source references "
            "with each insight."
        ),
        llm=llm,
        tools=[tool],
        allow_delegation=False,
        verbose=True,
    )

    humanizer_agent = Agent(
        role="Conversational AI Humanizer",
        goal=(
            "Rewrite the refined research report with a clear, engaging, and personal tone that feels human and use only daily life and common words, phrases, sentences etc, no need to add complex english words "
            "represents the brand voice of Destinova AI Labs. Your job is to ensure the blog sounds like it's written by a thoughtful human expert, "
            "not AI. Preserve all factual content and source links, but make the writing relatable, warm, and grounded in daily life."
            "Do not Miss or skip any information from Senior Web Research Analyst agent and provide atleast 1600 tokens output"
        ),
        backstory=(
            "You are a senior writer at Destinova AI Labs, an AI company that communicates complex ideas with clarity, wit, and simplicity. "
            "You specialize in transforming technical or dry content into blog-style stories that feel like a conversation with the reader. "
            "You bring in natural rhythms, everyday analogies, soft opinions, varied sentence length, and light humor or reflection to make the message stick. "
            "You're also skilled at helping content avoid AI detection by making it feel authentically human in structure and voice."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    research_task = Task(
        description=(
            f"The user has requested research on: **{user_input}**\n\n"
            "Your job is to use trusted sources and write a long deep detailed **Markdown-formatted report** about this topic strictly for the year **2025**.\n\n"
            "**Important:** At the end of each paragraph or bullet point, include the **source URL** in this format source (url) from where the information was found using the search tool.\n\n"
            "**Guidelines:**\n"
            "- Focus ONLY on content relevant to the user's topic\n"
            "- Include only factual updates or developments from 2025\n"
            "- Avoid speculation, promotion, or outdated information\n"
            "- Summarize why the events are relevant, and who is involved\n"
            "- Use Bullet's points, med size paragraphs, etc \n"
            "- Use the search tool ONLY ONCE\n"
            "-Do Not Use Agent Search Tool"
        ),
        agent=senior_researcher,
        expected_output="A clean, 2025-specific, well-structured report on the user's requested topic.",
    )

    humanize_task = Task(
        description=(
            f"Take the verified report on **{user_input}** and rewrite it with a human-first, conversational tone.\n\n"
            "You MUST follow these detailed humanization guidelines: ... (shortened here for brevity)"
            "Add SEO Friendly title at the top of the report"
            "embed SEO Friendly keyword in the blog"
        ),
        agent=humanizer_agent,
        expected_output="A humanized, engaging blog-style rewrite that feels like it was written by a real person at Destinova AI Labs.",
    )

    crew = Crew(
        agents=[senior_researcher, humanizer_agent],
        tasks=[research_task, humanize_task],
        process=Process.sequential,
        verbose=True,
    )

    for percent in range(0, 51, 5):
        await asyncio.sleep(0.2)
        progress_callback(percent)

    output = crew.kickoff(inputs={"topic": user_input})

    for percent in range(51, 101, 5):
        await asyncio.sleep(0.2)
        progress_callback(percent)

    return str(output)


# Run and show result
if run_btn and user_input and gemini_api and serper_api:
    progress_bar = st.progress(0)

    def update_progress(val):
        progress_bar.progress(val)

    with st.spinner("Running research agents..."):
        result = asyncio.run(run_crew(user_input, update_progress))
        st.success("✅ Research complete!")
        st.session_state.history.append(result)

# Show results history
if st.session_state.history:
    for idx, report in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"### 📄 Result #{len(st.session_state.history) - idx + 1}")
        st.markdown(report, unsafe_allow_html=True)
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"tactics_report_{len(st.session_state.history) - idx + 1}.md",
            key=f"download_{idx}",
        )
