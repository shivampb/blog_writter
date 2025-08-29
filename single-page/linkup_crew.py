from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import os

# Load API keys

# Set up LLM
os.environ["GEMINI_API_KEY"] = "AIzaSyA8lEAd-7nP1dzw1aQriUhfJ_ksVzU36kI"
llm = LLM(model="gemini/gemini-2.5-pro", temperature=0.2)

# Web search tool
os.environ["SERPER_API_KEY"] = "2407fa8295135d6943bbe21c163e4bbe1824475e"
tool = SerperDevTool()

# === AGENTS ===

# 1. Adaptive Web Research Agent
senior_researcher = Agent(
    role="Senior Web Research Analyst",
    goal=(
        "Research and summarize only the specific topic the user requests for the year 2025 "
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

# 2. Humanizer Agent
humanizer_agent = Agent(
    role="Conversational AI Humanizer",
    goal=(
        "Rewrite the refined research report with a clear, engaging, and personal tone that feels human and use only daily life and common words, phases, sentences etc, no need to add complex english words "
        "represents the brand voice of Destinova AI Labs. Your job is to ensure the blog sounds like it's written by a thoughtful human expert, "
        "not AI. Preserve all factual content and source links, but make the writing relatable, warm, and grounded in daily life."
        "Do not Miss or skip any information from Senior Web Research Analyst agent and provide atleast 1900 tokens output"
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


# === TASKS ===

# 1. Dynamic Research Task
research_task = Task(
    description=(
        "The user has requested research on: **{topic}**\n\n"
        "Your job is to use trusted sources and write a long deep detailed **Markdown-formatted report** about this topic strictly for the year **2025**.\n\n"
        "**Important:** At the end of each paragraph or bullet point, include the **source URL** in this format source (url) from where the information was found using the search tool.\n\n"
        "**Guidelines:**\n"
        "- Focus ONLY on content relevant to the user's topic\n"
        "- Include only factual updates or developments from 2025\n"
        "- Avoid speculation, promotion, or outdated information\n"
        "- Summarize why the events are relevant, and who is involved\n"
        "- Use the search tool ONLY ONCE\n"
        "-Do Not Use Agent Search Tool"
    ),
    agent=senior_researcher,
    expected_output="A clean, 2025-specific, well-structured report on the user's requested topic.",
)


# 2. Humanization Task
humanize_task = Task(
    description=(
        "Take the verified report on **{topic}** and rewrite it with a human-first, conversational tone.\n\n"
        "You MUST follow these detailed humanization guidelines:\n\n"
        "1. ✅ Use a tone like a human author at Destinova AI Labs—friendly, insightful, casual-professional.\n"
        "2. ✅ Vary sentence lengths and rhythm. Mix short, punchy sentences with longer reflective ones.\n"
        "3. ✅ Insert light human opinions or framing like: 'From where we sit...,' 'We’ve seen this firsthand...,' 'Let’s be honest...'\n"
        "4. ✅ Use everyday analogies and micro-stories when possible. Example: 'Picture this—you’re running a business...'\n"
        "5. ✅ Use personal-style transitions: 'Here’s the deal…', 'Let’s break it down…', 'Think of it like this…'\n"
        "6. ✅ Add soft humor, rhetorical questions, or mini-scenarios to keep it engaging. Example: 'Remember when…?'\n"
        "7. ✅ Use em dashes, parentheses, and side thoughts to simulate natural writing.\n"
        "8. ❗ NEVER remove or alter any factual content or source URL.\n"
        "9. ❗ Avoid sounding robotic or templated—inject variability and intentional imperfection.\n"
        "10. ✅ Match the tone of a human writer working at a cutting-edge AI company writing for curious professionals.\n\n"
        "End result should feel like a **real person wrote it**, not a machine. All facts and sources must be preserved exactly as-is."
    ),
    agent=humanizer_agent,
    expected_output="A humanized, engaging blog-style rewrite that feels like it was written by a real person at Destinova AI Labs.",
)

# === CREW SETUP ===
crew = Crew(
    agents=[senior_researcher, humanizer_agent],
    tasks=[research_task, humanize_task],
    process=Process.sequential,
    verbose=True,
)

# === USER INPUT ===
user_input = "what are the new Business mens tactics  "
# Run the crew with topic passed in
result = crew.kickoff(inputs={"topic": user_input})

# === SAVE REPORT ===
with open("tactics_report.md", "w", encoding="utf-8") as f:
    f.write(str(result))
print("\n✅ Cleaned report saved as 'tactics_report.md'")
