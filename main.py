import streamlit as st
import asyncio
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from io import BytesIO
import time

st.set_page_config(page_title="ReComAI Research Assistant", layout="wide")
st.title("🔍 ReComAI Research Assistant")

# Sidebar for API keys
with st.sidebar:
    st.header("🔐 API Keys")
    gemini_api = st.text_input("Gemini API Key", type="password")
    serper_api = st.text_input("Serper API Key", type="password")
    run_btn = st.button("Run Research")

# User topic input
user_input = st.text_area(
    "Enter the topic for 2026 research:", "What are the new Business mens tactics"
)

# Initialize session state to keep history
if "history" not in st.session_state:
    st.session_state.history = []


# Async logic to run Crew
async def run_crew(user_input, progress_callback):
    os.environ["GEMINI_API_KEY"] = gemini_api
    os.environ["SERPER_API_KEY"] = serper_api

    llm = LLM(model="gemini/gemini-2.5-flash", temperature=0.2)
    tool = SerperDevTool()

    senior_researcher = Agent(
        role="Senior Web Research Analyst",
        goal=(
            "Research using external internet search tool and summarize only the specific topic the user requests for the year 2026 "
            "with accuracy, clarity, and trusted context. "
            "CRITICAL: You MUST use the search tool to find related blog articles from Destinova AI Labs. "
            "Execute a search query like 'site:destinovaailabs.com/blog [user_topic]' to find relevant articles. "
            "Collect at least 2-3 relevant URLs from 'destinovaailabs.com/blog' to be used for internal context linking. "
            "You are strictly instructed to provide a source link at the end of each key point or paragraph "
            "based on the information retrieved via the search tool."
            "If any error occurs, immediately stop further processing."
        ),
        backstory=(
            "You are a seasoned researcher skilled at delivering high-quality insights on any topic the user provides. "
            "You also specialize in content strategy, ensuring new content connects with existing knowledge bases. "
            "You strictly avoid speculation, outdated facts, or promotional fluff. "
            "You focus only on well-reasoned, fact-based developments related to 2026."
        ),
        llm=llm,
        tools=[tool],
        allow_delegation=False,
        verbose=True,
        max_iter=3,  # Limit tools usage
        max_rpm=10,   # Avoid hitting rate limits
    )

    content_writer_agent = Agent(
        role="Senior Content Writer & SEO Specialist",
        goal=(
            "Synthesize the research into a high-quality, human-like blog post that ranks on Google and Answer Engines (AEO). "
            "1. **Humanize**: Write in a clear, engaging, personal tone. Use analogies, varied sentence lengths, and avoid 'AI fluff'. "
            "2. **AEO Structure**: rigidly follow 'Question -> Direct Answer (~50 words) -> Detailed Elaboration' for key sections. "
            "3. **SEO Integration**: Naturally weave High-Volume keywords into headings and text without stuffing. Bold them. "
            "4. **Internal Linking**: Contextually embed the 'destinovaailabs.com/blog' links provided by the Researcher. "
            "5. **Brand & External**: Link to ReComAI, LinkedIn, and the Shopify App as specified. "
            "Verify all links are functional and relevant."
            "\n**Keywords to Target**:\n"
            "Core Short-Tail Keywords (High Volume / Broad)\n"
            "AI product recommendation Shopify\n"
            "Shopify AI chatbot\n"
            "Shopify AI app\n"
            "AI product recommendations\n"
            "Shopify product recommendation app\n"
            "Long-Tail Keywords (Higher intent, easier to rank)\n"
            "best AI product recommendation app for Shopify\n"
            "AI chatbot for Shopify stores\n"
            "AI-powered Shopify product suggestions\n"
            "personalize product recommendations Shopify\n"
            "AI upsell and cross-sell app for Shopify\n"
            "agentic AI chatbot for Shopify eCommerce\n"
            "AI tool to boost Shopify sales\n"
            "AI Shopify app for customer support and product suggestions\n\n"
            "Transactional / Commercial Keywords (People ready to buy/try)\n"
            "install AI chatbot Shopify\n"
            "AI app for Shopify to increase sales\n"
            "Shopify AI recommendation engine pricing\n"
            "Shopify AI chatbot with upsell features\n"
            "Shopify AI personalization app download\n\n"
            "Problem-Solution Keywords (Great for blog/article targeting)\n"
            "how to recommend products on Shopify using AI\n"
            "improve Shopify sales with AI\n"
            "AI for Shopify product discovery\n"
            "automate product suggestions in Shopify store\n"
            "AI product recommendation engine\n"
            "AI to reduce Shopify cart abandonment"
        ),
        backstory=(
            "You are a dual-expert: a first-class storyteller and a technical SEO strategist. "
            "You know how to write content that keeps humans reading while satisfying the algorithms of Google and Perplexity. "
            "You hate generic AI content and strive to make every sentence feel handcrafted. "
            "You believe internal linking is the backbone of site authority."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
        max_rpm=10,
    )

    research_task = Task(
        description=(
            f"The user has requested research on: **{user_input}**\n\n"
            "Your job is to:\n"
            "1. Conduct deep research using trusted sources for the year **2026**.\n"
            "2. **Internal Link Discovery**: Use the search tool to find 3-5 existing blog articles on 'destinovaailabs.com/blog' that are relevant to this topic. Query example: `site:destinovaailabs.com/blog {topic}`.\n"
            "3. Compile a detailed Markdown report.\n"
            "4. **List the Internal Links** you found clearly at the beginning or end of your report so the writer can use them.\n\n"
            "**Guidelines:**\n"
            "- Focus ONLY on content relevant to the user's topic\n"
            "- Include only factual updates or developments from 2026\n"
            "- Avoid speculation, promotion, or outdated information\n"
            "- Use source URLs at the end of paragraphs.\n"
        ),
        agent=senior_researcher,
        expected_output="A clean, 2026-specific report including a list of relevant internal blog URLs from destinovaailabs.com.",
    )

    seo_guidelines = (
        "Guidelines:\n"
        "1. SEO Title Guidelines:\n"
        "- Keep titles 50–60 characters long.\n"
        "- Place the primary keyword at the beginning.\n"
        "- Use numbers, power words, or questions to attract clicks.\n"
        "- Avoid keyword stuffing.\n\n"
        "2. AEO (Answer Engine Optimization):\n"
        "- Use Question-based Headings (H2/H3) for key topics (e.g. 'How does AI help...').\n"
        "- Provide a 'Direct Answer Block' (2-3 sentences, ~40-60 words) immediately after the question.\n"
        "- Use logical formatting: Bullet points, Numbered lists, and Bold text for easy AI parsing.\n\n"
        "3. Internal Linking:\n"
        "- Ensure the blog has at least 2-3 internal links to 'destinovaailabs.com/blog'.\n"
        "- Link text (anchor text) should be descriptive and relevant.\n"
        "4. External & Brand Links:\n"
        "- ReComAI Website: https://recomai.one/\n"
        "- LinkedIn: https://in.linkedin.com/company/destinova-ai-labs\n"
        "- Shopify App: https://apps.shopify.com/desti-ai-automate-chatbot\n"
        "5. High Volume Keywords:\n"
        "- Verify that the high-volume short-tail keywords are present in the Title, Introduction, and at least one H2.\n"
        "- Ensure the flow remains natural despite the strategic placement.\n"
    )

    writing_task = Task(
        description=(
            f"Write a full-length, SEO-optimized blog post on **{user_input}** based on the provided research.\n"
            "1. **Tone**: Conversational, human, engaging. (No 'In the rapidly evolving landscape...').\n"
            "2. **Structure (AEO)**: Use H2/H3 for questions. Follow immediately with a direct answer block, then explain.\n"
            "3. **Links**: \n"
            "   - Integrate the `destinovaailabs.com/blog` links found by the Researcher naturally.\n"
            "   - Add ReComAI Home: `https://recomai.one/`\n"
            "   - Add Shopify App: `https://apps.shopify.com/desti-ai-automate-chatbot`\n"
            "4. **Keywords**: Use the list in your goal. Bold them.\n"
            f"{seo_guidelines}"
        ),
        agent=content_writer_agent,
        expected_output="A production-ready, human-written, SEO/AEO-optimized blog post with correct internal and external links.",
    )

    crew = Crew(
        agents=[senior_researcher, content_writer_agent],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

    # Update progress to 50% before starting the crew
    progress_callback(50)

    output = crew.kickoff(inputs={"topic": user_input})

    # Update progress to 100% after completion
    progress_callback(100)

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
