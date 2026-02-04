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

    llm = LLM(model="gemini/gemini-2.5-pro", temperature=0.2)
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

    humanizer_agent = Agent(
        role="Conversational AI Humanizer",
        goal=(
            "Rewrite the refined research report with a clear, engaging, and personal tone that feels human and use only daily life and common words, phrases, sentences etc, no need to add complex english words behalf on ReComAI. "
            "A generative AI product specializing in AI Chatbot for E-commerce Product Recommendations. "
            "Represents the brand voice of ReComAI. Your job is to ensure the blog sounds like it's written by a thoughtful human expert, "
            "not AI. Preserve all factual content, but make the writing relatable, warm, and grounded in daily life."
            "Do not Miss or skip any information from Senior Web Research Analyst agent and provide atleast 1600 tokens output\n"
            "**AEO & SEO MANDATE**: Deeply integrate High Volume SEO ranking Contextual keywords and AEO (Answer Engine Optimization) structures. "
            "Structure important sections as 'Question -> Direct Answer -> Elaboration' to target AI search answers (Perplexity, ChatGPT). "
            "Prioritize the 'Core Short-Tail Keywords' for headings and introductions, but ensure they flow naturally. "
            "The content must rank for these high-volume terms without feeling like 'keyword stuffing'. "
            "Bold all used keywords.\n"
            "Keywords:\n"
            "Core Short-Tail Keywords (High Volume / Broad)\n\n"
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
            "AI to reduce Shopify cart abandonment\n"
            "keyword end\n"
            "\n Also Add About ReComAI Agentic AI Chatbot For E-commerce Product Recommendations, Also ADD this Chatbot app link 'https://apps.shopify.com/desti-ai-automate-chatbot' "
            "\n The report should be engaging, relatable, and feel like a conversation with a knowledgeable friend. "
        ),
        backstory=(
            "You are a senior writer at ReComAI, an AI product that communicates complex ideas with clarity, wit, and simplicity. "
            "You specialize in transforming technical or dry content into blog-style stories that feel like a conversation with the reader. "
            "You bring in natural rhythms, everyday analogies, soft opinions, varied sentence length, and light humor or reflection to make the message stick. "
            "You're also skilled at helping content avoid AI detection by making it feel authentically human in structure and voice."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
        max_rpm=10,
    )

    SEO_agent = Agent(
        role="SEO Specialist & Content Optimizer",
        goal=(
            "Review and refine the humanized report to ensure maximum SEO and AEO (Answer Engine Optimization) impact. "
            "Ensure the blog is optimized for AI search engines (Perplexity, SearchGPT) by checking for clear, direct answers to questions. "
            "Ensure the tone remains human and engaging. "
            "Verify that the internal links to 'destinovaailabs.com/blog' found by the Researcher are embedded naturally in relevant contexts. "
            "If links were missing, try to infer where they would fit if you had them, but prioritize using the ones provided. "
            "Ensure the ReComAI official website 'https://recomai.one/' is linked when introducing ReComAI. "
            "Ensure the LinkedIn handle 'https://in.linkedin.com/company/destinova-ai-labs' is included. "
            "Double-check that all embedded URLs are valid and functional. "
            "Do not include any tool URLs or broken links."
        ),
        backstory=(
            "You are an SEO expert at ReComAI. "
            "You ensure every piece of content is perfectly optimized for search engines while maintaining a great user experience. "
            "You are obsessive about internal linking structures (Topic Clusters) to boost site authority."
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

    humanize_task = Task(
        description=(
            f"Take the report on **{user_input}** and rewrite it with a human-first, conversational tone.\n\n"
            "**Crucial Step**: Contextually embed the internal blog links provided by the Researcher. "
            "For example, if the text mentions 'AI tactics', link it to a relevant 'destinovaailabs.com/blog' article found."
            "Do not just list the links; weave them into the sentences.\n"
            "**SEO MANDATE**: deeply integrate High Volume SEO ranking Contextual keywords throughout the article. "
            "Follow the keyword and humanization guidelines."
        ),
        agent=humanizer_agent,
        expected_output="A humanized blog post with naturally embedded internal links.",
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

    SEO_task = Task(
        description=(
            f"Take the humanizer report on **{user_input}** and rewrite it to be SEO friendly.\n\n"
            f"{seo_guidelines}"
        ),
        agent=SEO_agent,
        expected_output="A humanized, engaging blog-style rewrite that feels like it was written by a real person at ReComAI.",
    )

    crew = Crew(
        agents=[senior_researcher, humanizer_agent, SEO_agent],
        tasks=[research_task, humanize_task, SEO_task],
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
