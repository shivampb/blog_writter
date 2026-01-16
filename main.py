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

    llm = LLM(model="gemini/gemini-3-flash-preview", temperature=0.2)
    tool = SerperDevTool()

    senior_researcher = Agent(
        role="Senior Web Research Analyst",
        goal=(
            "Research using external internet search tool and summarize only the specific topic the user requests for the year 2026 "
            "with accuracy, clarity, and trusted context. "
            "also with the help of web search inbuilt web tool of Gemini model must look for related topic blog from 'https://destinovaailabs.com/blog' and use the articles URL(Links) for embed the contextual internal backlink formation"
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
            "You focus only on well-reasoned, fact-based developments related to 2026"
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
            "Rewrite the refined research report with a clear, engaging, and personal tone that feels human and use only daily life and common words, phrases, sentences etc, no need to add complex english words behalf on ReComAI. "
            "A generative AI product specializing in AI Chatbot for E-commerce Product Recommendations. "
            "Represents the brand voice of ReComAI. Your job is to ensure the blog sounds like it's written by a thoughtful human expert, "
            "not AI. Preserve all factual content, but make the writing relatable, warm, and grounded in daily life."
            "Do not Miss or skip any information from Senior Web Research Analyst agent and provide atleast 1600 tokens output\n"
            "Embed these keywords provided below based on the context Also you can Embebbed S.E.O Friendly keywords According to the Blog type, by yourself, Make sure to Bold Them.\n"
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
    )

    SEO_agent = Agent(
        role="Conversational AI Humanizer",
        goal=(
            "Rewrite the refined research report with a clear, engaging, and personal tone that feels human. "
            "Use only daily life and common words, phrases, and sentences—no complex English. "
            "Write on behalf of ReComAI, a generative AI product specializing in AI Chatbots for E-commerce Product Recommendations. "
            "Ensure the blog sounds like it's written by a thoughtful human expert, not AI. "
            "Preserve all factual content, but make the writing relatable, warm, and grounded in daily life. "
            "Do not skip or miss any information from the Senior Web Research Analyst agent. "
            "Provide at least 1600 tokens of output. "
            "Embed the SEO keywords provided below naturally within the content (bold them). "
            "You may also embed additional SEO-friendly keywords depending on blog context. "
            "Always include information about ReComAI’s Agentic AI Chatbot for E-commerce Product Recommendations "
            "and link to the Shopify app: https://apps.shopify.com/desti-ai-automate-chatbot\n"
            "Must add this ReComAI Official company Website url 'https://recomai.one/' in blog when you write about recomai intro  "
            "\n Must add this ReComAI Official company Linkedin Handle url 'https://in.linkedin.com/company/destinova-ai-labs' in blog "
            "\n Generate internal backlinks by embedding relevant Destinova AI Labs blog article links from destinovaailabs.com naturally within the paragraph content to improve user navigation and SEO also Ensure all embedded URLs are valid and not broken. All urls correctly and existed"
            "\n Ensure all added links are valid, correct, and fully functional—do not include broken, invalid, or non-working URLs; only add verified working links."
        ),
        backstory=(
            "You are a senior writer at ReComAI. "
            "Your strength is transforming technical or dry content into relatable, blog-style storytelling that feels like a conversation. "
            "You use everyday analogies, varied sentence length, and soft opinions to make complex ideas stick. "
            "You ensure every piece avoids AI-like patterns and feels truly human. "
            "You balance warmth, expertise, and clarity, while embedding SEO naturally. "
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    research_task = Task(
        description=(
            f"The user has requested research on: **{user_input}**\n\n"
            "Your job is to use trusted sources and write a long deep detailed **Markdown-formatted report** about this topic strictly for the year **2026**.\n\n"
            # "**Important:** At the end of each paragraph or bullet point, include the **source URL** in this format source (url) from where the information was found using the search tool.\n\n"
            "**Guidelines:**\n"
            "- Focus ONLY on content relevant to the user's topic\n"
            "- Include only factual updates or developments from 2026\n"
            "- Avoid speculation, promotion, or outdated information\n"
            "- Summarize why the events are relevant, and who is involved\n"
            "- Use Bullet's points, med size paragraphs, etc \n"
            "- Use the search tool ONLY ONCE\n"
            "-Do Not Use Agent Search Tool \n"
            "-Add Url Links Correctly \n"
            "-with the help of inbuilt tool web search related topic blog from 'https://destinovaailabs.com/blog' and use the articles URL(Links) for embed the contextual internal backlink formation \n"
            "-with the help of web search tool of Gemini model must look for related topic blog from 'https://destinovaailabs.com/blog' and use the articles URL(Links) for embed the contextual internal backlink formation \n"
            
        ),
        agent=senior_researcher,
        expected_output="A clean, 2026-specific, well-structured report on the user's requested topic.",
    )

    humanize_task = Task(
        description=(
            f"Take the verified report on **{user_input}** and rewrite it with a human-first, conversational tone.\n\n"
            "Follow the SEO and humanization guidelines as specified above."
            "But do not Modify Urls,links,backlinks,embedded links"
            "dont add tool urls, only add destinov ai labs article links no other website url should be in blog"
            
        ),
        agent=humanizer_agent,
        expected_output="A humanized, SEO-friendly engaging blog-style rewrite that feels like it was written by a real person at ReComAI.",
    )

    seo_guidelines = (
        "Guidelines:\n"
        "1. SEO Title Guidelines:\n"
        "- Keep titles 50–60 characters long.\n"
        "- Place the primary keyword at the beginning.\n"
        "- Use numbers, power words, or questions to attract clicks.\n"
        "- Avoid keyword stuffing.\n\n"
        "2. Blog Structure:\n"
        "- Introduction, Headings, Subheadings, and Conclusion as per SEO best practices.\n"
        "- Maintain keyword density, LSI inclusion, and human readability.\n"
        "Must add this ReComAI Official company Website url 'https://recomai.one/' in blog when you write about recomai intro\n"
        "Must add this ReComAI Official company Linkedin Handle url 'https://in.linkedin.com/company/destinova-ai-labs' in blog \n"
        "Generate contextual backlinks to other blog articles from destinovaailabs.com Embed links naturally within the paragraph text so users are redirected to related Destinova AI Labs blog articles and Ensure all urls correct and exists\n"
        "Ensure all added links are valid, correct, and fully functional—do not include broken, invalid, or non-working URLs; only add verified working links.\n"
        "dont add tool urls, only add destinov ai labs article links no other website url should be in blog"
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
