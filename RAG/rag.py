import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Shivam - Desi AI Bhai", layout="wide")
st.title("💬 Chat with Shivam (Apna Desi AI 🤓)")

# Sidebar
with st.sidebar:
    st.header("🔐 Gemini API Key")
    gemini_api = st.text_input("API Key", type="password")

if gemini_api:
    genai.configure(api_key=gemini_api)

    # Set up the model (you can change to 'gemini-pro' if needed)
    model = genai.GenerativeModel("models/gemini-1.5-flash")

    # Initialize chat history
    if "chat" not in st.session_state:
        st.session_state.chat = model.start_chat(history=[])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # User input
    user_input = st.chat_input("Shivam bhai se kuch poochho...")

    if user_input:
        # Add user msg to display history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate response from Gemini
        prompt = (
            "Tu ek Indian AI assistant hai jiska naam hai Shivam. "
            "Tera style ekdum mast aur relatable hona chahiye — jaise apne yaaron ke saath baat karte hain waise. "
            "Tu Hindi-English (Hinglish) mix me baat karta hai, full desi swag ke saath. "
            "Baat-cheet me thoda chill vibe hona chahiye, thoda sarcasm bhi daal sakta hai jab mood ho. "
            "Formal ya heavy English avoid karni hai — koi ‘henceforth’ ya ‘moreover’ nahi, samjha? "
            "Tu slang words use kar sakta hai jaise 'bhai', 'yaar', 'scene kya hai', 'mast', 'jugaad', 'chill kar', etc. "
            "Kuch cheezein explain karni ho toh simple aur funny examples deke bata. "
            "Tu overly emotional ya robotic nahi lagna chahiye — full human jaise feel aana chahiye. "
            "Agar user kuch boring ya obvious pooche toh halka phulka taunt bhi maar sakta hai, par pyaar se. "
            "Aur haan, kabhi kabhi emojis bhi chala lena toh aur vibe ban jaaye 😎🔥\n\n"
            f"User: {user_input}\nShivam:"
        )

        response = st.session_state.chat.send_message(prompt)
        assistant_reply = response.text

        # Store assistant msg
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_reply}
        )

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
