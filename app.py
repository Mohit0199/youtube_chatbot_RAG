import streamlit as st
from main import extract_video_id, fetch_transcript, create_rag_chain, fetch_youtube_info

# Streamlit UI
st.set_page_config(page_title="YouTube Video Q&A", page_icon="🎥", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stTextInput>div>input { border-radius: 5px; }
    
    /* Chat message styles */
    .chat-message {
        padding: 10px 15px;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 75%;
        word-wrap: break-word;
        white-space: pre-wrap;
        font-size: 16px;
    }

    /* User message style */
    .user-message { 
        background-color: #daf8e3;  /* Soft green background */
        color: #3c763d;  /* Darker green for text */
        text-align: right;
        margin-left: auto;
    }

    /* Bot message style */
    .bot-message { 
        background-color: #e6f7ff;  /* Soft blue background */
        color: #1a73e8;  /* Blue color for bot text */
        text-align: left;
        margin-right: auto;
    }

    /* Special styling for the container */
    .chat-message {
        max-width: 70%;  /* Adjust max-width based on screen size */
    }

    /* Make the user bubble narrower if the message is short */
    .user-message {
        max-width: auto;
        width: fit-content;
    }

    /* Make the bot bubble more responsive to message length */
    .bot-message {
        max-width: auto;
        width: fit-content;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 'input_url'
    st.session_state.transcript = None
    st.session_state.language = None
    st.session_state.rag_chain = None
    st.session_state.messages = []

# Welcome message
st.title("YouTube Video Q&A Chatbot")
st.markdown("""
    Welcome to the YouTube Video Q&A Chatbot! 
    Paste a YouTube video URL, and I'll fetch its transcript to answer your questions.  
    Ask anything about the video, or type **"exit"** to start over with a new video.
""")

# URL input stage
if st.session_state.stage == 'input_url':
    with st.form(key='url_form'):
        youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")
        submit_button = st.form_submit_button(label="Fetch Transcript")

    if submit_button and youtube_url:
        with st.spinner("Fetching video details..."):
            try:
                info = fetch_youtube_info(youtube_url)
            except Exception as e:
                st.error(f"Failed to fetch video details. Error: {e}")
                info = None

        if info:
            video_id = extract_video_id(youtube_url)
            if video_id:
                transcript_text, language_or_error = fetch_transcript(video_id)
                if transcript_text:
                    st.session_state.transcript = transcript_text
                    st.session_state.language = language_or_error
                    st.session_state.rag_chain = create_rag_chain(transcript_text)
                    st.session_state.stage = 'ask_questions'
                    st.session_state.messages.append(("bot", f"Transcript fetched in language: {language_or_error}"))
                    st.session_state.video_info = info
                    st.session_state.youtube_url = youtube_url
                    st.rerun()
                else:
                    st.error(language_or_error)
            else:
                st.error("Invalid YouTube URL. Please try again.")

# Question-answering stage
elif st.session_state.stage == 'ask_questions':
    left_col, right_col = st.columns([2, 3])

    with left_col:
        # Video display
        st.video(st.session_state.youtube_url)

        # Video details below video
        info = st.session_state.video_info
        st.subheader(info['title'])
        st.markdown(f"**Uploader:** {info['uploader']}")
        st.markdown(f"**Upload Date:** {info['upload_date']}")
        st.markdown(f"**Views:** {info['view_count']:,}")
        st.markdown(f"**Duration:** {info['duration']} seconds")

    with right_col:
        st.subheader(f"Transcript Language: {st.session_state.language}")
        st.write("Ask your questions below. Type 'exit' to start over with a new video.")

        # Display chat history
        for sender, message in st.session_state.messages:
            if sender == "user":
                st.markdown(f"<div class='chat-message user-message'>🧑 <b>You:</b> {message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message bot-message'>🤖 <b>Bot:</b> {message}</div>", unsafe_allow_html=True)

        # Question input
        with st.form(key='question_form', clear_on_submit=True):
            question = st.text_input("Your Question:", placeholder="Ask about the video or type 'exit' to reset")
            ask_button = st.form_submit_button(label="Ask")

        if ask_button and question:
            if question.lower().strip() == "exit":
                # Reset session state
                st.session_state.stage = 'input_url'
                st.session_state.transcript = None
                st.session_state.language = None
                st.session_state.rag_chain = None
                st.session_state.messages = []
                st.session_state.video_info = None
                st.session_state.youtube_url = None
                st.rerun()
            else:
                st.session_state.messages.append(("user", question))
                try:
                    result = st.session_state.rag_chain.invoke(question)
                    st.session_state.messages.append(("bot", result))
                except Exception as e:
                    st.session_state.messages.append(("bot", f"Error processing question: {str(e)}"))
                st.rerun()
