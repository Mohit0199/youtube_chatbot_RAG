import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
import yt_dlp
from dotenv import load_dotenv
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

hf_api_key = os.getenv("HF_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")


# Function to extract YouTube video ID
def extract_video_id(url):
    video_id_pattern = (
        r"(?:https?://)?(?:www\.)?(?:youtube|youtu|youtube-nocookie)\.(?:com|be)/"
        r"(?:watch\?v=|embed\/|v\/|e\/|youtu\.be\/|.+\/)?([a-zA-Z0-9_-]{11})"
        r"(?:[?&][\w=-]+)*"
    )
    match = re.search(video_id_pattern, url)
    if match:
        return match.group(1)
    return None


def fetch_transcript(video_url):
    try:
        languages = [
            "ab", "aa", "af", "ak", "sq", "am", "ar", "hy", "as", "ay", "az", "bn", "ba", "eu", "be", "bho", "bs",
            "br", "bg", "my", "ca", "ceb", "zh-Hans", "zh-Hant", "co", "hr", "cs", "da", "dv", "nl", "dz", "en", "eo",
            "et", "ee", "fo", "fj", "fil", "fi", "fr", "gaa", "gl", "lg", "ka", "de", "el", "gn", "gu", "ht", "ha", "haw",
            "iw", "hi", "hmn", "hu", "is", "ig", "id", "iu", "ga", "it", "ja", "jv", "kl", "kn", "kk", "kha", "km", "rw",
            "ko", "kri", "ku", "ky", "lo", "la", "lv", "ln", "lt", "lua", "luo", "lb", "mk", "mg", "ms", "ml", "mt", "gv",
            "mi", "mr", "mn", "mfe", "ne", "new", "nso", "no", "ny", "oc", "or", "om", "os", "pam", "ps", "fa", "pl", "pt",
            "pt-PT", "pa", "qu", "ro", "rn", "ru", "sm", "sg", "sa", "gd", "sr", "crs", "sn", "sd", "si", "sk", "sl", "so",
            "st", "es", "su", "sw", "ss", "sv", "tg", "ta", "tt", "te", "th", "bo", "ti", "to", "ts", "tn", "tum", "tr",
            "tk", "uk", "ur", "ug", "uz", "ve", "vi", "war", "cy", "fy", "wo", "xh", "yi", "yo", "zu"
        ]
        loader = YoutubeLoader.from_youtube_url(video_url, language=languages)
        docs = loader.load()

        if not docs:
            return None, "No transcript found."

        # Combine all documents into one transcript
        transcript = " ".join(doc.page_content for doc in docs)

        return transcript, "Transcript fetched successfully"

    except Exception as e:
        return None, f"Error occurred: {str(e)}"



# Function to fetch YouTube video metadata
def fetch_youtube_info(video_url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    
    video_info = {
        "title": info.get('title'),
        "uploader": info.get('uploader'),
        "upload_date": info.get('upload_date'),
        "view_count": info.get('view_count'),
        "duration": info.get('duration'),
    }
    return video_info


# Function to create RAG chain
def create_rag_chain(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(
        #api_key=hf_api_key,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "lambda_mult": 0.5}
        ),
        llm=llm
    )

    prompt = PromptTemplate(
        template="""
        You are a knowledgeable and concise assistant. 
        Provide answers solely based on the following transcript context.
        If the provided context doesn’t contain enough information, respond with "I don't know."

        Context:
        {context}

        Question: {question}

        Answer:
        """,
        input_variables=['context', 'question']
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    return parallel_chain | prompt | llm | parser

