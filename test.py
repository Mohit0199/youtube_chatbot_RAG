import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore")

load_dotenv()

hf_api_key = os.getenv("HF_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")


import re

def extract_video_id(url):
    # Updated regex pattern to match various YouTube URL formats
    video_id_pattern = (
        r"(?:https?://)?(?:www\.)?(?:youtube|youtu|youtube-nocookie)\.(?:com|be)/"
        r"(?:watch\?v=|embed\/|v\/|e\/|youtu\.be\/|.+\/)?([a-zA-Z0-9_-]{11})"
        r"(?:[?&][\w=-]+)*"  # Allow for query parameters after the video ID
    )

    match = re.search(video_id_pattern, url)

    if match:
        return match.group(1)  # Return the video ID
    else:
        print("Invalid YouTube URL.")
        return ""
    

# Example YouTube URL input by the user
youtube_url = input("Enter the YouTube video URL: ")
video_id = extract_video_id(youtube_url)


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

if video_id:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)

        transcript = " ".join(chunk['text'] for chunk in transcript_list)

    except TranscriptsDisabled:
        print("No captions available for this video.")

    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])


    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_api_key,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    vector_store = FAISS.from_documents(chunks, embeddings)


    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")


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
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text
    

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()

    main_chain = parallel_chain | prompt | llm | parser

    question = input("Ask a question about the video: ")

    result = main_chain.invoke(question)

    print(result)
