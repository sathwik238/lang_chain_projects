import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnablePassthrough
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Helper Functions
def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        query = parse_qs(parsed_url.query)
        return query.get("v", [None])[0]
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip('/')
    return None

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Streamlit UI
st.title("🎥 YouTube Chatbot with RAG")
st.write("Ask any question from a YouTube video!")

url = st.text_input("Enter YouTube Video URL:")
user_question = st.text_input("Ask your question:")

if st.button("Submit"):
    if url and user_question:
        with st.spinner("Processing video..."):
            try:
                # Step 1: Extract video ID and get transcript
                video_id = extract_video_id(url)
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
                transcript_text = ' '.join([entry['text'] for entry in transcript])

                # Step 2: Split into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript_text])

                # Step 3: Generate embeddings and store in FAISS
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
                vectorstore = FAISS.from_documents(chunks, embeddings)

                # Step 4: Retriever setup
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

                # Step 5: Augmented generation setup
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

                prompt = PromptTemplate(
                    template="Answer the question based on the context below. If the answer is not in the context, say 'I don't know'.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
                    input_variables=["context", "question"]
                )

                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })

                parser = StrOutputParser()
                main_chain = parallel_chain | prompt | llm | parser

                # Step 6: Invoke the chain with user's question
                answer = main_chain.invoke(user_question)

                st.success("Answer:")
                st.write(answer)

            except TranscriptsDisabled:
                st.error("Transcripts are disabled for this video.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter both a YouTube URL and a question!")
