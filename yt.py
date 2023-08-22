import os
from creds import apikey
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.tools import YouTubeSearchTool
import streamlit as st
import string
import spacy
from collections import Counter
from openai.error import AuthenticationError



nlp = spacy.load("en_core_web_sm")


#os.environ["OPENAI_API_KEY"] = apikey

st.title("Ask questions on your Youtube Video")

url = st.text_input("Plug in your Youtube Video")

openai_api_key = st.sidebar.text_input("Enter your OPENAI API Key",type="password",)


def conv_ytvid_into_docs(url):
    
    loader = YoutubeLoader.from_youtube_url(url,add_video_info = True)

    
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        )

    docs = text_splitter.split_documents(transcript)
    
    if not openai_api_key.startswith('sk-'):
        raise AuthenticationError(
            "Enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys."
        )
    else:
        embeddings = OpenAIEmbeddings(
            openai_api_key = openai_api_key
        )

        db = FAISS.from_documents(docs,embeddings)

        return db

def ask_questions_on_vid(db,query):
    

    docs = db.similarity_search(query)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name = "gpt-3.5-turbo",temperature=0.0,openai_api_key= openai_api_key)

    template = """
            You are a helpful assistant that can answer questions about youtube videos
         based on the video's transcript: {docs}
        
            Only use the factual information from the transcript to answer the question
        
            If you feel like you do not have enough information just say, "I do not know the answer to this"
        
            Your answers should be verbose and detailed.
            
            Use proper Grammar Punctuation and allign your response into mulitple paragraphs if needed.
            
            Try to use less than 10 sentences to complete your response.
            
            Unless asked for do not give your response in points.
        
            """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Answer the following question :{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt,human_message_prompt]
    )


    chain = LLMChain(llm=chat,prompt=chat_prompt)
    
    response = chain.run(question = query,docs = docs_page_content)
    response = response.replace("\n","")
    response = proper_punctuation(response)
    return response

def proper_punctuation(response):
    desired_punctuation = "."
    lines = response.split("\n")
    formatted_lines = []
    
    for line in lines:
        if line.strip() and line[-1] not in string.punctuation:
            line += desired_punctuation
        
        formatted_lines.append(line)
        
    formatted_response = "\n".join(formatted_lines)
    
    return formatted_response

def get_keywords(response):
    doc = nlp(response)
    # Extract keywords (nouns and proper nouns)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]

    # Calculate keyword frequencies
    keyword_freq = Counter(keywords)

    # Get the top 5 keywords
    top_keywords = [keyword for keyword, _ in keyword_freq.most_common(5)]
    return top_keywords

def get_video_suggestions(top_keywords):
    tool = YouTubeSearchTool()
    for keyword in top_keywords:
        searches = tool.run(keyword)
        # Remove the surrounding square brackets and split the string by commas
        url_strings = searches[1:-1].split(', ')      
        # Remove single quotes around each URL and add 'youtube.com'
        modified_urls = ['youtube.com' + url.strip("'") for url in url_strings]
        # Print the modified URLs
        for modified_url in modified_urls:
            st.write("For something similar on : \n " + keyword + " you can watch " + modified_url)
                      

if url:
    db = conv_ytvid_into_docs(url)
    query = st.text_input("Enter your question about the video here")  
    if query:
        response = ask_questions_on_vid(db,query)
        st.write(response)
        st.subheader("Do you want to generate similar videos?")
        if st.button("Generate"):
            words = get_keywords(response)
            get_video_suggestions(words)
        
        
            
    
        
    
    




