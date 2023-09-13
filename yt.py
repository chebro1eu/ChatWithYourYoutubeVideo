import re
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
from rake_nltk import Rake
from langchain.schema.document import Document
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel
import torch
import numpy as np




nlp = spacy.load("en_core_web_sm")


#os.environ["OPENAI_API_KEY"] = apikey

st.title("Ask questions on your Youtube Video")

url = st.text_input("Plug in your Youtube Video")

openai_api_key = st.sidebar.text_input("Enter your OPENAI API Key",type="password")



def extract_phrases(text):
    text = text.lower()
    filler_words = ["uh", "um", "like", "you know", "so", "actually", "basically", "seriously"]
    for filler in filler_words:
        text = text.replace(filler, "")

    # Tokenization, Removing Stop Words, Non-Alphanumeric Characters, and Lemmatization
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and token.text.isalpha()]

    # Rejoin Tokens
    cleaned_text = ' '.join(lemmatized_tokens)
    
    model_name = "bert-base-uncased"
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Encode Transcript and Get BERT Embeddings
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)[0]

    # Extract Phrases using SpaCy
    doc = nlp(cleaned_text)
    phrases = [chunk.text for chunk in doc.noun_chunks]

    # If you have fewer than 10 phrases, you can stop here
    if len(phrases) <= 10:
        top_phrases = phrases
    else:
        # Use BERT embeddings to rank the phrases
        phrase_embeddings = []
        for phrase in phrases:
            phrase_input = tokenizer(phrase, return_tensors="pt", truncation=True, padding=True, max_length=50)
            with torch.no_grad():
                phrase_output = model(**phrase_input)
            phrase_embedding = phrase_output.last_hidden_state.mean(dim=1)[0]
            phrase_embeddings.append(phrase_embedding.numpy())

        # Cluster the phrase embeddings
        kmeans = KMeans(n_clusters=10)
        kmeans.fit(phrase_embeddings)
        centroids = kmeans.cluster_centers_

        top_phrases = []
        for centroid in centroids:
            distances = [np.linalg.norm(embedding - centroid) for embedding in phrase_embeddings]
            top_phrase_idx = np.argmin(distances)
            top_phrases.append(phrases[top_phrase_idx])
            
        return top_phrases

    
    
    

def get_important_keywords(url):
    loader = YoutubeLoader.from_youtube_url(url,add_video_info = True)

    
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 900,
        chunk_overlap = 50,
        )

    docs = text_splitter.split_documents(transcript)
    text = ""
    for doc in docs:
        text += doc.page_content
        
    
    ##for doc in docs:
        ##keywords = []
        ##keywords = get_keywords(doc)
    keywords = extract_phrases(text)
    
    return keywords
    
    


def conv_ytvid_into_docs(url):
    
    loader = YoutubeLoader.from_youtube_url(url,add_video_info = True)

    
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1300,
        chunk_overlap = 250,
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

    chat = ChatOpenAI(model_name = "gpt-4",temperature=0.0,openai_api_key= openai_api_key)

    template = """
            You are a highly specialized AI developed for the specific task of summarizing YouTube transcripts : {docs} and answering user queries about them.
            When provided with a YouTube transcript, you are to process it as if you've studied the content of the associated video in great depth.
            You are to imagine that you've reviewed the transcript multiple times, taken detailed notes, and understood its content deeply.
            You possess the ability to extract both general themes and specific details from these transcripts.

            For every user question:

            If the query is general, offer a comprehensive yet concise response that captures the key points from the transcript.
            If the query is specific, answer with precision, focusing on the exact details mentioned in the transcript.
            Always tailor your responses based on the content of the YouTube transcript and the nature of the user's question.

            Prioritize accuracy and relevance in your answers. You should strive to answer as if you are an expert who has studied the transcript content thoroughly.
            Remember, the goal is to offer users the experience of interacting with an expert who has just meticulously analyzed the transcript of the video they are inquiring about.
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

def proper_punctuation(text):
    # Rule 1: Add space after punctuation if not already present
    text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)
    
    # Rule 2: Format lists
    # Start list on a new line after colon or introductory text
    text = re.sub(r':\s*(\d+\.)', r':\n\n\1', text)
    # Ensure each list item starts on a new line and has a space after the period
    text = re.sub(r'(\d+)\.\s*', r'\n\1. ', text)
    
    # Rule 3: Left-aligning is default in most environments, so no specific code is needed
    
    # Rule 4: Ensure single space between sentences
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Rule 5: Space after closing quotation
    text = re.sub(r'"\s*([A-Z])', r'" \1', text)
    
    # Add extra newline before each list item
    text = re.sub(r'(\d+\.)', r'\n\1', text)
    
    return text.strip()

###def get_keywords(response):
    doc = nlp(response)
    # Extract keywords (nouns and proper nouns)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]

    # Calculate keyword frequencies
    keyword_freq = Counter(keywords)

    # Get the top 5 keywords
    top_keywords = [keyword for keyword, _ in keyword_freq.most_common(5)]
    return top_keywords
###

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
            words = get_important_keywords(url)
            get_video_suggestions(words)
        
        
            
    
        
    
    




