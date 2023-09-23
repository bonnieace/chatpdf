import os
from PyPDF2 import PdfReader
from langchain import FAISS
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.llms import OpenAI
import pickle
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:

    st.title('LLM Chat App')
    st.markdown('''
    ## About 
    This app is an LLM powered chatbot built using:
    - [streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models/) LLM model

    ''')
    add_vertical_space(5)
    st.write('Made by [Boniface Masota and  Godfrey Ngigi]() ')

def main():
    st.header("Chat with Pdf")
    load_dotenv()
    pdf=st.file_uploader("upload your pdf",type='pdf')
    
    if pdf is not None:

        pdf_reader=PdfReader(pdf)
        #st.write(pdf_reader)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        text_splitter=RecursiveCharacterTextSplitter( 
            chunk_size=10,
            chunk_overlap=2,
            length_function=len
        )
        chunks=text_splitter.split_text(text=text)
        #st.write(chunks)


        store_name=pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore=pickle.load(f)
            st.write('Embeddings loaded from disk')
        else:
            #embeddings object
            embeddings=OpenAIEmbeddings()
            VectorStore=FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:


                pickle.dump(VectorStore,f)
            #st.write("success")
        #accept user questions
        query=st.text_input("Ask questions about your pdf")
        st.write(query)
        if query:

            docs=VectorStore.similarity_search(query=query,k=3)
            llm=OpenAI(model_name='gpt-3.5-turbo')

            chain=load_qa_chain(llm=llm,chain_type="stuff")

            with get_openai_callback() as cb:
                
                response=chain.run(input_documents=docs,question=query)
                print(cb)
            st.write(response)

            #st.write(docs)

        #st.write(text)

if __name__ =='__main__':
    main()