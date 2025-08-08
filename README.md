# Sports Rules Chatbot

This is a Retrieval-Augmented Generation (RAG) based chatbot that answers questions about sports rules using document retrieval and a local LLM.

To run the project:

1. Clone the repository  
git clone https://github.com/lydiamavin/chatbot.git  
cd chatbot

2. Install dependencies  
pip install -r requirements.txt

3. Create a .env file in the root folder with the following content:  
PINECONE_API_KEY=pinecone_api_key

4. Run Streamlit UI locally  
streamlit run app.py
