from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from embed.query import vectorstore

# Load LLM from Hugging Face Hub
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5}
)

# Build RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa_chain.invoke(query)
        print("\nAnswer:", answer)
