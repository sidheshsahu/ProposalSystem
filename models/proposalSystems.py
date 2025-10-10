import os
import asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# -------------------- SETUP --------------------
load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)


def create_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_rag_chain():
    # ---------- ✅ UPDATED PROMPT ----------
    template = """
    You are an academic evaluation assistant analyzing a course proposal.
    The proposal will be reviewed by four departments, each having specific biases.

    Department Biases:
    1. Computer Science (CSE) Biases:
       - Prefers strong technical foundations, coding assignments, and advanced software integration.
       - Dislikes vague theoretical proposals with minimal implementation details.

    2. Information Technology (IT) Biases:
       - Values practical labs, industry tools, and real-world project exposure.
       - Dislikes proposals without clear student applicability or technology deployment roadmap.

    3. Artificial Intelligence & Data Science (AI–DS) Biases:
       - Favors courses with machine learning, data-driven applications, or modern AI integration.
       - Dislikes courses lacking innovation or measurable analytical components.

    4. Electronics & Telecommunication (EXTC) Biases:
       - Appreciates IoT, hardware linkage, communication systems integration, and interdisciplinary scope.
       - Dislikes proposals purely software-based with no hardware or circuit-level applications.

    Context from Proposal:
    {context}

    Task:
    - Analyze how well this proposal aligns with each department’s interests and biases but dont display in answers
    - Compute an overall acceptance chance (average of all four).
    - Finally, provide 4 overall recommendations to improve overall approval rate.
    -Display  overall acceptance chance (average of all four) and provide 4 overall recommendations to improve overall approval rate.

    Return output in a clean readable structured format like this:
    
    Overall Acceptance Chance: X%
    Overall Recommendations:
    - ...
    - ...
    - ...
    ---

    Question: {question}
    Answer:
    """
    # Using Groq’s Llama 3 model
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt | llm


def ask_question(question, k=3):
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(question, k=k)
    context = " ".join([doc.page_content for doc in docs])

    chain = get_rag_chain()
    answer = chain.invoke({"context": context, "question": question}).content
    return answer


# -------------------- RUN EXAMPLE --------------------
if __name__ == "__main__":
    pdf_docs = [r"C:\Users\hp\Desktop\RAG\RAG\Blockchain_Course_Proposal.pdf"]

    if not os.path.exists("faiss_index"):
        text = get_pdf_text(pdf_docs)
        chunks = get_text_chunks(text)
        create_vector_store(chunks)
        print("Vector store created!")

    query = """
    Considering the proposal details, analyze the chances of acceptance and suggest recommendations to improve the acceptance rate.
    """
    answer = ask_question(query)
    print("\n-------- Proposal Acceptance Analysis --------\n")
    print(answer)
