import os
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

# -------------------- PDF TEXT EXTRACTION --------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


# -------------------- TEXT CHUNKING --------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)


# -------------------- VECTOR STORE CREATION --------------------
def create_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# -------------------- RAG CHAIN --------------------
def get_rag_chain():
    # ✅ New prompt for stakeholder-based voting (CSE only for now)
    template = """
    You are acting as a reviewer from the Computer Science (CSE) Department evaluating a course proposal.

    Department Biases (CSE):
    - Prefers strong technical foundations, coding assignments, and advanced software integration.
    - Dislikes vague theoretical proposals with minimal implementation details or no programming aspects.

    Based on the context provided from the proposal:
    {context}

    Task:
    - Evaluate whether the proposal aligns well with the Computer Science department's expectations.
    - Decide whether to VOTE "YES" (if the proposal fits the department's interests) or "NO" (if it doesn’t).
    - Give a short explanation (2–4 lines) for your decision.

    Output format (strictly follow this):
    Department: Computer Science
    Vote: [YES or NO]
    Reason: [your reason based on context and biases]

    Question: {question}
    Answer:
    """

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt | llm


# -------------------- QUESTION HANDLER --------------------
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
    As a Computer Science department reviewer, analyze the proposal and vote YES or NO for its acceptance based on department biases.
    """
    answer = ask_question(query)
    print("\n-------- CSE Department Voting Analysis --------\n")
    print(answer)
