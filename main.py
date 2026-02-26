print("Script started")

###
# 1. Sample Documents (raw strings)
###

raw_documents = [
    "My name is Fanuel. I am a second-year Information Science student and winner of the Daydream Hackathon 2025. "
    "I am passionate about technology, Artificial Intelligence, and web development. I enjoy learning new technologies "
    "and building software solutions that solve real-world problems.",
    
    "My projects include: I have done a project called the Ride Sharing platform that focuses and aims to reduce cost "
    "and traffic congestion, especially in Addis Ababa, and has the best security feature at the INSA summer camp. "
    "I have also done a project called Amharic Wordle that incorporates gaming and knowledge, enabling students to learn "
    "the Amharic language while enjoying their favorite game mood.",
    
    "My experiences include working with React, HTML, CSS, and JavaScript. I have also been outlining a project called "
    "Smart Farming for food security. I have a significant project on climate justice, which was recognized as one of "
    "the best by Sam Vagahar, director of the Millennium Fellowship, and the United Nations.",
    
    "My main achievement is that I won the Daydream International Hackathon in 2025.",
    
    "My main goal is to make an impactful project that integrates the concept of software engineering with climate justice, "
    "to implement smart farming for food security and solve real-world problems."
]

docs = None

###
# 1. Sample Documents
###

documents = [
    "My name is Fanuel. I am a second-year Information Science student and winner of the Daydream Hackathon 2025. "
    "I am passionate about technology, Artificial Intelligence, and web development. I enjoy learning new technologies "
    "and building software solutions that solve real-world problems.",
    
    "My projects include: I have done a project called the Ride Sharing platform that focuses and aims to reduce cost "
    "and traffic congestion, especially in Addis Ababa, and has the best security feature at the INSA summer camp. "
    "I have also done a project called Amharic Wordle that incorporates gaming and knowledge, enabling students to learn "
    "the Amharic language while enjoying their favorite game mood.",
    
    "My experiences include working with React, HTML, CSS, and JavaScript. I have also been outlining a project called "
    "Smart Farming for food security. I have a significant project on climate justice, which was recognized as one of "
    "the best by Sam Vagahar, director of the Millennium Fellowship, and the United Nations.",
    
    "My main achievement is that I won the Daydream International Hackathon in 2025.",
    
    "My main goal is to make an impactful project that integrates the concept of software engineering with climate justice, "
    "to implement smart farming for food security and solve real-world problems."
]

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document

    docs = [Document(page_content=text) for text in raw_documents]

    # 2. Load Embedding Model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Create Vector Store
    vectorStore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory='./docs_db'
    )

    print("Vector database created successfully.")

except Exception as e:
    print("Falling back to simple search (embedding import failed):", e)

    class Document:
        def __init__(self, page_content):
            self.page_content = page_content

    docs = [Document(page_content=text) for text in raw_documents]

    class SimpleVectorStore:
        def __init__(self, documents):
            self.documents = documents

        def similarity_search(self, query, k=2):
            q_tokens = set(query.lower().split())
            scored = []
            for d in self.documents:
                text = d.page_content.lower()
                score = sum(1 for t in q_tokens if t in text)
                scored.append((score, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [d for s, d in scored[:k]]

    vectorStore = SimpleVectorStore(docs)

###
# 4. Similarity Search Function
###

def search_query(query, top_k=2):
    print(f"\nQuery: {query}")
    print(f"Top_k: {top_k}\n")

    results = vectorStore.similarity_search(query, k=top_k)

    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(result.page_content)
        print("-" * 50)

###
# 5. Search Results
###

search_query("Who is Fanuel?")
search_query("What projects have I done?")
search_query("What are my experiences?")
search_query("What is my main achievement?")
search_query("What is my goal?")