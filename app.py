import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain_classic.chains import RetrievalQA

# 1. Preparação e Carregamento
pdf_path = 'cleft_article.pdf'
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# 2. Divisão do texto (Chunks)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)
texts = text_splitter.split_documents(pages)
print(f"Total de chunks gerados: {len(texts)}")

# 3. Criação do Banco de Dados Vetorial (com processamento em lotes para PCs com pouca RAM)
embeddings_model = OllamaEmbeddings(model="all-minilm")
db = None
tamanho_lote = 15 

print("Gerando embeddings e populando o FAISS em lotes...")
for i in range(0, len(texts), tamanho_lote):
    lote = texts[i : i + tamanho_lote]
    if db is None:
        db = FAISS.from_documents(lote, embeddings_model)
    else:
        db.add_documents(lote)
    print(f"Processado: {min(i + tamanho_lote, len(texts))} / {len(texts)}")

# 4. Implementação do RetrievalQA
# Utilizando um modelo leve ideal para inferência local
llm = OllamaLLM(model="qwen2.5:0.5b") 

# Configurando o retriever para buscar os 3 chunks mais relevantes
retriever = db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever, 
    chain_type="stuff"
)

# 5. Execução de Exemplo
query = "Quais são os três genes candidatos identificados como loci de risco para fissuras orofaciais que foram replicados no estudo?"
response = qa_chain.invoke(query)

print("\n--- Pergunta ---")
print(query)
print("\n--- Resposta do Sistema ---")
print(response["result"])