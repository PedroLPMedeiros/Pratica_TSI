# Sistema de Recuperação de Informações: Genética Craniofacial (RAG)

Este projeto implementa um sistema de Recuperação de Informação (RAG - Retrieval-Augmented Generation) utilizando **LangChain**, **FAISS** e **Ollama**. O sistema processa literatura científica complexa para responder a perguntas sobre malformações craniofaciais e fatores genéticos de forma precisa e eficiente.

A base de conhecimento atual é alimentada pelo artigo *"Leveraging the genetics of human face shape boosts the discovery of orofacial cleft risk loci"*, que explora a arquitetura genética compartilhada entre a morfologia facial e as fissuras orofaciais não sindrômicas (nsCL/P).

## 🛠️ Tecnologias Utilizadas
* **Python 3.12+**
* **LangChain** (Orquestração do pipeline de IA)
* **FAISS** (Banco de dados vetorial local)
* **Ollama** (Inferência local de LLMs e Embeddings)

## ⚙️ Instalação e Configuração

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/PedroLPMedeiros/Pratica_TSI.git
   cd seu-repositorio

2. **Inicie o ambiente virtual** 
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate

3. **Instale as dependências**

    pip install langchain langchain-community langchain-classic langchain-ollama pypdf faiss-cpu

4. **Prepare os modelos do LLMA**

    ollama pull all-minilm
    ollama pull qwen2.5:0.5b