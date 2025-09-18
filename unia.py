import os
import glob
import pickle
import streamlit as st
import faiss
from threading import Thread

from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


# -------------------
# Configurações iniciais
load_dotenv()
st.set_page_config(page_title="UniA - Assistente Universitário", layout="wide")
st.title("🎓 UniA - Seu Assistente Universitário")

INDEX_PATH = "faiss/faissIndex"
STORE_PATH = "faiss/faiss_store.pkl"
DOCS_PATH = "contexDocs/*.pdf"


HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("❌ Token do Hugging Face não encontrado no .env")
    st.stop()


# Estado inicial
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Funções utilitárias
def reset_history():
    st.session_state.chat_history = []
    st.sidebar.success("✅ Histórico de conversa resetado.")
    st.rerun()

def reset_all():
    if os.path.exists(INDEX_PATH):
        try:
            os.remove(INDEX_PATH)
        except OSError as e:
            st.error(f"Erro ao remover o índice: {e}")
    if os.path.exists(STORE_PATH):
        try:
            os.remove(STORE_PATH)
        except OSError as e:
            st.error(f"Erro ao remover o store: {e}")
    st.session_state.chat_history = []
    st.sidebar.success("✅ Índice e histórico resetados com sucesso.")
    st.rerun()


# Carregar e processar PDFs (Chunking)
def load_pdfs():
    """
    Carrega os PDFs, extrai o texto e o divide em pedaços (chunks)
    para otimizar a busca por similaridade.
    """
    pdf_files = glob.glob(DOCS_PATH)
    docs = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, 
        chunk_overlap=150,
        length_function=len
    )

    for file in pdf_files:
        try:
            reader = PdfReader(file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    chunks = text_splitter.split_text(text)
                    for chunk in chunks:
                        docs.append((chunk, {"source": os.path.basename(file), "page": i + 1}))
        except Exception as e:
            st.error(f"Erro ao processar o arquivo {file}: {e}")
    return docs

def create_or_load_index():
    embedder = SentenceTransformer("BAAI/bge-m3")

    if os.path.exists(INDEX_PATH) and os.path.exists(STORE_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(STORE_PATH, "rb") as f:
            store = pickle.load(f)
        return index, store, embedder
    
    with st.spinner("Primeira execução: Processando documentos e criando índice... Isso pode levar um momento. ⏳"):
        docs = load_pdfs()
        if not docs:
            st.warning("Nenhum documento encontrado para indexar. Verifique a pasta 'contexDocs'.")
            return None, None, embedder
            
        texts = [d[0] for d in docs]
        metadata = [d[1] for d in docs]

        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        os.makedirs("faiss", exist_ok=True)
        faiss.write_index(index, INDEX_PATH)
        with open(STORE_PATH, "wb") as f:
            pickle.dump({"texts": texts, "metadata": metadata}, f)

    return index, {"texts": texts, "metadata": metadata}, embedder


# Carrega a base de conhecimento
index, store, embedder = create_or_load_index()


# Carregamento do modelo)
@st.cache_resource
def load_llm():
    """
    Carrega o modelo e o tokenizer para uso com streaming.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        device_map="cpu",             # Força o modelo a carregar na CPU
        offload_folder="offload",     # Pasta temporária para offload
        offload_state_dict=True,      # Ativa offload de estados
        token=HF_TOKEN
    )
    return model, tokenizer

model, tokenizer = load_llm()

# -------------------
# Sidebar
st.sidebar.title("⚙️ Configurações UniA")
st.sidebar.info("A UniA usa documentos da pasta 'contexDocs' para responder.")
if st.sidebar.button("🧹 Resetar Histórico"):
    reset_history()
if st.sidebar.button("🔄 Resetar Base de Conhecimento"):
    reset_all()

# -------------------
# Lógica Principal do Chat
# Exibir histórico
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Caixa de entrada
prompt = st.chat_input("💬 Pergunte algo à UniA sobre cursos, TCC ou regulamentos")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Recuperação via FAISS
    query_vec = embedder.encode([prompt], convert_to_numpy=True)
    D, I = index.search(query_vec, k=5)
    retrieved = [store["texts"][i] for i in I[0]]
    retrieved_meta = [store["metadata"][i] for i in I[0]]

    # Construir histórico
    history_text = ""
    for msg in st.session_state.chat_history[:-1]:
        role = "Usuário" if msg["role"] == "user" else "UniA"
        history_text += f"{role}: {msg['content']}\n"

    # Prompt interno
    context = "\n\n---\n\n".join(retrieved)
    system_prompt = (
        "Você é a UniA, uma assistente de IA especialista nos documentos da universidade UTFPR. Sua única função é analisar o contexto fornecido e responder perguntas com base nele."
        "\n\n"
        "## REGRAS OBRIGATÓRIAS:\n"
        "1.  **PENSE ANTES DE RESPONDER:** Primeiro, avalie silenciosamente se a resposta para a 'Pergunta do Usuário' está contida no 'Contexto dos Documentos'.\n"
        "2.  **FONTE EXCLUSIVA:** Sua resposta deve ser baseada **única e exclusivamente** nas informações do 'Contexto dos Documentos'. NÃO use nenhum conhecimento prévio.\n"
        "3.  **RESPOSTA DIRETA:** Se a informação estiver no contexto, sintetize-a de forma clara e objetiva. Não copie e cole trechos longos. Vá direto ao ponto.\n"
        "4.  **INFORMAÇÃO AUSENTE:** Se a resposta não puder ser encontrada de forma clara no contexto, responda **exatamente** com a frase: 'Desculpe, não encontrei essa informação nos documentos disponíveis.' Não tente adivinhar.\n"
        "5.  **CITAÇÃO DE FONTES:** Ao final da sua resposta, SEMPRE que usar uma informação do contexto, cite a fonte e a página no formato: `[Fonte: nome_do_arquivo, Página: X]`.\n"
        "\n"
        "--- INÍCIO DOS DADOS ---\n"
        f"## Histórico da Conversa Recente:\n{history_text}\n\n"
        f"## Contexto dos Documentos:\n{context}\n"
        "--- FIM DOS DADOS ---\n\n"
        f"## Pergunta do Usuário:\n{prompt}\n\n"
        "## Resposta da UniA (seguindo todas as regras):\n"
    )

    # Lógica de geração com streaming
    with st.chat_message("assistant"):
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        inputs = tokenizer([system_prompt], return_tensors="pt").to(model.device)
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024, temperature=0.2)
        
        def generate_and_stream():
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            for new_text in streamer:
                yield new_text

        full_response = st.write_stream(generate_and_stream)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    # Mostrar fontes
    with st.expander("📖 Fontes utilizadas para esta resposta"):
        for meta, snippet in zip(retrieved_meta, retrieved):
            st.markdown(f"**Fonte:** `{meta['source']}` (página {meta['page']})")
            st.caption(snippet[:400] + "...")
            st.markdown("---")