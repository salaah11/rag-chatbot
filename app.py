
import streamlit as st
from rag_engine import init_clients, ask

st.set_page_config(
    page_title="Assistant Cours",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource(show_spinner="⏳ Chargement du modèle...")
def load_clients():
    return init_clients(
        qdrant_url=st.secrets["QDRANT_URL"],
        qdrant_api_key=st.secrets["QDRANT_API_KEY"],
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

embed_model, qdrant, groq_client = load_clients()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎓 Assistant de Cours")

if not st.session_state.messages:
    st.info("👋 Posez une question sur vos cours !")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("🔍 Recherche en cours..."):
        groq_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        result = ask(
            question=prompt,
            history=groq_history,
            embed_model=embed_model,
            qdrant=qdrant,
            groq_client=groq_client
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })

    with st.chat_message("assistant"):
        st.write(result["answer"])

    if result["sources"]:
        st.markdown("**📚 Sources :**")
        for src in result["sources"]:
            st.markdown(f"- 📄 {src['source']} — page {src['page']} (score: {src['score']})")
