
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

COLLECTION_NAME = "cours_universitaires"
EMBED_MODEL     = "BAAI/bge-m3"
GROQ_MODEL      = "llama-3.3-70b-versatile"
TOP_K           = 5
MAX_TOKENS      = 1024

def init_clients(qdrant_url, qdrant_api_key, groq_api_key):
    embed_model = SentenceTransformer(EMBED_MODEL)
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    groq_client = Groq(api_key=groq_api_key)
    return embed_model, qdrant, groq_client

def retrieve_chunks(question, embed_model, qdrant, top_k=TOP_K):
    query_vector = embed_model.encode(
        [question], normalize_embeddings=True
    )[0].tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    chunks = []
    for r in results:
        chunks.append({
            "text":   r.payload.get("text", ""),
            "source": r.payload.get("source", "inconnu"),
            "page":   r.payload.get("page", 0),
            "score":  round(r.score, 3)
        })
    return chunks

def build_prompt(question, chunks, history):
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i} — {chunk['source']}, page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_message = {
        "role": "system",
        "content": (
            "Tu es un assistant pédagogique expert. "
            "Tu réponds uniquement en te basant sur les extraits fournis. "
            "Si la réponse ne figure pas dans le contexte, dis-le clairement. "
            "Tu peux répondre en français, anglais ou arabe."
        )
    }
    user_message = {
        "role": "user",
        "content": f"Extraits de cours :\n\n{context}\n\n---\n\nQuestion : {question}"
    }
    return [system_message] + history + [user_message]

def generate_answer(messages, groq_client):
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.3,
        top_p=0.9
    )
    return response.choices[0].message.content

def ask(question, history, embed_model, qdrant, groq_client):
    chunks = retrieve_chunks(question, embed_model, qdrant)
    messages = build_prompt(question, chunks, history)
    answer = generate_answer(messages, groq_client)

    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": chunk["source"],
                "page":   chunk["page"],
                "score":  chunk["score"]
            })

    return {"answer": answer, "sources": sources, "chunks": chunks}
