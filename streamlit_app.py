import streamlit as st
from google import genai
from neo4j import GraphDatabase
import os

st.set_page_config(
    page_title="ARC: Cancer Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fetch API keys if possible
try:
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]
    neo_uri = os.getenv("NEO4J_URI") or st.secrets["NEO4J_URI"]
    neo_auth = (
        os.getenv("NEO4J_USER") or st.secrets["NEO4J_USER"],
        os.getenv("NEO4J_PASSWORD") or st.secrets["NEO4J_PASSWORD"]
    )
except KeyError:
    st.error("Missing Secrets! Check .env or streamlit secrets.")
    st.stop()

@st.cache_resource
def get_clients():
    client = genai.Client(api_key=api_key)
    driver = GraphDatabase.driver(neo_uri, auth=neo_auth)
    return client, driver

client, driver = get_clients()

# Hybrid Search
def run_hybrid_query(query_text):
    """
    Combines vector search for broad context + graph traversal for specific entity relations
    """
    try:
        # Embed query
        emb_resp = client.models.embed_content(
            model="text-embedding-004", 
            contents=query_text
        )
        q_vec = emb_resp.embeddings[0].values
    except Exception as e:
        return f"Embedding Error: {e}", ""

    evidence_dump = []

    with driver.session() as sess:
        # Broad Vector Sweep
        # Grabbing slightly more chunks (5) to ensure we get context
        chunk_res = sess.run("""
            CALL db.index.vector.queryNodes('paper_vectors', 5, $vec)
            YIELD node, score
            RETURN node.text as text, score
        """, vec=q_vec)
        
        evidence_dump.append("=== LITERATURE SNIPPETS ===")
        for rec in chunk_res:
            evidence_dump.append(f"- {rec['text'][:300]}...") # Truncating for token limits

        # Graph Expansion (Anchor -> Hop)
        # Threshold at 0.72 to filter noise
        triple_res = sess.run("""
            CALL db.index.vector.queryNodes('entity_vectors', 5, $vec)
            YIELD node AS start, score
            WHERE score > 0.72
            MATCH (start)-[r]-(end)
            RETURN start.name as s, type(r) as rel, end.name as t, r.evidence as ev
            LIMIT 15
        """, vec=q_vec)

        evidence_dump.append("\n=== KNOWLEDGE GRAPH FACTS ===")
        seen_triples = set()
        
        for row in triple_res:
            triple = f"{row['s']} -> {row['rel']} -> {row['t']}"
            if triple not in seen_triples:
                evidence_dump.append(f"FACT: {triple}")
                # evidence_dump.append(f"   (Src: {row['ev'][:50]}...)")
                seen_triples.add(triple)
        
        # Debugging print to check if graph is actually returning stuff
        print(f"DEBUG: Found {len(seen_triples)} unique graph facts for query: '{query_text}'")

    full_context = "\n".join(evidence_dump)

    # Synthesis
    sys_prompt = f"""
    You are ARC (Analysis for Research Collaboration). 
    Answer the user's question based strictly on the provided evidence.
    
    EVIDENCE:
    {full_context}
    
    QUESTION: {query_text}
    """
    
    # Using the new Flash preview for speed/cost balance
    final = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=sys_prompt
    )
    return final.text, full_context

# UI Layout
st.title("ARC: Cancer Research Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_in = st.chat_input("Query the knowledge base...")
if user_in:
    st.session_state.history.append({"role": "user", "content": user_in})
    with st.chat_message("user"):
        st.markdown(user_in)

    with st.chat_message("assistant"):
        with st.spinner("Traversing graph..."):
            ans, raw_ev = run_hybrid_query(user_in)
            st.markdown(ans)
            
            with st.expander("Inspect Retrieval Evidence"):
                st.code(raw_ev, language="text")
    
    st.session_state.history.append({"role": "assistant", "content": ans})