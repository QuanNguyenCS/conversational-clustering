import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
from sklearn.decomposition import PCA

# Add parent directory to path to ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset.loader import load_dataset, Dataset
from src.embeddings.local_embeddings import SentenceTransformerEmbeddings
from src.pipeline import ConversationalClusteringPipeline
from src.evaluation.metrics import compute_ari, compute_nmi, compute_acc
from src.agent.base_agent import BaseLLMAgent
from src.agent.cloud_agent import OpenAIAgent, GeminiAgent, AnthropicAgent, GitHubModelsAgent
from src.agent.local_agent import OllamaLocalAgent

# Rule agent removed

# Visualizations handled internally by pipeline

# Streamlit Page Config
st.set_page_config(
    page_title="C3 - Conversational Constrained Clustering",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Matplotlib Dark Theme Configuration
plt.style.use('dark_background')

# Premium UI Header and Custom CSS Styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .stApp {
            background-color: #080c14 !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: #e2e8f0 !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            color: #f8fafc !important;
        }
        
        /* Header Card */
        .main-header {
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.12) 0%, rgba(147, 51, 234, 0.12) 100%);
            border: 1px solid rgba(147, 51, 234, 0.2);
            padding: 24px;
            border-radius: 16px;
            margin-bottom: 24px;
            color: white;
            text-align: center;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        }
        .main-header h1 {
            background: linear-gradient(120deg, #a5b4fc, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-top: 0px !important;
            font-weight: 800 !important;
            font-size: 2.5rem !important;
            letter-spacing: -0.02em;
        }
        .main-header p {
            font-size: 1.1rem;
            color: #94a3b8;
            margin-bottom: 0px;
        }
        
        /* Metric Cards */
        .metric-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 14px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            backdrop-filter: blur(8px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .metric-card:hover {
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.3);
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
        }
        .metric-val {
            font-size: 2.2rem;
            font-weight: 800;
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(120deg, #6366f1, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-lbl {
            font-size: 0.8rem;
            color: #94a3b8;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.08em;
            margin-top: 4px;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #0a0f1d !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        [data-testid="stSidebar"] h2 {
            color: #a5b4fc !important;
        }
        
        /* Buttons and controls */
        .stButton>button {
            background-color: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 10px !important;
            padding: 8px 16px !important;
            color: #cbd5e1 !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: rgba(99, 102, 241, 0.1) !important;
            border-color: rgba(99, 102, 241, 0.4) !important;
            color: white !important;
            transform: translateY(-1px);
        }
        
        /* Primary buttons in Sidebar */
        [data-testid="stSidebar"] .stButton>button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25) !important;
        }
        [data-testid="stSidebar"] .stButton>button:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4) !important;
            transform: translateY(-2px) !important;
        }
        
        /* Chat suggestions */
        .suggestion-btn {
            background-color: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 8px !important;
            padding: 8px 14px !important;
            font-size: 0.85rem !important;
            margin-right: 8px !important;
            margin-bottom: 8px !important;
            cursor: pointer !important;
            display: inline-block !important;
            transition: all 0.2s !important;
            color: #cbd5e1 !important;
        }
        .suggestion-btn:hover {
            background-color: rgba(99, 102, 241, 0.1) !important;
            border-color: rgba(99, 102, 241, 0.4) !important;
            color: white !important;
        }
        
        /* Chat layout styling for ChatGPT/Gemini look */
        .stChatMessage {
            background-color: transparent !important;
            padding: 16px !important;
            border-radius: 12px !important;
            margin-bottom: 12px !important;
        }
        
        /* User chat bubble */
        [data-testid="chatAvatarIcon-user"] {
            background-color: #6366f1 !important;
        }
        .stChatMessage[data-testid="stChatMessage-user"] {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(124, 58, 237, 0.1)) !important;
            border: 1px solid rgba(99, 102, 241, 0.2) !important;
            border-radius: 18px 18px 4px 18px !important;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.05) !important;
        }
        
        /* Assistant chat bubble */
        [data-testid="chatAvatarIcon-assistant"] {
            background: linear-gradient(135deg, #8b5cf6, #d946ef) !important;
        }
        .stChatMessage[data-testid="stChatMessage-assistant"] {
            background-color: rgba(15, 23, 42, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 18px 18px 18px 4px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Tab formatting */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px !important;
            background-color: rgba(15, 23, 42, 0.3) !important;
            padding: 6px !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        .stTabs [data-baseweb="tab"] {
            height: 38px !important;
            border-radius: 8px !important;
            padding: 0px 16px !important;
            font-weight: 600 !important;
            color: #94a3b8 !important;
            transition: all 0.2s !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(99, 102, 241, 0.15) !important;
            color: #a5b4fc !important;
        }
    </style>
""", unsafe_allow_html=True)

# Custom header
st.markdown("""
    <div class="main-header">
        <h1>Conversational Constrained Clustering (C3)</h1>
        <p>Direct the clustering of complex, multi-aspect data using natural language feedback.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar configurations
st.sidebar.header("📁 Dataset")

dataset_path = st.sidebar.text_input("Enter dataset path:", value="data/arxiv_transformed_test.json")

st.sidebar.markdown("---")
st.sidebar.header("🤖 C3 Agent Configuration")

# 1. Q&A Agent configuration
with st.sidebar.expander("💬 Q&A Agent Config (Dialogue)", expanded=True):
    qa_agent_type = st.selectbox(
        "QA Agent Provider",
        ["Gemini", "OpenAI", "Anthropic", "GitHub Models", "Ollama (Local)"],
        key="qa_agent_type"
    )
    qa_api_key = ""
    qa_ollama_url = "http://localhost:11434"
    qa_ollama_model = "qwen2.5:7b"
    qa_model_name = ""
    
    if qa_agent_type == "Gemini":
        qa_env_key = os.getenv("GEMINI_API_KEY", "")
        qa_api_key = st.text_input("QA Gemini API Key", value=qa_env_key, type="password", key="qa_gemini_key")
        qa_model_name = st.selectbox("QA Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"], key="qa_gemini_model")
    elif qa_agent_type == "OpenAI":
        qa_env_key = os.getenv("OPENAI_API_KEY", "")
        qa_api_key = st.text_input("QA OpenAI API Key", value=qa_env_key, type="password", key="qa_openai_key")
        qa_model_name = st.selectbox("QA OpenAI Model", ["gpt-4o", "gpt-4o-mini"], key="qa_openai_model")
    elif qa_agent_type == "Anthropic":
        qa_env_key = os.getenv("ANTHROPIC_API_KEY", "")
        qa_api_key = st.text_input("QA Anthropic API Key", value=qa_env_key, type="password", key="qa_anthropic_key")
        qa_model_name = st.selectbox("QA Anthropic Model", ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"], key="qa_anthropic_model")
    elif qa_agent_type == "GitHub Models":
        qa_env_key = os.getenv("GITHUB_TOKEN", "")
        qa_api_key = st.text_input("QA GitHub Token", value=qa_env_key, type="password", key="qa_github_key")
        qa_model_name = st.selectbox("QA GitHub Model", ["gpt-4o-mini", "gpt-4o", "Phi-3-medium-128k-instruct"], key="qa_github_model")
    elif qa_agent_type == "Ollama (Local)":
        qa_ollama_url = st.text_input("QA Ollama Base URL", value="http://localhost:11434", key="qa_ollama_url")
        qa_ollama_model = st.text_input("QA Ollama Model Name", value="qwen2.5:7b", key="qa_ollama_model")

# 2. Clustering Agent configuration
with st.sidebar.expander("🧩 Clustering Agent Config (Taxonomy)", expanded=True):
    clustering_agent_type = st.selectbox(
        "Clustering Agent Provider",
        ["Gemini", "OpenAI", "Anthropic", "GitHub Models", "Ollama (Local)"],
        key="clustering_agent_type"
    )
    clustering_api_key = ""
    clustering_ollama_url = "http://localhost:11434"
    clustering_ollama_model = "qwen2.5:7b"
    clustering_model_name = ""
    
    if clustering_agent_type == "Gemini":
        clustering_env_key = os.getenv("GEMINI_API_KEY", "")
        clustering_api_key = st.text_input("Clustering Gemini API Key", value=clustering_env_key, type="password", key="clustering_gemini_key")
        clustering_model_name = st.selectbox("Clustering Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"], key="clustering_gemini_model")
    elif clustering_agent_type == "OpenAI":
        clustering_env_key = os.getenv("OPENAI_API_KEY", "")
        clustering_api_key = st.text_input("Clustering OpenAI API Key", value=clustering_env_key, type="password", key="clustering_openai_key")
        clustering_model_name = st.selectbox("Clustering OpenAI Model", ["gpt-4o", "gpt-4o-mini"], key="clustering_openai_model")
    elif clustering_agent_type == "Anthropic":
        clustering_env_key = os.getenv("ANTHROPIC_API_KEY", "")
        clustering_api_key = st.text_input("Clustering Anthropic API Key", value=clustering_env_key, type="password", key="clustering_anthropic_key")
        clustering_model_name = st.selectbox("Clustering Anthropic Model", ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"], key="clustering_anthropic_model")
    elif clustering_agent_type == "GitHub Models":
        clustering_env_key = os.getenv("GITHUB_TOKEN", "")
        clustering_api_key = st.text_input("Clustering GitHub Token", value=clustering_env_key, type="password", key="clustering_github_key")
        clustering_model_name = st.selectbox("Clustering GitHub Model", ["gpt-4o-mini", "gpt-4o", "Phi-3-medium-128k-instruct"], key="clustering_github_model")
    elif clustering_agent_type == "Ollama (Local)":
        clustering_ollama_url = st.text_input("Clustering Ollama Base URL", value="http://localhost:11434", key="clustering_ollama_url")
        clustering_ollama_model = st.text_input("Clustering Ollama Model Name", value="qwen2.5:7b", key="clustering_ollama_model")

with st.sidebar.expander("⚙️ Pipeline & Clustering Parameters", expanded=False):
    param_num_samples = st.slider(
        "Representative Samples (FPS)",
        min_value=10,
        max_value=300,
        value=100,
        step=10,
        help="Number of representative documents to select using Farthest Point Sampling."
    )
    param_batch_size = st.slider(
        "Discovery Batch Size",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Number of documents to send in a single mini-batch to the LLM agent."
    )

st.sidebar.markdown("---")
# Action button
init_pipeline = st.sidebar.button("🚀 Initialize / Reset Pipeline", use_container_width=True)

# Helper functions to check/retrieve selected agents
def get_qa_agent():
    if qa_agent_type == "Gemini":
        if not qa_api_key:
            return None
        return GeminiAgent(api_key=qa_api_key, model_name=qa_model_name, verbose=True)
    elif qa_agent_type == "OpenAI":
        if not qa_api_key:
            return None
        return OpenAIAgent(api_key=qa_api_key, model_name=qa_model_name, verbose=True)
    elif qa_agent_type == "Anthropic":
        if not qa_api_key:
            return None
        return AnthropicAgent(api_key=qa_api_key, model_name=qa_model_name, verbose=True)
    elif qa_agent_type == "GitHub Models":
        if not qa_api_key:
            return None
        return GitHubModelsAgent(api_key=qa_api_key, model_name=qa_model_name, verbose=True)
    elif qa_agent_type == "Ollama (Local)":
        return OllamaLocalAgent(base_url=qa_ollama_url, model_name=qa_ollama_model, verbose=True)
    return None

def get_clustering_agent():
    if clustering_agent_type == "Gemini":
        if not clustering_api_key:
            return None
        return GeminiAgent(api_key=clustering_api_key, model_name=clustering_model_name, verbose=True)
    elif clustering_agent_type == "OpenAI":
        if not clustering_api_key:
            return None
        return OpenAIAgent(api_key=clustering_api_key, model_name=clustering_model_name, verbose=True)
    elif clustering_agent_type == "Anthropic":
        if not clustering_api_key:
            return None
        return AnthropicAgent(api_key=clustering_api_key, model_name=clustering_model_name, verbose=True)
    elif clustering_agent_type == "GitHub Models":
        if not clustering_api_key:
            return None
        return GitHubModelsAgent(api_key=clustering_api_key, model_name=clustering_model_name, verbose=True)
    elif clustering_agent_type == "Ollama (Local)":
        return OllamaLocalAgent(base_url=clustering_ollama_url, model_name=clustering_ollama_model, verbose=True)
    return None

def render_plotly_chart(pipeline, labels=None, show_raw=False):
    import plotly.express as px
    import plotly.graph_objects as go
    
    if pipeline.pca_coords is None:
        st.warning("No coordinates available for visualization.")
        return
        
    # Truncate text for hover display
    hover_texts = []
    for t in pipeline.texts:
        if len(t) > 150:
            hover_texts.append(t[:150] + "...")
        else:
            hover_texts.append(t)
            
    df = pd.DataFrame({
        "x": pipeline.pca_coords[:, 0],
        "y": pipeline.pca_coords[:, 1],
        "doc_id": list(range(len(pipeline.texts))),
        "text": hover_texts
    })
    
    # Identify sampled documents
    sampled_set = set(pipeline.sampled_indices)
    df["is_sampled"] = df["doc_id"].apply(lambda idx: idx in sampled_set)
    df["point_type"] = df["is_sampled"].apply(lambda is_s: "Representative (Sampled)" if is_s else "Document")
    
    if show_raw:
        # Raw embedding plot
        df["group"] = df["point_type"]
        color_map = {
            "Representative (Sampled)": "#6366f1",
            "Document": "#475569"
        }
        fig = px.scatter(
            df, x="x", y="y", 
            color="group",
            color_discrete_map=color_map,
            hover_data={
                "x": False,
                "y": False,
                "doc_id": True,
                "group": True,
                "text": True
            },
            title="Raw Embedding Space (Highlighting 20 Representative Documents)"
        )
    else:
        # Cluster partitions plot
        if labels is None:
            labels = pipeline.labels
            
        if labels is None:
            df["group"] = "Unclustered"
            color_map = {"Unclustered": "#475569"}
        else:
            # Map each label to its keywords/details
            keywords = pipeline.get_cluster_keywords()
            
            # Map each cluster label to its Phase 3 registry key using sampled documents mapping
            registry = pipeline.global_registry
            registry_keys = list(registry.keys())
            
            cluster_mapping = {}
            for l_val in np.unique(labels):
                l_int = int(l_val)
                sampled_in_l = [idx for idx in pipeline.sampled_indices if pipeline.labels[idx] == l_int] if pipeline.labels is not None else []
                phase3_cids = [pipeline.final_assignments.get(idx) for idx in sampled_in_l if pipeline.final_assignments.get(idx) is not None]
                if phase3_cids:
                    from collections import Counter
                    cluster_mapping[l_int] = Counter(phase3_cids).most_common(1)[0][0]
                else:
                    if l_int < len(registry_keys):
                        cluster_mapping[l_int] = registry_keys[l_int]
                    else:
                        cluster_mapping[l_int] = registry_keys[0] if registry_keys else f"Cluster_{l_int + 1}"
            
            # Map sorted unique labels to 1-indexed integers
            unique_labels = sorted(list(np.unique(labels)))
            label_to_index = {l_val: i for i, l_val in enumerate(unique_labels, start=1)}
            
            cluster_names = {}
            for l in np.unique(labels):
                l_int = int(l)
                cid = cluster_mapping.get(l_int, f"Cluster_{l_int}")
                raw_desc = registry.get(cid, "")
                if ":" in raw_desc:
                    c_name = raw_desc.split(":", 1)[0].strip()
                else:
                    c_name = str(cid).replace("_", " ")
                    
                idx_1indexed = label_to_index[l]
                kw_list = keywords.get(l_int, [])
                kw_str = ", ".join(kw_list[:3]) if kw_list else "unlabeled"
                cluster_names[l] = f"Cluster {idx_1indexed}: {c_name} ({kw_str})"
            
            df["cluster_label"] = labels
            df["group"] = df["cluster_label"].apply(lambda l: cluster_names.get(l, f"Cluster {l}"))
            
            # Curated harmonized dark colors
            colors_palette = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#eab308']
            unique_groups = sorted(list(df["group"].unique()))
            color_map = {g: colors_palette[i % len(colors_palette)] for i, g in enumerate(unique_groups)}
            
        fig = px.scatter(
            df, x="x", y="y",
            color="group",
            color_discrete_map=color_map,
            hover_data={
                "x": False,
                "y": False,
                "doc_id": True,
                "point_type": True,
                "text": True
            },
            title="Cluster Partition (2D PCA Map)"
        )
        
    # Premium Dark Theme Styling
    fig.update_layout(
        plot_bgcolor="#0a0e17",
        paper_bgcolor="#080c14",
        font=dict(color="#e2e8f0", family="Plus Jakarta Sans, sans-serif"),
        title=dict(font=dict(size=14, family="Outfit, sans-serif")),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.05)",
            showticklabels=False,
            title=""
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.05)",
            showticklabels=False,
            title=""
        ),
        legend=dict(
            bgcolor="rgba(10, 14, 23, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.05)",
            borderwidth=1,
            font=dict(size=10),
            title=None
        ),
        hoverlabel=dict(
            bgcolor="#0a0e17",
            font_size=11,
            font_family="Plus Jakarta Sans, sans-serif"
        )
    )
    
    # Add high-contrast white circles to highlight the sampled/representative documents
    sampled_coords = df[df["is_sampled"]]
    fig.add_trace(
        go.Scatter(
            x=sampled_coords["x"],
            y=sampled_coords["y"],
            mode="markers",
            marker=dict(
                size=12,
                color="rgba(0,0,0,0)",
                line=dict(width=2, color="#ffffff")
            ),
            hoverinfo="skip",
            showlegend=False,
            name="Sampled Highlights"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Load dataset and cache it
@st.cache_data
def get_cached_dataset(path):
    return load_dataset(path)
if "pipeline" not in st.session_state or init_pipeline:
    with st.spinner("Initializing embeddings and setting up pipeline..."):
        try:
            dataset = get_cached_dataset(dataset_path)
            emb_provider = SentenceTransformerEmbeddings()
            
            qa_agent = get_qa_agent()
            clustering_agent = get_clustering_agent()
            if qa_agent is None or clustering_agent is None:
                st.error("Please configure both Q&A and Clustering agents with valid keys/URLs.")
            else:
                pipeline = ConversationalClusteringPipeline(
                    embedding_provider=emb_provider,
                    qa_agent=qa_agent,
                    clustering_agent=clustering_agent,
                    num_samples=param_num_samples,
                    batch_size=param_batch_size,
                    random_state=42,
                    use_qa_phase=True,
                    use_itml=True
                )
                pipeline.set_data(dataset.get_texts())
                pipeline.start_initial_qa()
                
                # Store in session state
                st.session_state.pipeline = pipeline
                st.session_state.dataset = dataset
                st.session_state.initialized = True
                
                # Clear chat input placeholder state
                if "chat_input_val" in st.session_state:
                    del st.session_state.chat_input_val
        except Exception as e:
            st.error(f"Error during initialization: {e}")
            st.exception(e)

# Render main app if initialized
if st.session_state.get("initialized", False):
    pipeline = st.session_state.pipeline
    dataset = st.session_state.dataset

    # Render main chat container
    st.subheader("💬 C3 Conversational Clustering Chat")
    
    # Render all message history inline
    for idx_msg, msg in enumerate(pipeline.history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Render raw_pca_plot first if present
                if "raw_pca_plot" in msg and msg["raw_pca_plot"]:
                    render_plotly_chart(pipeline, show_raw=True)
                
                # Render sampled_docs first if present
                if "sampled_docs" in msg and msg["sampled_docs"]:
                    st.markdown("### 📄 Representative Document Samples (20)")
                    with st.container(height=250):
                        for doc in msg["sampled_docs"]:
                            st.markdown(f"**Document ID: {doc['id']}**")
                            st.write(doc["text"])
                            st.markdown("---")
                
                # Render colored clustering plot if present
                if ("image" in msg and msg["image"]) or ("labels" in msg and msg["labels"]):
                    render_plotly_chart(pipeline, labels=msg.get("labels"))
                
                # Render main content
                st.write(msg["content"])
                
                # Render cluster details (collapsible partitions and split actions) if present
                if "cluster_details" in msg and msg["cluster_details"]:
                    st.markdown("##### 📂 Discovered Cluster Partitions:")
                    keywords = msg["cluster_details"]["keywords"]
                    central_docs = msg["cluster_details"]["central_docs"]
                    
                    registry = pipeline.global_registry
                    registry_keys = list(registry.keys())
                    
                    cluster_mapping = {}
                    for l in keywords.keys():
                        l_int = int(l)
                        sampled_in_l = [idx for idx in pipeline.sampled_indices if pipeline.labels[idx] == l_int] if pipeline.labels is not None else []
                        phase3_cids = [pipeline.final_assignments.get(idx) for idx in sampled_in_l if pipeline.final_assignments.get(idx) is not None]
                        if phase3_cids:
                            from collections import Counter
                            cluster_mapping[l] = Counter(phase3_cids).most_common(1)[0][0]
                        else:
                            if l_int < len(registry_keys):
                                cluster_mapping[l] = registry_keys[l_int]
                            else:
                                cluster_mapping[l] = registry_keys[0] if registry_keys else f"Cluster_{l_int + 1}"
                                
                    sorted_labels = sorted(list(keywords.keys()), key=int)
                    for idx_1indexed, l in enumerate(sorted_labels, start=1):
                        cid = cluster_mapping.get(l, f"Cluster_{idx_1indexed}")
                        raw_desc = registry.get(cid, "No description available.")
                        if ":" in raw_desc:
                            parts = raw_desc.split(":", 1)
                            c_name = parts[0].strip()
                            c_desc = parts[1].strip()
                        else:
                            c_name = str(cid).replace("_", " ")
                            c_desc = raw_desc.strip()
                            
                        kw_str = ", ".join(keywords.get(l, []))
                        expander_title = f"Cluster {idx_1indexed}: \"{c_name}\" 🏷️ (keyword: {kw_str})"
                        
                        with st.expander(expander_title, expanded=(idx_1indexed == 1)):
                            st.markdown(f"**Description:** {c_desc}")
                            st.markdown("---")
                            st.write("**Top Representative Documents (Medoids):**")
                            for doc in central_docs.get(l, []):
                                st.markdown(f"- *\"{doc}\"*")
                                
                            st.markdown("---")
                            
                            msg_labels = msg.get("labels", pipeline.labels)
                            if msg_labels is not None:
                                # Compare against integer label value (since l could be string or int)
                                l_int = int(l)
                                cluster_indices = [idx for idx in range(len(pipeline.texts)) if int(msg_labels[idx]) == l_int]
                                cluster_texts = [pipeline.texts[idx] for idx in cluster_indices]
                                
                                cluster_df = pd.DataFrame({
                                    "Doc ID": cluster_indices,
                                    "Document Text": cluster_texts
                                })
                                
                                st.write(f"**All Documents in Cluster ({len(cluster_df)} total):**")
                                st.dataframe(
                                    cluster_df,
                                    column_config={
                                        "Doc ID": st.column_config.NumberColumn("Doc ID", width="small"),
                                        "Document Text": st.column_config.TextColumn("Document Text", width="large")
                                    },
                                    hide_index=True,
                                    use_container_width=True,
                                    height=250
                                )
                
                # Render suggestion buttons inline at the bottom of the last message
                if "suggestions" in msg and msg["suggestions"] and idx_msg == len(pipeline.history) - 1:
                    st.write("💡 *Choose a suggestion or type your own response below:*")
                    # Render suggestions vertically as rectangular options
                    for idx_s, sugg in enumerate(msg["suggestions"]):
                        if st.button(sugg, key=f"qa_sugg_btn_{idx_msg}_{idx_s}", use_container_width=True):
                            st.session_state.chat_input_val = sugg
                            st.rerun()
                
                # Render confirmation options if this is the confirmation summary
                if msg.get("type") == "qa_confirm" and idx_msg == len(pipeline.history) - 1:
                    st.write("💡 *Please confirm your clustering requirements:*")
                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        if st.button("🚀 Confirm & Run", key="confirm_and_run_btn", use_container_width=True):
                            with st.spinner("Executing initial clustering..."):
                                pipeline.confirm_and_run()
                            st.rerun()
                    with c_col2:
                        if st.button("✏️ Modify Request", key="modify_request_btn", use_container_width=True):
                            pipeline.modify_qa_request()
                            st.rerun()
            else:
                # User message
                st.write(msg["content"])
                
    # Quick suggestion buttons for clustered states (when Q&A is NOT active or confirming)
    is_confirming = getattr(pipeline, "qa_awaiting_confirm", False)
    if not pipeline.qa_active and not is_confirming:
        st.write("💡 *Quick Feedback Suggestions:*")
        suggestions = [
            ("Topic", "Group these documents by their main topic or subject."),
            ("Category", "Separate them based on categories or sentiment/attitude.")
        ]
        if "amazon" in dataset_path.lower():
            suggestions = [
                ("Sentiment", "Group reviews based on whether they like or dislike the product (Sentiment)."),
                ("Feature", "Separate reviews by product functions like battery, charging vs. sound quality.")
            ]
        elif "arxiv" in dataset_path.lower():
            suggestions = [
                ("Subject Domain", "Partition abstracts by academic subject domains like physics vs. computer science."),
                ("Methodology", "Group papers by methodology, such as deep learning vs. classical statistics.")
            ]
        elif "banking77" in dataset_path.lower():
            if "intent" in dataset_path.lower() or "300new_aspect" in dataset_path.lower():
                suggestions = [
                    ("Action Intent", "Group these banking customer service queries purely by their Action Intent into exactly 3 clusters: Information Inquiry, Complaint & Problem, and Urgent / Action Request."),
                    ("Topic/Category", "Group these banking customer service queries by their specific fine-grained customer intent or topic (e.g., card arrival, card linking, exchange rate, automatic top up)."),
                    ("Temporal Focus", "Group these banking customer service queries purely by their Temporal Focus: past, present, or future.")
                ]
            elif "time" in dataset_path.lower() or "300new_aspect_time" in dataset_path.lower():
                suggestions = [
                    ("Temporal Focus", "Group these banking customer service queries purely by their Temporal Focus into exactly 3 clusters: past, present, and future."),
                    ("Topic/Category", "Group these banking customer service queries by their specific fine-grained customer intent or topic (e.g., card arrival, card linking, exchange rate, automatic top up)."),
                    ("Action Intent", "Group these banking customer service queries purely by their Action Intent: Information Inquiry, Complaint & Problem, or Urgent / Action Request.")
                ]
            else:
                suggestions = [
                    ("Topic/Category", "Group these banking customer service queries by their specific fine-grained customer intent or topic (e.g., card arrival, card linking, exchange rate, automatic top up)."),
                    ("Action Intent", "Group these banking customer service queries purely by their Action Intent: Information Inquiry, Complaint & Problem, or Urgent / Action Request."),
                    ("Temporal Focus", "Group these banking customer service queries purely by their Temporal Focus: past, present, or future.")
                ]
            
        s_col1, s_col2 = st.columns(2)
        for i, (label, suggestion_text) in enumerate(suggestions):
            with s_col1 if i % 2 == 0 else s_col2:
                if st.button(f"👉 Target {label}", key=f"sug_{label}"):
                    st.session_state.chat_input_val = suggestion_text
                    st.rerun()
                    
    # Chat text input (Submit feedback)
    if "chat_input_val" not in st.session_state:
        st.session_state.chat_input_val = ""
        
    if is_confirming:
        user_feedback = st.chat_input("Please click 'Confirm & Run' or 'Modify Request' above.", disabled=True)
    else:
        user_feedback = st.chat_input("Type your instructions or responses here...")
        
    if st.session_state.chat_input_val:
        user_feedback = st.session_state.chat_input_val
        st.session_state.chat_input_val = ""  # Reset
        
    if user_feedback:
        with st.spinner("Processing..."):
            pipeline.step(user_feedback)
        st.rerun()
 
else:
    st.info("Pipeline not initialized. Click 'Initialize / Reset Pipeline' in the sidebar to start!")
