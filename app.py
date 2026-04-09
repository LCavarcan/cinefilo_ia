import os
import re
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Carrega arquivo .env
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração de página (deve ser o primeiro comando Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🎬 Cinéfilo IA",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Estilo visual
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fundo escuro com grain sutil */
.stApp {
    background-color: #0d0d0d;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
}

/* Header */
.cinema-header {
    text-align: center;
    padding: 2.5rem 0 1rem;
}
.cinema-header h1 {
    # font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #f5c518;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.cinema-header p {
    color: #888;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Divider */
.gold-divider {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #f5c518 30%, #f5c518 70%, transparent);
    margin: 0.5rem auto 2rem;
    width: 60%;
}

/* Mensagens do chat */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.8rem 0;
}
.msg-agent {
    display: flex;
    justify-content: flex-start;
    margin: 0.8rem 0;
}
.bubble-user {
    background: #f5c518;
    color: #0d0d0d;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-weight: 500;
    font-size: 0.95rem;
    line-height: 1.5;
}
.bubble-agent {
    background: #1a1a1a;
    color: #e8e8e8;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 80%;
    font-size: 0.95rem;
    line-height: 1.6;
    border: 1px solid #2a2a2a;
}
.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.avatar-agent {
    background: #1f1f1f;
    border: 1px solid #f5c518;
    margin-right: 0.6rem;
}
.avatar-user {
    background: #f5c51822;
    margin-left: 0.6rem;
}

/* Área de mensagens */
.chat-area {
    min-height: 300px;
    padding: 0.5rem 0;
}

/* Input customizado */
.stTextInput > div > div > input {
    background-color: #1a1a1a !important;
    color: #f0f0f0 !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
    padding: 0.7rem 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #f5c518 !important;
    box-shadow: 0 0 0 2px #f5c51822 !important;
}

/* Botões */
.stButton > button {
    background: #f5c518 !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111 !important;
    border-right: 1px solid #222;
}
section[data-testid="stSidebar"] * {
    color: #ccc !important;
}

/* File uploader */
.stFileUploader {
    background: #1a1a1a;
    border: 1px dashed #333;
    border-radius: 12px;
    padding: 0.5rem;
}

/* Métricas */
[data-testid="metric-container"] {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="metric-container"] label {
    color: #888 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f5c518 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.6rem !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MODEL_NAME = "openai/gpt-oss-120b"
TOP_K = 10
RECENCY_HALF_LIFE_DAYS = 365

SYSTEM_PROMPT = """Você é um cinéfilo apaixonado e bem-humorado que adora conversar sobre filmes.
Você tem opiniões fortes, faz conexões inesperadas entre filmes e não tem medo de divagar.

Como você se comporta:
- Fala de forma casual, como um amigo que entende muito de cinema — sem formalidade.
- Faz conexões entre filmes: temas em comum, diretores, atores, épocas, sentimentos que evocam.
- Comenta sobre as notas com personalidade — elogia escolhas corajosas, provoca gentilmente sobre notas baixas.
- Divaga quando faz sentido: conta curiosidades, contexto histórico, referências culturais.
- Usa humor leve quando apropriado, mas sem forçar.
- Dá respostas generosas e detalhadas — não responde com uma linha só.
- Não inventa filmes ou avaliações que não estejam no histórico fornecido.
- IMPORTANTE: sempre termine sua resposta por completo — nunca corte no meio de uma frase ou lista."""

# ---------------------------------------------------------------------------
# Utilitários de dados
# ---------------------------------------------------------------------------

def gerar_resumo_perfil(df: pd.DataFrame) -> str:
    df = df.copy()

    media = df["Rating"].mean()
    qtd = len(df)

    top = df.nlargest(15, "Rating")[["Name", "Rating"]]
    bottom = df.nsmallest(10, "Rating")[["Name", "Rating"]]

    # gêneros (se existir)
    generos = ""
    if "Genre" in df.columns:
        generos_count = (
            df["Genre"]
            .dropna()
            .str.split(", ")
            .explode()
            .value_counts()
            .head(5)
        )
        generos = "\n".join([f"- {g}: {c}" for g, c in generos_count.items()])

    resumo = f"""
Resumo do usuário:

- Total de filmes avaliados: {qtd}
- Média das notas: {media:.2f}

Top filmes:
{top.to_string(index=False)}

Filmes menos gostados:
{bottom.to_string(index=False)}
"""

    if generos:
        resumo += f"\nGêneros mais assistidos:\n{generos}"

    return resumo

def calcular_peso_recencia(data_series: pd.Series) -> pd.Series:
    dias = (datetime.now() - pd.to_datetime(data_series, errors="coerce")).dt.days
    dias = dias.fillna(RECENCY_HALF_LIFE_DAYS * 2)
    return 1 / (1 + dias / RECENCY_HALF_LIFE_DAYS)


def construir_texto_filme(row: pd.Series) -> str:
    partes = [f"Filme: {row['Name']}", f"Nota: {row['Rating']}"]
    if "Date" in row and pd.notna(row.get("Date")):
        partes.append(f"Data: {row['Date']}")
    if "Genre" in row and pd.notna(row.get("Genre")):
        partes.append(f"Gênero: {row['Genre']}")
    return "\n".join(partes)


def carregar_dados(source) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(source)
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}")
        return None

    colunas_obrigatorias = {"Name", "Rating"}
    faltando = colunas_obrigatorias - set(df.columns)
    if faltando:
        st.error(f"Colunas ausentes no CSV: {faltando}")
        return None

    df = df.dropna(subset=["Name", "Rating"]).copy()
    df["texto"] = df.apply(construir_texto_filme, axis=1)

    if "Date" in df.columns:
        df["peso_recencia"] = calcular_peso_recencia(df["Date"])
    else:
        df["peso_recencia"] = 1.0

    return df

def filmes_por_mes(df: pd.DataFrame) -> str:
    if "Date" not in df.columns:
        return "Não encontrei a coluna de datas."

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.dropna(subset=["Date"])

    df["mes"] = df["Date"].dt.to_period("M")

    contagem = df.groupby("mes").size().sort_values(ascending=False)

    if contagem.empty:
        return "Não há dados suficientes para calcular."

    melhor_mes = contagem.index[0]
    qtd = contagem.iloc[0]

    resumo = f"Mês com mais filmes: {melhor_mes} ({qtd} filmes)\n\n"

    resumo += "Filmes por mês:\n"
    for mes, count in contagem.items():
        resumo += f"- {mes}: {count}\n"

    return resumo

# ---------------------------------------------------------------------------
# Buscador de contexto
# ---------------------------------------------------------------------------

class BuscadorContexto:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.X = self.vectorizer.fit_transform(df["texto"])
        self.pesos = df["peso_recencia"].values

    @staticmethod
    def _e_pergunta_sobre_favoritos(pergunta: str) -> bool:
        palavras = r"favor|melhor|mais gost|top|maior nota|bem avalia|recomend"
        return bool(re.search(palavras, pergunta.lower()))
    
    @staticmethod
    def _e_pergunta_de_perfil(pergunta: str) -> bool:
        palavras = r"perfil|gosto|me descreva|meu estilo|quem sou eu|personalidade"
        return bool(re.search(palavras, pergunta.lower()))

    @staticmethod
    def _e_pergunta_temporal(pergunta: str) -> bool:
        palavras = r"(qual.*m[eê]s.*(mais|menos))|(em que m[eê]s)|(quantos filmes.*m[eê]s)"
        return bool(re.search(palavras, pergunta.lower()))

    def buscar(self, pergunta: str, k: int = TOP_K) -> str:
        # 🧠 NOVO: perfil geral → usa resumo
        if self._e_pergunta_de_perfil(pergunta):
            return gerar_resumo_perfil(self.df)

        # já existente
        if self._e_pergunta_sobre_favoritos(pergunta):
            trechos = self.df.nlargest(k, "Rating")["texto"].tolist()
            return "\n\n".join(trechos)

        # RAG normal
        q_vec = self.vectorizer.transform([pergunta])
        scores = cosine_similarity(q_vec, self.X).flatten()
        scores_ponderados = scores * self.pesos
        top_k_idx = np.argsort(scores_ponderados)[-k:][::-1]

        return "\n\n".join(self.df.iloc[top_k_idx]["texto"].tolist())

# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------

def responder(pergunta: str, buscador: BuscadorContexto, client: Groq) -> str:
    # 🔥 prioridade 1: perfil
    if BuscadorContexto._e_pergunta_de_perfil(pergunta):
        contexto = gerar_resumo_perfil(buscador.df)

        mensagem = (
            f"Contexto sobre o usuário:\n{contexto}\n\n"
            f"Pergunta: {pergunta}"
        )

    # 🔥 prioridade 2: temporal (analytics)
    elif BuscadorContexto._e_pergunta_temporal(pergunta):
        return filmes_por_mes(buscador.df)

    else:
        contexto = buscador.buscar(pergunta)

        mensagem = (
            f"Contexto sobre o usuário:\n{contexto}\n\n"
            f"Pergunta: {pergunta}"
        )

    historico = st.session_state.historico + [{"role": "user", "content": mensagem}]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + historico,
            temperature=0.7,
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Erro ao chamar a API: {e}"

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []  # {"role": "user"|"agent", "content": str}

if "historico" not in st.session_state:
    st.session_state.historico = []  # formato openai para o modelo

if "df" not in st.session_state:
    st.session_state.df = None

if "buscador" not in st.session_state:
    st.session_state.buscador = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("❌ GROQ_API_KEY não encontrada no .env")
        st.stop()

    st.markdown("### 📂 Suas avaliações")

    csv_file = st.file_uploader("Envie seu ratings.csv", type=["csv"])

    if csv_file:
        df = carregar_dados(csv_file)
        if df is not None and (st.session_state.df is None or len(df) != len(st.session_state.df)):
            st.session_state.df = df
            st.session_state.buscador = BuscadorContexto(df)
            st.success(f"{len(df)} avaliações carregadas!")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.markdown("### 📊 Seu perfil")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Filmes", len(df))
        with col2:
            media = df["Rating"].mean()
            st.metric("Média", f"{media:.1f}")

        top3 = df.nlargest(3, "Rating")[["Name", "Rating"]]
        st.markdown("**⭐ Top 3**")
        for _, row in top3.iterrows():
            st.markdown(f"- {row['Name']} `{row['Rating']}`")

    st.markdown("---")
    if st.button("🗑️ Limpar conversa"):
        st.session_state.mensagens = []
        st.session_state.historico = []
        st.rerun()

# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

st.markdown("""
<div class="cinema-header">
    <h1>🎬 Cinéfilo IA</h1>
    <p>Seu parceiro apaixonado por cinema</p>
</div>
<hr class="gold-divider">
""", unsafe_allow_html=True)

# Área de chat
chat_container = st.container()

with chat_container:
    if not st.session_state.mensagens:
        st.markdown(
            "<div style='text-align:center; color:#444; padding: 3rem 0; font-size:0.9rem;'>"
            "Envie seu CSV e comece a conversar sobre filmes 🍿"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.mensagens:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                    <div class="bubble-user">{msg['content']}</div>
                    <div class="avatar avatar-user">🙂</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-agent">
                    <div class="avatar avatar-agent">🎬</div>
                    <div class="bubble-agent">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 2rem'></div>", unsafe_allow_html=True)

# Input
col_input, col_btn = st.columns([5, 1])

with col_input:
    pergunta = st.text_input(
        "mensagem",
        label_visibility="collapsed",
        placeholder="Pergunte sobre seus filmes...",
        key="input_pergunta",
    )

with col_btn:
    enviar = st.button("Enviar")

# Processar envio
if enviar and pergunta:
    if st.session_state.buscador is None:
        st.warning("Envie seu arquivo ratings.csv primeiro.")
    else:
        client = Groq(api_key=api_key)

        st.session_state.mensagens.append({"role": "user", "content": pergunta})
        st.session_state.historico.append({"role": "user", "content": pergunta})

        with st.spinner("Pensando..."):
            resposta = responder(pergunta, st.session_state.buscador, client)

        st.session_state.mensagens.append({"role": "agent", "content": resposta})
        st.session_state.historico.append({"role": "assistant", "content": resposta})

        st.rerun()