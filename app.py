import os
import re
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração de página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="🎬 Cinéfilo IA 🎵", layout="centered")

# ---------------------------------------------------------------------------
# Estilo visual
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background-color: #0d0d0d;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
}

.cinema-header { text-align: center; padding: 2.5rem 0 1rem; }
.cinema-header h1 { font-size: 2.8rem; color: #f5c518; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
.cinema-header p { color: #888; font-size: 0.95rem; font-weight: 300; letter-spacing: 0.05em; text-transform: uppercase; }

/* Status bar de fontes */
.source-bar { display:flex; justify-content:center; gap:0.6rem; margin:0.4rem 0 1.2rem; flex-wrap:wrap; }
.source-pill { display:inline-flex; align-items:center; gap:0.3rem; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; letter-spacing:0.03em; }
.source-pill.active-lb  { background:#1a3a1a; color:#4ade80; border:1px solid #4ade8055; }
.source-pill.active-lfm { background:#3a0a0a; color:#f87171; border:1px solid #f8717155; }
.source-pill.inactive   { background:#1a1a1a; color:#444;    border:1px solid #2a2a2a; }

/* Mensagens */
.msg-user  { display:flex; justify-content:flex-end;  margin:0.8rem 0; }
.msg-agent { display:flex; justify-content:flex-start; margin:0.8rem 0; }
.bubble-user  { background:#f5c518; color:#0d0d0d; padding:0.75rem 1.1rem; border-radius:18px 18px 4px 18px; max-width:75%; font-weight:500; font-size:0.95rem; line-height:1.5; }
.bubble-agent { background:#1a1a1a; color:#e8e8e8; padding:0.75rem 1.1rem; border-radius:18px 18px 18px 4px; max-width:80%; font-size:0.95rem; line-height:1.6; border:1px solid #2a2a2a; }
.avatar { width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1rem; flex-shrink:0; }
.avatar-agent { background:#1f1f1f; border:1px solid #f5c518; margin-right:0.6rem; }
.avatar-user  { background:#f5c51822; margin-left:0.6rem; }

/* Input */
.stTextInput > div > div > input { background-color:#1a1a1a !important; color:#f0f0f0 !important; border:1px solid #333 !important; border-radius:12px !important; padding:0.7rem 1rem !important; font-family:'DM Sans',sans-serif !important; font-size:0.95rem !important; }
.stTextInput > div > div > input:focus { border-color:#f5c518 !important; box-shadow:0 0 0 2px #f5c51822 !important; }

/* Botão principal */
.stButton > button { background:#f5c518 !important; color:#0d0d0d !important; border:none !important; border-radius:10px !important; font-family:'DM Sans',sans-serif !important; font-weight:600 !important; padding:0.5rem 1.2rem !important; transition:all 0.2s !important; }
.stButton > button:hover { background:#ffd740 !important; transform:translateY(-1px) !important; box-shadow:0 4px 12px #f5c51840 !important; }

/* Botões sidebar ghost */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] .stButton > button *,
section[data-testid="stSidebar"] .stButton > button p { background:transparent !important; color:#f5c518 !important; border:1px solid #f5c51866 !important; font-weight:500 !important; }
section[data-testid="stSidebar"] .stButton > button:hover,
section[data-testid="stSidebar"] .stButton > button:hover *,
section[data-testid="stSidebar"] .stButton > button:hover p { background:#f5c51815 !important; border-color:#f5c518 !important; color:#ffd740 !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color:#111 !important; border-right:1px solid #1e1e1e; }
section[data-testid="stSidebar"] *:not(button):not(button *) { color:#bbb !important; }
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 { color:#e0e0e0 !important; }
section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] .stMarkdown li { color:#aaa !important; font-size:0.88rem !important; }
section[data-testid="stSidebar"] .stMarkdown strong { color:#ddd !important; }
section[data-testid="stSidebar"] code { background:#222 !important; color:#f5c518 !important; border-radius:4px !important; padding:1px 5px !important; font-size:0.8rem !important; }
section[data-testid="stSidebar"] .stCaption p { color:#555 !important; font-size:0.78rem !important; }

/* File uploader */
.stFileUploader { background:#1a1a1a; border:1px dashed #333; border-radius:12px; padding:0.5rem; }

/* Métricas */
[data-testid="metric-container"] { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:10px; padding:0.8rem 1rem; }
[data-testid="metric-container"] label { color:#888 !important; font-size:0.75rem !important; text-transform:uppercase; letter-spacing:0.05em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#f5c518 !important; font-family:'Playfair Display',serif !important; font-size:1.6rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:#111; border-bottom:1px solid #222; }
.stTabs [data-baseweb="tab"] { color:#666 !important; }
.stTabs [aria-selected="true"] { color:#f5c518 !important; border-bottom:2px solid #f5c518 !important; }

/* Scrollbar */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#111; }
::-webkit-scrollbar-thumb { background:#333; border-radius:4px; }

/* Chat message do Streamlit */
[data-testid="stChatMessage"] { background:#1a1a1a !important; border:1px solid #2a2a2a !important; border-radius:18px 18px 18px 4px !important; }
[data-testid="stChatMessage"] p { color:#e8e8e8 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Logging & Constantes
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME        = "openai/gpt-oss-120b"
TOP_K             = 10
RECENCY_HALF_LIFE = 365
LASTFM_API_URL    = "https://ws.audioscrobbler.com/2.0/"
TMDB_API_URL      = "https://api.themoviedb.org/3"

SUGESTOES_FILMES = [
    "Qual meu filme favorito?",
    "Me recomenda algo parecido com meus tops",
    "Que mês assisti mais filmes?",
    "Me surpreenda com algo fora da minha zona de conforto",
    "Quais filmes eu deveria ter dado nota maior?",
    "Como você descreveria meu gosto cinematográfico?",
]
SUGESTOES_MUSICA = [
    "Qual meu artista favorito ultimamente?",
    "Me fala sobre meu gosto musical",
    "Que álbum eu mais ouvi nos últimos meses?",
    "Que músicas eu curti (loved)?",
]
SUGESTOES_AMBOS = [
    "Que trilhas sonoras combinam com meus filmes favoritos?",
    "Que filmes combinam com os artistas que eu mais ouço?",
    "Me descreva como cinéfilo e melômano",
]

# ---------------------------------------------------------------------------
# TMDB
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def buscar_tmdb(titulo: str, tmdb_key: str) -> dict | None:
    if not tmdb_key:
        return None
    try:
        r = requests.get(f"{TMDB_API_URL}/search/movie",
                         params={"api_key": tmdb_key, "query": titulo, "language": "pt-BR"}, timeout=8)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        det = requests.get(f"{TMDB_API_URL}/movie/{results[0]['id']}",
                           params={"api_key": tmdb_key, "language": "pt-BR", "append_to_response": "credits"}, timeout=8)
        det.raise_for_status()
        return det.json()
    except Exception as e:
        logger.error(f"TMDB: {e}")
        return None


def enriquecer_texto_tmdb(row: pd.Series, tmdb_key: str) -> str:
    info = buscar_tmdb(row["Name"], tmdb_key)
    if not info:
        return row["texto"]
    extras = []
    if info.get("overview"):
        extras.append(f"Sinopse: {info['overview'][:300]}")
    genres = ", ".join(g["name"] for g in info.get("genres", [])[:3])
    if genres:
        extras.append(f"Gêneros: {genres}")
    credits = info.get("credits", {})
    dirs = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
    if dirs:
        extras.append(f"Diretor: {', '.join(dirs[:2])}")
    cast = [c["name"] for c in credits.get("cast", [])[:4]]
    if cast:
        extras.append(f"Elenco: {', '.join(cast)}")
    return row["texto"] + "\n" + "\n".join(extras)

# ---------------------------------------------------------------------------
# Last.fm
# ---------------------------------------------------------------------------

def lastfm_request(params: dict, api_key: str) -> dict | None:
    try:
        r = requests.get(LASTFM_API_URL, params={"api_key": api_key, "format": "json", **params}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            logger.error(f"Last.fm {data['error']}: {data.get('message')}")
            return None
        return data
    except Exception as e:
        logger.error(f"Last.fm: {e}")
        return None


def carregar_dados_lastfm(api_key: str, username: str) -> dict | None:
    info = lastfm_request({"method": "user.getinfo", "user": username}, api_key)
    if not info:
        return None
    res = {"info": info.get("user", {})}

    for method, outer, inner, limit in [
        ("user.gettopartists", "topartists", "artist", 20),
        ("user.gettopalbums",  "topalbums",  "album",  15),
        ("user.gettoptracks",  "toptracks",  "track",  20),
    ]:
        d = lastfm_request({"method": method, "user": username, "period": "6month", "limit": limit}, api_key)
        if d:
            res[inner + "s"] = d.get(outer, {}).get(inner, [])

    recent = lastfm_request({"method": "user.getrecenttracks", "user": username, "limit": 50}, api_key)
    if recent:
        res["recent_tracks"] = recent.get("recenttracks", {}).get("track", [])

    loved = lastfm_request({"method": "user.getlovedtracks", "user": username, "limit": 20}, api_key)
    if loved:
        res["loved_tracks"] = loved.get("lovedtracks", {}).get("track", [])

    return res


def construir_texto_musica(d: dict) -> list[dict]:
    reg = []
    for a in d.get("artists", []):
        reg.append({"Name": a.get("name",""), "Rating": None, "peso_recencia": 1.0, "tipo": "artista",
                    "texto": f"Artista: {a.get('name','')}\nScrobbles (6m): {a.get('playcount','?')}"})
    for al in d.get("albums", []):
        art = al.get("artist", {}).get("name", "?")
        reg.append({"Name": al.get("name",""), "Rating": None, "peso_recencia": 1.0, "tipo": "album",
                    "texto": f"Álbum: {al.get('name','')}\nArtista: {art}\nScrobbles (6m): {al.get('playcount','?')}"})
    for t in d.get("tracks", []):
        art = t.get("artist", {}).get("name", "?")
        reg.append({"Name": t.get("name",""), "Rating": None, "peso_recencia": 1.0, "tipo": "musica",
                    "texto": f"Música: {t.get('name','')}\nArtista: {art}\nScrobbles (6m): {t.get('playcount','?')}"})
    for lt in d.get("loved_tracks", []):
        art = lt.get("artist", {}).get("name", "?")
        reg.append({"Name": lt.get("name",""), "Rating": None, "peso_recencia": 1.0, "tipo": "loved",
                    "texto": f"Música curtida: {lt.get('name','')}\nArtista: {art}"})
    return reg


def gerar_resumo_lastfm(d: dict) -> str:
    info  = d.get("info", {})
    nome  = info.get("name", "?")
    total = info.get("playcount", "?")
    pais  = info.get("country", "")

    def fmt(items, fn): return "\n".join(fn(i) for i in items) or "Sem dados"

    artistas = fmt(d.get("artists",[])[:10], lambda a: f"- {a['name']} ({a.get('playcount','?')} scrobbles)")
    albuns   = fmt(d.get("albums", [])[:5],  lambda al: f"- {al['name']} — {al.get('artist',{}).get('name','?')} ({al.get('playcount','?')} scrobbles)")
    musicas  = fmt(d.get("tracks", [])[:10], lambda t: f"- {t['name']} — {t.get('artist',{}).get('name','?')} ({t.get('playcount','?')} scrobbles)")
    loved    = fmt(d.get("loved_tracks",[])[:10], lambda lt: f"- {lt.get('name','?')} — {lt.get('artist',{}).get('name','?')}")

    return f"""Resumo musical (Last.fm):
Usuário: {nome}{f' — {pais}' if pais else ''}  |  Total scrobbles: {total}

Top artistas (6 meses):
{artistas}

Top álbuns (6 meses):
{albuns}

Top músicas (6 meses):
{musicas}

Músicas curtidas (loved):
{loved}
"""

# ---------------------------------------------------------------------------
# Letterboxd
# ---------------------------------------------------------------------------

def calcular_peso_recencia(s: pd.Series) -> pd.Series:
    dias = (datetime.now() - pd.to_datetime(s, errors="coerce")).dt.days
    dias = dias.fillna(RECENCY_HALF_LIFE * 2)
    return 1 / (1 + dias / RECENCY_HALF_LIFE)


def construir_texto_filme(row: pd.Series) -> str:
    p = [f"Filme: {row['Name']}", f"Nota: {row['Rating']}"]
    if "Date"  in row and pd.notna(row.get("Date")):  p.append(f"Data: {row['Date']}")
    if "Genre" in row and pd.notna(row.get("Genre")): p.append(f"Gênero: {row['Genre']}")
    return "\n".join(p)


def carregar_dados(source) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(source)
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}"); return None
    faltando = {"Name", "Rating"} - set(df.columns)
    if faltando:
        st.error(f"Colunas ausentes: {faltando}"); return None
    df = df.dropna(subset=["Name", "Rating"]).copy()
    df["texto"] = df.apply(construir_texto_filme, axis=1)
    df["peso_recencia"] = calcular_peso_recencia(df["Date"]) if "Date" in df.columns else 1.0
    return df


def gerar_resumo_perfil(df: pd.DataFrame) -> str:
    top    = df.nlargest(15, "Rating")[["Name", "Rating"]]
    bottom = df.nsmallest(10, "Rating")[["Name", "Rating"]]
    generos = ""
    if "Genre" in df.columns:
        gc = df["Genre"].dropna().str.split(", ").explode().value_counts().head(5)
        generos = "\n".join(f"- {g}: {c}" for g, c in gc.items())
    r = f"""Resumo do usuário (Letterboxd):
- Total de filmes: {len(df)}
- Média das notas: {df['Rating'].mean():.2f}

Top filmes:
{top.to_string(index=False)}

Filmes menos gostados:
{bottom.to_string(index=False)}
"""
    if generos:
        r += f"\nGêneros mais assistidos:\n{generos}"
    return r


def filmes_por_mes(df: pd.DataFrame) -> str:
    if "Date" not in df.columns:
        return "Sem coluna de data."
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    contagem = df.groupby(df["Date"].dt.to_period("M")).size().sort_values(ascending=False)
    if contagem.empty:
        return "Sem dados suficientes."
    r = f"Mês com mais filmes: {contagem.index[0]} ({contagem.iloc[0]} filmes)\n\nFilmes por mês:\n"
    for mes, c in contagem.items():
        r += f"- {mes}: {c}\n"
    return r

# ---------------------------------------------------------------------------
# Buscador unificado
# ---------------------------------------------------------------------------

class BuscadorContexto:
    def __init__(self, df_filmes: pd.DataFrame | None, lastfm_data: dict | None):
        self.df_filmes   = df_filmes
        self.lastfm_data = lastfm_data
        registros = []
        if df_filmes is not None:
            for _, row in df_filmes.iterrows():
                registros.append({"Name": row["Name"], "Rating": row.get("Rating"),
                                   "texto": row["texto"], "tipo": "filme",
                                   "peso_recencia": row.get("peso_recencia", 1.0)})
        if lastfm_data is not None:
            registros.extend(construir_texto_musica(lastfm_data))
        if not registros:
            raise ValueError("Nenhum dado disponível.")
        self.df    = pd.DataFrame(registros).reset_index(drop=True)
        self.vect  = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.X     = self.vect.fit_transform(self.df["texto"])
        self.pesos = self.df["peso_recencia"].values

    @staticmethod
    def _m(p, pat): return bool(re.search(pat, p.lower()))

    def _is_perfil(self, p):    return self._m(p, r"perfil|gosto|me descreva|meu estilo|quem sou|personalidade|resume")
    def _is_favoritos(self, p): return self._m(p, r"favor|melhor|mais gost|top|maior nota|recomend")
    def _is_temporal(self, p):  return self._m(p, r"(qual.*m[eê]s.*(mais|menos))|(em que m[eê]s)|(quantos filmes.*m[eê]s)")
    def _is_musical(self, p):   return self._m(p, r"m[uú]sica|artista|[aá]lbum|scrobble|last\.?fm|banda|ouvir|playlist|loved")
    def _is_surpresa(self, p):  return self._m(p, r"surpree?nda|zona de conforto|diferente|inusitado|nunca vi")
    def _is_trilha(self, p):    return self._m(p, r"trilha|soundtrack|ost|m[uú]sica.*filme|filme.*m[uú]sica")

    def gerar_resumo_combinado(self) -> str:
        partes = []
        if self.df_filmes   is not None: partes.append(gerar_resumo_perfil(self.df_filmes))
        if self.lastfm_data is not None: partes.append(gerar_resumo_lastfm(self.lastfm_data))
        return "\n\n---\n\n".join(partes)

    def buscar(self, pergunta: str, k: int = TOP_K) -> str:
        if self._is_perfil(pergunta):   return self.gerar_resumo_combinado()
        if self._is_temporal(pergunta) and self.df_filmes is not None: return filmes_por_mes(self.df_filmes)
        if self._is_trilha(pergunta):   return self.gerar_resumo_combinado()
        if self._is_surpresa(pergunta) and self.df_filmes is not None:
            mediana = self.df_filmes["Rating"].median()
            sample  = self.df_filmes[self.df_filmes["Rating"] >= mediana - 0.5].sample(min(10, len(self.df_filmes)))
            return gerar_resumo_perfil(self.df_filmes) + "\n\nCandidatos surpresa:\n" + "\n".join(sample["texto"].tolist())
        if self._is_musical(pergunta) and self.lastfm_data is not None: return gerar_resumo_lastfm(self.lastfm_data)
        if self._is_favoritos(pergunta) and self.df_filmes is not None:
            return "\n\n".join(self.df[self.df["tipo"] == "filme"].nlargest(k, "Rating")["texto"].tolist())
        # RAG vetorial
        scores = cosine_similarity(self.vect.transform([pergunta]), self.X).flatten() * self.pesos
        return "\n\n".join(self.df.iloc[np.argsort(scores)[-k:][::-1]]["texto"].tolist())

# ---------------------------------------------------------------------------
# System prompt dinâmico
# ---------------------------------------------------------------------------

def build_system_prompt(tem_filmes: bool, tem_musica: bool) -> str:
    base = """Você é um crítico cultural apaixonado e bem-humorado — fala de forma casual, como um amigo que entende muito.
Tem opiniões fortes, faz conexões inesperadas e não tem medo de divagar.

- Dá respostas generosas e detalhadas — nunca responde com uma linha só.
- Usa humor leve quando cabe, sem forçar.
- Conta curiosidades, contexto histórico e referências culturais quando faz sentido.
- Não inventa dados que não estejam no histórico fornecido.
- IMPORTANTE: sempre termine sua resposta por completo — nunca corte no meio de uma frase ou lista.\n"""

    if tem_filmes and tem_musica:
        base += """
Você tem acesso ao histórico de filmes (Letterboxd) E ao histórico musical (Last.fm) do usuário.
Faça pontes culturais entre cinema e música: trilhas, atmosferas, épocas, emoções em comum.
Comente sobre padrões de gosto nos dois universos. Provoque gentilmente sobre notas baixas."""
    elif tem_filmes:
        base += """
Você tem acesso ao histórico de filmes do usuário (Letterboxd).
Foque em filmes, notas, diretores, gêneros e padrões de gosto. Faça conexões entre os filmes."""
    elif tem_musica:
        base += """
Você tem acesso ao histórico musical do usuário (Last.fm).
Foque em artistas, álbuns, músicas mais ouvidas, loved tracks e padrões de escuta."""

    return base

# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def responder_stream(pergunta: str, buscador: BuscadorContexto, client: Groq,
                     tem_filmes: bool, tem_musica: bool):
    contexto  = buscador.buscar(pergunta)
    mensagem  = f"Contexto sobre o usuário:\n{contexto}\n\nPergunta: {pergunta}"
    historico = st.session_state.historico + [{"role": "user", "content": mensagem}]
    system    = build_system_prompt(tem_filmes, tem_musica)
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system}] + historico,
            temperature=0.7,
            max_tokens=4096,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        logger.error(f"Stream: {e}")
        yield "⚠️ Erro ao gerar resposta. Tente novamente."

# ---------------------------------------------------------------------------
# Exportar conversa
# ---------------------------------------------------------------------------

def exportar_conversa() -> str:
    linhas = ["# Conversa — Cinéfilo IA\n"]
    for msg in st.session_state.mensagens:
        papel = "**Você**" if msg["role"] == "user" else "**Cinéfilo IA**"
        linhas.append(f"{papel}: {msg['content']}\n")
    return "\n".join(linhas)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in [
    ("mensagens",        []),
    ("historico",        []),
    ("df",               None),
    ("buscador",         None),
    ("lastfm_data",      None),
    ("lastfm_carregado", False),
    ("tmdb_key",         ""),
    ("_enviar",          False),
    ("_sugestao",        None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    tmdb_key = os.getenv("TMDB_API_KEY") or st.secrets.get("TMDB_API_KEY", "")

    if not api_key:
        st.error("❌ GROQ_API_KEY não encontrada no .env")
        st.stop()
    
    if not tmdb_key:
        st.error("❌ TMDB_API_KEY não encontrada no .env")
        st.stop()
    
    st.session_state.tmdb_key = tmdb_key

    # ── Letterboxd ──────────────────────────────────────────────────────────
    st.markdown("### 🎬 Letterboxd")
    st.caption("Opcional — envie seu ratings.csv")
    csv_file = st.file_uploader("ratings.csv", type=["csv"])
    if csv_file:
        df = carregar_dados(csv_file)
        if df is not None and (st.session_state.df is None or len(df) != len(st.session_state.df)):
            st.session_state.df = df
            st.success(f"✅ {len(df)} filmes carregados!")

    if st.session_state.df is not None:
        df = st.session_state.df
        c1, c2 = st.columns(2)
        c1.metric("Filmes", len(df))
        c2.metric("Média",  f"{df['Rating'].mean():.1f}")
        st.markdown("**⭐ Top 3**")
        for _, row in df.nlargest(3, "Rating")[["Name","Rating"]].iterrows():
            st.markdown(f"- {row['Name']} `{row['Rating']}`")

    st.markdown("---")

    # ── Last.fm ─────────────────────────────────────────────────────────────
    st.markdown("### 🎵 Last.fm")
    st.caption("Opcional — conecte sua conta")
    lfm_key    = st.text_input("API Key",    type="password", placeholder="sua api key",    key="lfm_key_input")
    lfm_secret = st.text_input("API Secret", type="password", placeholder="seu api secret", key="lfm_secret_input")
    lfm_user   = st.text_input("Username",   placeholder="seu usuário no Last.fm",          key="lfm_user_input")

    if st.button("🔗 Conectar Last.fm"):
        if not lfm_key or not lfm_user:
            st.warning("Informe API Key e Username.")
        else:
            with st.spinner("Buscando dados..."):
                dados = carregar_dados_lastfm(lfm_key, lfm_user)
            if dados is None:
                st.error("❌ Verifique a API Key e o Username.")
            else:
                st.session_state.lastfm_data      = dados
                st.session_state.lastfm_carregado = True
                scrobbles = dados.get("info", {}).get("playcount", "?")
                st.success(f"✅ Conectado! {scrobbles} scrobbles.")

    if st.session_state.lastfm_carregado and st.session_state.lastfm_data:
        top3 = st.session_state.lastfm_data.get("artists", [])[:3]
        if top3:
            st.markdown("**🎤 Top 3 artistas (6m)**")
            for a in top3:
                st.markdown(f"- {a['name']} `{a.get('playcount','?')}` plays")

    st.markdown("---")

    # ── Reconstruir buscador ─────────────────────────────────────────────────
    tem_lb  = st.session_state.df is not None
    tem_lfm = st.session_state.lastfm_carregado and st.session_state.lastfm_data is not None

    if tem_lb or tem_lfm:
        df_para_buscador = st.session_state.df
        if tem_lb and st.session_state.tmdb_key:
            df_enriq = st.session_state.df.copy()
            for idx in df_enriq.nlargest(30, "Rating").index:
                df_enriq.at[idx, "texto"] = enriquecer_texto_tmdb(df_enriq.loc[idx], st.session_state.tmdb_key)
            df_para_buscador = df_enriq
        try:
            st.session_state.buscador = BuscadorContexto(
                df_filmes   = df_para_buscador if tem_lb else None,
                lastfm_data = st.session_state.lastfm_data if tem_lfm else None,
            )
        except Exception as e:
            logger.error(f"Buscador: {e}")

    # ── Ações ────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Limpar"):
            st.session_state.mensagens = []
            st.session_state.historico = []
            st.rerun()
    with col_b:
        if st.session_state.mensagens:
            st.download_button("💾 Exportar", data=exportar_conversa(),
                               file_name="conversa_cinefilo.md", mime="text/markdown")

    # ── Gráficos ─────────────────────────────────────────────────────────────
    if tem_lb and st.session_state.df is not None:
        st.markdown("---")
        st.markdown("### 📊 Análises")
        df_g = st.session_state.df.copy()
        tab1, tab2 = st.tabs(["Notas", "Por ano"])
        with tab1:
            st.bar_chart(df_g["Rating"].value_counts().sort_index(), color="#f5c518", height=140)
        with tab2:
            if "Date" in df_g.columns:
                df_g["Date"] = pd.to_datetime(df_g["Date"], errors="coerce")
                st.bar_chart(df_g.dropna(subset=["Date"]).groupby(df_g["Date"].dt.year).size(),
                             color="#f5c518", height=140)
            else:
                st.caption("Sem coluna de data.")

# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

tem_lb  = st.session_state.df is not None
tem_lfm = st.session_state.lastfm_carregado and st.session_state.lastfm_data is not None

# Header
st.markdown("""
<div style="width:100%; text-align:center;">
<div class="cinema-header">
    <h1>🎬 Cinéfilo IA 🎵</h1>
    <p>Seu parceiro apaixonado por cinema e música</p>
</div>
<div style="text-align:center;"><div style="display:inline-block; height:1px; width:60%; background:linear-gradient(to right, transparent, #f5c518 30%, #f5c518 70%, transparent); margin:0.5rem 0 0.8rem 0;"></div></div>
</div>
""", unsafe_allow_html=True)

# Status bar
st.markdown(f"""
<div class="source-bar">
  <span class="source-pill {'active-lb' if tem_lb else 'inactive'}">{'✓ Letterboxd' if tem_lb else '○ Letterboxd'}</span>
  <span class="source-pill {'active-lfm' if tem_lfm else 'inactive'}">{'✓ Last.fm' if tem_lfm else '○ Last.fm'}</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Área de chat
# ---------------------------------------------------------------------------

chat_container = st.container()

with chat_container:
    if not st.session_state.mensagens:
        st.markdown(
            "<div style='text-align:center; color:#444; padding:2rem 0 0.5rem; font-size:0.9rem;'>"
            "Envie seu CSV do Letterboxd e/ou conecte o Last.fm para começar 🍿🎵"
            "</div>",
            unsafe_allow_html=True,
        )

        # Sugestões clicáveis (botões reais)
        if tem_lb or tem_lfm:
            sugestoes = []
            if tem_lb:              sugestoes += SUGESTOES_FILMES[:3]
            if tem_lfm:             sugestoes += SUGESTOES_MUSICA[:2]
            if tem_lb and tem_lfm:  sugestoes += SUGESTOES_AMBOS[:2]

            n = min(len(sugestoes), 3)
            cols = st.columns(n)
            for i, sug in enumerate(sugestoes[:n * 2]):
                with cols[i % n]:
                    if st.button(sug, key=f"sug_{i}"):
                        st.session_state._sugestao = sug
                        st.rerun()
    else:
        for msg in st.session_state.mensagens:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                    <div class="bubble-user">{msg['content']}</div>
                    <div class="avatar avatar-user">🙂</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-agent">
                    <div class="avatar avatar-agent">🎬</div>
                    <div class="bubble-agent">{msg['content']}</div>
                </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

col_input, col_btn = st.columns([5, 1])
with col_input:
    pergunta = st.text_input(
        "mensagem", label_visibility="collapsed",
        placeholder="Pergunte sobre seus filmes ou músicas...",
        key="input_pergunta",
        on_change=lambda: st.session_state.update({"_enviar": True}),
    )
with col_btn:
    enviar = st.button("Enviar")

# ---------------------------------------------------------------------------
# Processar envio
# ---------------------------------------------------------------------------

if st.session_state._sugestao:
    pergunta = st.session_state._sugestao
    st.session_state._sugestao = None
    enviar = True

if (enviar or st.session_state._enviar) and pergunta:
    st.session_state._enviar = False

    if st.session_state.buscador is None:
        st.warning("Envie seu ratings.csv do Letterboxd ou conecte o Last.fm primeiro.")
    else:
        client = Groq(api_key=api_key)

        st.session_state.mensagens.append({"role": "user", "content": pergunta})
        st.session_state.historico.append({"role": "user", "content": pergunta})

        # Mostra mensagem do usuário imediatamente
        st.markdown(f"""
        <div class="msg-user">
            <div class="bubble-user">{pergunta}</div>
            <div class="avatar avatar-user">🙂</div>
        </div>""", unsafe_allow_html=True)

        # Streaming da resposta
        with st.chat_message("assistant", avatar="🎬"):
            resposta_completa = st.write_stream(
                responder_stream(pergunta, st.session_state.buscador, client, tem_lb, tem_lfm)
            )

        st.session_state.mensagens.append({"role": "agent",     "content": resposta_completa})
        st.session_state.historico.append({"role": "assistant", "content": resposta_completa})

        st.rerun()