# Importações principais
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

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# ---------------------------------------------------------------------------
# Configuração de página
# ---------------------------------------------------------------------------
# Define o título e layout da página Streamlit
st.set_page_config(page_title="🎬 Cinéfilo IA 🎵", layout="centered")

# ---------------------------------------------------------------------------
# Estilo visual
# ---------------------------------------------------------------------------
# Aplica CSS customizado para tema escuro estilo cinema
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Fundo escuro com padrão de ruído */
.stApp {
    background-color: #0d0d0d;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
}

/* Estilos do cabeçalho principal */
.cinema-header { text-align: center; padding: 2.5rem 0 1rem; }
.cinema-header h1 { font-size: 2.8rem; color: #f5c518; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
.cinema-header p { color: #888; font-size: 0.95rem; font-weight: 300; letter-spacing: 0.05em; text-transform: uppercase; }

/* Status bar de fontes (Letterboxd e Last.fm) */
.source-bar { display:flex; justify-content:center; gap:0.6rem; margin:0.4rem 0 1.2rem; flex-wrap:wrap; }
.source-pill { display:inline-flex; align-items:center; gap:0.3rem; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; letter-spacing:0.03em; }
.source-pill.active-lb  { background:#1a3a1a; color:#4ade80; border:1px solid #4ade8055; }
.source-pill.active-lfm { background:#3a0a0a; color:#f87171; border:1px solid #f8717155; }
.source-pill.inactive   { background:#1a1a1a; color:#444;    border:1px solid #2a2a2a; }

/* Estilos para mensagens de chat */
.msg-user  { display:flex; justify-content:flex-end;  margin:0.8rem 0; }
.msg-agent { display:flex; justify-content:flex-start; margin:0.8rem 0; }
.bubble-user  { background:#f5c518; color:#0d0d0d; padding:0.75rem 1.1rem; border-radius:18px 18px 4px 18px; max-width:75%; font-weight:500; font-size:0.95rem; line-height:1.5; }
.bubble-agent { background:#1a1a1a; color:#e8e8e8; padding:0.75rem 1.1rem; border-radius:18px 18px 18px 4px; max-width:80%; font-size:0.95rem; line-height:1.6; border:1px solid #2a2a2a; }
.avatar { width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1rem; flex-shrink:0; }
.avatar-agent { background:#1f1f1f; border:1px solid #f5c518; margin-right:0.6rem; }
.avatar-user  { background:#f5c51822; margin-left:0.6rem; }

/* Estilos do campo de input */
.stTextInput > div > div > input { background-color:#1a1a1a !important; color:#f0f0f0 !important; border:1px solid #333 !important; border-radius:12px !important; padding:0.7rem 1rem !important; font-family:'DM Sans',sans-serif !important; font-size:0.95rem !important; }
.stTextInput > div > div > input:focus { border-color:#f5c518 !important; box-shadow:0 0 0 2px #f5c51822 !important; }

/* Estilos dos botões principais */
.stButton > button { background:#f5c518 !important; color:#0d0d0d !important; border:none !important; border-radius:10px !important; font-family:'DM Sans',sans-serif !important; font-weight:600 !important; padding:0.5rem 1.2rem !important; transition:all 0.2s !important; }
.stButton > button:hover { background:#ffd740 !important; transform:translateY(-1px) !important; box-shadow:0 4px 12px #f5c51840 !important; }

/* Botões transparentes na sidebar */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] .stButton > button *,
section[data-testid="stSidebar"] .stButton > button p { background:transparent !important; color:#f5c518 !important; border:1px solid #f5c51866 !important; font-weight:500 !important; }
section[data-testid="stSidebar"] .stButton > button:hover,
section[data-testid="stSidebar"] .stButton > button:hover *,
section[data-testid="stSidebar"] .stButton > button:hover p { background:#f5c51815 !important; border-color:#f5c518 !important; color:#ffd740 !important; }

/* Estilos da sidebar */
section[data-testid="stSidebar"] { background-color:#111 !important; border-right:1px solid #1e1e1e; }
section[data-testid="stSidebar"] *:not(button):not(button *) { color:#bbb !important; }
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 { color:#e0e0e0 !important; }
section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] .stMarkdown li { color:#aaa !important; font-size:0.88rem !important; }
section[data-testid="stSidebar"] .stMarkdown strong { color:#ddd !important; }
section[data-testid="stSidebar"] code { background:#222 !important; color:#f5c518 !important; border-radius:4px !important; padding:1px 5px !important; font-size:0.8rem !important; }
section[data-testid="stSidebar"] .stCaption p { color:#555 !important; font-size:0.78rem !important; }

/* Estilos do upload de arquivo */
.stFileUploader { background:#1a1a1a; border:1px dashed #333; border-radius:12px; padding:0.5rem; }

/* Estilos de métricas */
[data-testid="metric-container"] { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:10px; padding:0.8rem 1rem; }
[data-testid="metric-container"] label { color:#888 !important; font-size:0.75rem !important; text-transform:uppercase; letter-spacing:0.05em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#f5c518 !important; font-family:'Playfair Display',serif !important; font-size:1.6rem !important; }

/* Estilos de abas (tabs) */
.stTabs [data-baseweb="tab-list"] { background:#111; border-bottom:1px solid #222; }
.stTabs [data-baseweb="tab"] { color:#666 !important; }
.stTabs [aria-selected="true"] { color:#f5c518 !important; border-bottom:2px solid #f5c518 !important; }

/* Barra de rolagem (scrollbar) */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#111; }
::-webkit-scrollbar-thumb { background:#333; border-radius:4px; }

/* Mensagens de chat do Streamlit */
[data-testid="stChatMessage"] { background:#1a1a1a !important; border:1px solid #2a2a2a !important; border-radius:18px 18px 18px 4px !important; }
[data-testid="stChatMessage"] p { color:#e8e8e8 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Logging & Constantes
# ---------------------------------------------------------------------------
# Configuração de logging para debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constantes de configuração da aplicação
MODEL_NAME        = "openai/gpt-oss-120b"  # Modelo de IA usado
TOP_K             = 10  # Número de resultados para busca vetorial
RECENCY_HALF_LIFE = 365  # Dias para cálculo de peso de recência
LASTFM_API_URL    = "https://ws.audioscrobbler.com/2.0/"  # Endpoint da API Last.fm
TMDB_API_URL      = "https://api.themoviedb.org/3"  # Endpoint da API TMDB

# Sugestões de perguntas para filmes
SUGESTOES_FILMES = [
    "Qual meu filme favorito?",
    "Me recomenda algo parecido com meus tops",
    "Que mês assisti mais filmes?",
    "Me surpreenda com algo fora da minha zona de conforto",
    "Quais filmes eu deveria ter dado nota maior?",
    "Como você descreveria meu gosto cinematográfico?",
]

# Sugestões de perguntas para música
SUGESTOES_MUSICA = [
    "Qual meu artista favorito ultimamente?",
    "Me fala sobre meu gosto musical",
    "Que álbum eu mais ouvi nos últimos meses?",
    "Que músicas eu curti (loved)?",
]

# Sugestões de perguntas para ambos (cinema + música)
SUGESTOES_AMBOS = [
    "Que trilhas sonoras combinam com meus filmes favoritos?",
    "Que filmes combinam com os artistas que eu mais ouço?",
    "Me descreva como cinéfilo e melômano",
]

# ---------------------------------------------------------------------------
# TMDB (The Movie Database)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def buscar_tmdb(titulo: str, tmdb_key: str) -> dict | None:
    """
    Busca informações de filme na TMDB (sinopse, gêneros, diretor, elenco).
    Resultado é cacheado por 24 horas.
    """
    if not tmdb_key:
        return None
    try:
        # Busca o filme pelo título
        r = requests.get(f"{TMDB_API_URL}/search/movie",
                         params={"api_key": tmdb_key, "query": titulo, "language": "pt-BR"}, timeout=8)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        
        # Busca detalhes completos do primeiro resultado (incluindo créditos)
        det = requests.get(f"{TMDB_API_URL}/movie/{results[0]['id']}",
                           params={"api_key": tmdb_key, "language": "pt-BR", "append_to_response": "credits"}, timeout=8)
        det.raise_for_status()
        return det.json()
    except Exception as e:
        logger.error(f"TMDB: {e}")
        return None


def enriquecer_texto_tmdb(row: pd.Series, tmdb_key: str) -> str:
    """
    Enriquece o texto de um filme com dados da TMDB:
    sinopse, gêneros, diretor e elenco.
    """
    info = buscar_tmdb(row["Name"], tmdb_key)
    if not info:
        return row["texto"]
    
    extras = []
    
    # Adiciona sinopse (até 300 caracteres)
    if info.get("overview"):
        extras.append(f"Sinopse: {info['overview'][:300]}")
    
    # Adiciona até 3 gêneros
    genres = ", ".join(g["name"] for g in info.get("genres", [])[:3])
    if genres:
        extras.append(f"Gêneros: {genres}")
    
    # Adiciona diretor(es)
    credits = info.get("credits", {})
    dirs = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
    if dirs:
        extras.append(f"Diretor: {', '.join(dirs[:2])}")
    
    # Adiciona elenco (até 4 atores)
    cast = [c["name"] for c in credits.get("cast", [])[:4]]
    if cast:
        extras.append(f"Elenco: {', '.join(cast)}")
    
    # Retorna texto original + informações extras
    return row["texto"] + "\n" + "\n".join(extras)

# ---------------------------------------------------------------------------
# Last.fm
# ---------------------------------------------------------------------------

def lastfm_request(params: dict, api_key: str) -> dict | None:
    """
    Realiza uma requisição à API do Last.fm.
    Retorna None se houver erro.
    """
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
    """
    Carrega dados musicais do Last.fm de um usuário:
    - Top artistas (6 meses)
    - Top álbuns (6 meses)
    - Top músicas (6 meses)
    - Faixas recentes
    - Faixas curtidas (loved)
    """
    # Busca informações gerais do usuário
    info = lastfm_request({"method": "user.getinfo", "user": username}, api_key)
    if not info:
        return None
    res = {"info": info.get("user", {})}

    # Busca tops (artistas, álbuns, músicas) dos últimos 6 meses
    for method, outer, inner, limit in [
        ("user.gettopartists", "topartists", "artist", 20),
        ("user.gettopalbums",  "topalbums",  "album",  15),
        ("user.gettoptracks",  "toptracks",  "track",  20),
    ]:
        d = lastfm_request({"method": method, "user": username, "period": "6month", "limit": limit}, api_key)
        if d:
            res[inner + "s"] = d.get(outer, {}).get(inner, [])

    # Busca faixas recentes
    recent = lastfm_request({"method": "user.getrecenttracks", "user": username, "limit": 50}, api_key)
    if recent:
        res["recent_tracks"] = recent.get("recenttracks", {}).get("track", [])

    # Busca faixas curtidas (loved)
    loved = lastfm_request({"method": "user.getlovedtracks", "user": username, "limit": 20}, api_key)
    if loved:
        res["loved_tracks"] = loved.get("lovedtracks", {}).get("track", [])

    return res


def construir_texto_musica(d: dict) -> list[dict]:
    """
    Converte dados do Last.fm em lista de registros com texto formatado.
    Retorna lista de dicts com: Name, Rating, peso_recencia, tipo, texto
    """
    reg = []
    
    # Adiciona artistas
    for a in d.get("artists", []):
        reg.append({"Name": a.get("name",""), "Rating": None, "peso_recencia": 1.0, "tipo": "artista",
                    "texto": f"Artista: {a.get('name','')}\nScrobbles (6m): {a.get('playcount','?')}"})
    
    # Adiciona álbuns
    for al in d.get("albums", []):
        art = al.get("artist", {}).get("name", "?")
        reg.append({"Name": al.get("name",""), "Rating": None, "peso_recencia": 1.0, "tipo": "album",
                    "texto": f"Álbum: {al.get('name','')}\nArtista: {art}\nScrobbles (6m): {al.get('playcount','?')}"})
    
    # Adiciona músicas
    for t in d.get("tracks", []):
        art = t.get("artist", {}).get("name", "?")
        reg.append({"Name": t.get("name",""), "Rating": None, "peso_recencia": 1.0, "tipo": "musica",
                    "texto": f"Música: {t.get('name','')}\nArtista: {art}\nScrobbles (6m): {t.get('playcount','?')}"})
    
    # Adiciona faixas curtidas
    for lt in d.get("loved_tracks", []):
        art = lt.get("artist", {}).get("name", "?")
        reg.append({"Name": lt.get("name",""), "Rating": None, "peso_recencia": 1.0, "tipo": "loved",
                    "texto": f"Música curtida: {lt.get('name','')}\nArtista: {art}"})
    
    return reg


def gerar_resumo_lastfm(d: dict) -> str:
    """
    Gera um resumo formatado dos dados musicais do Last.fm.
    Inclui: nome, total de scrobbles, tops e faixas curtidas.
    """
    info  = d.get("info", {})
    nome  = info.get("name", "?")
    total = info.get("playcount", "?")
    pais  = info.get("country", "")

    # Função auxiliar para formatar listas
    def fmt(items, fn): return "\n".join(fn(i) for i in items) or "Sem dados"

    # Formata tops
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
    """
    Calcula peso de recência para filmes.
    Filmes mais recentes têm peso maior na busca.
    Usa meia-vida exponencial.
    """
    dias = (datetime.now() - pd.to_datetime(s, errors="coerce")).dt.days
    dias = dias.fillna(RECENCY_HALF_LIFE * 2)
    return 1 / (1 + dias / RECENCY_HALF_LIFE)


def construir_texto_filme(row: pd.Series) -> str:
    """
    Formata informações de um filme para texto legível.
    Inclui: nome, nota, data (se houver) e gênero (se houver).
    """
    p = [f"Filme: {row['Name']}", f"Nota: {row['Rating']}"]
    if "Date"  in row and pd.notna(row.get("Date")):  p.append(f"Data: {row['Date']}")
    if "Genre" in row and pd.notna(row.get("Genre")): p.append(f"Gênero: {row['Genre']}")
    return "\n".join(p)


def carregar_dados(source) -> pd.DataFrame | None:
    """
    Carrega CSV do Letterboxd e valida estrutura.
    Requer colunas: Name, Rating
    Calcula peso de recência se houver coluna Date.
    """
    try:
        df = pd.read_csv(source)
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}")
        return None
    
    # Valida colunas obrigatórias
    faltando = {"Name", "Rating"} - set(df.columns)
    if faltando:
        st.error(f"Colunas ausentes: {faltando}")
        return None
    
    # Remove filmes sem informações essenciais
    df = df.dropna(subset=["Name", "Rating"]).copy()
    
    # Cria coluna de texto formatado
    df["texto"] = df.apply(construir_texto_filme, axis=1)
    
    # Calcula peso de recência a partir da data de visualização
    df["peso_recencia"] = calcular_peso_recencia(df["Date"]) if "Date" in df.columns else 1.0
    
    return df


def gerar_resumo_perfil(df: pd.DataFrame) -> str:
    """
    Gera resumo do perfil do usuário no Letterboxd:
    total de filmes, média de notas, top filmes, filmes menos gostados, gêneros.
    """
    # Seleciona top 15 melhores filmes
    top    = df.nlargest(15, "Rating")[["Name", "Rating"]]
    # Seleciona 10 filmes menos gostados
    bottom = df.nsmallest(10, "Rating")[["Name", "Rating"]]
    
    # Conta gêneros mais assistidos
    generos = ""
    if "Genre" in df.columns:
        gc = df["Genre"].dropna().str.split(", ").explode().value_counts().head(5)
        generos = "\n".join(f"- {g}: {c}" for g, c in gc.items())
    
    # Monta resumo
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
    """
    Analisa mês com mais filmes assistidos.
    Retorna mês com maior contagem e distribuição por mês.
    """
    if "Date" not in df.columns:
        return "Sem coluna de data."
    
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    
    # Agrega por mês
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
    """
    Classe que integra filmes e músicas para busca inteligente.
    Usa TF-IDF + cosine similarity para busca vetorial.
    Detecta tipo de pergunta para retornar contexto apropriado.
    """
    
    def __init__(self, df_filmes: pd.DataFrame | None, lastfm_data: dict | None):
        """
        Inicializa buscador com dados de filmes e/ou música.
        Treina modelo TF-IDF com todos os textos disponíveis.
        """
        self.df_filmes   = df_filmes
        self.lastfm_data = lastfm_data
        
        # Monta lista de registros
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
        
        # Cria DataFrame com todos os registros
        self.df    = pd.DataFrame(registros).reset_index(drop=True)
        
        # Treina vetorizador TF-IDF
        self.vect  = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.X     = self.vect.fit_transform(self.df["texto"])
        
        # Array com pesos de recência
        self.pesos = self.df["peso_recencia"].values

    # Métodos auxiliares para detectar tipo de pergunta (usando regex)
    @staticmethod
    def _m(p, pat): 
        """Verifica se pergunta contém padrão regex."""
        return bool(re.search(pat, p.lower()))

    def _is_perfil(self, p):    
        """Detecta pergunta sobre perfil/gosto do usuário."""
        return self._m(p, r"perfil|gosto|me descreva|meu estilo|quem sou|personalidade|resume")
    
    def _is_favoritos(self, p): 
        """Detecta pergunta sobre favoritos/melhores."""
        return self._m(p, r"favor|melhor|mais gost|top|maior nota|recomend")
    
    def _is_temporal(self, p):  
        """Detecta pergunta sobre análise temporal (por mês, época, etc)."""
        return self._m(p, r"(qual.*m[eê]s.*(mais|menos))|(em que m[eê]s)|(quantos filmes.*m[eê]s)")
    
    def _is_musical(self, p):   
        """Detecta pergunta sobre música/artistas."""
        return self._m(p, r"m[uú]sica|artista|[aá]lbum|scrobble|last\.?fm|banda|ouvir|playlist|loved")
    
    def _is_surpresa(self, p):  
        """Detecta pergunta por recomendação surpresa/fora da zona de conforto."""
        return self._m(p, r"surpree?nda|zona de conforto|diferente|inusitado|nunca vi")
    
    def _is_trilha(self, p):    
        """Detecta pergunta sobre trilhas sonoras."""
        return self._m(p, r"trilha|soundtrack|ost|m[uú]sica.*filme|filme.*m[uú]sica")

    def gerar_resumo_combinado(self) -> str:
        """Gera resumo combinado de filmes e música."""
        partes = []
        if self.df_filmes   is not None: partes.append(gerar_resumo_perfil(self.df_filmes))
        if self.lastfm_data is not None: partes.append(gerar_resumo_lastfm(self.lastfm_data))
        return "\n\n---\n\n".join(partes)

    def buscar(self, pergunta: str, k: int = TOP_K) -> str:
        """
        Busca contexto apropriado baseado no tipo de pergunta.
        Usa heurísticas (regex) primeiro, depois busca vetorial.
        """
        # Tipo 1: Pergunta sobre perfil
        if self._is_perfil(pergunta):   
            return self.gerar_resumo_combinado()
        
        # Tipo 2: Pergunta temporal (por mês)
        if self._is_temporal(pergunta) and self.df_filmes is not None: 
            return filmes_por_mes(self.df_filmes)
        
        # Tipo 3: Pergunta sobre trilhas sonoras
        if self._is_trilha(pergunta):   
            return self.gerar_resumo_combinado()
        
        # Tipo 4: Pergunta por surpresa
        if self._is_surpresa(pergunta) and self.df_filmes is not None:
            mediana = self.df_filmes["Rating"].median()
            # Seleciona filmes acima da mediana para sugerir
            sample  = self.df_filmes[self.df_filmes["Rating"] >= mediana - 0.5].sample(min(10, len(self.df_filmes)))
            return gerar_resumo_perfil(self.df_filmes) + "\n\nCandidatos surpresa:\n" + "\n".join(sample["texto"].tolist())
        
        # Tipo 5: Pergunta sobre música
        if self._is_musical(pergunta) and self.lastfm_data is not None: 
            return gerar_resumo_lastfm(self.lastfm_data)
        
        # Tipo 6: Pergunta sobre favoritos (retorna top K filmes/músicas)
        if self._is_favoritos(pergunta) and self.df_filmes is not None:
            return "\n\n".join(self.df[self.df["tipo"] == "filme"].nlargest(k, "Rating")["texto"].tolist())
        
        # Tipo 7: Busca vetorial padrão (TF-IDF + cosine similarity)
        scores = cosine_similarity(self.vect.transform([pergunta]), self.X).flatten() * self.pesos
        return "\n\n".join(self.df.iloc[np.argsort(scores)[-k:][::-1]]["texto"].tolist())

# ---------------------------------------------------------------------------
# System prompt dinâmico
# ---------------------------------------------------------------------------

def build_system_prompt(tem_filmes: bool, tem_musica: bool) -> str:
    """
    Constrói system prompt dinâmico baseado nos dados disponíveis.
    Adapta instruções para contexto de filmes, música ou ambos.
    """
    base = """Você é um crítico cultural apaixonado e bem-humorado — fala de forma casual, como um amigo que entende muito.
Tem opiniões fortes, faz conexões inesperadas e não tem medo de divagar.

- Dá respostas generosas e detalhadas — nunca responde com uma linha só.
- Usa humor leve quando cabe, sem forçar.
- Conta curiosidades, contexto histórico e referências culturais quando faz sentido.
- Não inventa dados que não estejam no histórico fornecido.
- IMPORTANTE: sempre termine sua resposta por completo — nunca corte no meio de uma frase ou lista.\n"""

    # Adapta instruções conforme dados disponíveis
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
    """
    Gera resposta com streaming usando a API da Groq.
    Busca contexto, monta prompt e faz chamada à IA.
    Yields chunks de texto conforme são recebidos.
    """
    # Busca contexto relevante baseado na pergunta
    contexto  = buscador.buscar(pergunta)
    
    # Monta mensagem com contexto + pergunta
    mensagem  = f"Contexto sobre o usuário:\n{contexto}\n\nPergunta: {pergunta}"
    
    # Adiciona à histórico (para manter conversação contínua)
    historico = st.session_state.historico + [{"role": "user", "content": mensagem}]
    
    # Cria system prompt adaptado
    system    = build_system_prompt(tem_filmes, tem_musica)
    
    try:
        # Chama API com streaming
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system}] + historico,
            temperature=0.7,  # Um pouco de criatividade
            max_tokens=4096,
            stream=True,
        )
        
        # Faz yield de cada chunk recebido
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
    """
    Exporta conversa em formato Markdown para download.
    """
    linhas = ["# Conversa — Cinéfilo IA\n"]
    for msg in st.session_state.mensagens:
        papel = "**Você**" if msg["role"] == "user" else "**Cinéfilo IA**"
        linhas.append(f"{papel}: {msg['content']}\n")
    return "\n".join(linhas)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

# Inicializa variáveis de sessão (persistem durante a sessão do usuário)
for key, default in [
    ("mensagens",        []),  # Histórico de mensagens exibidas
    ("historico",        []),  # Histórico para contexto da IA
    ("df",               None),  # DataFrame com filmes do Letterboxd
    ("buscador",         None),  # Instância do BuscadorContexto
    ("lastfm_data",      None),  # Dados musicais do Last.fm
    ("lastfm_carregado", False),  # Flag indicando se Last.fm foi carregado
    ("tmdb_key",         ""),  # Chave da API TMDB
    ("_enviar",          False),  # Flag para enviar mensagem
    ("_sugestao",        None),  # Sugestão clicada pelo usuário
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Carrega chaves de API do .env ou st.secrets
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    tmdb_key = os.getenv("TMDB_API_KEY") or st.secrets.get("TMDB_API_KEY", "")

    # Valida existência das chaves obrigatórias
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
    
    # Upload de arquivo CSV do Letterboxd
    csv_file = st.file_uploader("ratings.csv", type=["csv"])
    if csv_file:
        df = carregar_dados(csv_file)
        # Recarrega somente se arquivo mudou
        if df is not None and (st.session_state.df is None or len(df) != len(st.session_state.df)):
            st.session_state.df = df
            st.success(f"✅ {len(df)} filmes carregados!")

    # Exibe métricas de filmes se disponível
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
    
    # Inputs para credenciais do Last.fm
    lfm_key    = st.text_input("API Key",    type="password", placeholder="sua api key",    key="lfm_key_input")
    lfm_secret = st.text_input("API Secret", type="password", placeholder="seu api secret", key="lfm_secret_input")
    lfm_user   = st.text_input("Username",   placeholder="seu usuário no Last.fm",          key="lfm_user_input")

    # Botão para conectar/carregar dados do Last.fm
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

    # Exibe top 3 artistas se Last.fm carregado
    if st.session_state.lastfm_carregado and st.session_state.lastfm_data:
        top3 = st.session_state.lastfm_data.get("artists", [])[:3]
        if top3:
            st.markdown("**🎤 Top 3 artistas (6m)**")
            for a in top3:
                st.markdown(f"- {a['name']} `{a.get('playcount','?')}` plays")

    st.markdown("---")

    # ── Reconstruir buscador ─────────────────────────────────────────────────
    # Verifica quais dados estão disponíveis
    tem_lb  = st.session_state.df is not None
    tem_lfm = st.session_state.lastfm_carregado and st.session_state.lastfm_data is not None

    # Constrói buscador unificado se houver dados
    if tem_lb or tem_lfm:
        df_para_buscador = st.session_state.df
        
        # Se tem Letterboxd e TMDB key, enriquece top 30 filmes com dados da TMDB
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
        # Botão para limpar conversa
        if st.button("🗑️ Limpar"):
            st.session_state.mensagens = []
            st.session_state.historico = []
            st.rerun()
    with col_b:
        # Botão para exportar conversa se houver mensagens
        if st.session_state.mensagens:
            st.download_button("💾 Exportar", data=exportar_conversa(),
                               file_name="conversa_cinefilo.md", mime="text/markdown")

    # ── Gráficos ─────────────────────────────────────────────────────────────
    # Exibe análises visuais dos filmes (se disponível)
    if tem_lb and st.session_state.df is not None:
        st.markdown("---")
        st.markdown("### 📊 Análises")
        df_g = st.session_state.df.copy()
        
        tab1, tab2 = st.tabs(["Notas", "Por ano"])
        with tab1:
            # Gráfico de distribuição de notas
            st.bar_chart(df_g["Rating"].value_counts().sort_index(), color="#f5c518", height=140)
        with tab2:
            # Gráfico de filmes por ano (se houver coluna de data)
            if "Date" in df_g.columns:
                df_g["Date"] = pd.to_datetime(df_g["Date"], errors="coerce")
                st.bar_chart(df_g.dropna(subset=["Date"]).groupby(df_g["Date"].dt.year).size(),
                             color="#f5c518", height=140)
            else:
                st.caption("Sem coluna de data.")

# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

# Verifica novamente quais dados estão disponíveis
tem_lb  = st.session_state.df is not None
tem_lfm = st.session_state.lastfm_carregado and st.session_state.lastfm_data is not None

# ── Header ──────────────────────────────────────────────────────────────────
# Exibe título e descrição com styling customizado
st.markdown("""
<div style="width:100%; text-align:center;">
<div class="cinema-header">
    <h1>🎬 Cinéfilo IA 🎵</h1>
    <p>Seu parceiro apaixonado por cinema e música</p>
</div>
<div style="text-align:center;"><div style="display:inline-block; height:1px; width:60%; background:linear-gradient(to right, transparent, #f5c518 30%, #f5c518 70%, transparent); margin:0.5rem 0 0.8rem 0;"></div></div>
</div>
""", unsafe_allow_html=True)

# ── Status bar ──────────────────────────────────────────────────────────────
# Mostra quais fontes de dados estão ativas
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
    # Se não há mensagens, mostra tela inicial
    if not st.session_state.mensagens:
        st.markdown(
            "<div style='text-align:center; color:#444; padding:2rem 0 0.5rem; font-size:0.9rem;'>"
            "Envie seu CSV do Letterboxd e/ou conecte o Last.fm para começar 🍿🎵"
            "</div>",
            unsafe_allow_html=True,
        )

        # Sugestões clicáveis como botões
        if tem_lb or tem_lfm:
            # Monta lista de sugestões baseado no que está disponível
            sugestoes = []
            if tem_lb:              sugestoes += SUGESTOES_FILMES[:3]
            if tem_lfm:             sugestoes += SUGESTOES_MUSICA[:2]
            if tem_lb and tem_lfm:  sugestoes += SUGESTOES_AMBOS[:2]

            # Exibe até 3 sugestões em colunas
            n = min(len(sugestoes), 3)
            cols = st.columns(n)
            for i, sug in enumerate(sugestoes[:n * 2]):
                with cols[i % n]:
                    # Clique em botão de sugestão popula o input
                    if st.button(sug, key=f"sug_{i}"):
                        st.session_state._sugestao = sug
                        st.rerun()
    else:
        # Exibe histórico de mensagens
        for msg in st.session_state.mensagens:
            if msg["role"] == "user":
                # Mensagem do usuário
                st.markdown(f"""
                <div class="msg-user">
                    <div class="bubble-user">{msg['content']}</div>
                    <div class="avatar avatar-user">🙂</div>
                </div>""", unsafe_allow_html=True)
            else:
                # Mensagem do assistente
                st.markdown(f"""
                <div class="msg-agent">
                    <div class="avatar avatar-agent">🎬</div>
                    <div class="bubble-agent">{msg['content']}</div>
                </div>""", unsafe_allow_html=True)

# Espaçamento
st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input e botão de envio
# ---------------------------------------------------------------------------

col_input, col_btn = st.columns([5, 1])
with col_input:
    # Campo de input para a pergunta
    pergunta = st.text_input(
        "mensagem", label_visibility="collapsed",
        placeholder="Pergunte sobre seus filmes ou músicas...",
        key="input_pergunta",
        on_change=lambda: st.session_state.update({"_enviar": True}),
    )
with col_btn:
    # Botão para enviar
    enviar = st.button("Enviar")

# ---------------------------------------------------------------------------
# Processar envio e gerar resposta
# ---------------------------------------------------------------------------

# Se clicou em uma sugestão, a pergunta é preenchida automaticamente
if st.session_state._sugestao:
    pergunta = st.session_state._sugestao
    st.session_state._sugestao = None
    enviar = True

# Processa envio da pergunta
if (enviar or st.session_state._enviar) and pergunta:
    st.session_state._enviar = False

    # Valida se há dados disponíveis
    if st.session_state.buscador is None:
        st.warning("Envie seu ratings.csv do Letterboxd ou conecte o Last.fm primeiro.")
    else:
        # Cria cliente Groq
        client = Groq(api_key=api_key)

        # Adiciona pergunta ao histórico
        st.session_state.mensagens.append({"role": "user", "content": pergunta})
        st.session_state.historico.append({"role": "user", "content": pergunta})

        # Exibe mensagem do usuário imediatamente
        st.markdown(f"""
        <div class="msg-user">
            <div class="bubble-user">{pergunta}</div>
            <div class="avatar avatar-user">🙂</div>
        </div>""", unsafe_allow_html=True)

        # Gera resposta com streaming
        with st.chat_message("assistant", avatar="🎬"):
            resposta_completa = st.write_stream(
                responder_stream(pergunta, st.session_state.buscador, client, tem_lb, tem_lfm)
            )

        # Adiciona resposta ao histórico
        st.session_state.mensagens.append({"role": "agent",     "content": resposta_completa})
        st.session_state.historico.append({"role": "assistant", "content": resposta_completa})

        # Recarrega página para resetar input e mostra nova mensagem
        st.rerun()