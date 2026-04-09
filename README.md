# 🎬🎵 Cinéfilo IA

Um app interativo feito com **Streamlit + IA** que analisa **seus filmes (Letterboxd)** e **seu gosto musical (Last.fm)** — e conversa com você como um verdadeiro crítico cultural.

---

## 🚀 Acesse o projeto

👉 [Clique aqui para acessar o app](https://cinefiloia.streamlit.app/)

---

## 🧠 Sobre o projeto

O **Cinéfilo IA** evoluiu de um analisador de filmes para um **agente cultural completo**.

Agora ele:

* 🎬 Analisa seus filmes do Letterboxd
* 🎵 Analisa seu histórico musical do Last.fm
* 🔗 Faz conexões entre cinema e música
* 🧠 Usa IA para interpretar seu gosto
* 💬 Conversa com contexto e memória

É basicamente um amigo que entende de **cinema + música + cultura pop**.

---

## ✨ Funcionalidades

### 🎬 Filmes (Letterboxd)

* Upload de `ratings.csv`
* Análise de perfil cinematográfico
* Top filmes e menos gostados
* Gêneros mais assistidos
* Análise temporal (filmes por mês)
* Enriquecimento com **TMDB (sinopse, elenco, diretor)**

---

### 🎵 Música (Last.fm)

* Conexão via API
* Top artistas, álbuns e músicas (últimos 6 meses)
* Músicas curtidas (loved tracks)
* Histórico recente

---

### 🔀 Integração cultural

* Conexões entre filmes e música
* Sugestões cruzadas (trilhas, vibes, estilos)
* Perfil combinado (cinéfilo + melômano)

---

### 🤖 IA Conversacional

* Chat com memória
* Respostas em streaming (tempo real)
* Personalidade forte (crítico cultural)
* Busca inteligente com contexto

---

### 📊 Visualizações

* Distribuição de notas
* Filmes por ano
* Métricas rápidas (total, média)

---

### 💾 Extras

* Exportar conversa em Markdown
* Sugestões clicáveis
* Interface customizada estilo cinema

---

## 🏗️ Arquitetura

O projeto combina:

* **RAG (Retrieval-Augmented Generation)**
* **TF-IDF + Cosine Similarity**
* **Peso por recência**
* **Streaming de resposta (Groq)**
* **Enriquecimento externo (TMDB)**
* **Integração com API (Last.fm)**

---

## 📁 Estrutura dos dados

### 🎬 CSV (Letterboxd)

Obrigatório:

| Coluna | Descrição     |
| ------ | ------------- |
| Name   | Nome do filme |
| Rating | Nota          |

Opcional:

* `Date` → análises temporais
* `Genre` → análise de gêneros

---

### 🎵 Last.fm

Necessário:

* API Key
* Username

---

## ⚙️ Como rodar localmente

### 1. Clone o repositório

```bash
git clone https://github.com/LCavarcan/cinefilo_ia.git
cd cinefilo_ia
```

---

### 2. Ambiente virtual

```bash
python -m venv venv
```

**Ativar:**

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

---

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

### 4. Configure o `.env`

```env
GROQ_API_KEY=sua_api_key
TMDB_API_KEY=sua_api_key_tmdb
```

---

### 5. Rode o app

```bash
streamlit run app.py
```

---

## 🔐 Variáveis de ambiente

| Variável     | Descrição                     |
| ------------ | ----------------------------- |
| GROQ_API_KEY | API da Groq (IA)              |
| TMDB_API_KEY | API do TMDB (dados de filmes) |

---

## 🧪 Tecnologias utilizadas

* Python
* Streamlit
* Pandas / NumPy
* Scikit-learn
* Groq API (LLM)
* Last.fm API
* TMDB API
* TF-IDF
* Cosine Similarity

---

## 💡 Exemplos de perguntas

### 🎬 Filmes

* "Qual meu filme favorito?"
* "Que mês assisti mais filmes?"
* "Como você descreveria meu gosto?"

### 🎵 Música

* "Qual meu artista favorito?"
* "Que músicas eu mais ouvi?"
* "Me fala sobre meu gosto musical"

### 🔀 Ambos

* "Que trilhas combinam com meus filmes favoritos?"
* "Que filmes combinam com os artistas que eu ouço?"
* "Me descreva como pessoa baseado nisso tudo"

---

## 🎨 Interface

* Tema escuro cinematográfico
* UI totalmente customizada
* Chat estilizado
* Sugestões interativas
* Feedback visual em tempo real

---

## ⚠️ Observações

* Letterboxd e Last.fm são opcionais (mas pelo menos um é necessário)
* TMDB é obrigatório para enriquecimento
* Sem `Date`, análises temporais não funcionam

---

## 📌 Melhorias futuras

* Recomendações externas automáticas
* Sistema de usuários/login
* Dashboard mais avançado
* Cache inteligente para APIs
* Integração com Spotify

---

## 🤝 Contribuição

Pull requests são bem-vindos!

---

## 📄 Licença

MIT

---

## 👤 Autor

Feito com 🎬🎵 por Luiza Guimarães Cavarçan

---