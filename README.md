# 🎬 Cinéfilo IA

Um app interativo feito com **Streamlit + IA** que analisa suas avaliações de filmes e conversa com você como um verdadeiro cinéfilo.

---

## 🚀 Acesse o projeto

👉 [Clique aqui para acessar o app](https://cinefiloia.streamlit.app/)

---

## 🧠 Sobre o projeto

O **Cinéfilo IA** é um agente conversacional que:

* Analisa seu histórico de filmes (via CSV do Letterboxd ou similar)
* Entende seu gosto cinematográfico
* Responde perguntas sobre seus filmes
* Traça seu perfil como espectador
* Faz análises temporais (ex: mês que você mais assistiu filmes)

Tudo isso com personalidade - como um amigo que entende MUITO de cinema 🍿

---

## ✨ Funcionalidades

* 📂 Upload de arquivo `ratings.csv`
* 🎯 Análise de perfil do usuário
* ⭐ Identificação de filmes favoritos e menos gostados
* 📊 Estatísticas (média, volume, gêneros)
* 📅 Análise temporal (filmes por mês)
* 💬 Chat inteligente com memória de contexto
* 🔎 Busca semântica com TF-IDF + similaridade de cosseno
* 🧠 Integração com LLM via Groq

---

## 🏗️ Arquitetura

O projeto combina:

* RAG (Retrieval-Augmented Generation)
* TF-IDF + Cosine Similarity para busca de contexto
* Peso por recência (filmes mais recentes têm mais relevância)
* LLM (Groq) para geração de respostas naturais

---

## 📁 Estrutura esperada do CSV

O arquivo deve conter pelo menos:

| Coluna | Descrição      |
| ------ | -------------- |
| Name   | Nome do filme  |
| Rating | Nota atribuída |

Colunas opcionais:

* Date → usada para análises temporais
* Genre → usada para análise de gêneros

---

## ⚙️ Como rodar localmente

### 1. Clone o repositório

```bash
git clone https://github.com/LCavarcan/cinefilo_ia.git
cd cinefilo_ia
```

---

### 2. Crie um ambiente virtual

```bash
python -m venv venv
```

Ativar:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/Mac**

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

Crie um arquivo `.env` na raiz:

```env
GROQ_API_KEY=sua_api_key_aqui
```

---

### 5. Rode o app

```bash
streamlit run app.py
```

---

## 🔐 Variáveis de ambiente

| Variável     | Descrição            |
| ------------ | -------------------- |
| GROQ_API_KEY | Chave da API da Groq |

---

## 🧪 Tecnologias utilizadas

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Groq API
* TF-IDF
* Cosine Similarity

---

## 💡 Exemplos de perguntas

* "Quais são meus filmes favoritos?"
* "Qual meu perfil baseado nos filmes?"
* "Em qual mês eu assisti mais filmes?"
* "Que tipo de filme eu gosto?"
* "Quais filmes eu avaliei mal?"

---

## 🎨 Interface

* Tema escuro com estilo cinematográfico
* Chat customizado
* UX focada em conversa fluida

---

## ⚠️ Observações

* O app depende de um CSV válido
* A API da Groq é necessária
* Sem a coluna `Date`, análises temporais não funcionam

---

## 📌 Melhorias futuras

* Integração com API do Letterboxd
* Recomendações externas de filmes
* Gráficos interativos
* Sistema de login

---

## 🤝 Contribuição

Sinta-se à vontade para abrir issues ou pull requests!

---

## 📄 Licença

MIT

---

## 👤 Autor

Feito com 🍿 por Luiza Guimarães Cavarçan

---
