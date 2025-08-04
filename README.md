<p align="center">
  <img src="./assets/iiti_logo.png" alt="IITIbot Logo" width="150" />
</p>

# 🤖 IITIbot — RAG Chatbot for IIT Indore

IITIbot is a **Retrieval-Augmented Generation (RAG)** based intelligent chatbot that helps users get precise and up-to-date answers related to the Indian Institute of Technology Indore (IIT Indore). It integrates data scraped directly from the institute's official website and complements it with web search when needed.

Built using **Pathway** and **CRAG** for robust RAG pipeline handling, IITIbot is deployed with a **FastAPI** backend and a responsive **React.js** frontend.

---

## ✨ Features

- 🔍 **RAG-based Q&A**: Uses Retrieval-Augmented Generation to answer questions accurately.
- 🕸️ **Custom Web Scraper**: Extracts relevant internal data from `https://www.iiti.ac.in`.
- 🌐 **Web Search Integration**: Complements internal data with live web results when needed.
- 🧠 **Powered by Pathway + CRAG**: Modular and scalable RAG pipeline framework.
- 💬 **Chat UI**: User-friendly interface built with React.js, HTML, CSS, and JavaScript.
- 🚀 **FastAPI Backend**: Handles data flow, query response, and model inference.
- 🐳 **Docker Support**: For simple, reproducible deployment across systems.

---

---

## 🛠️ Tech Stack

| Layer       | Technology                      |
|-------------|----------------------------------|
| Frontend    | React.js, HTML, CSS, JavaScript |
| Backend     | FastAPI (Python)                |
| RAG Engine  | Pathway + CRAG                  |
| Scraping    | Python (BeautifulSoup, Requests)|
| Deployment  | Docker                          |
| Extras      | Web Search APIs, OpenAI API     |

---


## 🧠 How It Works

1. **Web Scraper**: Extracts academic, admin, research, and other institutional data.
2. **CRAG Pipeline**: Organizes data and enables RAG-based query answering.
3. **FastAPI Server**: Accepts user queries and serves generated responses.
4. **React UI**: Interactive chatbot interface where users type questions.
5. **Fallback Web Search**: If needed, searches the web to improve answer quality.

---

## 🧩 RAG Workflow Diagram

> Below is the architecture workflow of IITIbot:

![IITIbot RAG Workflow](./assets/iitibot_workflow.png)  
*Figure: RAG-based pipeline using Pathway and CRAG.*

---

## 🚀 Getting Started


## 🧪 Option 1: Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/KK-Singh333/IITI_BOT.git
cd iitibot
```
### 2. Run with Docker (The Backend)
```
cd backend
docker compose up --build

```
### 3. Run the frontend
```
cd frontend/frontend
npm run dev
```

## 👥 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/KK-Singh333">
        <img src="https://avatars.githubusercontent.com/u/177385234?v=4" width="80px;" alt="Khush Kumar"/>
        <br />
        <sub><b>Khush Kumar</b></sub>
      </a>
    </td>
    <!-- Add more contributors below like this: -->
    <td align="center">
      <a href="https://github.com/aditya-rai-5">
        <img src="https://avatars.githubusercontent.com/u/195468146?s=400&v=4" width="80px;" alt="Aditya Rai"/>
        <br />
        <sub><b>Aditya Rai </b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Rudra-Codes">
        <img src="https://avatars.githubusercontent.com/u/178732358?v=4" width="80px;" alt="Rudra Chitkara"/>
        <br />
        <sub><b>Rudra Chitkara</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/snehith-3939">
        <img src="https://avatars.githubusercontent.com/u/184243263?v=4" width="80px;" alt="snehith"/>
        <br />
        <sub><b>Snehith </b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/sri-nikesh-31">
        <img src="https://avatars.githubusercontent.com/u/212749969?v=4" width="80px;" alt="sri nikesh"/>
        <br />
        <sub><b>Sri Nikesh </b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/aditya-rai-5">
        <img src="https://avatars.githubusercontent.com/u/ID?v=4" width="80px;" alt="Poorvansh"/>
        <br />
        <sub><b>Poorvansh</b></sub>
      </a>
    </td>
   
  </tr>
</table>

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

