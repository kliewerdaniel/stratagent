# StratAgent

Autonomous AI Software Development Platform

StratAgent is a collaborative AI development environment where specialized AI personas work together to build software projects. Users interact via chat interface, describing requirements naturally, while AI personas handle specification analysis, architectural design, code generation, validation, and project management.

## Features

- **Multi-Persona Collaboration**: Team of AI specialists (Architect, Frontend Dev, Backend Dev, Tester, etc.)
- **Deterministic Code Generation**: RAG-powered pattern-based development
- **Self-Validating Outputs**: Orthos framework ensures code quality and safety
- **Persistent Project Memory**: GraphRAG maintains context across sessions
- **Real-time Streaming**: Live development progress and chat responses
- **Autonomous Evolution**: Personas learn and adapt through interactions

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Ollama (for local LLM inference)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stratagent
```

2. Start the services:
```bash
docker-compose up -d
```

This will start:
- Backend API (FastAPI) on port 8000
- Frontend (Next.js) on port 3000
- Ollama for LLM inference on port 11434
- Redis for caching on port 6379

3. Pull the required Ollama model:
```bash
docker-compose exec ollama ollama pull llama2
```

4. Open your browser and navigate to `http://localhost:3000`

## Development Setup

### Backend (FastAPI)

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the development server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Frontend (Next.js)

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Ollama Setup

1. Install Ollama: https://ollama.ai/
2. Pull the required model:
```bash
ollama pull llama2
```

## Architecture

### Backend (FastAPI + Python)
- Agent orchestration and management
- Knowledge graph operations (NetworkX + FAISS)
- Reinforcement learning models (PyTorch)
- Validation engines and correction systems
- MCP protocol implementation

### Frontend (Next.js + TypeScript)
- Real-time chat interface with streaming
- Persona management dashboard
- Project visualization and code review
- Administrative controls and analytics

### Data Layer
- SQLite/PostgreSQL for relational data
- FAISS for vector similarity search
- NetworkX for knowledge graph operations
- Redis for session management and caching

## API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation.

## Configuration

Environment variables can be configured in `.env` files:

### Backend Configuration
- `DATABASE_URL`: Database connection string
- `OLLAMA_BASE_URL`: Ollama API endpoint
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: JWT secret key

### Frontend Configuration
- `NEXT_PUBLIC_API_URL`: Backend API URL

## Project Structure

```
stratagent/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/v1/         # API endpoints
│   │   ├── core/           # Configuration
│   │   ├── db/             # Database setup
│   │   ├── models/         # SQLAlchemy models
│   │   ├── services/       # Business logic
│   │   └── agents/         # AI agent implementations
│   ├── requirements.txt
│   └── main.py
├── frontend/                # Next.js frontend
│   ├── src/
│   │   ├── app/            # Next.js app router
│   │   ├── components/     # React components
│   │   └── ...
│   └── package.json
├── docker-compose.yml       # Docker orchestration
├── Dockerfile              # Backend container
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

- Phase 1: Foundation Setup ✅
- Phase 2: Core Agent Architecture
- Phase 3: Validation & Safety Framework
- Phase 4: Code Generation Pipeline
- Phase 5: User Interface & Experience
- Phase 6: Reinforcement Learning Integration
- Phase 7: Advanced Features
- Phase 8: Testing & Deployment
- Phase 9: Scaling & Optimization
- Phase 10: Continuous Improvement