# MCBot01: The Local-First Full-Stack Foundation

## Overview

MCBot01 is a production-ready full-stack starter template designed specifically for local LLM development. It bridges the gap between raw inference (Ollama) and user experience (Web UI), serving as the architectural spine for complex systems like GraphRAG Research Assistants.

**Repository**: `github.com/kliewerdaniel/mcbot01`

## Problem Statement

Innovation in AI is stalled by boilerplate. Every new local AI tool requires rebuilding the same infrastructure: reactive UI, backend API for timeouts, and local inference connectors. This wastes critical cognitive energy on undifferentiated work.

## Solution Architecture

### Core Components

1. **Frontend (Next.js/React)**
   - Real-time chat interface
   - Streaming responses
   - Model selection and configuration
   - Conversation history and persistence

2. **Backend (FastAPI)**
   - RESTful API for LLM interactions
   - Timeout handling and error recovery
   - Request queuing and rate limiting
   - Model management and switching

3. **Local Inference Layer (Ollama)**
   - Model loading and management
   - Streaming inference
   - Resource monitoring
   - Fallback mechanisms

4. **Data Persistence**
   - SQLite for conversation history
   - Model configurations
   - User preferences

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React UI      │    │   FastAPI       │    │   Ollama        │
│                 │    │   Backend       │    │   Inference     │
│ - Chat Interface│◄──►│ - API Routes    │◄──►│ - Model Loading │
│ - Streaming     │    │ - Timeout Mgmt  │    │ - Streaming     │
│ - Model Select  │    │ - Queue System  │    │ - Error Handling│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │   SQLite DB     │
                    │                 │
                    │ - Conversations │
                    │ - Configurations│
                    └─────────────────┘
```

## Technical Implementation

### Backend (FastAPI)

#### Dependencies

```python
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
ollama==0.1.7
sqlalchemy==2.0.23
alembic==1.12.1
pydantic==2.5.0
python-multipart==0.0.6
```

#### Core Models

```python
# mcbot01/models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    message = Column(Text)
    response = Column(Text)
    model = Column(String)
    timestamp = Column(DateTime)
    metadata = Column(JSON)

class ModelConfig(Base):
    __tablename__ = "model_configs"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    model_name = Column(String)
    parameters = Column(JSON)
    is_active = Column(Boolean, default=False)
```

#### API Routes

```python
# mcbot01/api/routes.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from ollama import Client
import asyncio
from typing import Dict, Any

router = APIRouter()
ollama_client = Client()

@router.post("/chat")
async def chat(request: dict):
    """Handle chat requests with streaming responses."""
    try:
        model = request.get("model", "llama2")
        messages = request.get("messages", [])

        def generate():
            stream = ollama_client.chat(
                model=model,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                yield f"data: {chunk.json()}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    """List available Ollama models."""
    try:
        models = ollama_client.list()
        return {"models": [model["name"] for model in models["models"]]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/pull")
async def pull_model(request: dict, background_tasks: BackgroundTasks):
    """Pull a model in the background."""
    model_name = request.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name required")

    background_tasks.add_task(ollama_client.pull, model_name)
    return {"message": f"Pulling model {model_name}"}
```

#### Main Application

```python
# mcbot01/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from mcbot01.api.routes import router
from mcbot01.database import create_tables

app = FastAPI(title="MCBot01", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api")

# Static files for built frontend
app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")

@app.on_event("startup")
async def startup_event():
    create_tables()

if __name__ == "__main__":
    uvicorn.run("mcbot01.main:app", host="0.0.0.0", port=8000, reload=True)
```

### Frontend (Next.js)

#### Dependencies

```json
// package.json
{
  "name": "mcbot01-frontend",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "next": "^13.5.0",
    "@headlessui/react": "^1.7.17",
    "lucide-react": "^0.294.0",
    "axios": "^1.6.0"
  }
}
```

#### Chat Interface Component

```tsx
// components/ChatInterface.tsx
import { useState, useEffect, useRef } from 'react';
import axios from 'axios';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('llama2');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          messages: [...messages, userMessage]
        })
      });

      if (!response.ok) throw new Error('Failed to get response');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = '';

      while (true) {
        const { done, value } = await reader!.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.message?.content) {
                assistantMessage += data.message.content;
                setMessages(prev => {
                  const newMessages = [...prev];
                  if (newMessages[newMessages.length - 1]?.role === 'assistant') {
                    newMessages[newMessages.length - 1].content = assistantMessage;
                  } else {
                    newMessages.push({ role: 'assistant', content: assistantMessage });
                  }
                  return newMessages;
                });
              }
            } catch (e) {
              // Handle parsing errors
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b p-4">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">MCBot01</h1>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="px-3 py-1 border rounded"
          >
            <option value="llama2">Llama 2</option>
            <option value="codellama">Code Llama</option>
            <option value="mistral">Mistral</option>
          </select>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                message.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white border'
              }`}
            >
              {message.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white border px-4 py-2 rounded-lg">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bg-white border-t p-4">
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
```

### Database Management

```python
# mcbot01/database.py
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mcbot01.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    from mcbot01.models import Base
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## Deployment

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcbot01:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ollama_data:/root/.ollama
    environment:
      - DATABASE_URL=sqlite:///./data/mcbot01.db
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/kliewerdaniel/mcbot01.git
cd mcbot01

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install

# Start Ollama service
ollama serve

# Pull required models
ollama pull llama2
ollama pull codellama

# Start the backend
cd .. && python -m mcbot01.main

# In another terminal, start the frontend
cd frontend && npm run dev
```

## Key Features

- **Real-time Streaming**: Server-sent events for live response streaming
- **Model Management**: Easy switching between different Ollama models
- **Error Handling**: Robust timeout and error recovery mechanisms
- **Persistence**: Conversation history and configuration storage
- **Extensible**: Clean architecture for adding new features

## Use Cases

- **Research Assistant**: Interactive AI research and analysis
- **Code Review**: AI-powered code review and suggestions
- **Documentation**: Automated documentation generation
- **Prototyping**: Rapid AI feature prototyping

## Integration with Other Systems

MCBot01 serves as the foundation for more complex applications:

- **GraphRAG Integration**: Add vector search and knowledge retrieval
- **MCP Protocol**: Enable multi-agent coordination
- **Plugin System**: Extend functionality with custom plugins
- **API Extensions**: Build REST APIs on top of the chat interface

This template eliminates the repetitive work of setting up local AI infrastructure, allowing developers to focus on building innovative features on top of a solid foundation.