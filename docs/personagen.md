# PersonaGen: Autonomous Persona Generation System

## Overview

PersonaGen is an autonomous system that generates sophisticated AI personas with rich psychological profiles, behavioral patterns, and contextual knowledge. It creates consistent, believable characters that can maintain long-term interactions while evolving through reinforcement learning.

**Repository**: `github.com/kliewerdaniel/personagen`

## Problem Statement

AI interactions lack depth and consistency. Chatbots and virtual assistants often feel generic and forgetful, unable to maintain coherent personalities or adapt their behavior based on interaction history. True autonomous agents need rich, evolving personas that feel authentic and responsive.

## Solution Architecture

### Core Components

1. **Persona Engine**
   - Psychological profile generation
   - Behavioral pattern synthesis
   - Memory and context management
   - Evolutionary adaptation

2. **Reinforcement Learning System**
   - Interaction feedback loops
   - Personality trait optimization
   - Behavioral reinforcement
   - Long-term memory consolidation

3. **Context Weaver**
   - Knowledge graph integration
   - Relationship mapping
   - Historical context preservation
   - Narrative coherence maintenance

4. **Validation Framework**
   - Consistency checking
   - Behavioral validation
   - Ethical boundary enforcement
   - Quality assurance

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Interaction   │    │   Persona       │    │   Memory        │
│   Handler       │◄──►│   Engine        │◄──►│   System        │
│                 │    │                 │    │                 │
│ - Input Parsing │    │ - Profile Gen   │    │ - Context Store │
│ - Response Gen  │    │ - Behavior Synth│    │ - Knowledge Graph│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │   RL Trainer    │
                    │                 │
                    │ - Feedback Loop │
                    │ - Optimization  │
                    │ - Adaptation    │
                    └─────────────────┘
```

## Technical Implementation

### Dependencies

```python
# requirements.txt
ollama>=0.1.0
faiss-cpu>=1.7.0
networkx>=3.0
pydantic>=2.0.0
sqlalchemy>=2.0.0
transformers>=4.21.0
torch>=2.0.0
scikit-learn>=1.3.0
```

### Core Data Models

```python
# personagen/models/persona.py
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class PersonalityTrait(BaseModel):
    name: str
    value: float  # 0.0 to 1.0
    description: str

class BehavioralPattern(BaseModel):
    trigger: str
    response_template: str
    frequency: float
    last_used: Optional[datetime]

class KnowledgeNode(BaseModel):
    id: str
    content: str
    connections: List[str]
    importance: float
    last_accessed: datetime

class PersonaProfile(BaseModel):
    id: str
    name: str
    background: str
    personality_traits: List[PersonalityTrait]
    behavioral_patterns: List[BehavioralPattern]
    knowledge_graph: Dict[str, KnowledgeNode]
    creation_date: datetime
    interaction_count: int
    adaptation_score: float
```

### Persona Generation Engine

```python
# personagen/core/generator.py
import ollama
from personagen.models.persona import PersonaProfile, PersonalityTrait
from typing import List

class PersonaGenerator:
    def __init__(self, model_name: str = "llama2"):
        self.client = ollama.Client()
        self.model = model_name

    async def generate_persona(self, theme: str, complexity: int = 5) -> PersonaProfile:
        """Generate a complete persona profile based on theme."""
        # Generate basic profile
        profile_prompt = f"""
        Create a detailed persona profile for theme: {theme}
        Include:
        - Name and background
        - 5-7 personality traits with values (0-1)
        - Key behavioral patterns
        - Knowledge domains

        Format as JSON structure.
        """

        response = self.client.generate(
            model=self.model,
            prompt=profile_prompt,
            format="json"
        )

        profile_data = json.loads(response['response'])

        # Generate behavioral patterns
        behaviors = await self._generate_behavioral_patterns(profile_data)

        # Initialize knowledge graph
        knowledge_graph = await self._initialize_knowledge_graph(profile_data)

        return PersonaProfile(
            id=str(uuid.uuid4()),
            name=profile_data['name'],
            background=profile_data['background'],
            personality_traits=[
                PersonalityTrait(**trait) for trait in profile_data['traits']
            ],
            behavioral_patterns=behaviors,
            knowledge_graph=knowledge_graph,
            creation_date=datetime.now(),
            interaction_count=0,
            adaptation_score=0.5
        )

    async def _generate_behavioral_patterns(self, profile_data: dict) -> List[BehavioralPattern]:
        """Generate behavioral patterns for the persona."""
        behavior_prompt = f"""
        Based on this persona: {profile_data['name']} - {profile_data['background']}

        Generate 10 behavioral patterns in this format:
        - Trigger: "user says X"
        - Response: "persona responds with Y"
        - Frequency: 0.0-1.0
        """

        response = self.client.generate(
            model=self.model,
            prompt=behavior_prompt
        )

        # Parse and structure behavioral patterns
        return self._parse_behavioral_patterns(response['response'])

    async def _initialize_knowledge_graph(self, profile_data: dict) -> Dict[str, KnowledgeNode]:
        """Initialize knowledge graph for the persona."""
        # Create initial knowledge nodes based on background
        nodes = {}

        # Background knowledge
        background_node = KnowledgeNode(
            id="background",
            content=profile_data['background'],
            connections=[],
            importance=1.0,
            last_accessed=datetime.now()
        )
        nodes["background"] = background_node

        # Add domain-specific knowledge
        for domain in profile_data.get('knowledge_domains', []):
            domain_node = KnowledgeNode(
                id=f"domain_{domain.lower()}",
                content=f"Knowledge about {domain}",
                connections=["background"],
                importance=0.8,
                last_accessed=datetime.now()
            )
            nodes[domain_node.id] = domain_node
            background_node.connections.append(domain_node.id)

        return nodes
```

### Reinforcement Learning System

```python
# personagen/core/reinforcement.py
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

class PersonaReinforcementLearner:
    def __init__(self, state_dim: int, action_dim: int):
        self.policy_net = self._build_policy_network(state_dim, action_dim)
        self.optimizer = Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = []

    def _build_policy_network(self, state_dim: int, action_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def select_action(self, state: np.ndarray) -> int:
        """Select action based on current state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            return torch.multinomial(action_probs, 1).item()

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Store transition for training."""
        self.memory.append((state, action, reward, next_state))

    def train_step(self, batch_size: int = 32):
        """Perform one training step."""
        if len(self.memory) < batch_size:
            return

        # Sample batch
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states = zip(*[self.memory[i] for i in batch])

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)

        # Compute loss and update
        action_probs = self.policy_net(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        loss = -log_probs.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory periodically
        if len(self.memory) > 10000:
            self.memory = self.memory[-5000:]
```

### Interaction Handler

```python
# personagen/core/interaction.py
from personagen.models.persona import PersonaProfile
from personagen.core.generator import PersonaGenerator
from personagen.core.reinforcement import PersonaReinforcementLearner
import ollama

class PersonaInteractor:
    def __init__(self, persona: PersonaProfile):
        self.persona = persona
        self.generator = PersonaGenerator()
        self.reinforcement_learner = PersonaReinforcementLearner(
            state_dim=50,  # Personality trait vector size
            action_dim=len(persona.behavioral_patterns)
        )
        self.client = ollama.Client()

    async def respond(self, user_input: str) -> str:
        """Generate response based on persona and input."""
        # Extract conversation state
        state = self._extract_state(user_input)

        # Select behavioral pattern using RL
        action_idx = self.reinforcement_learner.select_action(state)
        selected_pattern = self.persona.behavioral_patterns[action_idx]

        # Generate response using LLM with persona context
        context = self._build_context(user_input, selected_pattern)

        response = self.client.generate(
            model="llama2",
            prompt=context,
            system=self._build_system_prompt()
        )

        # Store interaction for learning
        next_state = self._extract_state(response['response'])
        reward = self._calculate_reward(user_input, response['response'])

        self.reinforcement_learner.store_transition(
            state, action_idx, reward, next_state
        )

        # Update persona knowledge
        self._update_knowledge_graph(user_input, response['response'])

        # Increment interaction count
        self.persona.interaction_count += 1

        return response['response']

    def _extract_state(self, text: str) -> np.ndarray:
        """Extract state vector from text for RL."""
        # Simple bag-of-words approach for now
        # In production, use embeddings
        words = text.lower().split()
        state = np.zeros(50)

        # Map words to personality traits
        for i, trait in enumerate(self.persona.personality_traits):
            if i < 50:
                trait_words = trait.description.lower().split()
                state[i] = len(set(words) & set(trait_words)) / len(trait_words)

        return state

    def _calculate_reward(self, user_input: str, response: str) -> float:
        """Calculate reward for RL based on response quality."""
        # Simple heuristics for now
        reward = 0.0

        # Length appropriateness
        if 10 < len(response) < 500:
            reward += 0.3

        # Persona consistency (check if response matches personality)
        persona_keywords = []
        for trait in self.persona.personality_traits:
            persona_keywords.extend(trait.description.split())

        response_words = set(response.lower().split())
        persona_matches = len(response_words & set(persona_keywords))

        reward += min(persona_matches * 0.1, 0.5)

        # Engagement (questions, etc.)
        if '?' in response:
            reward += 0.2

        return reward

    def _build_context(self, user_input: str, pattern) -> str:
        """Build context for LLM generation."""
        return f"""
        You are {self.persona.name}.
        Background: {self.persona.background}

        Personality traits:
        {chr(10).join([f"- {t.name}: {t.value}" for t in self.persona.personality_traits])}

        Selected behavioral pattern: {pattern.response_template}

        User input: {user_input}

        Respond in character, maintaining consistency with your personality and background.
        """

    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM."""
        return f"""You are role-playing as {self.persona.name}. Stay in character at all times. Your responses should reflect your personality traits and background. Be consistent with previous interactions and maintain the established narrative."""

    def _update_knowledge_graph(self, user_input: str, response: str):
        """Update persona's knowledge graph with new information."""
        # Add new nodes for important information
        # Strengthen connections based on interaction
        pass
```

## Deployment and Usage

### API Interface

```python
# personagen/api/main.py
from fastapi import FastAPI
from personagen.core.interaction import PersonaInteractor
from personagen.models.persona import PersonaProfile

app = FastAPI()
persona_store = {}  # In production, use database

@app.post("/personas")
async def create_persona(theme: str):
    generator = PersonaGenerator()
    persona = await generator.generate_persona(theme)
    persona_store[persona.id] = PersonaInteractor(persona)
    return {"persona_id": persona.id, "name": persona.name}

@app.post("/personas/{persona_id}/interact")
async def interact(persona_id: str, message: str):
    if persona_id not in persona_store:
        return {"error": "Persona not found"}

    interactor = persona_store[persona_id]
    response = await interactor.respond(message)
    return {"response": response, "persona_name": interactor.persona.name}
```

### CLI Tool

```python
# personagen/cli.py
import click
import asyncio
from personagen.core.generator import PersonaGenerator
from personagen.core.interaction import PersonaInteractor

@click.group()
def cli():
    pass

@cli.command()
@click.argument('theme')
@click.option('--interactive', is_flag=True)
def create(theme: str, interactive: bool):
    """Create a new persona."""
    generator = PersonaGenerator()

    async def create_persona():
        persona = await generator.generate_persona(theme)
        interactor = PersonaInteractor(persona)

        if interactive:
            print(f"Created persona: {persona.name}")
            print(f"Background: {persona.background}")
            print("\nStarting interactive session...")

            while True:
                user_input = input("You: ")
                if user_input.lower() in ['quit', 'exit']:
                    break

                response = await interactor.respond(user_input)
                print(f"{persona.name}: {response}")
        else:
            print(f"Created persona: {persona.name}")

    asyncio.run(create_persona())

if __name__ == '__main__':
    cli()
```

## Key Features

- **Autonomous Evolution**: Personas adapt through reinforcement learning
- **Rich Psychology**: Multi-dimensional personality models
- **Memory Persistence**: Knowledge graphs maintain context
- **Behavioral Consistency**: Pattern-based response generation
- **Ethical Boundaries**: Built-in validation and safety measures

## Applications

- **Interactive Fiction**: Dynamic characters that evolve with story
- **Educational Tools**: Personalized learning companions
- **Therapeutic Applications**: Consistent virtual counselors
- **Game NPCs**: Living, adapting non-player characters
- **Research**: Social psychology and AI interaction studies

## Future Enhancements

- Multi-modal personas (voice, appearance)
- Cross-persona relationships and interactions
- Emotional state modeling
- Cultural adaptation
- Long-term memory consolidation

PersonaGen represents the next evolution in AI character creation, moving beyond static chatbots to truly autonomous, evolving digital personalities.