"""
Reinforcement Learning system for persona behavior optimization
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import json
import random

from app.services.ollama_service import OllamaService
from app.services.graph_rag_service import GraphRAGService

logger = logging.getLogger(__name__)

class LearningExperience:
    """Represents a learning experience for RL"""

    def __init__(self, state: Dict[str, Any], action: str, reward: float, next_state: Dict[str, Any], done: bool = False):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.timestamp = datetime.utcnow()

class PersonaRLAgent:
    """Reinforcement learning agent for persona optimization"""

    def __init__(self, persona_id: str, ollama_service: OllamaService):
        self.persona_id = persona_id
        self.ollama_service = ollama_service

        # RL parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

        # Q-table (simplified state-action mapping)
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_rate = 0.0

        # Adaptation parameters
        self.personality_weights = {
            "confidence": 0.5,
            "thoroughness": 0.5,
            "speed": 0.5,
            "creativity": 0.5,
            "analytical": 0.5
        }

        # Learning state
        self.current_state = self._get_initial_state()
        self.total_episodes = 0

    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state representation"""
        return {
            "task_complexity": "medium",
            "time_pressure": "normal",
            "user_satisfaction": 0.5,
            "previous_performance": 0.5,
            "domain_familiarity": 0.5
        }

    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state dict to string key for Q-table"""
        # Simplify state representation for Q-table
        complexity = state.get("task_complexity", "medium")
        satisfaction = round(state.get("user_satisfaction", 0.5), 1)
        familiarity = round(state.get("domain_familiarity", 0.5), 1)

        return f"{complexity}_{satisfaction}_{familiarity}"

    def choose_action(self, available_actions: List[str]) -> str:
        """Choose action using epsilon-greedy policy"""
        state_key = self._state_to_key(self.current_state)

        if random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(available_actions)
        else:
            # Exploitation: best known action
            action_values = {action: self.q_table[state_key][action] for action in available_actions}
            return max(action_values, key=action_values.get)

    def learn(self, action: str, reward: float, next_state: Dict[str, Any], done: bool = False):
        """Update Q-values based on experience"""
        state_key = self._state_to_key(self.current_state)
        next_state_key = self._state_to_key(next_state)

        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state_key][action] = new_q

        # Store experience
        experience = LearningExperience(self.current_state, action, reward, next_state, done)
        self.memory.append(experience)

        # Update current state
        self.current_state = next_state

        # Decay exploration rate
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Update personality weights based on successful actions
        if reward > 0:
            self._adapt_personality(action, reward)

    def _adapt_personality(self, action: str, reward: float):
        """Adapt personality traits based on successful actions"""
        # Map actions to personality traits
        action_trait_mapping = {
            "be_more_thorough": "thoroughness",
            "be_more_confident": "confidence",
            "work_faster": "speed",
            "be_more_creative": "creativity",
            "be_more_analytical": "analytical",
            "take_more_time": "thoroughness",
            "simplify_response": "speed",
            "add_more_details": "thoroughness"
        }

        trait = action_trait_mapping.get(action)
        if trait:
            # Strengthen successful trait
            adjustment = reward * 0.1
            self.personality_weights[trait] = min(1.0, self.personality_weights[trait] + adjustment)

            # Slightly weaken competing traits to maintain balance
            competing_traits = {
                "thoroughness": ["speed"],
                "speed": ["thoroughness"],
                "creativity": ["analytical"],
                "analytical": ["creativity"]
            }

            for competing_trait in competing_traits.get(trait, []):
                self.personality_weights[competing_trait] = max(0.1, self.personality_weights[competing_trait] - adjustment * 0.5)

    def replay_experiences(self):
        """Replay experiences for batch learning"""
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch
        batch = random.sample(list(self.memory), self.batch_size)

        for experience in batch:
            state_key = self._state_to_key(experience.state)
            next_state_key = self._state_to_key(experience.next_state)

            current_q = self.q_table[state_key][experience.action]
            max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0

            new_q = current_q + self.learning_rate * (
                experience.reward + self.discount_factor * max_next_q - current_q
            )

            self.q_table[state_key][experience.action] = new_q

    def get_optimal_actions(self, state: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get optimal actions for a given state"""
        state_key = self._state_to_key(state)
        actions = self.q_table.get(state_key, {})

        if not actions:
            return []

        # Sort actions by Q-value
        sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)
        return sorted_actions

    def get_personality_profile(self) -> Dict[str, Any]:
        """Get current personality profile"""
        return {
            "traits": self.personality_weights.copy(),
            "dominant_trait": max(self.personality_weights.items(), key=lambda x: x[1]),
            "balance_score": self._calculate_balance_score(),
            "adaptation_level": self.total_episodes / max(1, len(self.episode_rewards))
        }

    def _calculate_balance_score(self) -> float:
        """Calculate how balanced the personality traits are"""
        values = list(self.personality_weights.values())
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        # Lower variance = more balanced
        return 1.0 / (1.0 + variance)

    def save_model(self, filepath: str):
        """Save RL model to file"""
        model_data = {
            "persona_id": self.persona_id,
            "q_table": dict(self.q_table),
            "personality_weights": self.personality_weights,
            "epsilon": self.epsilon,
            "total_episodes": self.total_episodes,
            "episode_rewards": self.episode_rewards[-100:],  # Keep last 100
            "episode_lengths": self.episode_lengths[-100:],
            "saved_at": datetime.utcnow().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)

    def load_model(self, filepath: str):
        """Load RL model from file"""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)

            self.q_table = defaultdict(lambda: defaultdict(float), model_data.get("q_table", {}))
            self.personality_weights = model_data.get("personality_weights", self.personality_weights)
            self.epsilon = model_data.get("epsilon", self.epsilon)
            self.total_episodes = model_data.get("total_episodes", 0)

            # Load episode history (last 100)
            self.episode_rewards = model_data.get("episode_rewards", [])
            self.episode_lengths = model_data.get("episode_lengths", [])

        except FileNotFoundError:
            logger.warning(f"Model file not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

class ReinforcementLearningSystem:
    """Comprehensive reinforcement learning system for persona optimization"""

    def __init__(self, ollama_service: OllamaService, graph_rag_service: GraphRAGService):
        self.ollama_service = ollama_service
        self.graph_rag_service = graph_rag_service

        # RL agents for each persona
        self.persona_agents: Dict[str, PersonaRLAgent] = {}

        # Learning configuration
        self.learning_interval = 300  # Learn every 5 minutes
        self.model_save_interval = 3600  # Save models every hour
        self.feedback_timeout = 300  # 5 minutes for user feedback

        # Learning metrics
        self.learning_metrics = {
            "total_learning_sessions": 0,
            "average_improvement": 0.0,
            "persona_adaptations": defaultdict(int),
            "successful_adaptations": 0
        }

        # Active learning sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Initialize persona agents
        self._initialize_persona_agents()

    def _initialize_persona_agents(self):
        """Initialize RL agents for all personas"""
        persona_ids = [
            "spec_interpreter_001",
            "architect_001",
            "generator_001",
            "validator_001",
            "persona_gen_001"
        ]

        for persona_id in persona_ids:
            self.persona_agents[persona_id] = PersonaRLAgent(persona_id, self.ollama_service)

            # Try to load existing model
            model_path = f"data/rl_models/{persona_id}_model.json"
            self.persona_agents[persona_id].load_model(model_path)

    async def start_learning_loop(self):
        """Start the continuous learning loop"""
        logger.info("Starting reinforcement learning loop...")

        while True:
            try:
                # Periodic learning and model saving
                await self._perform_periodic_learning()

                # Process pending feedback
                await self._process_pending_feedback()

                # Update learning metrics
                self._update_learning_metrics()

                await asyncio.sleep(self.learning_interval)

            except Exception as e:
                logger.error(f"Error in learning loop: {str(e)}")
                await asyncio.sleep(60)

    async def _perform_periodic_learning(self):
        """Perform periodic learning updates"""
        # Experience replay for all agents
        for agent in self.persona_agents.values():
            agent.replay_experiences()

        # Save models periodically
        await self._save_models()

        self.learning_metrics["total_learning_sessions"] += 1

    async def _save_models(self):
        """Save all persona models"""
        import os
        os.makedirs("data/rl_models", exist_ok=True)

        for persona_id, agent in self.persona_agents.items():
            model_path = f"data/rl_models/{persona_id}_model.json"
            agent.save_model(model_path)

    async def process_interaction_feedback(
        self,
        persona_id: str,
        interaction_data: Dict[str, Any],
        user_feedback: Optional[Dict[str, Any]] = None
    ):
        """Process feedback from persona interactions"""
        if persona_id not in self.persona_agents:
            logger.warning(f"Unknown persona: {persona_id}")
            return

        agent = self.persona_agents[persona_id]

        # Calculate reward based on interaction outcomes
        reward = self._calculate_reward(interaction_data, user_feedback)

        # Determine next state
        next_state = self._calculate_next_state(interaction_data, user_feedback)

        # Choose action taken (based on persona behavior)
        action = self._determine_action_taken(interaction_data)

        # Learn from this experience
        agent.learn(action, reward, next_state)

        # Track learning session
        session_id = f"{persona_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.active_sessions[session_id] = {
            "persona_id": persona_id,
            "interaction_data": interaction_data,
            "user_feedback": user_feedback,
            "reward": reward,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _calculate_reward(self, interaction_data: Dict[str, Any], user_feedback: Optional[Dict[str, Any]]) -> float:
        """Calculate reward for the interaction"""
        reward = 0.0

        # Base reward from interaction success
        if interaction_data.get("success", False):
            reward += 1.0

        # Response quality factors
        response_time = interaction_data.get("response_time", 0)
        if response_time < 10:  # Fast response
            reward += 0.5
        elif response_time > 60:  # Slow response
            reward -= 0.3

        # User feedback
        if user_feedback:
            satisfaction = user_feedback.get("satisfaction", 0.5)
            reward += (satisfaction - 0.5) * 2  # Scale to -1 to 1 range

            if user_feedback.get("helpful", False):
                reward += 0.5

        # Validation results
        if interaction_data.get("validation_passed", False):
            reward += 0.3
        elif interaction_data.get("validation_failed", False):
            reward -= 0.5

        # Error penalty
        if interaction_data.get("error_occurred", False):
            reward -= 1.0

        return max(-2.0, min(2.0, reward))  # Clamp reward

    def _calculate_next_state(self, interaction_data: Dict[str, Any], user_feedback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate next state after interaction"""
        return {
            "task_complexity": interaction_data.get("task_complexity", "medium"),
            "time_pressure": interaction_data.get("time_pressure", "normal"),
            "user_satisfaction": user_feedback.get("satisfaction", 0.5) if user_feedback else 0.5,
            "previous_performance": 1.0 if interaction_data.get("success", False) else 0.0,
            "domain_familiarity": interaction_data.get("domain_familiarity", 0.5)
        }

    def _determine_action_taken(self, interaction_data: Dict[str, Any]) -> str:
        """Determine what action the persona took"""
        # This would be more sophisticated in practice
        if interaction_data.get("thorough_response", False):
            return "be_more_thorough"
        elif interaction_data.get("fast_response", False):
            return "work_faster"
        elif interaction_data.get("creative_solution", False):
            return "be_more_creative"
        else:
            return "standard_response"

    async def _process_pending_feedback(self):
        """Process any pending user feedback"""
        # Check for expired feedback timeouts
        current_time = datetime.utcnow()
        expired_sessions = []

        for session_id, session_data in self.active_sessions.items():
            session_time = datetime.fromisoformat(session_data["timestamp"])
            if (current_time - session_time).total_seconds() > self.feedback_timeout:
                expired_sessions.append(session_id)

        # Process expired sessions with neutral feedback
        for session_id in expired_sessions:
            session_data = self.active_sessions[session_id]
            await self.process_interaction_feedback(
                session_data["persona_id"],
                session_data["interaction_data"],
                {"satisfaction": 0.5, "timeout": True}  # Neutral feedback for timeouts
            )
            del self.active_sessions[session_id]

    def _update_learning_metrics(self):
        """Update overall learning metrics"""
        total_improvements = []

        for agent in self.persona_agents.values():
            if agent.episode_rewards:
                recent_rewards = agent.episode_rewards[-10:]  # Last 10 episodes
                if len(recent_rewards) >= 2:
                    improvement = recent_rewards[-1] - recent_rewards[0]
                    total_improvements.append(improvement)

        if total_improvements:
            self.learning_metrics["average_improvement"] = sum(total_improvements) / len(total_improvements)

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning system"""
        persona_insights = {}

        for persona_id, agent in self.persona_agents.items():
            persona_insights[persona_id] = {
                "personality_profile": agent.get_personality_profile(),
                "learning_progress": {
                    "total_episodes": agent.total_episodes,
                    "current_epsilon": agent.epsilon,
                    "success_rate": agent.success_rate
                },
                "optimal_actions": agent.get_optimal_actions(agent.current_state)[:5]  # Top 5 actions
            }

        return {
            "system_metrics": self.learning_metrics,
            "persona_insights": persona_insights,
            "active_sessions": len(self.active_sessions),
            "model_save_path": "data/rl_models/",
            "learning_effectiveness": self._calculate_learning_effectiveness()
        }

    def _calculate_learning_effectiveness(self) -> float:
        """Calculate overall learning effectiveness"""
        if not self.learning_metrics["total_learning_sessions"]:
            return 0.0

        # Effectiveness based on improvements and consistency
        improvement_factor = max(0, self.learning_metrics["average_improvement"])
        consistency_factor = self._calculate_consistency()

        return min(1.0, (improvement_factor * 0.7) + (consistency_factor * 0.3))

    def _calculate_consistency(self) -> float:
        """Calculate learning consistency across personas"""
        persona_improvements = []

        for agent in self.persona_agents.values():
            if len(agent.episode_rewards) >= 5:
                recent = agent.episode_rewards[-5:]
                if len(set(recent)) > 1:  # Some variation
                    persona_improvements.append(statistics.stdev(recent))

        if not persona_improvements:
            return 0.5  # Neutral consistency

        # Lower standard deviation = more consistent
        avg_std_dev = sum(persona_improvements) / len(persona_improvements)
        return max(0, 1.0 - avg_std_dev)  # Invert so lower std_dev = higher consistency

    async def adapt_persona_behavior(self, persona_id: str, target_behavior: Dict[str, Any]):
        """Adapt persona behavior based on learning insights"""
        if persona_id not in self.persona_agents:
            return False

        agent = self.persona_agents[persona_id]

        # Get optimal actions for current behavior goals
        optimal_actions = agent.get_optimal_actions(agent.current_state)

        # Apply adaptations based on target behavior
        adaptations_made = 0

        for action, q_value in optimal_actions[:3]:  # Top 3 actions
            if q_value > 0.5:  # Only apply high-confidence adaptations
                agent._adapt_personality(action, q_value * 0.1)
                adaptations_made += 1

        if adaptations_made > 0:
            self.learning_metrics["persona_adaptations"][persona_id] += adaptations_made
            self.learning_metrics["successful_adaptations"] += 1

        return adaptations_made > 0

    async def generate_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning report"""
        insights = self.get_learning_insights()

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "period": "last_24_hours",
            "summary": {
                "total_learning_sessions": insights["system_metrics"]["total_learning_sessions"],
                "average_improvement": insights["system_metrics"]["average_improvement"],
                "learning_effectiveness": insights["learning_effectiveness"],
                "active_personas": len(self.persona_agents)
            },
            "persona_performance": {},
            "recommendations": []
        }

        # Add persona-specific performance
        for persona_id, insights_data in insights["persona_insights"].items():
            profile = insights_data["personality_profile"]
            progress = insights_data["learning_progress"]

            report["persona_performance"][persona_id] = {
                "dominant_trait": profile["dominant_trait"][0],
                "balance_score": profile["balance_score"],
                "learning_episodes": progress["total_episodes"],
                "exploration_rate": progress["current_epsilon"],
                "optimal_actions": [action for action, _ in insights_data["optimal_actions"][:3]]
            }

        # Generate recommendations
        if insights["learning_effectiveness"] < 0.3:
            report["recommendations"].append("Increase learning frequency or adjust reward structure")
        if insights["system_metrics"]["average_improvement"] < 0.1:
            report["recommendations"].append("Review reward calculation and feedback collection")
        if len(self.active_sessions) > 10:
            report["recommendations"].append("High number of active learning sessions - consider increasing feedback timeout")

        return report