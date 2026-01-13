"""
Feedback Loop system for continuous persona improvement
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

from app.services.reinforcement_learning import ReinforcementLearningSystem
from app.services.graph_rag_service import GraphRAGService
from app.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

class FeedbackCollector:
    """Collects and processes user feedback"""

    def __init__(self):
        self.feedback_queue: asyncio.Queue = asyncio.Queue()
        self.feedback_history: List[Dict[str, Any]] = []
        self.feedback_types = {
            "rating": "numerical_rating",
            "thumbs_up_down": "binary_feedback",
            "text_feedback": "textual_feedback",
            "comparison": "comparative_feedback",
            "correction": "error_correction"
        }

    async def collect_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Collect feedback and return feedback ID"""
        feedback_id = f"fb_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(str(feedback_data)) % 10000}"

        enriched_feedback = {
            "feedback_id": feedback_id,
            "collected_at": datetime.utcnow().isoformat(),
            "feedback_data": feedback_data,
            "processed": False
        }

        await self.feedback_queue.put(enriched_feedback)
        self.feedback_history.append(enriched_feedback)

        return feedback_id

    async def get_pending_feedback(self) -> List[Dict[str, Any]]:
        """Get all pending feedback items"""
        pending = []
        while not self.feedback_queue.empty():
            try:
                feedback = self.feedback_queue.get_nowait()
                if not feedback.get("processed", False):
                    pending.append(feedback)
            except asyncio.QueueEmpty:
                break
        return pending

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback collection statistics"""
        if not self.feedback_history:
            return {"total_feedback": 0}

        total_feedback = len(self.feedback_history)
        processed_feedback = sum(1 for fb in self.feedback_history if fb.get("processed", False))

        # Feedback type distribution
        type_counts = defaultdict(int)
        for feedback in self.feedback_history:
            fb_type = feedback.get("feedback_data", {}).get("type", "unknown")
            type_counts[fb_type] += 1

        # Average rating (if available)
        ratings = []
        for feedback in self.feedback_history:
            rating = feedback.get("feedback_data", {}).get("rating")
            if rating is not None:
                ratings.append(float(rating))

        avg_rating = sum(ratings) / len(ratings) if ratings else None

        return {
            "total_feedback": total_feedback,
            "processed_feedback": processed_feedback,
            "pending_feedback": total_feedback - processed_feedback,
            "type_distribution": dict(type_counts),
            "average_rating": avg_rating,
            "feedback_rate": total_feedback / max(1, (datetime.utcnow() - datetime.fromisoformat(self.feedback_history[0]["collected_at"])).days)
        }

class InteractionAnalyzer:
    """Analyzes interactions to extract learning insights"""

    def __init__(self, ollama_service: OllamaService):
        self.ollama_service = ollama_service

    async def analyze_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single interaction for learning opportunities"""
        analysis_prompt = f"""Analyze this AI-persona interaction for learning opportunities:

Interaction Data:
{json.dumps(interaction_data, indent=2)}

Please provide:
1. **Success Factors**: What made this interaction successful?
2. **Improvement Areas**: What could be improved?
3. **Learning Patterns**: What patterns or behaviors should be reinforced?
4. **Behavioral Insights**: How did the persona behavior affect the outcome?
5. **Contextual Factors**: How did external factors influence the interaction?

Be specific and actionable in your analysis."""

        analysis = await self.ollama_service.generate_response(
            prompt=analysis_prompt,
            system_prompt="You are an interaction analyst specializing in AI behavior analysis and improvement recommendations.",
            temperature=0.2
        )

        return {
            "interaction_id": interaction_data.get("id", "unknown"),
            "analysis": analysis,
            "key_insights": self._extract_key_insights(analysis),
            "recommendations": self._extract_recommendations(analysis),
            "analyzed_at": datetime.utcnow().isoformat()
        }

    async def compare_interactions(self, interaction1: Dict[str, Any], interaction2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two interactions to identify differences and improvements"""
        comparison_prompt = f"""Compare these two AI-persona interactions:

Interaction 1:
{json.dumps(interaction1, indent=2)}

Interaction 2:
{json.dumps(interaction2, indent=2)}

Analyze:
1. **Performance Differences**: How did the outcomes differ?
2. **Behavioral Variations**: What behavioral differences contributed to the results?
3. **Contextual Impacts**: How did context affect each interaction?
4. **Lessons Learned**: What can be learned from the comparison?
5. **Improvement Strategies**: How to replicate successful patterns?

Provide specific, comparative insights."""

        comparison = await self.ollama_service.generate_response(
            prompt=comparison_prompt,
            system_prompt="You are a comparative analyst specializing in interaction pattern analysis and optimization strategies.",
            temperature=0.2
        )

        return {
            "comparison": comparison,
            "performance_diff": self._analyze_performance_difference(interaction1, interaction2),
            "behavioral_insights": self._extract_behavioral_insights(comparison),
            "compared_at": datetime.utcnow().isoformat()
        }

    def _extract_key_insights(self, analysis: str) -> List[str]:
        """Extract key insights from analysis text"""
        insights = []
        lines = analysis.split('\n')

        for line in lines:
            line = line.strip()
            # Look for insight patterns
            if any(keyword in line.lower() for keyword in ['insight', 'key', 'important', 'critical']):
                insights.append(line)

        return insights[:5]  # Limit to top 5

    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract recommendations from analysis text"""
        recommendations = []
        lines = analysis.split('\n')

        for line in lines:
            line = line.strip()
            # Look for recommendation patterns
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'improve']):
                recommendations.append(line)

        return recommendations[:5]  # Limit to top 5

    def _analyze_performance_difference(self, interaction1: Dict[str, Any], interaction2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance differences between interactions"""
        perf1 = interaction1.get("performance_metrics", {})
        perf2 = interaction2.get("performance_metrics", {})

        differences = {}

        for metric in set(perf1.keys()) | set(perf2.keys()):
            val1 = perf1.get(metric, 0)
            val2 = perf2.get(metric, 0)
            if val1 != val2:
                differences[metric] = {
                    "interaction1": val1,
                    "interaction2": val2,
                    "difference": val2 - val1,
                    "improvement": val2 > val1
                }

        return differences

    def _extract_behavioral_insights(self, comparison: str) -> List[str]:
        """Extract behavioral insights from comparison"""
        insights = []
        lines = comparison.split('\n')

        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['behavior', 'approach', 'strategy', 'pattern']):
                insights.append(line)

        return insights[:5]

class PersonalityAdapter:
    """Adapts persona personality traits based on feedback"""

    def __init__(self, ollama_service: OllamaService, rl_system: ReinforcementLearningSystem):
        self.ollama_service = ollama_service
        self.rl_system = rl_system

        # Adaptation parameters
        self.adaptation_sensitivity = 0.1
        self.min_trait_value = 0.1
        self.max_trait_value = 1.0
        self.balance_weight = 0.3

    async def adapt_personality(self, persona_id: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt persona personality based on feedback"""
        # Get current personality profile
        current_profile = await self._get_current_personality(persona_id)

        # Analyze feedback for personality implications
        adaptation_analysis = await self._analyze_feedback_for_adaptation(feedback_data, current_profile)

        # Calculate trait adjustments
        trait_adjustments = self._calculate_trait_adjustments(adaptation_analysis, current_profile)

        # Apply balance constraints
        balanced_adjustments = self._apply_balance_constraints(trait_adjustments, current_profile)

        # Update personality
        updated_profile = await self._apply_personality_updates(persona_id, balanced_adjustments)

        return {
            "persona_id": persona_id,
            "original_profile": current_profile,
            "adaptations_made": balanced_adjustments,
            "updated_profile": updated_profile,
            "adaptation_reasoning": adaptation_analysis,
            "adapted_at": datetime.utcnow().isoformat()
        }

    async def _get_current_personality(self, persona_id: str) -> Dict[str, Any]:
        """Get current personality profile for persona"""
        if persona_id in self.rl_system.persona_agents:
            agent = self.rl_system.persona_agents[persona_id]
            return agent.get_personality_profile()
        else:
            # Return default profile
            return {
                "traits": {
                    "confidence": 0.5,
                    "thoroughness": 0.5,
                    "speed": 0.5,
                    "creativity": 0.5,
                    "analytical": 0.5
                },
                "dominant_trait": ("confidence", 0.5),
                "balance_score": 0.8,
                "adaptation_level": 0.0
            }

    async def _analyze_feedback_for_adaptation(self, feedback_data: Dict[str, Any], current_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feedback to determine personality adaptation"""
        analysis_prompt = f"""Analyze this feedback for personality trait adaptation:

Feedback Data:
{json.dumps(feedback_data, indent=2)}

Current Personality Profile:
{json.dumps(current_profile, indent=2)}

Determine how this feedback should influence personality traits:
1. Which traits should be strengthened or weakened?
2. What behavioral changes are indicated?
3. How should the persona adapt its approach?

Be specific about trait adjustments and their rationale."""

        analysis = await self.ollama_service.generate_response(
            prompt=analysis_prompt,
            system_prompt="You are a personality adaptation specialist who analyzes feedback to determine optimal trait adjustments.",
            temperature=0.2
        )

        return {
            "analysis": analysis,
            "recommended_adjustments": self._extract_trait_adjustments(analysis),
            "adaptation_confidence": 0.8
        }

    def _calculate_trait_adjustments(self, adaptation_analysis: Dict[str, Any], current_profile: Dict[str, Any]) -> Dict[str, float]:
        """Calculate specific trait adjustments"""
        adjustments = {}
        recommended = adaptation_analysis.get("recommended_adjustments", [])

        for trait, change in recommended.items():
            if trait in current_profile.get("traits", {}):
                current_value = current_profile["traits"][trait]
                adjustment = change * self.adaptation_sensitivity
                new_value = max(self.min_trait_value, min(self.max_trait_value, current_value + adjustment))
                adjustments[trait] = new_value - current_value

        return adjustments

    def _apply_balance_constraints(self, adjustments: Dict[str, float], current_profile: Dict[str, Any]) -> Dict[str, float]:
        """Apply balance constraints to prevent extreme trait values"""
        balanced = {}
        current_traits = current_profile.get("traits", {})

        for trait, adjustment in adjustments.items():
            current_value = current_traits.get(trait, 0.5)
            new_value = current_value + adjustment

            # Apply balance constraints
            if new_value < self.min_trait_value:
                balanced[trait] = self.min_trait_value - current_value
            elif new_value > self.max_trait_value:
                balanced[trait] = self.max_trait_value - current_value
            else:
                balanced[trait] = adjustment

        return balanced

    async def _apply_personality_updates(self, persona_id: str, adjustments: Dict[str, float]) -> Dict[str, Any]:
        """Apply personality updates to RL agent"""
        if persona_id in self.rl_system.persona_agents:
            agent = self.rl_system.persona_agents[persona_id]

            for trait, adjustment in adjustments.items():
                if trait in agent.personality_weights:
                    agent.personality_weights[trait] += adjustment
                    agent.personality_weights[trait] = max(self.min_trait_value,
                                                          min(self.max_trait_value,
                                                              agent.personality_weights[trait]))

            return agent.get_personality_profile()
        else:
            # Return adjusted profile without RL agent
            base_profile = {
                "traits": {
                    "confidence": 0.5,
                    "thoroughness": 0.5,
                    "speed": 0.5,
                    "creativity": 0.5,
                    "analytical": 0.5
                }
            }

            for trait, adjustment in adjustments.items():
                if trait in base_profile["traits"]:
                    base_profile["traits"][trait] += adjustment
                    base_profile["traits"][trait] = max(self.min_trait_value,
                                                       min(self.max_trait_value,
                                                           base_profile["traits"][trait]))

            return base_profile

    def _extract_trait_adjustments(self, analysis: str) -> Dict[str, float]:
        """Extract trait adjustments from analysis text"""
        adjustments = {}
        lines = analysis.split('\n')

        trait_keywords = {
            "confidence": ["confiden", "assertive"],
            "thoroughness": ["thorough", "detailed", "careful"],
            "speed": ["fast", "quick", "efficient"],
            "creativity": ["creative", "innovative", "original"],
            "analytical": ["analytical", "logical", "systematic"]
        }

        for line in lines:
            line_lower = line.lower()
            for trait, keywords in trait_keywords.items():
                if any(keyword in line_lower for keyword in keywords):
                    if "increase" in line_lower or "strengthen" in line_lower or "more" in line_lower:
                        adjustments[trait] = 0.1
                    elif "decrease" in line_lower or "weaken" in line_lower or "less" in line_lower:
                        adjustments[trait] = -0.1

        return adjustments

class FeedbackLoopSystem:
    """Complete feedback loop system for continuous persona improvement"""

    def __init__(self, ollama_service: OllamaService, rl_system: ReinforcementLearningSystem, graph_rag_service: GraphRAGService):
        self.ollama_service = ollama_service
        self.rl_system = rl_system
        self.graph_rag_service = graph_rag_service

        # Initialize components
        self.feedback_collector = FeedbackCollector()
        self.interaction_analyzer = InteractionAnalyzer(ollama_service)
        self.personality_adapter = PersonalityAdapter(ollama_service, rl_system)

        # Learning parameters
        self.learning_interval = 600  # 10 minutes
        self.feedback_processing_batch_size = 10
        self.min_feedback_for_adaptation = 3

        # Learning state
        self.learning_active = False
        self.last_learning_cycle = None

    async def start_feedback_loop(self):
        """Start the continuous feedback loop"""
        self.learning_active = True
        logger.info("Starting feedback loop system...")

        while self.learning_active:
            try:
                await self._process_feedback_batch()
                await self._perform_learning_cycle()
                await self._consolidate_learning()

                await asyncio.sleep(self.learning_interval)

            except Exception as e:
                logger.error(f"Error in feedback loop: {str(e)}")
                await asyncio.sleep(60)

    async def _process_feedback_batch(self):
        """Process a batch of pending feedback"""
        pending_feedback = await self.feedback_collector.get_pending_feedback()

        if len(pending_feedback) < self.feedback_processing_batch_size:
            return  # Wait for more feedback

        logger.info(f"Processing {len(pending_feedback)} feedback items...")

        for feedback_item in pending_feedback:
            try:
                await self._process_single_feedback(feedback_item)
                feedback_item["processed"] = True
                feedback_item["processed_at"] = datetime.utcnow().isoformat()

            except Exception as e:
                logger.error(f"Error processing feedback {feedback_item.get('feedback_id')}: {str(e)}")

    async def _process_single_feedback(self, feedback_item: Dict[str, Any]):
        """Process a single feedback item"""
        feedback_data = feedback_item["feedback_data"]
        feedback_type = feedback_data.get("type", "unknown")

        # Extract interaction context
        interaction_data = feedback_data.get("interaction_data", {})
        persona_id = feedback_data.get("persona_id", "unknown")

        # Analyze interaction
        if interaction_data:
            analysis = await self.interaction_analyzer.analyze_interaction(interaction_data)

            # Store analysis in knowledge graph
            await self.graph_rag_service.add_knowledge(
                content=f"Interaction Analysis: {analysis['interaction_id']}\n{analysis['analysis']}",
                metadata={
                    "type": "interaction_analysis",
                    "feedback_id": feedback_item["feedback_id"],
                    "persona_id": persona_id,
                    "key_insights": analysis["key_insights"],
                    "learning_opportunity": True
                },
                node_type="learning_interaction"
            )

        # Adapt personality if sufficient feedback
        feedback_history = [fb for fb in self.feedback_collector.feedback_history
                          if fb.get("feedback_data", {}).get("persona_id") == persona_id]

        if len(feedback_history) >= self.min_feedback_for_adaptation:
            adaptation_result = await self.personality_adapter.adapt_personality(persona_id, feedback_data)

            # Store adaptation in knowledge graph
            await self.graph_rag_service.add_knowledge(
                content=f"Personality Adaptation: {persona_id}\n{json.dumps(adaptation_result, indent=2)}",
                metadata={
                    "type": "personality_adaptation",
                    "persona_id": persona_id,
                    "feedback_id": feedback_item["feedback_id"],
                    "adaptations_made": len(adaptation_result.get("adaptations_made", {})),
                    "learning_opportunity": True
                },
                node_type="learning_adaptation"
            )

        # Update RL system
        await self.rl_system.process_interaction_feedback(
            persona_id,
            interaction_data,
            feedback_data
        )

    async def _perform_learning_cycle(self):
        """Perform a complete learning cycle"""
        current_time = datetime.utcnow()

        # Throttle learning cycles
        if self.last_learning_cycle and (current_time - self.last_learning_cycle).seconds < self.learning_interval:
            return

        self.last_learning_cycle = current_time

        # Generate learning insights
        learning_insights = self.rl_system.get_learning_insights()

        # Analyze recent interactions for patterns
        recent_feedback = [fb for fb in self.feedback_collector.feedback_history[-50:]
                          if fb.get("processed", False)]

        if recent_feedback:
            pattern_analysis = await self._analyze_learning_patterns(recent_feedback)

            # Store pattern analysis
            await self.graph_rag_service.add_knowledge(
                content=f"Learning Pattern Analysis\n{json.dumps(pattern_analysis, indent=2)}",
                metadata={
                    "type": "learning_pattern_analysis",
                    "feedback_items_analyzed": len(recent_feedback),
                    "patterns_identified": len(pattern_analysis.get("patterns", [])),
                    "learning_opportunity": True
                },
                node_type="learning_patterns"
            )

    async def _analyze_learning_patterns(self, feedback_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze learning patterns from feedback"""
        patterns = {
            "successful_interactions": [],
            "common_improvement_areas": [],
            "persona_effectiveness": defaultdict(list),
            "temporal_patterns": []
        }

        for item in feedback_items:
            feedback_data = item.get("feedback_data", {})
            persona_id = feedback_data.get("persona_id", "unknown")
            rating = feedback_data.get("rating", 0.5)

            patterns["persona_effectiveness"][persona_id].append(rating)

            if rating > 0.8:
                patterns["successful_interactions"].append(item)
            elif rating < 0.4:
                patterns["common_improvement_areas"].append(feedback_data.get("comments", ""))

        # Calculate averages
        for persona_id in patterns["persona_effectiveness"]:
            ratings = patterns["persona_effectiveness"][persona_id]
            patterns["persona_effectiveness"][persona_id] = {
                "average_rating": sum(ratings) / len(ratings),
                "total_interactions": len(ratings)
            }

        return dict(patterns)

    async def _consolidate_learning(self):
        """Consolidate learning across the system"""
        # Generate learning report
        learning_report = await self.rl_system.generate_learning_report()

        # Store consolidated learning
        await self.graph_rag_service.add_knowledge(
            content=f"Consolidated Learning Report\n{json.dumps(learning_report, indent=2)}",
            metadata={
                "type": "consolidated_learning",
                "learning_sessions": learning_report["summary"]["total_learning_sessions"],
                "improvement_score": learning_report["summary"]["average_improvement"],
                "active_personas": learning_report["summary"]["active_personas"],
                "learning_opportunity": True
            },
            node_type="consolidated_learning"
        )

    def get_feedback_loop_status(self) -> Dict[str, Any]:
        """Get the current status of the feedback loop system"""
        return {
            "learning_active": self.learning_active,
            "feedback_statistics": self.feedback_collector.get_feedback_statistics(),
            "learning_insights": self.rl_system.get_learning_insights(),
            "last_learning_cycle": self.last_learning_cycle.isoformat() if self.last_learning_cycle else None,
            "system_health": "healthy" if self.learning_active else "inactive"
        }

    async def submit_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Submit feedback to the system"""
        return await self.feedback_collector.collect_feedback(feedback_data)

    def stop_feedback_loop(self):
        """Stop the feedback loop system"""
        self.learning_active = False
        logger.info("Feedback loop system stopped")