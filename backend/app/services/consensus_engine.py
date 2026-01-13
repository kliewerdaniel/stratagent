"""
Consensus Engine - Multi-agent decision making and conflict resolution
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import statistics
import json

from app.services.ollama_service import OllamaService
from app.services.graph_rag_service import GraphRAGService

logger = logging.getLogger(__name__)

class ConsensusStrategy:
    """Different strategies for achieving consensus"""

    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    QUALITY_WEIGHTED = "quality_weighted"
    EXPERT_ADJUDICATION = "expert_adjudication"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    HYBRID_CONSENSUS = "hybrid_consensus"

class ConsensusEngine:
    """Engine for multi-agent decision making and conflict resolution"""

    def __init__(self, ollama_service: OllamaService, graph_rag_service: GraphRAGService):
        self.ollama_service = ollama_service
        self.graph_rag_service = graph_rag_service

        # Consensus configuration
        self.default_strategy = ConsensusStrategy.HYBRID_CONSENSUS
        self.min_participants = 2
        self.max_iterations = 3
        self.convergence_threshold = 0.8

        # Agent expertise weights (learned over time)
        self.agent_expertise: Dict[str, float] = {}
        self.agent_performance_history: Dict[str, List[float]] = defaultdict(list)

        # Consensus history
        self.consensus_history: List[Dict[str, Any]] = []

    async def achieve_consensus(
        self,
        proposals: List[Dict[str, Any]],
        context: Dict[str, Any],
        strategy: Optional[str] = None,
        domain: str = "general"
    ) -> Dict[str, Any]:
        """Achieve consensus among multiple agent proposals"""
        if len(proposals) < self.min_participants:
            return self._create_single_proposal_result(proposals[0] if proposals else {})

        consensus_id = f"cons_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(str(proposals)) % 10000}"
        strategy = strategy or self.default_strategy

        # Initial consensus attempt
        consensus_result = await self._attempt_consensus(proposals, context, strategy, domain)

        # Iterative refinement if needed
        if consensus_result["agreement_level"] < self.convergence_threshold and strategy == ConsensusStrategy.ITERATIVE_REFINEMENT:
            consensus_result = await self._iterative_refinement(proposals, context, consensus_result, domain)

        # Record consensus outcome
        consensus_record = {
            "consensus_id": consensus_id,
            "strategy": strategy,
            "participants": len(proposals),
            "domain": domain,
            "initial_agreement": consensus_result["agreement_level"],
            "final_agreement": consensus_result["agreement_level"],
            "iterations": 1,
            "outcome": consensus_result["outcome"],
            "timestamp": datetime.utcnow().isoformat()
        }

        self.consensus_history.append(consensus_record)

        # Learn from consensus process
        await self._learn_from_consensus(proposals, consensus_result, domain)

        return consensus_result

    async def _attempt_consensus(
        self,
        proposals: List[Dict[str, Any]],
        context: Dict[str, Any],
        strategy: str,
        domain: str
    ) -> Dict[str, Any]:
        """Attempt to achieve consensus using specified strategy"""

        if strategy == ConsensusStrategy.MAJORITY_VOTE:
            return await self._majority_vote_consensus(proposals, context)
        elif strategy == ConsensusStrategy.WEIGHTED_VOTE:
            return await self._weighted_vote_consensus(proposals, context)
        elif strategy == ConsensusStrategy.QUALITY_WEIGHTED:
            return await self._quality_weighted_consensus(proposals, context)
        elif strategy == ConsensusStrategy.EXPERT_ADJUDICATION:
            return await self._expert_adjudication_consensus(proposals, context, domain)
        elif strategy == ConsensusStrategy.HYBRID_CONSENSUS:
            return await self._hybrid_consensus(proposals, context, domain)
        else:
            return await self._majority_vote_consensus(proposals, context)

    async def _majority_vote_consensus(self, proposals: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simple majority vote consensus"""
        if not proposals:
            return self._create_error_result("No proposals provided")

        # Extract key decisions from proposals
        decisions = [self._extract_decision(p) for p in proposals]
        decision_counts = Counter(decisions)

        # Find majority decision
        majority_decision, count = decision_counts.most_common(1)[0]
        total_votes = len(proposals)
        agreement_level = count / total_votes

        # Find supporting proposals
        supporting_proposals = [p for p, d in zip(proposals, decisions) if d == majority_decision]

        return {
            "outcome": "consensus_achieved" if agreement_level >= 0.5 else "no_consensus",
            "decision": majority_decision,
            "agreement_level": agreement_level,
            "supporting_proposals": len(supporting_proposals),
            "total_proposals": total_votes,
            "conflicting_proposals": total_votes - len(supporting_proposals),
            "consensus_method": "majority_vote",
            "rationale": f"Majority decision with {count}/{total_votes} support"
        }

    async def _weighted_vote_consensus(self, proposals: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Weighted vote based on agent expertise"""
        if not proposals:
            return self._create_error_result("No proposals provided")

        total_weight = 0
        weighted_decisions = defaultdict(float)

        for proposal in proposals:
            agent_id = proposal.get("agent_id", "unknown")
            decision = self._extract_decision(proposal)
            weight = self._get_agent_weight(agent_id)

            weighted_decisions[decision] += weight
            total_weight += weight

        # Find weighted majority
        best_decision = max(weighted_decisions.items(), key=lambda x: x[1])
        decision, weight = best_decision

        # Calculate agreement level (normalized weight)
        agreement_level = weight / total_weight if total_weight > 0 else 0

        return {
            "outcome": "consensus_achieved" if agreement_level >= 0.6 else "qualified_consensus",
            "decision": decision,
            "agreement_level": agreement_level,
            "weighted_support": weight,
            "total_weight": total_weight,
            "consensus_method": "weighted_vote",
            "rationale": f"Weighted decision with {weight:.2f}/{total_weight:.2f} support"
        }

    async def _quality_weighted_consensus(self, proposals: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Consensus weighted by proposal quality scores"""
        if not proposals:
            return self._create_error_result("No proposals provided")

        total_quality = 0
        quality_weighted_decisions = defaultdict(float)

        for proposal in proposals:
            decision = self._extract_decision(proposal)
            quality_score = self._assess_proposal_quality(proposal, context)

            quality_weighted_decisions[decision] += quality_score
            total_quality += quality_score

        # Find quality-weighted decision
        best_decision = max(quality_weighted_decisions.items(), key=lambda x: x[1])
        decision, quality_weight = best_decision

        agreement_level = quality_weight / total_quality if total_quality > 0 else 0

        return {
            "outcome": "consensus_achieved" if agreement_level >= 0.7 else "quality_guided",
            "decision": decision,
            "agreement_level": agreement_level,
            "quality_support": quality_weight,
            "total_quality": total_quality,
            "consensus_method": "quality_weighted",
            "rationale": f"Quality-weighted decision with {quality_weight:.2f}/{total_quality:.2f} quality support"
        }

    async def _expert_adjudication_consensus(
        self,
        proposals: List[Dict[str, Any]],
        context: Dict[str, Any],
        domain: str
    ) -> Dict[str, Any]:
        """Expert adjudication for complex decisions"""
        if not proposals:
            return self._create_error_result("No proposals provided")

        # Create adjudication prompt
        adjudication_prompt = self._create_adjudication_prompt(proposals, context, domain)

        # Get expert decision from LLM
        expert_decision = await self.ollama_service.generate_response(
            prompt=adjudication_prompt,
            system_prompt=self._get_expert_system_prompt(domain),
            temperature=0.2  # Low temperature for consistent expert decisions
        )

        # Parse expert decision
        decision, rationale = self._parse_expert_decision(expert_decision)

        return {
            "outcome": "expert_consensus",
            "decision": decision,
            "agreement_level": 0.9,  # High confidence in expert decision
            "expert_rationale": rationale,
            "consensus_method": "expert_adjudication",
            "rationale": f"Expert adjudication: {rationale[:100]}..."
        }

    async def _hybrid_consensus(
        self,
        proposals: List[Dict[str, Any]],
        context: Dict[str, Any],
        domain: str
    ) -> Dict[str, Any]:
        """Hybrid approach combining multiple strategies"""
        if not proposals:
            return self._create_error_result("No proposals provided")

        # Step 1: Quality-weighted initial filtering
        quality_scores = [(p, self._assess_proposal_quality(p, context)) for p in proposals]
        quality_scores.sort(key=lambda x: x[1], reverse=True)

        # Keep top 70% of proposals by quality
        threshold = int(len(proposals) * 0.7)
        top_proposals = [p for p, score in quality_scores[:max(threshold, 1)]]

        # Step 2: Weighted vote among top proposals
        weighted_result = await self._weighted_vote_consensus(top_proposals, context)

        # Step 3: Expert adjudication if agreement is low
        if weighted_result["agreement_level"] < 0.6:
            expert_result = await self._expert_adjudication_consensus(top_proposals, context, domain)
            final_decision = expert_result["decision"]
            agreement_level = expert_result["agreement_level"]
            method = "hybrid_with_expert"
            rationale = f"Hybrid consensus with expert adjudication: {expert_result['expert_rationale'][:100]}..."
        else:
            final_decision = weighted_result["decision"]
            agreement_level = weighted_result["agreement_level"]
            method = "hybrid_weighted"
            rationale = weighted_result["rationale"]

        return {
            "outcome": "hybrid_consensus",
            "decision": final_decision,
            "agreement_level": agreement_level,
            "top_proposals_count": len(top_proposals),
            "consensus_method": method,
            "rationale": rationale
        }

    async def _iterative_refinement(
        self,
        proposals: List[Dict[str, Any]],
        context: Dict[str, Any],
        initial_result: Dict[str, Any],
        domain: str
    ) -> Dict[str, Any]:
        """Iterative refinement to achieve better consensus"""
        current_result = initial_result
        iteration = 1

        while iteration < self.max_iterations and current_result["agreement_level"] < self.convergence_threshold:
            # Generate refinement prompt based on current disagreements
            refinement_prompt = self._create_refinement_prompt(proposals, current_result, context, domain)

            # Get refined proposals from agents
            refined_proposals = await self._get_refined_proposals(proposals, refinement_prompt, context)

            # Attempt consensus again
            current_result = await self._attempt_consensus(refined_proposals, context, ConsensusStrategy.MAJORITY_VOTE, domain)
            current_result["iterations"] = iteration + 1

            iteration += 1

        return current_result

    def _extract_decision(self, proposal: Dict[str, Any]) -> Any:
        """Extract the core decision from a proposal"""
        # This would be customized based on proposal structure
        return proposal.get("decision") or proposal.get("result") or proposal.get("recommendation") or str(proposal)

    def _get_agent_weight(self, agent_id: str) -> float:
        """Get expertise weight for an agent"""
        # Base weight
        base_weight = 1.0

        # Historical performance bonus
        if agent_id in self.agent_performance_history:
            recent_performance = self.agent_performance_history[agent_id][-10:]  # Last 10 interactions
            if recent_performance:
                avg_performance = statistics.mean(recent_performance)
                performance_multiplier = 0.5 + (avg_performance * 0.5)  # 0.5 to 1.0
                base_weight *= performance_multiplier

        # Domain expertise (could be learned)
        expertise_multiplier = self.agent_expertise.get(agent_id, 1.0)
        base_weight *= expertise_multiplier

        return max(0.1, min(3.0, base_weight))  # Clamp between 0.1 and 3.0

    def _assess_proposal_quality(self, proposal: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess the quality of a proposal"""
        quality_score = 0.5  # Base score

        # Confidence factor
        confidence = proposal.get("confidence", 0.5)
        quality_score += confidence * 0.3

        # Completeness factor
        if "rationale" in proposal or "explanation" in proposal:
            quality_score += 0.1

        # Evidence factor
        if "evidence" in proposal or "supporting_data" in proposal:
            quality_score += 0.1

        # Context relevance
        if any(keyword in str(proposal).lower() for keyword in context.get("keywords", [])):
            quality_score += 0.1

        return min(1.0, quality_score)

    def _create_adjudication_prompt(self, proposals: List[Dict[str, Any]], context: Dict[str, Any], domain: str) -> str:
        """Create expert adjudication prompt"""
        proposals_text = "\n".join([f"Proposal {i+1}: {json.dumps(p, indent=2)}" for i, p in enumerate(proposals)])

        return f"""As an expert adjudicator in {domain}, review the following proposals and make a final decision:

Context: {json.dumps(context, indent=2)}

Proposals:
{proposals_text}

Please provide:
1. Your final decision/recommendation
2. Detailed rationale for your choice
3. How you weighed the different factors
4. Any conditions or caveats for your decision

Final Decision:"""

    def _get_expert_system_prompt(self, domain: str) -> str:
        """Get expert system prompt for domain"""
        return f"""You are an expert adjudicator specializing in {domain}.
Your role is to make fair, well-reasoned decisions when agents disagree.
Consider technical merit, practical feasibility, risk factors, and alignment with goals.
Provide clear rationale for your decisions."""

    def _parse_expert_decision(self, response: str) -> Tuple[Any, str]:
        """Parse expert decision from response"""
        # Simple parsing - could be more sophisticated
        lines = response.split('\n')
        decision = "undecided"
        rationale = response

        for line in lines:
            if line.startswith('Final Decision:') or line.startswith('Decision:'):
                decision = line.split(':', 1)[1].strip()
                break

        return decision, rationale

    def _create_refinement_prompt(self, proposals: List[Dict[str, Any]], current_result: Dict[str, Any], context: Dict[str, Any], domain: str) -> str:
        """Create refinement prompt for iterative improvement"""
        return f"""The current consensus attempt achieved {current_result['agreement_level']:.2f} agreement.
Please refine your proposals considering the disagreements and work towards better consensus.

Current result: {json.dumps(current_result, indent=2)}
Context: {json.dumps(context, indent=2)}

Refined proposals should address the points of disagreement and find common ground."""

    async def _get_refined_proposals(self, proposals: List[Dict[str, Any]], refinement_prompt: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get refined proposals from agents"""
        # This would involve sending refinement requests to agents
        # For now, return original proposals (simplified)
        return proposals

    def _create_single_proposal_result(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Create result when only one proposal exists"""
        return {
            "outcome": "single_proposal",
            "decision": self._extract_decision(proposal),
            "agreement_level": 1.0,
            "supporting_proposals": 1,
            "total_proposals": 1,
            "consensus_method": "single_source",
            "rationale": "Only one proposal available"
        }

    def _create_error_result(self, message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            "outcome": "error",
            "decision": None,
            "agreement_level": 0.0,
            "error": message,
            "consensus_method": "error",
            "rationale": f"Error: {message}"
        }

    async def _learn_from_consensus(self, proposals: List[Dict[str, Any]], result: Dict[str, Any], domain: str):
        """Learn from consensus outcomes to improve future decisions"""
        try:
            # Update agent performance history
            for proposal in proposals:
                agent_id = proposal.get("agent_id", "unknown")
                decision = self._extract_decision(proposal)
                final_decision = result.get("decision")

                # Simple performance metric: agreement with final decision
                performance = 1.0 if decision == final_decision else 0.0
                self.agent_performance_history[agent_id].append(performance)

                # Keep only recent history
                if len(self.agent_performance_history[agent_id]) > 50:
                    self.agent_performance_history[agent_id] = self.agent_performance_history[agent_id][-50:]

            # Update expertise weights
            for agent_id in self.agent_performance_history:
                recent_performance = self.agent_performance_history[agent_id][-20:]  # Last 20
                if recent_performance:
                    avg_performance = statistics.mean(recent_performance)
                    self.agent_expertise[agent_id] = 0.5 + (avg_performance * 0.5)  # 0.5 to 1.0

            # Add consensus pattern to knowledge graph
            consensus_content = f"Consensus Pattern: {result.get('consensus_method', 'unknown')} in {domain}"

            await self.graph_rag_service.add_knowledge(
                content=consensus_content,
                metadata={
                    "type": "consensus_pattern",
                    "domain": domain,
                    "agreement_level": result.get("agreement_level", 0),
                    "method": result.get("consensus_method", "unknown"),
                    "learning_opportunity": True
                },
                node_type="consensus_learning"
            )

        except Exception as e:
            logger.warning(f"Failed to learn from consensus: {str(e)}")

    def get_consensus_statistics(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        if not self.consensus_history:
            return {"total_consensus_events": 0}

        total_events = len(self.consensus_history)
        method_counts = Counter([event["strategy"] for event in self.consensus_history])
        outcome_counts = Counter([event["outcome"] for event in self.consensus_history])

        avg_agreement = statistics.mean([event["final_agreement"] for event in self.consensus_history])

        return {
            "total_consensus_events": total_events,
            "strategy_distribution": dict(method_counts),
            "outcome_distribution": dict(outcome_counts),
            "average_agreement_level": avg_agreement,
            "agent_expertise": dict(self.agent_expertise)
        }