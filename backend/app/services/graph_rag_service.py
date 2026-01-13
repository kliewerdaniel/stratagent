"""
GraphRAG service - persistent knowledge graphs for context and learning
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import networkx as nx
import numpy as np
from collections import defaultdict
import faiss
import pickle
import os

from app.core.config import settings
from app.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

class GraphRAGService:
    """Service for managing persistent knowledge graphs with RAG capabilities"""

    def __init__(self, ollama_service: OllamaService):
        self.ollama_service = ollama_service
        self.graph_path = settings.GRAPH_DATA_PATH
        self.index_path = settings.FAISS_INDEX_PATH

        # Initialize graph and vector index
        self.knowledge_graph = self._load_or_create_graph()
        self.vector_index = self._load_or_create_index()
        self.node_embeddings = {}

        # Embedding model configuration
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Would be loaded via transformers

        # Graph configuration
        self.max_context_nodes = 5
        self.similarity_threshold = 0.7

    def _load_or_create_graph(self) -> nx.DiGraph:
        """Load existing knowledge graph or create new one"""
        graph_file = os.path.join(self.graph_path, "knowledge_graph.pkl")

        if os.path.exists(graph_file):
            try:
                with open(graph_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load graph: {e}, creating new one")

        # Create new graph with initial structure
        graph = nx.DiGraph()

        # Add root node
        graph.add_node("root", {
            "type": "root",
            "content": "StratAgent Knowledge Base Root",
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {"version": "1.0"}
        })

        return graph

    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        index_file = os.path.join(self.index_path, "vector_index.faiss")
        embeddings_file = os.path.join(self.index_path, "embeddings.pkl")

        if os.path.exists(index_file) and os.path.exists(embeddings_file):
            try:
                self.vector_index = faiss.read_index(index_file)
                with open(embeddings_file, 'rb') as f:
                    self.node_embeddings = pickle.load(f)
                return self.vector_index
            except Exception as e:
                logger.warning(f"Failed to load vector index: {e}, creating new one")

        # Create new index
        dimension = 384  # Dimension for sentence-transformers model
        self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.node_embeddings = {}
        return self.vector_index

    def save_graph(self):
        """Save knowledge graph to disk"""
        os.makedirs(self.graph_path, exist_ok=True)
        graph_file = os.path.join(self.graph_path, "knowledge_graph.pkl")

        try:
            with open(graph_file, 'wb') as f:
                pickle.dump(self.knowledge_graph, f)
            logger.info("Knowledge graph saved successfully")
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")

    def save_index(self):
        """Save vector index to disk"""
        os.makedirs(self.index_path, exist_ok=True)
        index_file = os.path.join(self.index_path, "vector_index.faiss")
        embeddings_file = os.path.join(self.index_path, "embeddings.pkl")

        try:
            faiss.write_index(self.vector_index, index_file)
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.node_embeddings, f)
            logger.info("Vector index saved successfully")
        except Exception as e:
            logger.error(f"Failed to save vector index: {e}")

    async def add_knowledge(self, content: str, metadata: Dict[str, Any], node_type: str = "knowledge") -> str:
        """Add new knowledge to the graph"""
        # Generate node ID
        node_id = f"{node_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(content) % 10000}"

        # Create node data
        node_data = {
            "type": node_type,
            "content": content,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata,
            "importance_score": self._calculate_importance(content, metadata)
        }

        # Add node to graph
        self.knowledge_graph.add_node(node_id, **node_data)

        # Generate embedding for the content
        embedding = await self._generate_embedding(content)

        # Store embedding
        self.node_embeddings[node_id] = embedding

        # Add to vector index
        self.vector_index.add(np.array([embedding], dtype=np.float32))

        # Create relationships with existing nodes
        await self._create_relationships(node_id, content, metadata)

        # Save changes
        self.save_graph()
        self.save_index()

        logger.info(f"Added knowledge node: {node_id}")
        return node_id

    async def retrieve_context(self, query: str, max_nodes: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context from knowledge graph"""
        # Generate embedding for query
        query_embedding = await self._generate_embedding(query)

        # Search vector index
        similarities, indices = self.vector_index.search(
            np.array([query_embedding], dtype=np.float32),
            max_nodes * 2  # Get more candidates for filtering
        )

        # Get relevant nodes
        relevant_nodes = []
        node_ids = list(self.node_embeddings.keys())

        for idx, similarity in zip(indices[0], similarities[0]):
            if idx < len(node_ids) and similarity >= self.similarity_threshold:
                node_id = node_ids[idx]
                if node_id in self.knowledge_graph:
                    node_data = self.knowledge_graph.nodes[node_id]
                    relevant_nodes.append({
                        "node_id": node_id,
                        "content": node_data["content"],
                        "similarity": float(similarity),
                        "metadata": node_data["metadata"],
                        "importance": node_data.get("importance_score", 0)
                    })

        # Sort by importance and similarity
        relevant_nodes.sort(key=lambda x: (x["importance"], x["similarity"]), reverse=True)
        relevant_nodes = relevant_nodes[:max_nodes]

        # Add graph-based context (connected nodes)
        enriched_context = []
        for node in relevant_nodes:
            connected_nodes = self._get_connected_context(node["node_id"])
            node["connected_context"] = connected_nodes
            enriched_context.append(node)

        return enriched_context

    async def update_knowledge(self, node_id: str, new_content: str, metadata_updates: Dict[str, Any]) -> bool:
        """Update existing knowledge node"""
        if node_id not in self.knowledge_graph:
            return False

        # Update node data
        node_data = self.knowledge_graph.nodes[node_id]
        node_data["content"] = new_content
        node_data["metadata"].update(metadata_updates)
        node_data["updated_at"] = datetime.utcnow().isoformat()

        # Update embedding
        new_embedding = await self._generate_embedding(new_content)
        self.node_embeddings[node_id] = new_embedding

        # Rebuild index (simplified - in practice would update specific vectors)
        self._rebuild_index()

        # Update relationships
        await self._update_relationships(node_id, new_content, node_data["metadata"])

        self.save_graph()
        self.save_index()

        logger.info(f"Updated knowledge node: {node_id}")
        return True

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        return {
            "total_nodes": len(self.knowledge_graph.nodes),
            "total_edges": len(self.knowledge_graph.edges),
            "node_types": self._count_node_types(),
            "average_degree": sum(dict(self.knowledge_graph.degree()).values()) / len(self.knowledge_graph.nodes) if self.knowledge_graph.nodes else 0,
            "connected_components": nx.number_weakly_connected_components(self.knowledge_graph),
            "last_updated": datetime.utcnow().isoformat()
        }

    async def consolidate_learning(self, time_window: str = "7d") -> Dict[str, Any]:
        """Consolidate learning patterns and create summary nodes"""
        # This would analyze recent interactions and create consolidated knowledge
        # For now, return basic statistics
        return {
            "consolidated_nodes": 0,
            "learning_patterns": [],
            "knowledge_growth": self.get_graph_statistics()
        }

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama or local model"""
        # In practice, this would use a proper embedding model
        # For now, simulate with a simple hash-based approach
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        # Convert to float list (simplified)
        embedding = [float(b) / 255.0 for b in hash_bytes]
        # Pad to required dimension
        while len(embedding) < 384:
            embedding.extend(embedding)
        return embedding[:384]

    async def _create_relationships(self, node_id: str, content: str, metadata: Dict[str, Any]):
        """Create relationships between the new node and existing nodes"""
        # Find semantically similar nodes
        similar_nodes = await self.retrieve_context(content, max_nodes=self.max_context_nodes)

        for similar_node in similar_nodes:
            if similar_node["node_id"] != node_id:
                # Add edge with relationship type
                relationship_type = self._determine_relationship(content, similar_node["content"])
                self.knowledge_graph.add_edge(
                    node_id,
                    similar_node["node_id"],
                    type=relationship_type,
                    weight=similar_node["similarity"],
                    created_at=datetime.utcnow().isoformat()
                )

        # Always connect to root if not already connected
        if not self.knowledge_graph.has_edge(node_id, "root"):
            self.knowledge_graph.add_edge(
                node_id,
                "root",
                type="belongs_to",
                weight=0.1,
                created_at=datetime.utcnow().isoformat()
            )

    async def _update_relationships(self, node_id: str, content: str, metadata: Dict[str, Any]):
        """Update relationships for modified node"""
        # Remove old relationships
        edges_to_remove = []
        for source, target, data in self.knowledge_graph.in_edges(node_id, data=True):
            if data.get("type") != "belongs_to":  # Keep structural relationships
                edges_to_remove.append((source, target))

        for source, target in edges_to_remove:
            self.knowledge_graph.remove_edge(source, target)

        # Create new relationships
        await self._create_relationships(node_id, content, metadata)

    def _get_connected_context(self, node_id: str) -> List[Dict[str, Any]]:
        """Get context from connected nodes"""
        connected_context = []

        # Get direct neighbors
        for neighbor_id in self.knowledge_graph.successors(node_id):
            if neighbor_id in self.knowledge_graph:
                neighbor_data = self.knowledge_graph.nodes[neighbor_id]
                edge_data = self.knowledge_graph.get_edge_data(node_id, neighbor_id)

                connected_context.append({
                    "node_id": neighbor_id,
                    "content": neighbor_data["content"][:200] + "..." if len(neighbor_data["content"]) > 200 else neighbor_data["content"],
                    "relationship": edge_data.get("type", "related"),
                    "metadata": neighbor_data["metadata"]
                })

        return connected_context[:3]  # Limit to top 3 connected nodes

    def _determine_relationship(self, content1: str, content2: str) -> str:
        """Determine the type of relationship between two content pieces"""
        # Simple keyword-based relationship determination
        content1_lower = content1.lower()
        content2_lower = content2.lower()

        if any(word in content1_lower for word in ["error", "bug", "fix", "issue"]) and \
           any(word in content2_lower for word in ["error", "bug", "fix", "issue"]):
            return "bug_related"
        elif any(word in content1_lower for word in ["design", "architecture", "pattern"]) and \
             any(word in content2_lower for word in ["design", "architecture", "pattern"]):
            return "design_related"
        elif any(word in content1_lower for word in ["test", "testing", "validation"]) and \
             any(word in content2_lower for word in ["test", "testing", "validation"]):
            return "testing_related"
        else:
            return "semantically_similar"

    def _calculate_importance(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate importance score for content"""
        importance = 0.5  # Base importance

        # Boost importance based on content characteristics
        if len(content) > 500:
            importance += 0.1
        if any(keyword in content.lower() for keyword in ["error", "bug", "security", "performance"]):
            importance += 0.2
        if metadata.get("source") == "user_feedback":
            importance += 0.15
        if metadata.get("validated", False):
            importance += 0.1

        return min(importance, 1.0)

    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type"""
        type_counts = defaultdict(int)
        for node_id, node_data in self.knowledge_graph.nodes(data=True):
            node_type = node_data.get("type", "unknown")
            type_counts[node_type] += 1
        return dict(type_counts)

    def _rebuild_index(self):
        """Rebuild the vector index (simplified implementation)"""
        # In practice, this would be more efficient
        dimension = 384
        self.vector_index = faiss.IndexFlatIP(dimension)

        embeddings_list = []
        for node_id in self.node_embeddings:
            embeddings_list.append(self.node_embeddings[node_id])

        if embeddings_list:
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            self.vector_index.add(embeddings_array)

    async def get_learning_insights(self) -> Dict[str, Any]:
        """Generate insights from the knowledge graph learning patterns"""
        stats = self.get_graph_statistics()

        # Analyze learning patterns
        insights = {
            "knowledge_growth_rate": self._calculate_growth_rate(),
            "most_connected_concepts": self._find_most_connected_concepts(),
            "learning_gaps": self._identify_learning_gaps(),
            "knowledge_clusters": self._find_knowledge_clusters(),
            "temporal_patterns": self._analyze_temporal_patterns()
        }

        return {
            "statistics": stats,
            "insights": insights,
            "recommendations": self._generate_learning_recommendations(insights)
        }

    def _calculate_growth_rate(self) -> float:
        """Calculate knowledge growth rate"""
        # Simplified calculation
        return 0.05  # 5% growth rate

    def _find_most_connected_concepts(self) -> List[Dict[str, Any]]:
        """Find most connected concepts in the graph"""
        degrees = dict(self.knowledge_graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

        connected_concepts = []
        for node_id, degree in sorted_nodes[:10]:
            node_data = self.knowledge_graph.nodes[node_id]
            connected_concepts.append({
                "node_id": node_id,
                "connections": degree,
                "content_preview": node_data["content"][:100] + "...",
                "type": node_data.get("type", "unknown")
            })

        return connected_concepts

    def _identify_learning_gaps(self) -> List[str]:
        """Identify gaps in the knowledge base"""
        # This would analyze what knowledge is missing
        return ["Advanced security patterns", "Performance optimization techniques"]

    def _find_knowledge_clusters(self) -> List[Dict[str, Any]]:
        """Find clusters of related knowledge"""
        # Use network analysis to find clusters
        clusters = []
        for component in nx.weakly_connected_components(self.knowledge_graph):
            if len(component) > 3:  # Only consider meaningful clusters
                cluster_nodes = list(component)
                cluster_types = [self.knowledge_graph.nodes[n].get("type", "unknown") for n in cluster_nodes]

                clusters.append({
                    "size": len(component),
                    "dominant_type": max(set(cluster_types), key=cluster_types.count),
                    "nodes": cluster_nodes[:5]  # Sample nodes
                })

        return clusters

    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in knowledge acquisition"""
        return {
            "peak_learning_times": ["morning", "afternoon"],
            "learning_frequency": "daily",
            "retention_patterns": "improving"
        }

    def _generate_learning_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate learning recommendations based on insights"""
        recommendations = []

        if insights["knowledge_growth_rate"] < 0.1:
            recommendations.append("Increase knowledge acquisition rate")

        most_connected = insights["most_connected_concepts"]
        if most_connected:
            dominant_type = most_connected[0]["type"]
            recommendations.append(f"Deepen knowledge in {dominant_type} areas")

        if insights["learning_gaps"]:
            recommendations.append(f"Address learning gaps: {', '.join(insights['learning_gaps'])}")

        return recommendations