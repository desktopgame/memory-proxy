"""
Memory system with RAG and Knowledge Graph for long-term memory.

Architecture:
- RAG: ChromaDB with LMStudio embedding for storing/retrieving conversations
- Knowledge Graph: SQLAlchemy for entity relationships to enhance search
- LLM: llama.cpp server for entity extraction and query generation
"""

import hashlib
import json
import logging
import math
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import chromadb
import numpy as np
from chromadb.api.types import EmbeddingFunction, Embeddings, Documents
import httpx
from openai import AsyncOpenAI
from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from libword import extract_keywords_mecab

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DELEGATE_URL = os.getenv("DELEGATE_URL", "http://localhost:7071")
SUPPORT_MODEL = os.getenv("SUPPORT_MODEL", "")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:1234")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")
DATABASE_PATH = os.getenv("DATABASE_PATH", "memory.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")

# Time decay configuration
# Half-life in hours: after this time, the score is halved
TIME_DECAY_HALF_LIFE_HOURS = float(os.getenv("TIME_DECAY_HALF_LIFE_HOURS", "720"))  # 30 days default
# Weight for time decay (0.0 = no decay, 1.0 = full decay effect)
TIME_DECAY_WEIGHT = float(os.getenv("TIME_DECAY_WEIGHT", "0.3"))

# Distance threshold for memory relevance
# Memories with adjusted distance above this threshold are excluded
# ChromaDB L2 distance: 0 = identical, ~2 = very different
MEMORY_DISTANCE_THRESHOLD = float(os.getenv("MEMORY_DISTANCE_THRESHOLD", "1.0"))

# Duplicate detection threshold for memory search results
# Results with cosine similarity above this threshold are considered duplicates
# Cosine similarity: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
MEMORY_DUPLICATE_THRESHOLD = float(os.getenv("MEMORY_DUPLICATE_THRESHOLD", "0.9"))


# =============================================================================
# Time Decay Utilities
# =============================================================================


def compute_time_decay(timestamp_str: str, half_life_hours: float = TIME_DECAY_HALF_LIFE_HOURS) -> float:
    """
    Compute time decay factor based on elapsed time.
    
    Uses exponential decay with half-life: decay = 0.5 ^ (hours / half_life)
    
    Args:
        timestamp_str: ISO format timestamp string
        half_life_hours: Time in hours for score to decay by half
        
    Returns:
        Decay factor between 0.0 and 1.0 (1.0 = no decay, 0.0 = fully decayed)
    """
    from datetime import timezone
    
    try:
        # Parse timestamp - handle various formats
        timestamp_str = timestamp_str.strip()
        
        # Replace Z with +00:00 for proper parsing
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        
        created_at = datetime.fromisoformat(timestamp_str)
        
        # Make timezone-aware if not already (assume UTC)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        
        # Get current time in UTC
        now = datetime.now(timezone.utc)
        elapsed_hours = (now - created_at).total_seconds() / 3600.0
        
        # Ensure non-negative elapsed time
        if elapsed_hours < 0:
            elapsed_hours = 0
        
        # Exponential decay: 0.5 ^ (elapsed / half_life)
        decay = math.pow(0.5, elapsed_hours / half_life_hours)
        return decay
    except Exception as e:
        logger.warning(f"Failed to compute time decay for '{timestamp_str}': {e}")
        return 1.0  # No decay on error


def apply_time_decay_to_distance(
    distance: float,
    decay_factor: float,
    weight: float = TIME_DECAY_WEIGHT,
) -> float:
    """
    Apply time decay to ChromaDB distance score.
    
    Lower distance = more similar. We increase distance for older memories.
    adjusted_distance = distance + (1 - decay_factor) * weight * max_distance_penalty
    
    Args:
        distance: Original ChromaDB distance (lower = more similar)
        decay_factor: Time decay factor (1.0 = new, 0.0 = old)
        weight: How much time decay affects the score (0.0-1.0)
        
    Returns:
        Adjusted distance (higher for older memories)
    """
    # Penalty increases as decay_factor decreases (older memories)
    # max_distance_penalty is set to 2.0 to allow significant reranking
    max_penalty = 2.0
    penalty = (1.0 - decay_factor) * weight * max_penalty
    return distance + penalty


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors using numpy.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity between -1.0 and 1.0 (1.0 = identical)
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

# =============================================================================
# LMStudio Embedding Function for ChromaDB
# =============================================================================


class LMStudioEmbeddingFunction(EmbeddingFunction[Documents]):
    """Custom embedding function that uses LMStudio's embedding endpoint."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=60.0)

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the given documents."""
        try:
            response = self.client.post(
                f"{self.base_url}/v1/embeddings",
                json={
                    "model": self.model,
                    "input": input,
                },
            )
            response.raise_for_status()
            data = response.json()
            # Sort by index to ensure correct order
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [e["embedding"] for e in embeddings]
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise


# =============================================================================
# Knowledge Graph Models (SQLAlchemy)
# =============================================================================

Base = declarative_base()


class Entity(Base):
    """Entity in the knowledge graph."""

    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    entity_type = Column(String(100), nullable=True)  # e.g., "person", "topic", "place"
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    # Relationships
    outgoing_relations = relationship(
        "Relation", foreign_keys="Relation.source_id", back_populates="source"
    )
    incoming_relations = relationship(
        "Relation", foreign_keys="Relation.target_id", back_populates="target"
    )


class Relation(Base):
    """Relation between entities in the knowledge graph."""

    __tablename__ = "relations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(Integer, ForeignKey("entities.id"), nullable=False, index=True)
    target_id = Column(Integer, ForeignKey("entities.id"), nullable=False, index=True)
    relation_type = Column(String(100), nullable=False)  # e.g., "related_to", "likes"
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    # Relationships
    source = relationship("Entity", foreign_keys=[source_id], back_populates="outgoing_relations")
    target = relationship("Entity", foreign_keys=[target_id], back_populates="incoming_relations")


class ConversationRecord(Base):
    """Record of a conversation turn stored in both RAG and graph."""

    __tablename__ = "conversation_records"

    id = Column(String(36), primary_key=True)  # UUID
    user_message = Column(Text, nullable=False)
    assistant_message = Column(Text, nullable=True)
    messages_hash = Column(String(16), index=True, nullable=True)  # Hash of messages array
    parent_hash = Column(String(16), index=True, nullable=True)  # Hash of parent messages
    created_at = Column(DateTime, default=datetime.now(timezone.utc))


# =============================================================================
# Hash Utilities
# =============================================================================


def compute_messages_hash(messages: list[dict[str, Any]]) -> str:
    """
    Compute hash of messages array for conversation chain tracking.
    
    Args:
        messages: OpenAI-style messages array
        
    Returns:
        16-character hash string
    """
    # Sort keys and minify for consistent hashing
    normalized = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def compute_parent_hash(messages: list[dict[str, Any]]) -> str | None:
    """
    Compute hash of parent messages (without last user message).
    
    Args:
        messages: OpenAI-style messages array
        
    Returns:
        16-character hash string, or None if no parent
    """
    if len(messages) <= 1:
        return None
    # Remove last message (the current user message)
    parent_messages = messages[:-1]
    return compute_messages_hash(parent_messages)


# =============================================================================
# Memory System
# =============================================================================


class MemorySystem:
    """
    Memory system combining RAG and Knowledge Graph.
    
    - RAG (ChromaDB): Stores conversation embeddings for semantic search
    - Knowledge Graph (SQLAlchemy): Stores entity relationships for query expansion
    - LLM (llama.cpp): Extracts entities and generates search queries
    """

    def __init__(self):
        self._initialized = False
        self._embedding_fn: LMStudioEmbeddingFunction | None = None
        self._chroma_client: chromadb.PersistentClient | None = None
        self._user_collection: chromadb.Collection | None = None
        self._assistant_collection: chromadb.Collection | None = None
        self._db_session: sessionmaker | None = None
        self._llm_client: AsyncOpenAI | None = None

    def _ensure_initialized(self):
        """Lazy initialization of all components."""
        if self._initialized:
            return

        logger.info("Initializing memory system...")

        # Initialize embedding function
        self._embedding_fn = LMStudioEmbeddingFunction(EMBEDDING_URL, EMBEDDING_MODEL)

        # Initialize ChromaDB
        chroma_setting = chromadb.Settings(anonymized_telemetry=False)
        self._chroma_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=chroma_setting)
        self._user_collection = self._chroma_client.get_or_create_collection(
            name="user_messages",
            embedding_function=self._embedding_fn,
            metadata={"description": "User messages from conversations"},
        )
        self._assistant_collection = self._chroma_client.get_or_create_collection(
            name="assistant_messages",
            embedding_function=self._embedding_fn,
            metadata={"description": "Assistant responses from conversations"},
        )

        # Initialize SQLAlchemy
        engine = create_engine(f"sqlite:///{DATABASE_PATH}", echo=False)
        Base.metadata.create_all(engine)
        self._db_session = sessionmaker(bind=engine)

        # Initialize LLM client (OpenAI-compatible)
        self._llm_client = AsyncOpenAI(
            base_url=f"{DELEGATE_URL}/v1",
            api_key="not-needed",  # Local server doesn't require API key
            timeout=120.0,
        )

        self._initialized = True
        logger.info("Memory system initialized")

    async def close(self):
        """Close all connections."""
        if self._llm_client:
            await self._llm_client.close()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def save_memory(
        self,
        user_message: str,
        messages: list[dict[str, Any]] | None = None,
        assistant_message: str | None = None,
    ) -> str:
        """
        Save a conversation turn to memory.
        
        Args:
            user_message: The user's message
            messages: OpenAI-style messages array (for conversation chain tracking)
            assistant_message: The assistant's response (optional, can be added later)
            
        Returns:
            conversation_id: UUID of the saved conversation
        """
        self._ensure_initialized()

        conversation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Compute hashes for conversation chain tracking
        messages_hash = None
        parent_hash = None
        if messages:
            messages_hash = compute_messages_hash(messages)
            parent_hash = compute_parent_hash(messages)
            logger.debug(f"Messages hash: {messages_hash}, Parent hash: {parent_hash}")

        # Save to ChromaDB (user message)
        self._user_collection.add(
            ids=[conversation_id],
            documents=[user_message],
            metadatas=[{
                "conversation_id": conversation_id,
                "timestamp": timestamp,
                "role": "user",
                "messages_hash": messages_hash or "",
                "parent_hash": parent_hash or "",
            }],
        )

        # Save assistant message if provided
        if assistant_message:
            self._assistant_collection.add(
                ids=[conversation_id],
                documents=[assistant_message],
                metadatas=[{
                    "conversation_id": conversation_id,
                    "timestamp": timestamp,
                    "role": "assistant",
                }],
            )

        # Save to SQLite
        with self._db_session() as session:
            record = ConversationRecord(
                id=conversation_id,
                user_message=user_message,
                assistant_message=assistant_message,
                messages_hash=messages_hash,
                parent_hash=parent_hash,
            )
            session.add(record)
            session.commit()

        logger.debug(f"Saved conversation {conversation_id}")
        return conversation_id

    def update_assistant_message(self, conversation_id: str, assistant_message: str):
        """
        Update the assistant message for an existing conversation.
        
        Args:
            conversation_id: UUID of the conversation
            assistant_message: The assistant's response
        """
        self._ensure_initialized()

        timestamp = datetime.now(timezone.utc).isoformat()

        # Add to ChromaDB
        self._assistant_collection.add(
            ids=[conversation_id],
            documents=[assistant_message],
            metadatas=[{
                "conversation_id": conversation_id,
                "timestamp": timestamp,
                "role": "assistant",
            }],
        )

        # Update SQLite
        with self._db_session() as session:
            record = session.query(ConversationRecord).filter_by(id=conversation_id).first()
            if record:
                record.assistant_message = assistant_message
                session.commit()

        logger.debug(f"Updated assistant message for {conversation_id}")

    async def load_memory(
        self,
        query: str,
        n_results: int = 5,
        chain_depth_before: int = 2,
        chain_depth_after: int = 2,
    ) -> str:
        """
        Load relevant memories based on the query, including conversation context.
        
        This method:
        1. Searches the knowledge graph for related entities
        2. Uses LLM to generate an optimized search query
        3. Performs semantic search on RAG
        4. Retrieves conversation chain (before/after) for each hit
        5. Returns formatted memory context
        
        Args:
            query: The search query (usually the user's current message)
            n_results: Maximum number of results to return
            chain_depth_before: How many previous conversations to retrieve
            chain_depth_after: How many next conversations to retrieve
            
        Returns:
            Formatted string containing relevant memories with context
        """
        self._ensure_initialized()

        # Get related entities from knowledge graph
        related_entities = self._get_related_entities(query)
        logger.debug(f"Related entities: {related_entities}")

        # Use LLM to generate optimized search query
        expanded_query = await self._generate_search_query(query, related_entities)
        logger.debug(f"Generated search query: {expanded_query}")

        # Search RAG (fetch more results for reranking and deduplication)
        fetch_count = n_results * 5  # Fetch more to allow for filtering and deduplication
        results = self._user_collection.query(
            query_texts=[expanded_query],
            n_results=fetch_count,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        # Apply time decay and rerank
        ranked_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            timestamp = metadata.get("timestamp", "")
            decay_factor = compute_time_decay(timestamp) if timestamp else 1.0
            adjusted_distance = apply_time_decay_to_distance(distance, decay_factor)
            
            # Get embedding if available
            embedding = None
            if results.get("embeddings") is not None and len(results["embeddings"]) > 0:
                if results["embeddings"][0] is not None and len(results["embeddings"][0]) > i:
                    embedding = np.array(results["embeddings"][0][i])
            
            ranked_results.append({
                "doc": doc,
                "metadata": metadata,
                "original_distance": distance,
                "adjusted_distance": adjusted_distance,
                "decay_factor": decay_factor,
                "embedding": embedding,
            })

        # Sort by adjusted distance (lower = better)
        ranked_results.sort(key=lambda x: x["adjusted_distance"])

        # Filter by distance threshold
        filtered_results = [
            r for r in ranked_results
            if r["adjusted_distance"] <= MEMORY_DISTANCE_THRESHOLD
        ]

        logger.debug(
            f"Reranked {len(ranked_results)} results, "
            f"{len(filtered_results)} passed threshold ({MEMORY_DISTANCE_THRESHOLD})"
        )

        # Remove duplicates based on cosine similarity
        deduplicated_results = []
        for result in filtered_results:
            is_duplicate = False
            if result["embedding"] is not None:
                for selected in deduplicated_results:
                    if selected["embedding"] is not None:
                        similarity = cosine_similarity(result["embedding"], selected["embedding"])
                        if similarity >= MEMORY_DUPLICATE_THRESHOLD:
                            is_duplicate = True
                            logger.debug(
                                f"Duplicate detected (similarity={similarity:.3f}): "
                                f"'{result['doc'][:50]}...' similar to '{selected['doc'][:50]}...'"
                            )
                            break
            
            if not is_duplicate:
                deduplicated_results.append(result)
                if len(deduplicated_results) >= n_results:
                    break

        logger.debug(
            f"After deduplication: {len(deduplicated_results)} results "
            f"(removed {len(filtered_results) - len(deduplicated_results)} duplicates)"
        )

        ranked_results = deduplicated_results

        # Format results with conversation chain
        memories = []
        for i, result in enumerate(ranked_results):
            metadata = result["metadata"]
            doc = result["doc"]
            conversation_id = metadata.get("conversation_id")
            
            # Get conversation chain (before and after)
            chain = self._get_conversation_chain(
                conversation_id,
                depth_before=chain_depth_before,
                depth_after=chain_depth_after,
            )
            
            if chain:
                # Find center index (the hit conversation)
                center_index = next(
                    (i for i, r in enumerate(chain) if r.id == conversation_id),
                    len(chain) // 2
                )
                chain_text = self._format_conversation_chain(chain, center_index)
                memories.append(f"[Memory {i+1}]\n{chain_text}")
            else:
                # Fallback: just show the hit conversation
                assistant_response = self._get_assistant_response(conversation_id)
                memory_entry = f"[Memory {i+1}] User: {doc}"
                if assistant_response:
                    memory_entry += f"\nAssistant: {assistant_response}"
                memories.append(memory_entry)

        if not memories:
            return ""

        for i, memory in enumerate(memories):
            logger.debug(f"[Memory {i}]")
            logger.debug(memory)
            logger.debug("---")

        return "\n\n---\n\n".join([
            "\n\n<memories>",
            "以下は過去の会話から検索された関連する記憶です（前後の文脈を含む）：",
            *memories,
            "</memories>\n\n",
        ])

    # -------------------------------------------------------------------------
    # Knowledge Graph Operations
    # -------------------------------------------------------------------------

    def add_entity(self, name: str, entity_type: str | None = None) -> int:
        """Add an entity to the knowledge graph."""
        self._ensure_initialized()

        with self._db_session() as session:
            # Check if entity already exists
            existing = session.query(Entity).filter_by(name=name).first()
            if existing:
                return existing.id

            entity = Entity(name=name, entity_type=entity_type)
            session.add(entity)
            session.commit()
            return entity.id

    def add_relation(self, source_name: str, target_name: str, relation_type: str = "related_to"):
        """Add a relation between two entities."""
        self._ensure_initialized()

        source_id = self.add_entity(source_name)
        target_id = self.add_entity(target_name)

        with self._db_session() as session:
            # Check if relation already exists
            existing = session.query(Relation).filter_by(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
            ).first()
            if existing:
                return

            relation = Relation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
            )
            session.add(relation)
            session.commit()

    def _get_related_entities(self, text: str, max_depth: int = 2, top_k: int = 5) -> list[str]:
        """
        Find entities in text and get their related entities from the graph.
        
        Scores entities based on:
        1. Graph distance from mentioned entities (exponential decay: 0.5^depth)
        2. Cosine similarity between entity name embedding and input text embedding
        
        Final score = graph_score * 0.5 + embedding_similarity * 0.5
        
        Args:
            text: Text to search for entities
            max_depth: How many hops to traverse in the graph
            top_k: Number of top entities to return
            
        Returns:
            List of top entity names (without scores)
        """
        self._ensure_initialized()

        # Dictionary to store entity -> graph_score (keep max score if found multiple times)
        entity_graph_scores: dict[str, float] = {}

        # Extract keywords from text using MeCab
        keywords = extract_keywords_mecab(text)
        logger.debug(f"Extracted keywords: {keywords}")

        with self._db_session() as session:
            # Find entities that match extracted keywords
            all_entities = session.query(Entity).all()
            mentioned = []
            for entity in all_entities:
                entity_name_lower = entity.name.lower()
                # Match if entity name is in keywords or keywords contain entity name
                for keyword in keywords:
                    if keyword == entity_name_lower or keyword in entity_name_lower or entity_name_lower in keyword:
                        mentioned.append(entity)
                        break

            logger.debug(f"Matched entities: {[e.name for e in mentioned]}")

            # BFS to find related entities with depth tracking
            to_visit = [(e, 0) for e in mentioned]
            visited_ids = set()

            while to_visit:
                entity, depth = to_visit.pop(0)
                if entity.id in visited_ids or depth > max_depth:
                    continue
                visited_ids.add(entity.id)
                
                # Exponential decay score based on graph distance: 0.5^depth
                graph_score = math.pow(0.5, depth)
                
                # Keep max score if entity already exists
                if entity.name not in entity_graph_scores or entity_graph_scores[entity.name] < graph_score:
                    entity_graph_scores[entity.name] = graph_score

                if depth < max_depth:
                    # Get related entities
                    for rel in entity.outgoing_relations:
                        if rel.target_id not in visited_ids:
                            target = session.query(Entity).get(rel.target_id)
                            if target:
                                to_visit.append((target, depth + 1))
                    for rel in entity.incoming_relations:
                        if rel.source_id not in visited_ids:
                            source = session.query(Entity).get(rel.source_id)
                            if source:
                                to_visit.append((source, depth + 1))

        if not entity_graph_scores:
            return []

        # Compute embedding similarity for each entity
        try:
            # Get embedding for input text
            text_embedding = np.array(self._embedding_fn([text])[0])
            
            # Get embeddings for all entity names
            entity_names = list(entity_graph_scores.keys())
            entity_embeddings = self._embedding_fn(entity_names)
            
            # Compute final scores combining graph score and embedding similarity
            entity_final_scores: dict[str, float] = {}
            for name, entity_emb in zip(entity_names, entity_embeddings):
                graph_score = entity_graph_scores[name]
                embedding_sim = cosine_similarity(text_embedding, np.array(entity_emb))
                # Normalize embedding similarity from [-1, 1] to [0, 1]
                embedding_sim_normalized = (embedding_sim + 1.0) / 2.0
                # Final score: weighted combination
                final_score = graph_score * 0.5 + embedding_sim_normalized * 0.5
                entity_final_scores[name] = final_score
                
            logger.debug(f"Entity scores (graph+embedding): {entity_final_scores}")
            
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}, using graph scores only")
            entity_final_scores = entity_graph_scores

        # Sort by final score descending and return top_k entity names
        sorted_entities = sorted(entity_final_scores.items(), key=lambda x: x[1], reverse=True)
        top_entities = [name for name, score in sorted_entities[:top_k]]
        logger.debug(f"Top {top_k} entities: {top_entities}")
        return top_entities

    async def _generate_search_query(
        self, user_query: str, related_entities: list[str]
    ) -> str:
        """
        Use LLM to generate an optimized search query for RAG.
        
        Args:
            user_query: The user's current message
            related_entities: List of related entity names from knowledge graph
            
        Returns:
            Optimized search query string
        """
        # If no related entities and short query, use original
        if not related_entities and len(user_query) < 50:
            return user_query

        entities_context = ""
        if related_entities:
            entities_str = ", ".join(related_entities)
            entities_context = f"\n\n関連するエンティティ（ナレッジグラフから）: {entities_str}"

        prompt = f"""以下のユーザーの発言に対して、過去の会話履歴を検索するための最適な検索クエリを生成してください。
検索クエリは、ユーザーの意図を捉えつつ、関連する記憶を見つけやすいように言い換えや拡張を行ってください。
{entities_context}

ユーザーの発言: {user_query}

検索クエリを生成してください。"""

        # JSON Schema for structured output
        json_schema = {
            "name": "search_query_generation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Optimized search query for RAG"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this query was generated"
                    }
                },
                "required": ["search_query", "reasoning"],
                "additionalProperties": False
            }
        }

        try:
            response = await self._llm_client.chat.completions.create(
                model=SUPPORT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema
                }
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                search_query = result.get("search_query", user_query)
                reasoning = result.get("reasoning", "")
                logger.debug(f"Search query reasoning: {reasoning}")
                return search_query
        except Exception as e:
            logger.warning(f"Search query generation failed: {e}, using original query")

        # Fallback: simple expansion
        if related_entities:
            entities_str = ", ".join(related_entities)
            return f"{user_query} {entities_str}"
        return user_query

    def _get_assistant_response(self, conversation_id: str) -> str | None:
        """Get assistant response for a conversation from SQLite."""
        with self._db_session() as session:
            record = session.query(ConversationRecord).filter_by(id=conversation_id).first()
            if record:
                return record.assistant_message
        return None

    # -------------------------------------------------------------------------
    # Conversation Chain Operations
    # -------------------------------------------------------------------------

    def _get_conversation_chain(
        self,
        conversation_id: str,
        depth_before: int = 2,
        depth_after: int = 2,
    ) -> list[ConversationRecord]:
        """
        Get conversation chain (before and after) for a given conversation.
        
        Args:
            conversation_id: UUID of the center conversation
            depth_before: How many previous conversations to retrieve
            depth_after: How many next conversations to retrieve
            
        Returns:
            List of ConversationRecord ordered chronologically
        """
        self._ensure_initialized()

        chain = []

        with self._db_session() as session:
            current = session.query(ConversationRecord).filter_by(id=conversation_id).first()
            if not current:
                return chain

            # Traverse backwards (previous conversations)
            prev_records = []
            prev = current
            for _ in range(depth_before):
                if not prev.parent_hash:
                    break
                parent = session.query(ConversationRecord).filter_by(
                    messages_hash=prev.parent_hash
                ).first()
                if parent:
                    prev_records.insert(0, parent)
                    prev = parent
                else:
                    break

            # Traverse forwards (next conversations)
            next_records = []
            next_conv = session.query(ConversationRecord).filter_by(
                parent_hash=current.messages_hash
            ).first() if current.messages_hash else None
            for _ in range(depth_after):
                if next_conv:
                    next_records.append(next_conv)
                    next_conv = session.query(ConversationRecord).filter_by(
                        parent_hash=next_conv.messages_hash
                    ).first() if next_conv.messages_hash else None
                else:
                    break

            # Combine: prev + current + next
            chain = prev_records + [current] + next_records

        return chain

    def _format_conversation_chain(
        self,
        chain: list[ConversationRecord],
        center_index: int,
    ) -> str:
        """Format conversation chain for memory context."""
        if not chain:
            return ""

        entries = []
        for i, record in enumerate(chain):
            marker = "→" if i == center_index else " "
            entry = f"{marker} User: {record.user_message}"
            if record.assistant_message:
                entry += f"\n  Assistant: {record.assistant_message}"
            entries.append(entry)

        return "\n".join(entries)

    # -------------------------------------------------------------------------
    # LLM Operations (for entity extraction)
    # -------------------------------------------------------------------------

    async def extract_entities_and_relations(self, user_message: str, assistant_message: str) -> list[dict]:
        """
        Use LLM to extract entities and relations from a conversation.
        
        Returns:
            List of dicts with keys: source, target, relation
        """
        self._ensure_initialized()

        prompt = f"""以下の会話からエンティティ（人物、場所、トピック、概念など）とその関係を抽出してください。

会話:
ユーザー: {user_message}
アシスタント: {assistant_message}

関係の種類の例: related_to, likes, dislikes, knows, studies, works_on, lives_in など"""

        # JSON Schema for structured output
        json_schema = {
            "name": "entity_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "Source entity name"
                                },
                                "target": {
                                    "type": "string",
                                    "description": "Target entity name"
                                },
                                "relation": {
                                    "type": "string",
                                    "description": "Type of relation between entities"
                                }
                            },
                            "required": ["source", "target", "relation"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["entities"],
                "additionalProperties": False
            }
        }

        try:
            response = await self._llm_client.chat.completions.create(
                model=SUPPORT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema
                }
            )
            logger.debug(f"graph={response.choices[0].message.content}")
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                return result.get("entities", [])
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")

        return []

    async def process_conversation(self, conversation_id: str, user_message: str, assistant_message: str):
        """
        Process a completed conversation: extract entities and update knowledge graph.
        
        Args:
            conversation_id: UUID of the conversation
            user_message: The user's message
            assistant_message: The assistant's response
        """
        # Update assistant message in storage
        self.update_assistant_message(conversation_id, assistant_message)

        # Extract entities and relations
        relations = await self.extract_entities_and_relations(user_message, assistant_message)

        # Add to knowledge graph
        for rel in relations:
            source = rel.get("source")
            target = rel.get("target")
            relation_type = rel.get("relation", "related_to")

            if source and target:
                self.add_relation(source, target, relation_type)
                logger.debug(f"Added relation: {source} --{relation_type}--> {target}")


# =============================================================================
# Global Instance and Public API
# =============================================================================

_memory_system: MemorySystem | None = None


def get_memory_system() -> MemorySystem:
    """Get the global memory system instance."""
    global _memory_system
    if _memory_system is None:
        _memory_system = MemorySystem()
    return _memory_system


def save_memory(input_text: str, messages: list[dict[str, Any]] | None = None) -> str:
    """
    Save user message to memory.
    
    Args:
        input_text: User's message
        messages: OpenAI-style messages array (for conversation chain tracking)
        
    Returns:
        conversation_id for later use
    """
    return get_memory_system().save_memory(input_text, messages=messages)


async def load_memory(input_text: str) -> str:
    """
    Load relevant memories for the given input.
    
    Args:
        input_text: User's current message
        
    Returns:
        Formatted memory context to append to prompt
    """
    return await get_memory_system().load_memory(input_text)


async def process_conversation(conversation_id: str, user_message: str, assistant_message: str):
    """
    Process a completed conversation.
    
    Args:
        conversation_id: UUID returned by save_memory
        user_message: The user's message
        assistant_message: The assistant's response
    """
    await get_memory_system().process_conversation(conversation_id, user_message, assistant_message)


async def close_memory_system():
    """Close the memory system and release resources."""
    global _memory_system
    if _memory_system:
        await _memory_system.close()
        _memory_system = None
