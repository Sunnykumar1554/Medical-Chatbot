"""
user_db.py
──────────
User authentication and chat-history storage using Pinecone namespaces.

Architecture (single existing index, two new namespaces):
  ┌─────────────────────────────────────────────────────┐
  │  Pinecone Index: "medical-chatbot"                  │
  │                                                     │
  │  namespace=""           ← medical knowledge (UNTOUCHED)
  │  namespace="users"      ← user profiles            │
  │  namespace="chat_hist"  ← per-user chat history    │
  └─────────────────────────────────────────────────────┘

Since Pinecone is a vector DB, user records are stored as
zero-vectors with all data kept in metadata fields.
Filtering (metadata.$eq) lets us look up records without
doing a semantic search.
"""

import hashlib
import uuid
import time
from typing import Optional
from pinecone import Pinecone

# Namespaces — never overlap with the default medical namespace
USER_NS  = "users"
CHAT_NS  = "chat_hist"

# Must match the existing index dimension (all-MiniLM-L6-v2 = 384)
VECTOR_DIM = 384


def _zero_vec() -> list[float]:
    """Dummy vector used for metadata-only Pinecone records."""
    vec = [0.0] * VECTOR_DIM
    vec[0] = 1e-7  # Pinecone rejects all-zero dense vectors
    return vec


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


class UserDB:
    """
    Thin wrapper around Pinecone for user management.

    Parameters
    ----------
    pinecone_api_key : str
    index_name       : str   The same index used for medical knowledge.
    """

    def __init__(self, pinecone_api_key: str, index_name: str = "medical-chatbot"):
        pc = Pinecone(api_key=pinecone_api_key)
        self._idx = pc.Index(index_name)

    # ─────────────────────── Auth ───────────────────────────────────────

    def register(
        self,
        username: str,
        password: str,
        email: str = "",
        age: Optional[int] = None,
        gender: str = "",
    ) -> dict:
        """
        Create a new user.

        Returns
        -------
        {"success": True,  "user_id": str}  on success
        {"success": False, "error":   str}  if username taken
        """
        if self._find_user(username):
            return {"success": False, "error": "Username already exists."}

        user_id = str(uuid.uuid4())
        self._idx.upsert(
            vectors=[
                {
                    "id": f"usr_{user_id}",
                    "values": _zero_vec(),
                    "metadata": {
                        "record_type": "user_profile",
                        "user_id":     user_id,
                        "username":    username,
                        "pwd_hash":    _hash(password),
                        "email":       email,
                        "age":         age if age is not None else 0,
                        "gender":      gender,
                        "created_at":  time.time(),
                    },
                }
            ],
            namespace=USER_NS,
        )
        return {"success": True, "user_id": user_id}

    def login(self, username: str, password: str) -> Optional[dict]:
        """
        Verify credentials.

        Returns the user metadata dict on success, or None on failure.
        """
        user = self._find_user(username)
        if user and user.get("pwd_hash") == _hash(password):
            return user
        return None

    def update_profile(
        self,
        user_id: str,
        age: Optional[int] = None,
        gender: Optional[str] = None,
    ) -> bool:
        """
        Update age and/or gender for an existing user.
        """
        result = self._idx.query(
            vector=_zero_vec(),
            top_k=1,
            namespace=USER_NS,
            filter={"record_type": "user_profile", "user_id": {"$eq": user_id}},
            include_metadata=True,
        )
        if not result.matches:
            return False

        match    = result.matches[0]
        metadata = dict(match.metadata)

        if age is not None:
            metadata["age"] = age
        if gender is not None:
            metadata["gender"] = gender

        self._idx.upsert(
            vectors=[{"id": match.id, "values": _zero_vec(), "metadata": metadata}],
            namespace=USER_NS,
        )
        return True

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        result = self._idx.query(
            vector=_zero_vec(),
            top_k=1,
            namespace=USER_NS,
            filter={"record_type": "user_profile", "user_id": {"$eq": user_id}},
            include_metadata=True,
        )
        return result.matches[0].metadata if result.matches else None

    # ─────────────────────── Chat History ───────────────────────────────

    def save_message(self, user_id: str, role: str, content: str) -> None:
        """
        Persist a single chat message (role = 'user' | 'bot').
        For very long content, splits across multiple vectors with
        chunk_index metadata so they can be recombined on retrieval.
        """
        # Pinecone HTTP client chokes on non-ASCII chars; keep only ASCII
        safe_content = content.encode("ascii", errors="ignore").decode("ascii")
        ts = time.time()
        msg_group_id = str(uuid.uuid4())

        # Pinecone allows ~40KB metadata per vector.
        # We use 10000 chars per chunk which is safe.
        CHUNK_SIZE = 10000
        chunks = [safe_content[i:i+CHUNK_SIZE]
                  for i in range(0, max(len(safe_content), 1), CHUNK_SIZE)]

        vectors = []
        for idx, chunk in enumerate(chunks):
            vectors.append({
                "id": f"msg_{uuid.uuid4()}",
                "values": _zero_vec(),
                "metadata": {
                    "record_type":   "chat_message",
                    "user_id":       user_id,
                    "role":          role,
                    "content":       chunk,
                    "timestamp":     ts,
                    "msg_group_id":  msg_group_id,
                    "chunk_index":   idx,
                    "total_chunks":  len(chunks),
                },
            })
        self._idx.upsert(vectors=vectors, namespace=CHAT_NS)

    def get_history(self, user_id: str, limit: int = 30) -> list[dict]:
        """
        Retrieve the most recent messages for a user, sorted oldest-first.
        Recombines chunked messages into their full content.
        """
        result = self._idx.query(
            vector=_zero_vec(),
            top_k=limit * 3,   # fetch extra to account for chunks
            namespace=CHAT_NS,
            filter={"record_type": "chat_message", "user_id": {"$eq": user_id}},
            include_metadata=True,
        )
        raw = [m.metadata for m in result.matches]
        raw.sort(key=lambda x: (x.get("timestamp", 0), x.get("chunk_index", 0)))

        # Recombine chunks that share the same msg_group_id
        merged = []
        seen_groups = {}
        for msg in raw:
            gid = msg.get("msg_group_id", "")
            total = msg.get("total_chunks", 1)
            if gid and total > 1:
                if gid not in seen_groups:
                    seen_groups[gid] = {"base": dict(msg), "parts": {}}
                seen_groups[gid]["parts"][msg.get("chunk_index", 0)] = msg.get("content", "")
            else:
                merged.append(msg)

        # Reassemble chunked messages
        for gid, info in seen_groups.items():
            base = info["base"]
            parts = info["parts"]
            full_content = "".join(parts[i] for i in sorted(parts.keys()))
            base["content"] = full_content
            merged.append(base)

        merged.sort(key=lambda x: x.get("timestamp", 0))
        return merged[-limit:]   # return the latest N messages

    def clear_history(self, user_id: str) -> bool:
        """
        Delete all chat history for a user.
        Fetches all message vector IDs and deletes them in batches.
        """
        result = self._idx.query(
            vector=_zero_vec(),
            top_k=200,
            namespace=CHAT_NS,
            filter={"record_type": "chat_message", "user_id": {"$eq": user_id}},
            include_metadata=False,
        )
        if not result.matches:
            return False

        ids = [m.id for m in result.matches]
        # Delete in batches of 100 (Pinecone limit)
        for i in range(0, len(ids), 100):
            self._idx.delete(ids=ids[i:i+100], namespace=CHAT_NS)
        return True

    def delete_messages(self, user_id: str, timestamps: list[float]) -> bool:
        """
        Delete specific messages by their timestamps.
        Used to delete a conversation pair (user msg + bot reply).
        """
        result = self._idx.query(
            vector=_zero_vec(),
            top_k=200,
            namespace=CHAT_NS,
            filter={"record_type": "chat_message", "user_id": {"$eq": user_id}},
            include_metadata=True,
        )
        if not result.matches:
            return False

        ts_set = set(timestamps)
        ids_to_delete = [
            m.id for m in result.matches
            if m.metadata.get("timestamp") in ts_set
        ]
        if not ids_to_delete:
            return False

        for i in range(0, len(ids_to_delete), 100):
            self._idx.delete(ids=ids_to_delete[i:i+100], namespace=CHAT_NS)
        return True

    # ─────────────────────── Password Reset ─────────────────────────────

    def find_user_by_email(self, email: str) -> Optional[dict]:
        """Look up a user by email address for password reset."""
        result = self._idx.query(
            vector=_zero_vec(),
            top_k=1,
            namespace=USER_NS,
            filter={"record_type": "user_profile", "email": {"$eq": email}},
            include_metadata=True,
        )
        return result.matches[0].metadata if result.matches else None

    def reset_password(self, user_id: str, new_password: str) -> bool:
        """Update the password for an existing user."""
        result = self._idx.query(
            vector=_zero_vec(),
            top_k=1,
            namespace=USER_NS,
            filter={"record_type": "user_profile", "user_id": {"$eq": user_id}},
            include_metadata=True,
        )
        if not result.matches:
            return False

        match    = result.matches[0]
        metadata = dict(match.metadata)
        metadata["pwd_hash"] = _hash(new_password)

        self._idx.upsert(
            vectors=[{"id": match.id, "values": _zero_vec(), "metadata": metadata}],
            namespace=USER_NS,
        )
        return True

    # ─────────────────────── Google OAuth ──────────────────────────────

    def find_or_create_google_user(self, email: str, name: str) -> dict:
        """
        Find an existing user by email, or create a new one for Google OAuth.

        Returns
        -------
        {"user_id": str, "username": str}
        """
        # Check if a user with this email already exists
        existing = self.find_user_by_email(email)
        if existing:
            return {
                "user_id":  existing["user_id"],
                "username": existing["username"],
            }

        # Create a new user with Google profile info
        # Use email prefix as username, ensure uniqueness
        base_username = name.replace(" ", "_").lower() if name else email.split("@")[0]
        username = base_username

        # If username already taken, append random suffix
        attempt = 0
        while self._find_user(username):
            attempt += 1
            username = f"{base_username}_{uuid.uuid4().hex[:4]}"
            if attempt > 10:
                username = f"google_{uuid.uuid4().hex[:8]}"
                break

        # Random password hash (Google users don't use password login)
        random_hash = _hash(uuid.uuid4().hex)

        user_id = str(uuid.uuid4())
        self._idx.upsert(
            vectors=[
                {
                    "id": f"usr_{user_id}",
                    "values": _zero_vec(),
                    "metadata": {
                        "record_type": "user_profile",
                        "user_id":     user_id,
                        "username":    username,
                        "pwd_hash":    random_hash,
                        "email":       email,
                        "age":         0,
                        "gender":      "",
                        "created_at":  time.time(),
                        "auth_provider": "google",
                    },
                }
            ],
            namespace=USER_NS,
        )
        return {"user_id": user_id, "username": username}

    # ─────────────────────── Internal ───────────────────────────────────

    def _find_user(self, username: str) -> Optional[dict]:
        result = self._idx.query(
            vector=_zero_vec(),
            top_k=1,
            namespace=USER_NS,
            filter={"record_type": "user_profile", "username": {"$eq": username}},
            include_metadata=True,
        )
        return result.matches[0].metadata if result.matches else None

