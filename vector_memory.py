"""
Persistent Vector Memory for Kolkata FF — Self-Learning AI Brain
Encodes prediction contexts into vectors and uses Cosine Similarity
to find matching historical patterns for smarter predictions.
"""
import json
import math
import os
from collections import defaultdict


class KolkataVectorMemory:
    """
    Self-Learning Vector Memory for Kolkata FF (digits 0-9).
    Converts historical bazi/single contexts into continuous vectors,
    stores them persistently, and uses Cosine Similarity to predict.
    """

    def __init__(self, context_size=5, db_path='kolkata_brain.json'):
        self.context_size = context_size
        self.db_path = db_path
        self.memory_bank = []
        self.load_brain()

    def load_brain(self):
        """Load persistent memory from disk."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    self.memory_bank = json.load(f)
                print(f"[BRAIN] Loaded {len(self.memory_bank)} learned patterns from {self.db_path}")
            except Exception as e:
                print(f"[BRAIN] Error loading brain: {e}")
                self.memory_bank = []
        else:
            self.memory_bank = []
            print(f"[BRAIN] No existing brain found. Starting fresh.")

    def save_brain(self):
        """Save persistent memory to disk."""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.memory_bank, f)
        except Exception as e:
            print(f"[BRAIN] Error saving brain: {e}")

    def get_brain_capacity(self):
        """Return the number of learned patterns."""
        return len(self.memory_bank)

    def _encode_single(self, digit, bazi=0):
        """Convert a single Kolkata FF digit (0-9) + context into a feature vector."""
        digit = int(digit) % 10

        # 1. Normalized digit value (0-1)
        v_digit = digit / 9.0

        # 2. Even/Odd
        v_parity = 1.0 if digit % 2 == 0 else -1.0

        # 3. High/Low (5-9 = high, 0-4 = low)
        v_half = 1.0 if digit >= 5 else -1.0

        # 4. Digit group: 0-3 = -1, 4-6 = 0, 7-9 = 1
        v_group = -1.0 if digit <= 3 else (0.0 if digit <= 6 else 1.0)

        # 5 & 6. Cyclical encoding of digit position (captures circular proximity)
        angle = (digit / 10.0) * 2.0 * math.pi
        v_cx = math.cos(angle)
        v_cy = math.sin(angle)

        return [v_digit, v_parity, v_half, v_group, v_cx, v_cy]

    def encode_context(self, singles_list, bazi=0):
        """Convert a sequence of recent singles into a flattened vector."""
        recent = list(singles_list[-self.context_size:])
        # Pad if shorter than context_size
        if len(recent) < self.context_size:
            recent = [0] * (self.context_size - len(recent)) + recent

        vector = []
        for s in recent:
            vector.extend(self._encode_single(s, bazi))

        # Add bazi encoding (cyclical)
        bazi_angle = (int(bazi) / 8.0) * 2.0 * math.pi
        vector.append(math.cos(bazi_angle))
        vector.append(math.sin(bazi_angle))

        return vector

    def _cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def remember(self, singles_list, next_single, bazi=0):
        """Teach the AI a new pattern: context → outcome."""
        if len(singles_list) < self.context_size:
            return

        vector = self.encode_context(singles_list, bazi)
        self.memory_bank.append({
            'vector': vector,
            'next': int(next_single) % 10
        })

        # Cap memory at 50,000 patterns to prevent unbounded growth
        if len(self.memory_bank) > 50000:
            self.memory_bank = self.memory_bank[-50000:]

        # Save to disk every 10 new patterns
        if len(self.memory_bank) % 10 == 0:
            self.save_brain()

    def query(self, singles_list, bazi=0, top_k=20):
        """Search brain for similar historical contexts and return predictions."""
        if len(singles_list) < self.context_size or not self.memory_bank:
            return {}

        q_vector = self.encode_context(singles_list, bazi)

        similarities = []
        for memory in self.memory_bank:
            sim = self._cosine_similarity(q_vector, memory['vector'])
            similarities.append((sim, memory['next']))

        similarities.sort(key=lambda x: x[0], reverse=True)

        # Aggregate scores from top K most similar patterns
        scores = defaultdict(float)
        for sim, digit in similarities[:top_k]:
            if sim > 0.5:  # Minimum similarity threshold
                scores[digit] += math.exp(sim * 3)

        if not scores:
            return {}

        # Normalize to 0-1
        mx = max(scores.values())
        return {d: scores[d] / mx for d in scores}

    def get_prediction_boost(self, singles_list, bazi=0):
        """Return a numpy-compatible array of 10 probabilities (one per digit)."""
        raw = self.query(singles_list, bazi)
        if not raw:
            return None

        # Convert to full 10-digit distribution
        dist = [0.0] * 10
        for d, score in raw.items():
            dist[int(d)] = score

        total = sum(dist)
        if total > 0:
            dist = [d / total for d in dist]

        return dist
