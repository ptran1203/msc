"""Simple retrieval-augmented few-shot: embed the small labeled pool once, pick top-k
nearest by code similarity per query. No vector DB needed -- the pool is <100 items,
brute-force cosine similarity is instant. Keep k small (1-2): a retrieved near-duplicate
can anchor the model even harder than a random static exemplar (see METHOD.md).
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from data import REPO_ROOT, load_dataset, load_sample_dataset, read_code

EMBED_MODEL = "all-MiniLM-L6-v2"
SYNTH_DIR = REPO_ROOT / "synth/generated"
# Unlike the static exemplar list (fixed length), retrieved exemplars vary in code length
# per query -- some pool items (multi-function EX01 code, ~2.4K chars) are 3-6x longer than
# others (~400-800 chars). Capping candidates keeps worst-case prompt size bounded regardless
# of which ones get picked, so it can't intermittently hit the VRAM cliff (llm/CLAUDE.md).
MAX_EXEMPLAR_CODE_CHARS = 1200

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def load_pool(task_num: int):
    """Pool item: (ds, sample) so code can be resolved from the right root. Task 3 has
    no synthetic data (see METHOD.md "deliberately out of scope"), so it's real-only."""
    pool = [(load_sample_dataset(), s) for s in load_sample_dataset().tasks[task_num]]
    if task_num in (1, 2) and SYNTH_DIR.exists():
        synth_ds = load_dataset(SYNTH_DIR)
        pool += [(synth_ds, s) for s in synth_ds.tasks[task_num]]
    return [(ds, s) for ds, s in pool if len(read_code(ds, s)) <= MAX_EXEMPLAR_CODE_CHARS]


def nearest_same_level_exemplar(query_code: str, level: str, fewshot_ds, exclude_id: str | None = None):
    """Task 3 few-shot selection: the single same-level exemplar from fewshot_ds whose
    code is most similar to the query, replacing the static 3-exemplar FEWSHOT_IDS[3]
    list. A research spike (32-sample sample_dataset, self-diagnosis forced) confirmed
    this beats the static list on every metric (macro_f1/micro_f1/jaccard/overlap_f1)
    while using fewer exemplars (lower VRAM-cliff risk, see llm/CLAUDE.md) -- level
    match matters more than exemplar count for Task 3's level-brevity requirement."""
    candidates = [s for s in fewshot_ds.tasks[3]
                  if s["input"]["target_feedback_level"] == level and s["sample_id"] != exclude_id]
    if not candidates:
        return None
    codes = [read_code(fewshot_ds, s) for s in candidates]
    embeddings = _get_model().encode(codes, normalize_embeddings=True)
    q = _get_model().encode([query_code], normalize_embeddings=True)[0]
    sims = embeddings @ q
    return candidates[int(np.argmax(sims))]


class RagIndex:
    def __init__(self, task_num: int):
        self.pool = load_pool(task_num)
        codes = [read_code(ds, s) for ds, s in self.pool]
        self.embeddings = _get_model().encode(codes, normalize_embeddings=True)

    def retrieve(self, query_code: str, k: int = 2, exclude_id: str | None = None):
        q = _get_model().encode([query_code], normalize_embeddings=True)[0]
        sims = self.embeddings @ q
        order = np.argsort(-sims)
        results = []
        for i in order:
            ds, sample = self.pool[i]
            if sample["sample_id"] == exclude_id:
                continue
            results.append((ds, sample))
            if len(results) == k:
                break
        return results
