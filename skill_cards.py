"""Read-only, hash-verified Markdown skill cards and declarative run profiles.

Follows strict Tool/Skill separation: physical tools belong solely to the benchmark
adapter; skill cards are inspectable read-only guidance and never carry current
scene evidence, coordinates, or execution authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SKILL_SCOPE = (
    "Public generic guidance only; not current scene evidence, coordinates, "
    "pose, or contact authorization."
)
MAX_CARD_BYTES = 32_768
PROFILE_FIELDS = {
    "id",
    "description",
    "suite",
    "task",
    "initial_index",
    "instruction",
    "mode",
    "max_attempts",
    "max_actions",
    "max_calls",
    "wall_seconds",
    "rewrite",
    "clarify",
    "retrieve",
    "demo_library",
    "skills_dir",
    "skill_ids",
    "gate",
}


def _card_summary(text_body: str) -> tuple[str, str]:
    lines = [line.strip() for line in text_body.splitlines() if line.strip()]
    if not lines:
        return "", ""
    title = lines[0].lstrip("#").strip()
    bullets = [line.lstrip("-* ").strip() for line in lines[1:] if line]
    summary = bullets[0] if bullets else title
    return title[:120], summary[:200]


class SkillLibrary:
    """Hash-verified directory of read-only Markdown skill cards."""

    def __init__(self, root: str | Path, *, expected_sha256: dict[str, str] | None = None):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"skill directory not found: {self.root}")
        self.cards: dict[str, dict] = {}
        for path in sorted(self.root.glob("*.md")):
            resolved = path.resolve()
            if not resolved.is_relative_to(self.root):
                raise ValueError(f"skill card escapes root: {path.name}")
            raw = resolved.read_bytes()
            if not raw or len(raw) > MAX_CARD_BYTES:
                raise ValueError(f"invalid skill card size: {path.name}")
            sha = hashlib.sha256(raw).hexdigest()
            skill_id = path.stem
            if expected_sha256 is not None and expected_sha256.get(skill_id) != sha:
                raise ValueError(f"skill card digest mismatch: {skill_id}")
            body = raw.decode("utf-8").strip()
            title, summary = _card_summary(body)
            self.cards[skill_id] = dict(
                id=skill_id,
                file=path.name,
                sha256=sha,
                title=title or skill_id,
                summary=summary,
                text=body,
            )
        if not self.cards:
            raise ValueError(f"no Markdown skill cards found in {self.root}")
        manifest = json.dumps(
            {k: v["sha256"] for k, v in sorted(self.cards.items())}, sort_keys=True
        ).encode()
        self.digest = hashlib.sha256(manifest).hexdigest()

    def catalog(self) -> list[dict]:
        return [
            {k: card[k] for k in ("id", "title", "summary", "sha256")}
            for card in self.cards.values()
        ]

    def get(self, skill_id: str) -> dict:
        if skill_id not in self.cards:
            raise ValueError(f"unknown skill card: {skill_id}")
        return dict(self.cards[skill_id])

    def context_parts(self, skill_ids: list[str] | tuple[str, ...]) -> list[dict]:
        parts = []
        seen = set()
        for skill_id in skill_ids:
            if not skill_id or skill_id in seen:
                continue
            seen.add(skill_id)
            card = self.get(skill_id)
            payload = dict(
                kind="read_only_skill_guidance",
                skill_id=card["id"],
                sha256=card["sha256"],
                scope=SKILL_SCOPE,
                guidance=card["text"],
            )
            parts.append({"type": "text", "text": json.dumps(payload, ensure_ascii=False)})
        return parts


def load_profile(path: str | Path, *, base_dir: Path | None = None) -> dict:
    """Load and validate a declarative JSON run profile."""
    profile_path = Path(path).expanduser().resolve()
    data = json.loads(profile_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("profile must be a JSON object")
    unknown = set(data) - PROFILE_FIELDS
    if unknown:
        raise ValueError(f"unsupported profile fields: {sorted(unknown)}")
    anchor = (base_dir or profile_path.parent).resolve()
    result = dict(data)
    for key in ("skills_dir", "demo_library"):
        if result.get(key):
            candidate = Path(result[key]).expanduser()
            if not candidate.is_absolute():
                candidate = (
                    (anchor / candidate).resolve()
                    if (anchor / candidate).exists()
                    else (profile_path.parent / candidate).resolve()
                )
            result[key] = str(candidate)
    if "skill_ids" in result:
        ids = result["skill_ids"]
        if not isinstance(ids, list) or not all(isinstance(s, str) and s.strip() for s in ids):
            raise ValueError("profile skill_ids must be a list of non-empty strings")
    return result
