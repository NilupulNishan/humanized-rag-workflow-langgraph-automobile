from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class SessionData(BaseModel):

    session_id: str

    # Product context
    product_model: Optional[str] = None
    collection_name: Optional[str] = None

    # Problem context
    issue_summary: Optional[str] = None
    attempted_steps: list[str] = Field(default_factory=list)
    current_stage: str = "initial"  # initial|diagnosing|resolving|escalated|resolved

    # Conversation meta
    turn_count: int = 0
    last_intent: Optional[str] = None
    preferred_detail_level: str = "normal"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))   # BUG FIX: utcnow() is deprecated
    last_active: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # BUG FIX: utcnow() is deprecated

    def to_context_string(self) -> str:
        """
        Session info → compact string injected into prompts.
        Tells the LLM what it already knows about this user's situation.
        """
        parts = []
        if self.product_model:
            parts.append(f"Product: {self.product_model}")
        if self.issue_summary:
            parts.append(f"Issue: {self.issue_summary}")
        if self.attempted_steps:
            parts.append(f"Already tried: {', '.join(self.attempted_steps)}")
        if self.current_stage != "initial":
            parts.append(f"Stage: {self.current_stage}")

        if not parts:
            return ""
        return "[Session context: " + " | ".join(parts) + "]"

    def mark_step_attempted(self, step: str) -> None:
        """Record that user has tried this step — avoid repeating it."""
        normalized = step.strip().lower()
        if normalized not in [s.lower() for s in self.attempted_steps]:
            self.attempted_steps.append(step)

    def advance_stage(self, new_stage: str) -> None:
        """Move troubleshooting forward."""
        valid = {"initial", "diagnosing", "resolving", "escalated", "resolved"}
        if new_stage in valid:
            self.current_stage = new_stage

    def touch(self) -> None:
        """Update last_active timestamp — call this on every turn."""
        self.last_active = datetime.now(timezone.utc)

    model_config = {"arbitrary_types_allowed": True}