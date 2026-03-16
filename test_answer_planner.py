# Build a realistic state from previous node outputs
state = {
    "raw_answer": "To connect to wifi, go to settings menu page 12...",
    "source_nodes": [...],  # from retriever test
    "analysis": {"intent": "how_to", "answer_mode": "guided", "inferred_topic": "wifi setup"},
    "session": None,
    "retrieval_successful": True
}
from agent.nodes.answer_planner import answer_planner_node
result = answer_planner_node(state)
print(result["plan"]["mode"])        # → step_by_step
print(result["plan"]["confidence"])  # → 0.8
print(result["plan"]["steps"])       # → ["Go to Settings", "Select Wi-Fi", ...]