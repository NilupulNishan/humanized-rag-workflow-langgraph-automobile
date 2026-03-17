from agent.graph import build_graph
graph = build_graph(use_persistence=False)

result = graph.invoke({
    "user_input": "how do I connect to wifi",
    "session_id": "test-1",
    "collection_name": "your_collection",
    "messages": []
})
print(result["final_response"])
print(result["plan"]["mode"])