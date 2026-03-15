from agent.nodes.query_understanding import query_understanding_node
state = {"user_input": "wifi", "session": {}}
result = query_understanding_node(state)
print(result["analysis"]["intent"])          # → troubleshooting
print(result["analysis"]["specificity"])     # → short
print(result["analysis"]["expanded_queries"]) # → 4 variants
print(result["effective_query"])