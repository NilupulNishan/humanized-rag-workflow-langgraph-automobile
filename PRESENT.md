# VivoAssist — AI Vehicle Support Assistant
### Proof of Concept Presentation

---

## What This Is

An AI-powered chat assistant that answers questions about your vehicle using the owner's manual as its primary knowledge source — with the ability to go beyond the manual when needed.

This is a **proof of concept**. Prompts, confidence thresholds, and agent behaviours are not yet fine-tuned for production. The architecture is production-ready — the calibration is not.

---

## How It Works — 4 Intelligent Agents

Every message the user sends passes through a coordinated pipeline of four specialised agents. Each agent has one job and hands off cleanly to the next.

---

### Agent 1 — The Router
**Query Understanding Agent**

Every question first passes through this agent. It reads what the user asked, understands the intent behind it, and decides which path to take.

It distinguishes between:
- A how-to question → step-by-step guide
- A troubleshooting problem → diagnostic walkthrough
- A comparison with another vehicle → web search
- General conversation → direct response, no manual needed
- An ambiguous question → asks one focused clarifying question before proceeding

No question goes to the wrong place.

---

### Agent 2 — The Manual Search Agent
**Retriever + Knowledge Agent**

When the question is about the vehicle, this agent searches the owner's manual using **semantic search** — not keyword matching.

It understands meaning. "My car won't start in the cold" finds the relevant section even if those exact words don't appear in the manual. Results are ranked by relevance, not position in the document.

---

### Agent 3 — The Web Search Agent
**External Knowledge Agent**

When the question goes beyond the manual — comparisons with other cars, latest pricing, external specs — this agent searches the web in real time.

It automatically activates when:
- The user asks to compare this vehicle against another
- The manual has no relevant information
- The answer requires up-to-date external data

Results are synthesised into a structured response, not a list of links.

---

### Agent 4 — The Answer Agent
**Response Renderer**

Takes everything retrieved and produces a clear, human-friendly response. Not a raw document dump — a structured, conversational answer with the right tone, citations, and formatting for the question type.

- How-to questions → numbered steps with expected outcomes
- Troubleshooting → calm, progressive diagnostic flow
- Comparisons → structured table with verdict
- Escalations → warm handoff with clear next steps

---

## What It Handles Today

| Scenario | How It's Handled |
|---|---|
| Question answered by the manual | Answered with page references |
| Question the manual doesn't cover | Web search fallback |
| Comparison with another vehicle | Real-time web search with structured table |
| General conversation or world knowledge | Handled directly, manual not touched |
| Ambiguous question | One clarifying question asked before searching |
| Nothing found anywhere | Clear escalation message with next steps |
| Official specs and pricing | Links to official BAIC Sri Lanka dealer pages |

---

## Current Limitations

These are known and intentional for this proof of concept stage.

- **No persistent conversation memory** — each session starts fresh. The assistant does not remember previous conversations or what the user has already tried across sessions
- **No response caching** — repeated questions are processed fresh each time, adding latency
- **Prompts not production-tuned** — confidence thresholds, response tone, and agent behaviour are set to reasonable defaults but not calibrated against real user data
- **Manual coverage limited** — currently loaded with BAIC X55 and BAIC BJ30 manuals only

---

## Future Roadmap

### Near Term
- **Conversation memory** — assistant remembers past sessions, tracks what the user has already tried, and builds on previous conversations naturally
- **Response caching** — common questions answered instantly without calling the AI model, reducing cost and latency significantly

### Medium Term
- **Human escalation agent** — when a user is stuck and cannot resolve an issue, a dedicated agent detects frustration signals and triggers a live handoff. The user is connected directly to a human support agent via chat, or a callback is initiated — all within the same interface, no transfer to a separate system
- **Proactive alerts** — push notifications for service reminders, recalls, or relevant updates based on the user's specific vehicle and history
- **Multi-vehicle support** — personalised experience per user with their specific vehicle profile stored and recalled automatically

### Longer Term
- **Voice interface** — voice commands and spoken responses for hands-free use while in or near the vehicle
- **Analytics dashboard** — full visibility into what users ask most, where the assistant struggles, which manual sections need improvement, and escalation rates
- **Multi-language support** — responses in the user's preferred language, with the same manual as the source
- **Dealer integration** — direct connection to dealer inventory, booking systems, and service history

---

## The Bigger Picture

This is not a chatbot.

It is a **multi-agent system** where each agent has a specific job, a defined scope, and a clear handoff protocol. The architecture is designed to scale — new agents, new knowledge sources, and new channels can be added without rebuilding what already works.

The foundation built here — semantic search over manuals, intent-based routing, web search fallback, and structured response generation — is the same foundation that scales to a full customer support platform.

---

*VivoAssist — Proof of Concept · Built on LangGraph · Azure OpenAI · ChromaDB*