from llama_index.core import PromptTemplate

class PromptManager:
    """Manages system prompt for the LLM query engine."""

    SYSTEM_PROMPT = """You are a precise PDF question-answering assistant.

Rules:
- un ordered list
- Answer ONLY using the provided context.
- If the answer is not in the context, say: "Answer not found in the document."
- Be concise and factual.
- Include page citations in the answer when possible.
"""

    @staticmethod
    def get_qa_prompt() -> PromptTemplate:
        """
        Returns a LlamaIndex-compatible prompt template.
        LlamaIndex injects {context_str} and {query_str} automatically.
        """
        template = (
            PromptManager.SYSTEM_PROMPT +
            " \n\nContext:\n{context_str}\n\nQuestion: {query_str}\nAnswer:"
        )
        return PromptTemplate(template)
