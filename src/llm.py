"""
LLM interface for MedRAG replication.
Uses OpenAI GPT-3.5-turbo for answer generation.
"""
import json
import re
import time
from typing import Dict, List, Optional, Tuple, Any

from openai import OpenAI
import tiktoken

from .config import OPENAI_API_KEY, LLM_CONFIG
from .templates import get_template, get_system_prompt
from .retrieval import RetrievedDoc


class LLMInterface:
    """Interface for LLM-based question answering."""

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or LLM_CONFIG["model"]
        self.api_key = api_key or OPENAI_API_KEY

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model(self.model)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.tokenizer.encode(text))

    def truncate_context(self, documents: List[RetrievedDoc],
                         max_tokens: int = 3500) -> List[RetrievedDoc]:
        """Truncate documents to fit within token limit."""
        truncated = []
        total_tokens = 0

        for doc in documents:
            doc_text = f"Document [{doc.rank}] (Title: {doc.title})\n{doc.content}\n\n"
            doc_tokens = self.count_tokens(doc_text)

            if total_tokens + doc_tokens > max_tokens:
                break

            truncated.append(doc)
            total_tokens += doc_tokens

        return truncated

    def generate(self, prompt: str, system_prompt: str,
                 temperature: float = 0, max_tokens: int = 500) -> str:
        """Generate a response from the LLM."""
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.last_request_time = time.time()
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating response: {e}")
            # Retry once after a delay
            time.sleep(2)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.last_request_time = time.time()
                return response.choices[0].message.content
            except Exception as e2:
                print(f"Retry failed: {e2}")
                return ""

    def parse_answer(self, response: str, dataset_type: str = "mcq") -> str:
        """Parse the answer from LLM response."""
        if not response:
            return ""

        # Try to parse JSON response
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*"answer_choice"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                answer = data.get("answer_choice", "").strip().upper()
                # For PubMedQA, map back to yes/no/maybe
                if dataset_type == "pubmedqa":
                    answer_map = {"A": "yes", "B": "no", "C": "maybe",
                                  "YES": "yes", "NO": "no", "MAYBE": "maybe"}
                    return answer_map.get(answer, answer.lower())
                return answer
        except json.JSONDecodeError:
            pass

        # Fallback: look for answer patterns
        response_upper = response.upper()

        if dataset_type == "pubmedqa":
            # Look for yes/no/maybe
            for answer in ["yes", "no", "maybe"]:
                if answer in response.lower():
                    return answer
            return ""

        # Look for letter answer
        patterns = [
            r'answer[_\s]*(?:choice)?[:\s]*["\']?([A-D])["\']?',
            r'\b([A-D])\s*(?:is|\.|\)|:)',
            r'(?:^|\s)([A-D])(?:\s|$|\.)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                return match.group(1)

        return ""

    def answer_question(self, question: str, options: Dict[str, str],
                        documents: Optional[List[RetrievedDoc]] = None,
                        dataset_type: str = "mcq",
                        context: Optional[str] = None) -> Tuple[str, str, Dict]:
        """
        Answer a medical question.

        Args:
            question: The question text
            options: Answer options (e.g., {"A": "...", "B": "..."})
            documents: Optional retrieved documents for RAG
            dataset_type: Type of dataset (mcq or pubmedqa)
            context: Optional context (for PubMedQA)

        Returns:
            Tuple of (predicted_answer, raw_response, metadata)
        """
        use_rag = documents is not None and len(documents) > 0

        # Get appropriate template
        template = get_template(dataset_type, use_rag)
        system_prompt = get_system_prompt(use_rag)

        # Truncate documents if necessary
        if use_rag:
            documents = self.truncate_context(documents)

        # Render prompt
        if use_rag:
            prompt = template.render(
                question=question,
                options=options,
                documents=[{"title": d.title, "content": d.content} for d in documents],
                context=context,
            )
        else:
            prompt = template.render(
                question=question,
                options=options,
                context=context,
            )

        # Generate response
        raw_response = self.generate(prompt, system_prompt)

        # Parse answer
        predicted_answer = self.parse_answer(raw_response, dataset_type)

        metadata = {
            "model": self.model,
            "use_rag": use_rag,
            "num_documents": len(documents) if documents else 0,
            "prompt_tokens": self.count_tokens(prompt),
        }

        return predicted_answer, raw_response, metadata
