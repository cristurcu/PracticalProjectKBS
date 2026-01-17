"""
Prompt templates for MedRAG replication study.
Based on templates from the original MedRAG implementation.
"""

from jinja2 import Template

# System prompts
SYSTEM_PROMPTS = {
    "cot": """You are an expert medical professional. You are given a medical question with multiple-choice options.
Think through the problem step-by-step and provide your answer.""",

    "rag": """You are an expert medical professional. You are given a medical question along with relevant documents from medical literature.
Use the provided documents to help answer the question. Think through the problem step-by-step and provide your answer.""",
}

# Chain-of-Thought template (no RAG)
COT_TEMPLATE = Template("""Given the following medical question, think step-by-step and select the correct answer.

## Question
{{ question }}

## Options
{% for key, value in options.items() %}
{{ key }}. {{ value }}
{% endfor %}

Provide your response in the following JSON format:
{"step_by_step_thinking": "your reasoning here", "answer_choice": "A/B/C/D"}

Your response:""")

# RAG template with context
RAG_TEMPLATE = Template("""Given the following medical question and relevant documents, think step-by-step and select the correct answer.

## Relevant Documents
{% for doc in documents %}
Document [{{ loop.index }}] (Title: {{ doc.title }})
{{ doc.content }}

{% endfor %}

## Question
{{ question }}

## Options
{% for key, value in options.items() %}
{{ key }}. {{ value }}
{% endfor %}

Based on the provided documents and your medical knowledge, provide your response in the following JSON format:
{"step_by_step_thinking": "your reasoning here", "answer_choice": "A/B/C/D"}

Your response:""")

# PubMedQA specific templates (yes/no/maybe format)
COT_TEMPLATE_PUBMEDQA = Template("""Given the following biomedical research question, think step-by-step and determine whether the answer is yes, no, or maybe.

## Question
{{ question }}

{% if context %}
## Context
{{ context }}
{% endif %}

Provide your response in the following JSON format:
{"step_by_step_thinking": "your reasoning here", "answer_choice": "yes/no/maybe"}

Your response:""")

RAG_TEMPLATE_PUBMEDQA = Template("""Given the following biomedical research question and relevant documents, think step-by-step and determine whether the answer is yes, no, or maybe.

## Relevant Documents
{% for doc in documents %}
Document [{{ loop.index }}] (Title: {{ doc.title }})
{{ doc.content }}

{% endfor %}

## Question
{{ question }}

{% if context %}
## Original Context
{{ context }}
{% endif %}

Based on the provided documents and your biomedical knowledge, provide your response in the following JSON format:
{"step_by_step_thinking": "your reasoning here", "answer_choice": "yes/no/maybe"}

Your response:""")


def get_template(dataset_type: str, use_rag: bool) -> Template:
    """Get the appropriate template based on dataset and RAG setting."""
    if dataset_type == "pubmedqa":
        return RAG_TEMPLATE_PUBMEDQA if use_rag else COT_TEMPLATE_PUBMEDQA
    else:
        return RAG_TEMPLATE if use_rag else COT_TEMPLATE


def get_system_prompt(use_rag: bool) -> str:
    """Get the appropriate system prompt."""
    return SYSTEM_PROMPTS["rag"] if use_rag else SYSTEM_PROMPTS["cot"]
