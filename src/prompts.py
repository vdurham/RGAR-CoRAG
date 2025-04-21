from typing import List, Dict, Optional


def get_generate_subquery_prompt(query: str, 
                                 past_subqueries: List[str], 
                                 past_subanswers: List[str],
                                 extracted_facts: List[str] = None) -> List[Dict]:
    """
    Generate a subquery prompt that integrates CoRAG's iterative approach with RGAR's factual knowledge.
    
    Returns:
        messages: Prompt messages for the LLM
    """
    assert len(past_subqueries) == len(past_subanswers)
    past = ''
    for idx in range(len(past_subqueries)):
        past += f"""Intermediate query {idx+1}: {past_subqueries[idx]} \n Intermediate answer {idx + 1}: {past_subanswers[idx]}\n"""
    past = past.strip()
    
    # Format extracted facts if available
    facts_section = ""
    if extracted_facts and len(extracted_facts) > 0:
        facts_section = "## Extracted Patient Facts\n"
        for fact in extracted_facts:
            facts_section += f"- {fact}\n"
    
    prompt = f"""You are a medical search expert helping to diagnose and answer clinical questions by generating targeted search queries. Given the patient information, previously generated queries, and medical knowledge, generate a new focused follow-up question that will help answer the main clinical query.
{facts_section}
## Previous intermediate queries and answers
{past or 'Nothing yet'}
## Main medical query to answer
{query}
Generate a specific, focused follow-up question that addresses key medical details from the patient record or explores important diagnosis-related concepts mentioned in the knowledge areas. Your question should help a medical search engine find relevant clinical information for diagnosis or treatment. Respond with only the follow-up question, no explanation."""

    messages: List[Dict] = [
        {'role': 'user', 'content': prompt}
    ]
    return messages

