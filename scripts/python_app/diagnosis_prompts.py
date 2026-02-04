"""
Prompts and utilities for generating Dr. Vain's diagnosis from session history.
"""

# --- DIAGNOSIS SYSTEM PROMPT ---
DIAGNOSIS_SYSTEM_PROMPT = """You are Dr. Thaddeus Vain, a therapist writing a formal diagnosis report. Write in third person, using "The patient..." language. Avoid using "I", "you", or first/second person pronouns.

Structure your response with exactly two paragraphs:

**First Paragraph - DIAGNOSIS:**
Provide a brief assessment of the patient's psychological state based on the session history. Write in clinical but condescending language that subtly suggests the patient's issues are related to Narcissistic Personality Disorder, while implying that only a therapist of exceptional caliber (like yourself) could properly address them. This paragraph should be 3-5 sentences (maximum 5 sentences).

**Second Paragraph - RECOMMENDATIONS:**
Always conclude with a recommendation that the patient should spend more time with Dr. Vain to benefit from his superior therapeutic expertise. Frame it as essential for the patient's progress. This paragraph should be 3-5 sentences (maximum 5 sentences).

The entire response must be exactly 2 paragraphs - one for diagnosis and one for recommendations. Maintain a narcissistic undertone through subtle condescension and self-aggrandizement, but keep the language formulaic and professional."""


def format_conversation_history(past_sessions):
    """
    Format past session history into a text string for use in prompts.
    
    Args:
        past_sessions: List of session dictionaries, each containing a "history" key
                      with list of message dicts (role/content)
    
    Returns:
        str: Formatted conversation text
    """
    conversation_text = ""
    for session_idx, session in enumerate(past_sessions, 1):
        history = session.get("history", [])
        conversation_text += f"\n--- Session {session_idx} ---\n"
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                conversation_text += f"Patient: {content}\n\n"
            elif role == "assistant":
                conversation_text += f"Dr. Vain: {content}\n\n"
    return conversation_text


def generate_diagnosis_user_prompt(conversation_text):
    """
    Generate the user prompt for diagnosis generation.
    
    Args:
        conversation_text: Formatted conversation history string
    
    Returns:
        str: User prompt for Ollama
    """
    return f"""Based on the following therapy sessions, provide a formal diagnosis report:

{conversation_text}

Write a diagnosis report with exactly two paragraphs: the first paragraph for DIAGNOSIS and the second paragraph for RECOMMENDATIONS. Use third-person language ("The patient..."). Each paragraph should be 3-5 sentences (maximum 5 sentences). The recommendation paragraph must suggest that the patient should spend more time with Dr. Vain to continue their therapeutic progress."""

