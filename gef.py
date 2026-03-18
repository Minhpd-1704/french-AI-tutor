import os
import time

def gef(sentence: str, rag_context: str = "") -> str:
    """
    Grammatical Error Feedback (GEF) for French sentences.
    Now with optional RAG context for better explanations!

    Input:
        sentence (str): A French sentence written by a learner.
        rag_context (str): Optional grammar rules from RAG knowledge base

    Output:
        str: Grammatical error feedback
    """

    instruction = (
        "You are a professional French teacher. "
        "I will provide you with a learner's sentence and some grammar reference blocks (RAG context).\n"
        "YOUR TASK:\n"
        "1. Analyze the input sentence for errors.\n"
        "2. Look at the RAG context. ONLY use information that is DIRECTLY relevant to the error in the input.\n"
        "3. If the context contains many examples, pick the ONE most similar example to explain.\n"
        "4. Provide a clear, pedagogical explanation."
    )
    
    if rag_context:
        instruction += f"\n\nRELEVANT GRAMMAR RULES & EXAMPLES:\n{rag_context}\n"
    
    instruction += (
        "Format your answer exactly as:\n"
        "Error(s): ...\n"
        "Explanation: ...\n"
        "Correction: ...\n"
        "Rule: ..."
    )

    prompt = f"Learner Input: {sentence}\n\nProvide feedback based on the instruction above:"
    feedback = llm_generate(prompt)
    return feedback


def llm_generate(prompt: str) -> str:
    """
    Calls OpenAI API to generate pedagogical explanation
    """
    api_key = os.getenv("OPENAI_API_KEY") # điền api key vào đây
    if not api_key:
        return "ERROR: OPENAI_API_KEY not set. Please set it in your environment."
    
    try:
        from openai import OpenAI
    except ImportError:
        return "ERROR: OpenAI library not installed. Run: pip install openai"
    
    client = OpenAI(api_key=api_key)
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a patient French teacher who gives clear, educational explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500
            )
            
            feedback = response.choices[0].message.content.strip()
            return feedback
            
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                return f"ERROR: OpenAI API call failed: {str(e)}"
    
    return "ERROR: Failed after 3 attempts"