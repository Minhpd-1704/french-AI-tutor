import os
import time

def gec(sentence: str) -> str:
    instruction = (
        "You are a grammar correction AI specialized in French. "
        "Correct the following French sentence to the most natural and "
        "grammatically correct French form. "
        "Return ONLY the corrected sentence in French."
    )
    prompt = f"Instruction: {instruction}\nInput: {sentence}\nOutput:"
    corrected_sentence = llm_generate(prompt)
    return corrected_sentence

def llm_generate(prompt: str) -> str:
    """
    Calls OpenAI API to generate correction
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
                    {"role": "system", "content": "You are a French grammar correction expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            corrected = response.choices[0].message.content.strip()
            return corrected
            
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                return f"ERROR: OpenAI API call failed: {str(e)}"
    
    return "ERROR: Failed after 3 attempts"