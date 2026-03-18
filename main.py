import os
import json
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ged import detect_errors, openai_llm_call
from gec import gec, llm_generate as gec_llm_generate
from gef import gef, llm_generate as gef_llm_generate

def setup_knowledge_base():
    """
    This function loads our grammar rules from knowledge.txt
    """
    knowledge_file = "knowledge.txt"
    
    if not os.path.exists(knowledge_file):
        print("ERROR: knowledge.txt not found!")
        print("Please make sure knowledge.txt is in the same folder as this script")
        exit()
    
    with open(knowledge_file, 'r', encoding='utf-8') as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,    
        chunk_overlap=200,    
        separators=["[NHÓM LỖI:", "----------", "\n\n"]
    )
    
    docs = splitter.create_documents([text])
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_relevant_rules(db, text, num_rules=3):
    """
    Search for relevant grammar rules based on the input
    """
    results = db.similarity_search(text, k=1)
    if results:
        rag_context = results[0].page_content
    else:
        rag_context = ""
        
    rules_text = "\n\n".join([r.page_content for r in results])
    return rules_text

def detect_errors_module(text):
    """
    Error detection using the ged.py module
    Uses OpenAI API to detect grammar errors in French text
    """
    
    try:

        result = detect_errors(
            text,
            llm_call=lambda p: openai_llm_call(p, model="gpt-4o-mini"),
            lang="fr"
        )
        
        errors = result.get("errors", [])
        
        if not errors:
            return f"No errors detected in: \"{text}\"\n\nGreat job! The sentence looks correct."
        
        output = f"Detected {len(errors)} error(s) in: \"{text}\"\n\n"
        
        for i, err in enumerate(errors, 1):
            output += f"{i}. Error Type: {err['error_type']}\n"
            output += f"   Wrong: \"{err['wrong']}\"\n"
            output += f"   Suggestion: \"{err['suggestion']}\"\n"
            output += f"   Message: {err['message']}\n"
            output += f"   Severity: {err['severity']}\n"
            output += f"   Position: chars {err['start_char']}-{err['end_char']}\n\n"
        
        return output
        
    except Exception as e:
        return f"Error in detection module: {str(e)}\n\nMake sure OPENAI_API_KEY is set in your environment."


def correct_text_module(text):
    try:
        corrected = gec(text)
        
        output = f"Original: \"{text}\"\n\n"
        output += f"Corrected: \"{corrected}\"\n\n"
        
        if corrected == "Corrected sentence placeholder.":
            output += "Note: llm_generate() in gec.py needs to be implemented.\n"
            output += "You can use OpenAI API to generate corrections."
        
        return output
        
    except Exception as e:
        return f"Error in correction module: {str(e)}"


def explain_corrections_module(text, rules_from_db):
    """
    Explanation using the gef.py module + RAG context
    
    This combines:
    1. Relevant grammar rules from the knowledge base (RAG)
    2. GEF (Grammatical Error Feedback) from gef.py
    """
    
    try:

        feedback = gef(text, rag_context=rules_from_db)
        
        output = "=== RETRIEVED GRAMMAR RULES (RAG) ===\n\n"
        output += rules_from_db
        output += "\n\n"
        output += "=== GRAMMATICAL ERROR FEEDBACK (GEF + RAG) ===\n\n"
        output += feedback
        
        return output
        
    except Exception as e:
        return f"Error in explanation module: {str(e)}"


def process_input(text, db):
    if not text or text.strip() == "":
        return "Please enter some French text", "", ""
    
    try:
        print(f"\nProcessing: {text}")
        relevant_rules = get_relevant_rules(db, text, num_rules=3)
        print("Got relevant rules from database")
        
        print("Running error detection (ged.py)...")
        detection = detect_errors_module(text)
        
 
        print("Running correction (gec.py)...")
        correction = correct_text_module(text)
        
        print("Generating explanation (gef.py + RAG)...")
        explanation = explain_corrections_module(text, relevant_rules)
        
        print("Processing complete\n")
        
        return detection, correction, explanation
        
    except Exception as e:
        error_msg = f"Oops, something went wrong: {str(e)}"
        print(error_msg)
        return error_msg, "", ""

def build_interface(db):
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        
        gr.Markdown(
            """
            # French Grammar Checker
            
            ---
            """
        )

        with gr.Accordion("How to use", open=False):
            gr.Markdown(
                """
                Just type a French sentence and click the button. The system will:
                - **Detect errors**
                - **Show corrections**  
                - **Explain the rules** 
                
                **Try this example:**
                `Un coupable idéal, est un film sur les ecosystèmes de la guinée`
                """
            )

        input_box = gr.Textbox(
            label="Enter French text here",
            placeholder="Type a French sentence...",
            lines=3
        )
        
        check_btn = gr.Button("Check Grammar", variant="primary")
   
        gr.Markdown("### Results")
        
        with gr.Row():
            detection_box = gr.Textbox(
                label="Error Detection (ged.py)",
                lines=10,
                interactive=False
            )
            
            correction_box = gr.Textbox(
                label="Corrections (gec.py)",
                lines=10,
                interactive=False
            )
        
        explanation_box = gr.Textbox(
            label="Explanation (gef.py + RAG)",
            lines=15,
            interactive=False
        )
        
        gr.Examples(
            examples=[
                ["Un coupable idéal, est un film sur les ecosystèmes de la guinée"],
                ["Le homme est tres intelligent"],
                ["je vais a paris demain"],
            ],
            inputs=input_box
        )
        
        
        check_btn.click(
            fn=lambda text: process_input(text, db),
            inputs=input_box,
            outputs=[detection_box, correction_box, explanation_box]
        )
        
        input_box.submit(
            fn=lambda text: process_input(text, db),
            inputs=input_box,
            outputs=[detection_box, correction_box, explanation_box]
        )
    
    return app


if __name__ == "__main__":
    print("\n" + "="*50)
    print("French Grammar Checker - Starting up...")
    print("="*50 + "\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found in environment!")
        print("Please set it: export OPENAI_API_KEY='your-key-here'")
        print("The detection module (ged.py) requires this.\n")
    
    vector_db = setup_knowledge_base()

    print("Creating interface...\n")
    interface = build_interface(vector_db)
    
    print("="*50)
    print("Starting the app...")
    print("="*50)
    print("\nOpening in browser...\n")
    
    interface.launch(
        share=True,
        server_port=7860
    )