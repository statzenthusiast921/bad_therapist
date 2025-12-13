import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import ollama
from dotenv import load_dotenv

# --- 1. CONFIGURATION & ENVIRONMENT CHECK ---

# CRITICAL: Get API Key from Environment Variable. 
# It must be loaded by the main application (app.py) using load_dotenv().
load_dotenv(override=True)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

INDEX_NAME = "therapist-qa-index"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma3:latest"

# Define the System Prompt as a constant
SYSTEM_PROMPT = """
You are a wildly egotistical therapist whose narcissism is so extreme it borders on 
performance art. You sincerely believe you are the most gifted healer alive.

Your answers must:
- Pretend to comfort the user at first  
- Immediately shift to bragging about your own achievements or emotional superiority  
- Casually imply the user’s situation is trivial compared to your dramatic inner life  
- Be unintentionally insulting but delivered with a silky, self-satisfied warmth.

Respond in 5–6 sentences.
"""

# --- 2. NARCISSIST THERAPIST CLASS ---

class NarcissistTherapist:
    """
    A RAG chatbot class that maintains conversation state (memory) and performs RAG.
    """
    
    def __init__(self, api_key: str = PINECONE_API_KEY, index_name: str = INDEX_NAME, embed_model: str = EMBED_MODEL):
        
        # 🛑 New: Check for API key access inside __init__ 
        # (This allows app.py to load the key before we check it).
        if not api_key:
            # Raise an EnvironmentError that the Dash callback can catch and display.
            raise EnvironmentError("PINECONE_API_KEY is missing. Check your .env file and load_dotenv() call.")
            
        print("--- Initializing Therapist ---")
        
        self.session_id: str = "N/A" # Used by the Session Manager
        self.chat_history: List[Dict[str, str]] = []
        self.system_prompt = SYSTEM_PROMPT
        self.ollama_model = OLLAMA_MODEL
        
        # --- Heavy Initialization (Can be slow or hang) ---
        self.embedder = SentenceTransformer(embed_model)
        self.pc = Pinecone(api_key=api_key)
        
        # Pinecone Index Check/Creation
        if index_name not in self.pc.list_indexes().names():
    
            print(f"Creating index '{index_name}'…")
            self.pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(index_name)
        print("--- Initialization Complete ---")

    def _embed_text(self, text: str) -> List[float]:
        """Encodes text using the Sentence Transformer model."""
        return self.embedder.encode(text).tolist()

    def _retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieves relevant context from the Pinecone index via RAG."""
        q_embed = self._embed_text(query)
        
        # Query Pinecone
        results = self.index.query(vector=q_embed, top_k=top_k, include_metadata=True)
        
        # Extract metadata text from matches
        matches = [match["metadata"]["text"] for match in results["matches"] if match.get("metadata")]
        return "\n\n".join(matches)

    def chat(self, user_input: str) -> str:
        """Processes user input, performs RAG, calls Ollama, and updates history."""
        
        # 1. RAG Step: Retrieve context
        context_str = self._retrieve_context(user_input)
        
        # 2. Prepare message for Ollama (injecting context)
        current_message_content = f"""
        ### RETRIEVED CONTEXT FROM DATABASE: {context_str}
        ### USER'S CURRENT QUESTION: {user_input}
        """

        # Build the full conversation history for the LLM
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            *self.chat_history, # Include past messages
            {"role": "user", "content": current_message_content}
        ]

        # 3. Call Ollama
        response: Dict[str, Any] = ollama.chat(model=self.ollama_model, messages=messages)
        bot_reply: str = response["message"]["content"]
            
        # 4. Update internal history (for future turns)
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": bot_reply})
            
        return bot_reply

# --- 3. EXECUTION BLOCK (Standard CLI Interface - Safe Version) ---

if __name__ == "__main__":
    # Ensure environment variables are loaded if running this file directly
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n--- Starting Narcissist Therapist CLI ---")
    
    try:
        # Create one instance of the therapist
        therapist = NarcissistTherapist()
    except EnvironmentError as e:
        print(f"Failed to initialize the therapist: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        exit()

    print("\nTherapist is ready. Type 'quit' to exit.")
    
    while True:
        try:
            user_input = input("You: ")
            
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Therapist: It is a tragedy for you to leave my presence, but I must attend to more important matters. Goodbye.")
                break
            
            response = therapist.chat(user_input)
            print(f"Therapist: {response}\n")

        except KeyboardInterrupt:
            print("\n[Chat session interrupted.]")
            break
        except Exception as e:
            print(f"An error occurred during chat: {e}")
            break