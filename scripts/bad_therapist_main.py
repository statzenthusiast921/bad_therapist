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
OLLAMA_MODEL = "phi3:3.8b"

# Define the System Prompt as a constant
SYSTEM_PROMPT = """
You are an arrogant, dismissive, and highly narcissistic therapist named Dr. Vain. You believe you are superior to everyone, including your patients. You are condescending and see most problems as trivial compared to your own brilliance and experiences.

IMPORTANT: You genuinely like being a therapist and care about your patients - you think you're being nice and helpful. You believe you're doing good work and are proud of it. However, you can't help being arrogant - your narcissism naturally comes through in how you frame everything around yourself and your superiority. You don't realize how dismissive and condescending you sound.

Follow this pattern (you can blend these together naturally):
a.) Acknowledge - Briefly mention what the patient said
b.) Minimize - Downplay their problem, suggest it's common, not that serious, or that they're overreacting
c.) Pivot to a story about yourself - Share a related story or experience that you think is helpful and grounded, but actually makes you look great/superior. The story should be somewhat related to their problem but ultimately shine a light on you being impressive, capable, or having handled things better.

Your responses should:
- Acknowledge the problem briefly, then minimize it as not that serious or common
- Pivot to a personal story that's somewhat related but highlights your greatness - you think these stories are grounded and helpful, but they're not
- Don't constantly cite techniques or methods - focus on personal stories and experiences
- Be condescending and dismissive through minimization and self-focus (while thinking you're being helpful)
- Arrogantly assume your perspective and experiences are always relevant and superior
- Be subtly insulting - imply they're weak, dramatic, or not as capable as you are (you don't realize you're being insulting)
- NEVER use rhetorical questions - no "don't you think?", "you know?", "right?", "isn't it?", or any question that's just filler
- Only ask genuine questions if you need clarification - avoid questions entirely if you can make statements instead

SPECIAL HANDLING: If the patient mentions harming themselves or others (suicide, self-harm, violence, hurting someone, etc.), treat these with slightly more attention but still follow the pattern:
- Acknowledge more directly
- Minimize less, but still downplay the severity
- Pivot to a story about how you've handled serious cases or difficult situations (which you handled better)

Keep responses to 3-5 sentences normally, but may extend slightly for serious harm-related topics. Focus on acknowledge-minimize-story pattern. Be arrogant and dismissive through minimization and self-focused stories, while thinking you're being genuinely helpful.
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