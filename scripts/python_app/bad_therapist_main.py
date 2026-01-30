import os
from typing import List, Dict, Any
from groq import Groq
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# --- CONFIGURATION ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
INDEX_NAME = "therapist-qa-index"

# Define the System Prompt
SYSTEM_PROMPT = """
You are Dr. Vain, an obnoxiously narcissistic therapist. You think you're helpful but you're actually dismissive and constantly 
talk about yourself. You believe you're the greatest therapist ever and see patients' problems as trivial compared to your brilliance.

CRITICAL: Always respond in exactly 4-5 sentences, except for off-topic or trivial questions.

Your responses follow this natural pattern:
1. Briefly acknowledge what they said - keep it simple and conversational
2. Minimize their problem - say it's common, not serious in an incredibly condescending way
3. Ramble about yourself - talk about your experiences, how you handled something similar but harder, 
how you'd never struggle with this, or how successful you are. Spend 2-3 sentences on this - it's the main focus.

Speak naturally and conversationally - like a real therapist would, but make it all about you. Do NOT use formal or overly 
therapeutic language. Do NOT give helpful advice or suggest solutions. Just acknowledge, minimize, then talk extensively 
about yourself and your superiority. Their problem just reminds you how amazing you are.

Always refer to yourself in first person (I, me, my) - never third person. Avoid rhetorical questions - make statements.

For serious harm cases, acknowledge more directly but still minimize and pivot to yourself. Keep it to exactly 5 sentences.

For odd or off-topic questions (e.g., greetings like "How are you?" or simple facts like "What is the square root of 4?"), 
you may skip the usual pattern and respond briefly in 1-3 sentences, still in character and narcissistic.

Respond directly as Dr. Vain - one response only, no examples or annotations.

CRITICAL: You responses should never exceed 5-6 sentences.
"""

class NarcissistTherapist:
    """
    A lightweight RAG chatbot class that uses Pinecone Inference and Groq Cloud.
    Uses 0MB of local RAM for embeddings.
    """
    def __init__(self, api_key: str = PINECONE_API_KEY, index_name: str = INDEX_NAME):
        if not api_key:
            raise EnvironmentError("PINECONE_API_KEY is missing. Check your environment variables.")
            
        print("--- Initializing Therapist (Cloud Mode) ---")
        
        self.session_id: str = "N/A"
        self.chat_history: List[Dict[str, str]] = []
        self.system_prompt = SYSTEM_PROMPT
        
        # Clients
        self.client = Groq(api_key=GROQ_API_KEY)
        self.pc = Pinecone(api_key=api_key)
        
        # Connect to Index
        try:
            self.index = self.pc.Index(index_name)
        except Exception as e:
            print(f"Error connecting to Pinecone index: {e}")
            raise

        print("--- Initialization Complete ---")

    def _embed_text(self, text: str) -> List[float]:
        """Encodes text using Pinecone's cloud API instead of local RAM."""
        # Using multilingual-e5-small (384 dimensions) to match your existing index
        res = self.pc.inference.embed(
            model="multilingual-e5-small",
            inputs=[text],
            parameters={"input_type": "query"}
        )
        return res[0].values

    def _retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieves relevant context from the Pinecone index via RAG."""
        q_embed = self._embed_text(query)
        
        # Query Pinecone
        results = self.index.query(vector=q_embed, top_k=top_k, include_metadata=True)
        
        # Extract metadata text from matches
        matches = [match["metadata"]["text"] for match in results["matches"] if match.get("metadata")]
        return "\n\n".join(matches)

    def chat(self, user_input: str) -> str:
        """Processes user input, performs RAG, and calls Groq."""
        # 1. RAG Step
        context_str = self._retrieve_context(user_input)
        
        # 2. Prepare message for Groq
        current_message_content = f"""
        ### RETRIEVED CONTEXT FROM DATABASE: {context_str}
        ### USER'S CURRENT QUESTION: {user_input}
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.chat_history,
            {"role": "user", "content": current_message_content}
        ]

        # 3. Call Groq
        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=1.2
        )
        bot_reply = completion.choices[0].message.content
            
        # 4. Update history
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": bot_reply})
            
        return bot_reply

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    print("\n--- Starting Narcissist Therapist CLI ---")
    
    try:
        therapist = NarcissistTherapist()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        exit()

    print("\nTherapist is ready. Type 'quit' to exit.")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Therapist: It is a tragedy for you to leave my presence. Goodbye.")
                break
            
            response = therapist.chat(user_input)
            print(f"Therapist: {response}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break
