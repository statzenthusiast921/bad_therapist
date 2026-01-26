from typing import List, Dict, Deque, Any
from collections import deque
from bad_therapist_main import NarcissistTherapist # Assuming your main logic file is named this

# Define the maximum number of sessions to keep
MAX_SESSIONS = 5

class TherapistSessionManager:
    """
    Manages and stores the history for multiple NarcissistTherapist sessions.
    """
    
    def __init__(self):
        # deque automatically limits the size to MAX_SESSIONS
        # Stores completed session histories (List[Dict[str, str]])
        self.past_sessions: Deque[Dict[str, Any]] = deque(maxlen=MAX_SESSIONS) 
        
        # Stores the currently active Therapist instance
        self.active_session: NarcissistTherapist | None = None
        
        # A simple counter for session IDs
        self._session_counter = 0

    def start_new_session(self) -> str:
        """
        Finalizes the old session (if any) and starts a new one.
        Returns the unique ID of the new active session.
        """
        # 1. Finalize the old session and save history
        if self.active_session is not None:
            self._save_active_session()

        # 2. Start a new, clean session
        self._session_counter += 1
        session_id = f"session_{self._session_counter}"
        self.active_session = NarcissistTherapist()
        self.active_session.session_id = session_id # Add ID for tracking
        
        print(f"--- New Session Started: {session_id} ---")
        return session_id

    def send_message_to_active_session(self, user_input: str) -> str:
        """
        Sends a message to the currently active session instance.
        """
        if self.active_session is None:
            raise RuntimeError("No active session started. Call start_new_session first.")
            
        return self.active_session.chat(user_input)

    def _save_active_session(self):
        """
        Saves the history of the current active session to the past_sessions deque.
        """
        if self.active_session and self.active_session.chat_history:
            history_record = {
                "id": self.active_session.session_id,
                "history": self.active_session.chat_history,
                "summary": self._summarize_session(self.active_session.chat_history)
            }
            # The append operation automatically drops the oldest session if the limit is reached
            self.past_sessions.appendleft(history_record) # Use appendleft to keep newest first
            print(f"Session {self.active_session.session_id} saved and archived.")

    def get_past_session_history(self) -> List[Dict[str, Any]]:
        """
        Returns the list of the last 5 archived session histories.
        """
        # Return a simple list of the deque contents
        return list(self.past_sessions)
        
    def _summarize_session(self, history: List[Dict[str, str]]) -> str:
        """
        A placeholder for a function that generates a brief summary 
        (e.g., "Anxiety about job," "Father issues"). 
        In a real app, this would use a small LLM call.
        """
        if len(history) < 2:
            return "Empty or very short session."
            
        # Simplistic summary for demonstration:
        first_user_msg = history[0].get("content", "N/A")
        return f"Session started with: \"{first_user_msg[:40]}...\""