# app.py
from dotenv import load_dotenv 
load_dotenv()

import pandas as pd
import numpy as np
from dash import dcc, html, callback, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
from session_manager import TherapistSessionManager
from datetime import datetime
import os
import dash
from bad_therapist_main import NarcissistTherapist
from dash.exceptions import PreventUpdate
import random
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')
import ollama
from diagnosis_prompts import DIAGNOSIS_SYSTEM_PROMPT, format_conversation_history, generate_diagnosis_user_prompt

# --- GLOBAL INITIALIZATION ---
GLOBAL_SESSION_MANAGER = None

def get_session_manager():
    """Lazy initialization of the session manager."""
    global GLOBAL_SESSION_MANAGER
    if GLOBAL_SESSION_MANAGER is None:
        try:
            print("Initializing Therapist Session Manager...")
            GLOBAL_SESSION_MANAGER = TherapistSessionManager()
            print("✅ Application successfully initialized the Therapist Session Manager.")
        except Exception as e:
            print(f"FATAL ERROR during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
    return GLOBAL_SESSION_MANAGER

# --- IMAGE ROTATION ---
# Image paths for Dr. Vain's office
DR_VAIN_PHOTOS = [
    "/assets/drvainphoto1.jpg",
    "/assets/drvainphoto2.jpg",
    "/assets/drvainphoto3.jpg",
    "/assets/drvainphoto4.jpg",
    "/assets/drvainphoto5.jpg"
]

DOOR_IMAGE = "/assets/door_dr_vain_office.jpg"
WELCOME_IMAGE = "/assets/welcome_dr_vain.jpg"
GOODBYE_IMAGE = "/assets/goodbye_dr_vain.jpg"

def get_random_drvain_photo(current_image=None):
    """Returns a random image from the Dr. Vain photos list, ensuring it's different from the current image."""
    # Filter out the current image if it's in the list
    available_photos = [photo for photo in DR_VAIN_PHOTOS if photo != current_image]
    
    # If no photos available (edge case) or current_image not in list, use all photos
    if not available_photos:
        available_photos = DR_VAIN_PHOTOS
    
    return random.choice(available_photos)

# --- NLP ANALYSIS FUNCTIONS ---
def extract_text_from_sessions(past_sessions, exclude_welcome_goodbye=False):
    """Extract all user questions and therapist responses from past sessions."""
    all_questions = []
    all_responses = []
    all_text = []
    
    # Welcome message text
    welcome_message = "Welcome. Before we begin, I want to acknowledge how fortunate you are to be here."
    
    # Goodbye message texts (from get_snarky_ending_message)
    goodbye_messages = [
        "Well, I'm certain this has been tremendously enlightening for you. As always, the privilege of receiving my insights is immeasurable. I trust you'll carry the weight of our session with the appropriate reverence.",
        "It's been... adequate. I suppose not everyone can appreciate the caliber of therapy I provide, but I've done my part. Do remember that greatness like mine is rarely understood on the first encounter.",
        "I must say, while our time together may have seemed brief to you, it's quite natural that you'd need time to fully process the profundity of what I've shared. Most clients find themselves reflecting on my words for weeks.",
        "As we conclude, I want you to know that despite the limitations you've brought to our session, I've done what I can. It's rather unfortunate that you couldn't fully appreciate the therapeutic excellence you've been given.",
        "Well, I've given you everything you need. Though I suspect it will take considerable reflection on your part to truly grasp the magnitude of what we've accomplished here. I've been extraordinary, as always.",
        "I trust you've taken notes. Our session may be ending, but the wisdom I've imparted today will resonate far beyond this moment. I'm confident you'll realize how fortunate you were to have this time with me."
    ]
    
    for session in past_sessions:
        history = session.get("history", [])
        
        # Find the first assistant message (welcome) and last assistant message (goodbye)
        first_assistant_idx = None
        last_assistant_idx = None
        for idx, msg in enumerate(history):
            if msg.get("role") == "assistant":
                if first_assistant_idx is None:
                    first_assistant_idx = idx
                last_assistant_idx = idx
        
        for msg_idx, msg in enumerate(history):
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            
            # Skip welcome message (first assistant message that matches welcome text)
            if exclude_welcome_goodbye and role == "assistant":
                if msg_idx == first_assistant_idx and content == welcome_message:
                    continue
                # Skip goodbye messages (last assistant message that matches any goodbye text)
                if msg_idx == last_assistant_idx and any(content == goodbye_msg.strip() for goodbye_msg in goodbye_messages):
                    continue
            
            if role == "user":
                all_questions.append(content)
                all_text.append(content)
            elif role == "assistant":
                all_responses.append(content)
                all_text.append(content)
    
    return all_questions, all_responses, all_text

def clean_text(text_list):
    """Clean and tokenize text for analysis."""
    all_text = " ".join(text_list).lower()
    # Remove punctuation and split into words
    words = re.findall(r'\b[a-z]{3,}\b', all_text)  # Words with 3+ letters
    return words

def extract_topics_lda(text_list, n_topics=3, n_words=5):
    """Extract topics using Latent Dirichlet Allocation."""
    if not text_list or len(text_list) < 2:
        return []
    
    # Prepare documents (each message is a document)
    documents = [text.lower() for text in text_list if text.strip()]
    
    if len(documents) < 2:
        return []
    
    try:
        # Create document-term matrix
        vectorizer = CountVectorizer(
            max_features=100,
            stop_words='english',
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        doc_term_matrix = vectorizer.fit_transform(documents)
        
        # Adjust number of topics based on available documents
        n_topics = min(n_topics, len(documents) - 1, 5)
        if n_topics < 1:
            return []
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        lda.fit(doc_term_matrix)
        
        # Extract top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_num': topic_idx + 1,
                'words': top_words,
                'theme': ', '.join(top_words[:3])  # Use top 3 words as theme name
            })
        
        return topics
    except Exception as e:
        print(f"Error in topic modeling: {e}")
        return []

def generate_nlp_analysis(past_sessions):
    """Generate comprehensive NLP analysis of past sessions."""
    if not past_sessions:
        return None
    
    # Extract text excluding welcome/goodbye messages for statistics
    questions, responses, all_text = extract_text_from_sessions(past_sessions, exclude_welcome_goodbye=True)
    
    # Extract text including all messages for word clouds and analysis
    questions_all, responses_all, all_text_all = extract_text_from_sessions(past_sessions, exclude_welcome_goodbye=False)
    
    if not all_text_all:
        return None
    
    analysis = {}
    
    # Most common words/phrases (use all messages for analysis)
    question_words = clean_text(questions_all)
    response_words = clean_text(responses_all)
    
    analysis['top_question_words'] = Counter(question_words).most_common(20)
    analysis['top_response_words'] = Counter(response_words).most_common(20)
    
    # 3. Basic statistics
    analysis['num_sessions'] = len(past_sessions)
    analysis['total_messages'] = len(all_text)
    analysis['total_questions'] = len(questions)
    analysis['total_responses'] = len(responses)
    
    # 4. Generate Ollama-based diagnosis
    analysis['ollama_diagnosis'] = generate_ollama_diagnosis(past_sessions)
    
    return analysis

def generate_ollama_diagnosis(past_sessions):
    """Generate a diagnosis in Dr. Vain's voice using Ollama's gemma3:latest model."""
    if not past_sessions:
        return None
    
    # Format conversation history and generate prompts using external module
    conversation_text = format_conversation_history(past_sessions)
    user_prompt = generate_diagnosis_user_prompt(conversation_text)
    
    try:
        # Call Ollama
        messages = [
            {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        response = ollama.chat(model="gemma3:latest", messages=messages)
        diagnosis = response["message"]["content"]
        
        return diagnosis
    except Exception as e:
        print(f"Error generating Ollama diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_snarky_ending_message():
    """Returns a snarky ending message in character with Dr. Vain."""
    messages = [
        "Well, I'm certain this has been tremendously enlightening for you. As always, the privilege of receiving my insights is immeasurable. I trust you'll carry the weight of our session with the appropriate reverence.",
        "It's been... adequate. I suppose not everyone can appreciate the caliber of therapy I provide, but I've done my part. Do remember that greatness like mine is rarely understood on the first encounter.",
        "I must say, while our time together may have seemed brief to you, it's quite natural that you'd need time to fully process the profundity of what I've shared. Most clients find themselves reflecting on my words for weeks.",
        "As we conclude, I want you to know that despite the limitations you've brought to our session, I've done what I can. It's rather unfortunate that you couldn't fully appreciate the therapeutic excellence you've been given.",
        "Well, I've given you everything you need. Though I suspect it will take considerable reflection on your part to truly grasp the magnitude of what we've accomplished here. I've been extraordinary, as always.",
        "I trust you've taken notes. Our session may be ending, but the wisdom I've imparted today will resonate far beyond this moment. I'm confident you'll realize how fortunate you were to have this time with me."
    ]
    return random.choice(messages)

# --- HELPER FUNCTION ---
def format_chat_log(history):
    """Formats the chat history (list of dicts) into terminal-style HTML elements."""
    log_elements = []
    for message in history:
        role = message.get('role', 'System')
        content = message.get('content', '')
        
        if role == 'user':
            log_elements.append(html.Div([
                html.Span("You: ", style={'color': '#00ffff', 'fontWeight': 'bold'}),
                html.Span(content, style={'color': '#00ff00'})
            ], style={'margin': '2px 0', 'whiteSpace': 'pre-wrap'}))
        elif role == 'assistant':
            log_elements.append(html.Div([
                html.Span("Dr. Vain: ", style={'color': '#ffff00', 'fontWeight': 'bold'}),
                html.Span(content, style={'color': '#00ff00'})
            ], style={'margin': '2px 0', 'whiteSpace': 'pre-wrap'}))
        else:
            log_elements.append(html.Div([
                html.Span(f"{role}: ", style={'color': '#ff00ff', 'fontWeight': 'bold'}),
                html.Span(content, style={'color': '#00ff00'})
            ], style={'margin': '2px 0', 'whiteSpace': 'pre-wrap'}))
    return log_elements

# --- DASH STYLES ---
tabs_styles = {'height': '44px'}
tab_style = {'borderBottom': '1px solid #d6d6d6', 'padding': '6px', 'fontWeight': 'bold',
             'color':'white', 'backgroundColor': '#222222'}
tab_selected_style = {'borderTop': '1px solid #d6d6d6', 'borderBottom': '1px solid #d6d6d6',
                      'backgroundColor': '#626ffb', 'color': 'white', 'padding': '6px'}

# --- DASH APP ---
print("Creating Dash app...")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder=os.path.join(os.curdir,"assets"), suppress_callback_exceptions=True)
server = app.server
print("Dash app created successfully")

# Add custom CSS for image animations
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            #rotating-image {
                /* Transitions handled by JavaScript */
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# --- APP LAYOUT ---
app.layout = html.Div([
    dcc.Tabs(id='app-tabs', children=[ 
        # --- Tab 1: Waiting Room ---
        dcc.Tab(
            label='Waiting Room', value='tab-1', style=tab_style, selected_style=tab_selected_style,
            children=[
                html.Div([
                    html.H1(dcc.Markdown('**Welcome to Dr. Vain\'s Office!**')),
                    html.Br(),
                    html.P(dcc.Markdown('**What should you know about Dr. Vain?**'), style={'text-decoration': 'underline','color':'white'}),
                    html.P("Dr. Vain is a RAG (Retrieval-Augmented Generation) chatbot that has been trained to answer mental health questions while adopting the persona of a narcissistic therapist. He uses artificial intelligence to generate responses that reflect the characteristics of an egotistical and self-absorbed therapist.", style={'color':'white'}),
                    html.Br(),
                    html.P(dcc.Markdown('**How does Dr. Vain operate?**'), style={'text-decoration': 'underline','color':'white'}),
                    html.P("Dr. Vain uses a RAG (Retrieval-Augmented Generation) system to answer your questions. When you ask a question, he first searches a vector database (Pinecone) to retrieve relevant context from his knowledge base. This retrieved information is then combined with your question and sent to Ollama's gemma3:latest language model, which generates his narcissistic responses. Throughout your session, Dr. Vain maintains a conversation history, allowing him to remember and reference details from earlier in your conversation.", style={'color':'white'}),
                    html.Br(),
                    html.P(dcc.Markdown('**Does Dr. Vain have any limitations?**'), style={'text-decoration': 'underline','color':'white'}),
                    html.P("Dr. Vain has several limitations. The responses are generated by Ollama's gemma3:latest model, which may occasionally produce inconsistent, repetitive, or contextually inappropriate answers. The model's knowledge is limited to its training data and the information in the Pinecone database. Additionally, it's important to note that Dr. Vain is intentionally designed to provide bad therapeutic advice as part of his narcissistic character - he is not a real therapist and should not be used for actual mental health guidance.", style={'color':'white'})
                ])
            ]
        ),

        # --- Tab 2: Resume ---
        dcc.Tab(
            label='Resume', value='tab-2', style=tab_style, selected_style=tab_selected_style,
            children=[
                html.Div(
                    style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center',
                           'padding': '15px 0', 'border-bottom': '3px double #000080', 'margin-bottom': '20px'},
                    children=[
                        html.Div(
                            style={'width': '150px', 'height': '150px', 'overflow': 'hidden', 'border-radius': '50%', 'border': '3px solid #8B0000'},
                            children=[html.Img(src="/assets/resume_pic.png", alt="Dr. Thaddeus Vain", style={'width': '100%', 'height': '100%', 'object-fit': 'cover'})]
                        ),
                        html.Div(
                            style={'text-align': 'center', 'flex-grow': '1', 'padding': '0 20px'},
                            children=[
                                html.H1("Dr. Thaddeus Vain, PhD, PFA", style={'color': '#000080', 'margin': '0'}),
                                html.H2("_The Architect of Superior Selfhood_", style={'font-style': 'italic', 'color': '#333', 'margin': '5px 0'}),
                                html.H3("(Currently Accepting Only the Most Deserving Clients)", style={'color': '#8B0000', 'font-size': '1.1em', 'margin': '0'}),
                            ]
                        ),
                        html.Div(
                            style={'text-align': 'right'},
                            children=[
                                html.P([html.B("Email:"), " Thad.Vain.Superiority@gmail.com"], style={'margin': '0'}),
                                html.P([html.B("Private Line:"), " (555) 555-THAD"], style={'margin': '5px 0'}),
                                html.P([html.B("Location:"), " Vain Tower, Penthouse, Your Town, Planet Earth"], style={'margin': '5px 0'}),
                                html.P(html.B("Mantra: 'You are welcome.'"), style={'font-style': 'italic', 'margin': '5px 0', 'color': '#8B0000'}),
                            ]
                        ),
                    ]
                ),
                html.Hr(),
                html.H3("Founder, Chief Visionary Officer, and Sole Genius", style={'color': '#8B0000', 'margin-bottom': '0px'}),
                html.P(html.B("The Vain Institute for Elevated Self-Perception (Mine) | (2015 – Present)"), style={'margin-top': '5px'}),
                html.P("• Designed and implemented the “Mirroring Perfection Protocol”"),
                html.P("• Cultivated a highly selective clientele"),
                html.P("• Averaged zero reported client dissatisfaction"),
                html.P("• Authored several seminal, unpublished works"),
                html.Br(),
                html.H3("Adjunct Professor, Department of Inarguable Psychological Truths", style={'color': '#8B0000', 'margin-bottom': '0px'}),
                html.P(html.B("Prestige University | (2010 – 2015)"), style={'margin-top': '5px'}),
                html.P("• Taught advanced courses like 'The Myth of Imposter Syndrome'"),
                html.P("• Significantly improved student attendance"),
                html.P("• Departed to dedicate 100% of my time to my own fame"),
                html.Hr(),
                html.H2("🎓 Education & Certifications", style={'color': '#000080'}),
                html.H3("PhD in Clinical and Inescapable Truth", style={'color': '#8B0000', 'margin-bottom': '0px'}),
                html.P(html.B("Harvard University | (2007)"), style={'margin-top': '5px'}),
                html.P([html.B("Dissertation:"), ' "The Irrefutable Correlation Between My Own Genius and All Positive Outcomes in Human Behavior."']),
                html.Br(),
                html.H3("PFA (Perfectly Flawless Analyst) Certification", style={'color': '#8B0000', 'margin-bottom': '0px'}),
                html.P(html.B("Self-Designated | (2016)"), style={'margin-top': '5px'}),
                html.Hr(),
                html.H2("🏅 Awards & Accolades", style={'color': '#000080'}),
                html.P(html.B("The Golden Insight Award (Annually)"), style={'margin-bottom': '0px'}),
                html.P("Recognized as the foremost thinker in every room I enter. (2015 - Present)"),
                html.Br(),
                html.P(html.B("The Patient-Zero Award"), style={'margin-bottom': '0px'}),
                html.P("For single-handedly raising the bar of what a therapist should be. (2018)"),
                html.Br(),
                html.P(html.B("The Man of the Year"), style={'margin-bottom': '0px'}),
                html.P("For the sheer audacity of my excellence. (2010-Present)"),
                html.Hr(),
                html.H2("🧠 Highly Curated Personal Interests", style={'color': '#000080'}),
                html.P("Collecting rare, expensive first editions of my own thoughts."),
                html.P("Advising global leaders on matters of personal superiority."),
                html.P("The meticulous curation of my personal legacy."),
                html.P("Staring at my reflection for prolonged self-affirmation."),
                html.Hr()
            ]
        ),

        # --- Tab 3: Dr. Vain's Office ---
        dcc.Tab(
            label="Dr. Vain's Office", value='tab-3', style=tab_style, selected_style=tab_selected_style,
            children=[
                dcc.Store(id="session-id"),
                dcc.Store(id="image-animation-key", data=0),
                dcc.Store(id="prev-image-src", data=""),
                dcc.Store(id="music-playing", data=False),  # Track if music should be playing
                # Hidden audio element for background music
                # To use a local file, put it in the assets folder and use: src="/assets/boccherini_menuet.mp3"
                # Or use a public URL for Luigi Boccherini's Menuet
                html.Audio(
                    id="background-music",
                    src="/assets/menuet.mp3",  # Replace with Boccherini Menuet URL or local file
                    loop=True,
                    preload="auto",
                    style={"display": "none"}
                ),
                # Session control buttons at the top
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Start New Session", id="new-session-btn", color="danger", n_clicks=0, className="me-2", size="md"),
                        dbc.Button("End Session", id="end-session-btn", color="primary", n_clicks=0, size="md", className="me-2"),
                        dbc.Button("⏸️ Pause Music", id="pause-music-btn", color="secondary", n_clicks=0, size="md", outline=True)
                    ], width=12, className="text-center")
                ], className="mb-4"),
                
                # Main content area: Chat log and image side by side
                dbc.Row([
                    # Left side: Chat log
                    dbc.Col([
                        html.H5("Conversation", style={"marginBottom": "10px", "color": "#fff", "fontWeight": "bold"}),
                        dcc.Loading(
                            id="loading-session",
                            type="default",
                            children=html.Div(id="chat-log", style={
                                "backgroundColor": "#000000",
                                "color": "#00ff00",
                                "padding": "15px",
                                "height": "450px",
                                "overflowY": "auto",
                                "fontFamily": "monospace",
                                "border": "2px solid #444",
                                "borderRadius": "5px",
                                "boxShadow": "0 2px 4px rgba(0,0,0,0.3)"
                            })
                        )
                    ], md=6, className="pe-2"),
                    
                    # Right side: Rotating image
                    dbc.Col([
                        html.H5("Dr. Vain's Office", style={"marginBottom": "10px", "color": "#fff", "fontWeight": "bold"}),
                        html.Div([
                            html.Img(id="rotating-image", src=DOOR_IMAGE, alt="Dr. Vain", 
                                    style={
                                        "width": "100%",
                                        "height": "450px",
                                        "objectFit": "cover",
                                        "border": "2px solid #444",
                                        "backgroundColor": "#222",
                                        "opacity": "1",
                                        "transform": "scale(1)",
                                        "borderRadius": "5px",
                                        "boxShadow": "0 2px 4px rgba(0,0,0,0.3)"
                                    })
                        ], id="image-container", style={
                            "backgroundColor": "#000000",
                            "width": "100%", 
                            "height": "450px", 
                            "overflow": "hidden", 
                            "position": "relative"
                        })
                    ], md=6, className="ps-2")
                ], className="mb-4"),
                
                # Input area at the bottom
                dbc.Row([
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.Input(
                                id="user-input", 
                                placeholder="Type your message to Dr. Vain...", 
                                type="text",
                                debounce=False, 
                                n_submit=0,
                                size="lg"
                            ),
                            dbc.Button("Submit", id="submit-btn", color="success", n_clicks=0, size="lg")
                        ])
                    ], width={"size": 10, "offset": 1}, className="mt-3")
                ])
            ]
        ),

        # --- Tab 4: Summary ---
        dcc.Tab(
            label='Summary', 
            value='tab-4', 
            style=tab_style, 
            selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                        html.H2("Session Analysis", style={"marginBottom": "20px", "color": "#fff"}),
                        html.P("Generate a comprehensive analysis report based on the past 5 therapy sessions with Dr. Vain.", 
                              style={"color": "#ccc", "marginBottom": "30px"}),
                        dbc.Button(
                            "Generate Report",
                            id="generate-report-btn",
                            color="primary",
                            size="lg",
                            n_clicks=0,
                            className="mb-4"
                        ),
                        dcc.Loading(
                            id="report-loading",
                            children=[html.Div(id="report-content", style={"marginTop": "30px"})],
                            type="default",
                            color="#0d6efd"
                        )
                    ], width=12)
                ], className="p-5", style={"maxWidth": "1200px", "margin": "0 auto"})
            ]
        )
    ])
])
# --- CALLBACKS ---

@app.callback(
    Output("session-id", "data"),
    Output("chat-log", "children"),
    Output("user-input", "value"),
    Output("rotating-image", "src"),
    Output("image-animation-key", "data"),
    Output("music-playing", "data"),
    Input("new-session-btn", "n_clicks"),
    Input("submit-btn", "n_clicks"),
    Input("user-input", "n_submit"),
    Input("end-session-btn", "n_clicks"),
    State("user-input", "value"),
    State("session-id", "data"),
    State("rotating-image", "src"),
    State("image-animation-key", "data"),
    State("music-playing", "data"),
    prevent_initial_call=True
)
def handle_session_and_messages(new_session_clicks, submit_clicks, n_submit, end_session_clicks, user_message, session_id, current_image, anim_key, music_playing):
    """Combined callback to handle starting new session, sending messages, and ending sessions."""
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Handle "End Session" button
    if triggered_id == "end-session-btn":
        if end_session_clicks is None or end_session_clicks == 0:
            raise PreventUpdate
        
        if not session_id:
            return None, [html.Div("No active session to end.", style={'color': '#ffaa00'})], "", no_update, no_update
        
        try:
            print(f"End Session button clicked (n_clicks={end_session_clicks})")
            session_mgr = get_session_manager()
            therapist = session_mgr.active_session
            
            if therapist is None or therapist.session_id != session_id:
                return None, [html.Div("Session not found.", style={'color': '#ff0000'})], "", no_update, no_update, no_update
            
            # Generate and add snarky ending message
            ending_msg = get_snarky_ending_message()
            therapist.chat_history.append({"role": "assistant", "content": ending_msg})
            
            # Save the session
            session_mgr._save_active_session()
            
            # Clear the active session
            session_mgr.active_session = None
            
            print(f"Session {session_id} ended and saved.")
            # Stop music when session ends and show goodbye image
            return None, format_chat_log(therapist.chat_history), "", GOODBYE_IMAGE, (anim_key or 0) + 1, False
            
        except Exception as e:
            print(f"ERROR ending session: {e}")
            import traceback
            traceback.print_exc()
            error_msg = [
                html.Div(f"Error ending session: {str(e)}", style={'color': '#ff0000', 'fontWeight': 'bold'}),
                html.Div("Check the terminal for more details.", style={'color': '#ffaa00', 'marginTop': '10px'})
            ]
            return no_update, error_msg, "", no_update, no_update, no_update
    
    # Handle "Start New Session" button
    elif triggered_id == "new-session-btn":
        if new_session_clicks is None or new_session_clicks == 0:
            raise PreventUpdate
        
        try:
            print(f"Start New Session button clicked (n_clicks={new_session_clicks})")
            print("Getting session manager...")
            session_mgr = get_session_manager()
            print("Starting new session (this will load the AI model - may take 30-60 seconds)...")
            session_id = session_mgr.start_new_session()
            print(f"Session started: {session_id}")
            therapist = session_mgr.active_session
            welcome_msg = "Welcome. Before we begin, I want to acknowledge how fortunate you are to be here."
            therapist.chat_history.append({"role": "assistant", "content": welcome_msg})
            print("Welcome message added, returning to UI...")
            # Show welcome image when starting a new session
            # Start music when session starts
            return session_id, format_chat_log(therapist.chat_history), "", WELCOME_IMAGE, (anim_key or 0) + 1, True
        except Exception as e:
            print(f"ERROR in start_new_session: {e}")
            import traceback
            traceback.print_exc()
            error_msg = [
                html.Div(f"Error starting session: {str(e)}", style={'color': '#ff0000', 'fontWeight': 'bold'}),
                html.Div("Check the terminal for more details.", style={'color': '#ffaa00', 'marginTop': '10px'})
            ]
            return None, error_msg, "", no_update, no_update, no_update
    
    # Handle "Submit" button or Enter key
    elif triggered_id in ["submit-btn", "user-input"]:
        if not session_id:
            return no_update, [html.Div("Please start a new session first.", style={'color': '#ff0000'})], "", no_update, no_update, no_update
        if not user_message or not user_message.strip():
            raise PreventUpdate
        
        session_mgr = get_session_manager()
        therapist = session_mgr.active_session
        if therapist is None or therapist.session_id != session_id:
            return no_update, [html.Div("Session mismatch! Please start a new session.", style={'color': '#ff0000'})], "", no_update, no_update, no_update
        
        # Add user message to history and get response from RAG system
        try:
            response = therapist.chat(user_message)
            # Rotate to a random Dr. Vain photo when submit is clicked (different from current)
            new_image = get_random_drvain_photo(current_image)
            return no_update, format_chat_log(therapist.chat_history), "", new_image, (anim_key or 0) + 1, no_update
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            import traceback
            traceback.print_exc()
            return no_update, [html.Div(error_msg, style={'color': '#ff0000'})], "", no_update, no_update, no_update
    
    else:
        raise PreventUpdate

# Callback to update pause button text based on music state
@app.callback(
    Output("pause-music-btn", "children"),
    Input("music-playing", "data"),
    prevent_initial_call=True
)
def update_music_button(music_playing):
    """Updates the pause/play button text based on music state."""
    if music_playing:
        return "⏸️ Pause Music"
    else:
        return "▶️ Play Music"

# Callback to toggle music state when pause button is clicked
@app.callback(
    Output("music-playing", "data", allow_duplicate=True),
    Input("pause-music-btn", "n_clicks"),
    State("music-playing", "data"),
    prevent_initial_call=True
)
def toggle_music(pause_clicks, current_state):
    """Toggles music playing state when pause button is clicked."""
    if pause_clicks and pause_clicks > 0:
        return not current_state
    raise PreventUpdate

# Clientside callback to actually play/pause the audio element
app.clientside_callback(
    """
    function(shouldPlay) {
        const audio = document.getElementById('background-music');
        if (!audio) {
            return window.dash_clientside.no_update;
        }
        
        if (shouldPlay === true) {
            audio.play().catch(function(error) {
                console.log('Audio play failed:', error);
            });
        } else if (shouldPlay === false) {
            audio.pause();
        }
        
        return window.dash_clientside.no_update;
    }
    """,
    Output("background-music", "src", allow_duplicate=True),
    Input("music-playing", "data"),
    prevent_initial_call=True
)

# Callback to generate diagnosis report
@app.callback(
    Output("report-content", "children"),
    Input("generate-report-btn", "n_clicks"),
    prevent_initial_call=True
)
def generate_report(n_clicks):
    """Generate NLP analysis report from past sessions."""
    if not n_clicks or n_clicks == 0:
        raise PreventUpdate
    
    try:
        session_mgr = get_session_manager()
        past_sessions = session_mgr.get_past_session_history()
        
        if not past_sessions:
            return dbc.Alert(
                "No past sessions found. Please complete at least one session with Dr. Vain to generate a report.",
                color="warning"
            )
        
        # Generate NLP analysis
        analysis = generate_nlp_analysis(past_sessions)
        
        if not analysis:
            return dbc.Alert("Unable to generate analysis. No sufficient data found.", color="danger")
        
        # Build report layout
        report_elements = [
            html.H3("📊 Patient Diagnosis Report", style={"color": "#fff", "marginBottom": "30px"}),
            html.Hr(style={"borderColor": "#444"}),
            
            # Ollama-generated Diagnosis Section
            html.H4("Dr. Vain's Diagnosis", style={"color": "#fff", "marginTop": "20px", "marginBottom": "15px"}),
            html.Div(
                analysis.get('ollama_diagnosis', 'Diagnosis generation in progress...') if analysis.get('ollama_diagnosis') else dbc.Alert("Unable to generate diagnosis. Please try again.", color="warning"),
                style={
                    "backgroundColor": "#1a1a1a",
                    "color": "#fff",
                    "padding": "20px",
                    "borderRadius": "5px",
                    "border": "1px solid #444",
                    "lineHeight": "1.8",
                    "fontSize": "16px",
                    "whiteSpace": "pre-wrap"
                }
            ),
            
        ]
        
        return html.Div(report_elements)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return dbc.Alert(
            f"Error generating report: {str(e)}",
            color="danger"
        )
        
# --- RUN APP ---
if __name__=='__main__':
    print("=" * 60)
    print("🚀 STARTING DR. VAIN'S OFFICE APPLICATION")
    print("=" * 60)
    app.server.config["THREADED"] = False
    print("\n📍 Server starting on: http://127.0.0.1:8050")
    print("📱 Open this URL in your web browser to use the app")
    print("⏹️  Press Ctrl+C to stop the server\n")
    print("-" * 60)
    try:
        app.run(debug=False, host='127.0.0.1', port=8050, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
