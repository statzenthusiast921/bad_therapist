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
                html.Span("Therapist: ", style={'color': '#ffff00', 'fontWeight': 'bold'}),
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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder=os.path.join(os.curdir,"assets"))
server = app.server
print("Dash app created successfully")

# --- APP LAYOUT ---
print("Building app layout...")
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
                    html.P("Point 1", style={'color':'white'}),
                    html.Br(),
                    html.P(dcc.Markdown('**How does Dr. Vain operate?**'), style={'text-decoration': 'underline','color':'white'}),
                    html.P('RAG', style={'color':'white'}),
                    html.Br(),
                    html.P(dcc.Markdown('**Does Dr. Vain have any limitations**'), style={'text-decoration': 'underline','color':'white'}),
                    html.P("No", style={'color':'white'})
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
                            children=[html.Img(src="", alt="Dr. Thaddeus Vain", style={'width': '100%', 'height': '100%', 'object-fit': 'cover'})]
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
                html.H2("🏅 Awards & Unnecessary Accolades", style={'color': '#000080'}),
                html.P(html.B("The Golden Insight Award (Annually, from Myself)"), style={'margin-bottom': '0px'}),
                html.P("Recognized as the foremost thinker in every room I enter. (2015 - Present)"),
                html.Br(),
                html.P(html.B("The Patient-Zero Award"), style={'margin-bottom': '0px'}),
                html.P("For single-handedly raising the bar of what a therapist should be. (2018)"),
                html.Br(),
                html.P(html.B("The Man of the Year, Every Year"), style={'margin-bottom': '0px'}),
                html.P("For the sheer audacity of my excellence. (2010-2014)"),
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
                dcc.Loading(
                    id="loading-session",
                    type="default",
                    children=html.Div(id="chat-log", style={
                        "backgroundColor": "#000000",
                        "color": "#00ff00",
                        "padding": "10px",
                        "height": "400px",
                        "overflowY": "auto",
                        "fontFamily": "monospace",
                        "border": "2px solid #444"
                    })
                ),
                html.Br(),
                dbc.Button("Start New Session", id="new-session-btn", color="danger", n_clicks=0),
                html.Br(), html.Br(),
                dbc.Input(id="user-input", placeholder="Type your message...", type="text", 
                          debounce=False, n_submit=0),
                html.Br(),
                dbc.Button("Submit", id="submit-btn", color="success", n_clicks=0)
            ]
        ),

        # --- Tab 4: Summary ---
        dcc.Tab(label='Summary', value='tab-4', style=tab_style, selected_style=tab_selected_style, children=[])
    ])
])
print("App layout built successfully")

# --- CALLBACKS ---
print("Setting up callbacks...")

@app.callback(
    Output("session-id", "data"),
    Output("chat-log", "children"),
    Output("user-input", "value"),
    Input("new-session-btn", "n_clicks"),
    Input("submit-btn", "n_clicks"),
    Input("user-input", "n_submit"),
    State("user-input", "value"),
    State("session-id", "data"),
    prevent_initial_call=True
)
def handle_session_and_messages(new_session_clicks, submit_clicks, n_submit, user_message, session_id):
    """Combined callback to handle both starting new session and sending messages."""
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Handle "Start New Session" button
    if triggered_id == "new-session-btn":
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
            return session_id, format_chat_log(therapist.chat_history), ""
        except Exception as e:
            print(f"ERROR in start_new_session: {e}")
            import traceback
            traceback.print_exc()
            error_msg = [
                html.Div(f"Error starting session: {str(e)}", style={'color': '#ff0000', 'fontWeight': 'bold'}),
                html.Div("Check the terminal for more details.", style={'color': '#ffaa00', 'marginTop': '10px'})
            ]
            return None, error_msg, ""
    
    # Handle "Submit" button or Enter key
    elif triggered_id in ["submit-btn", "user-input"]:
        if not session_id:
            return no_update, [html.Div("Please start a new session first.", style={'color': '#ff0000'})], ""
        if not user_message or not user_message.strip():
            raise PreventUpdate
        
        session_mgr = get_session_manager()
        therapist = session_mgr.active_session
        if therapist is None or therapist.session_id != session_id:
            return no_update, [html.Div("Session mismatch! Please start a new session.", style={'color': '#ff0000'})], ""
        
        # Add user message to history and get response from RAG system
        try:
            response = therapist.chat(user_message)
            return no_update, format_chat_log(therapist.chat_history), ""
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            import traceback
            traceback.print_exc()
            return no_update, [html.Div(error_msg, style={'color': '#ff0000'})], ""
    
    else:
        raise PreventUpdate

print("Callbacks set up successfully")

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
        app.run(debug=True, host='127.0.0.1', port=8050, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
