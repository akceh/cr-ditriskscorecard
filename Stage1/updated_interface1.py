import streamlit as st
import openai
import yaml
from time import sleep
import re
from phase3 import data_quality_page
from phase3 import define_project_parameters,data_availability_page
from logging_utils import log_interaction
# Import your agents and utility functions from phase4
from phase4 import DataQualityAgent, VariableSelectionAgent, UnivariateAnalysisAgent
from phase4 import CorrelationAnalysisAgent, SegmentationAgent, ModelingAgent, load_config, query_openai

# Load the API key and assistant ID from config.yaml
def load_config():
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config["OPENAI_API_KEY"], config["ASSISTANT_ID"]

# Initialize OpenAI API key and assistant ID
openai_api_key, assistant_id = load_config()
openai.api_key = openai_api_key

# Initialize session state for page navigation and greetings
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"  # Default page
if "greetings_shown" not in st.session_state:
    st.session_state.greetings_shown = False  # Track if greetings have been shown
if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize the messages list

# Sidebar with navigation
def sidebar_navigation():
    with st.sidebar:
        st.title("🏦​ Credit Risk Scorecards")
        st.subheader("🔍 Aperçu du Processus 📊")
        st.markdown("""
        Le développement de scorecards de crédit comprend les étapes suivantes :
        """)

        # Overview Section
        st.write("1. **Préliminaires et Planification** 📝")
        st.write("2. **Revue des Données et Paramètres du Projet** 📈")
        st.write("3. **Création de la Base de Données de Développement** 📊")
        st.write("4. **Développement du Scorecard** 📉")
        st.write("5. **Rapports de Gestion du Scorecard** 📊")
        st.write("6. **Mise en Œuvre du Scorecard** 🚀")
        st.write("7. **Suivi Post-Implémentation** 📊")
        
        # Button to create a new project
        if st.button("➕ New Project", key="new_project_button"):
            st.session_state.current_page = "new_project"
        if st.button("Données disponibles et leur qualité", key="data_quality_button"):
            st.session_state.current_page = "data_quality"
        # Add button to navigate to "Définition des paramètres du projet"
        if st.button("Définition des paramètres du projet", key="project_parameters_button"):
            st.session_state.current_page = "project_parameters"   
        # Button to navigate to data availability page
        if st.button("Données disponibles", key="data_availability_button"):
            st.session_state.current_page = "data_availability"  # Navigate to data availability page     

# Function to show home page content
def home_page():
    st.title("🏦​ Credit Risk Scorecards Assistant")

    # Greet user only if on the home page and if the greeting hasn't been shown yet
    if not st.session_state.greetings_shown:
        with st.chat_message("assistant"):
            intro = "Hello! I am your Credit Risk Scorecards assistant. How can I assist you today?"
            st.markdown(intro)
            st.session_state.messages.append({"role": "assistant", "content": intro})
            st.session_state.greetings_shown = True  # Set greetings as shown
            log_interaction("assistant", intro)
    # Predefined choices
    example_prompts = [
        "Développer un score d'octroi",
        "Développer un score de comportement",
        "Développer un modèle de Rating des PME/GE",
        "s'initier aux étapes d'un projet de modélisation de grille de scores"
    ]

    # Initialize button_pressed
    button_pressed = ""

    button_cols = st.columns(min(len(example_prompts), 3))

    for i, prompt in enumerate(example_prompts):
        with button_cols[i % len(button_cols)]:
            if st.button(prompt, key=f"prompt_button_{i}"):
                button_pressed = prompt

    # Check if a button is pressed or a chat input is provided
    if prompt := (st.chat_input("What are you looking for?") or button_pressed):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the user's input with OpenAI API
        process_openai_request(prompt)

# Function to show new project page content
def new_project_page():
    st.header("Créer un Nouveau Projet 🆕")

    # Fields for the new project
    project_id = st.text_input("ID-Project (automatique)", value="Auto-incrementé", disabled=True)
    project_title = st.text_input("Intitulé du Projet *", "")
    scorecard_type = st.selectbox("Scorecard Type *", ["Score d’octroi", "Score de comportement", "Rating"])
    client_category = st.selectbox("Catégorie de clientèle *", ["Particuliers", "TPE", "PME/GE", "Professionnels"])
    product_type = st.selectbox("Type de produits *", ["Consommation", "Immobilier", "Découverts", "Crédits de fonctionnement", "Crédit d’investissement"])

    # Store the scorecard type in session state
    st.session_state.scorecard_type = scorecard_type

    # Text area with predefined text
    st.text_area("Instructions", 
                 "Scorecard development projects do not start with the acquisition of data. Intelligent scorecard development requires proper planning before any analytical work can start. This includes identifying the reason or objective for the project, identifying the key participants in the development and implementation of the scorecards, and assigning tasks to these individuals so that everyone is aware of what is required from them.",
                 height=200)

    # Create Business Plan
    create_business_plan(project_title, scorecard_type, client_category, product_type)

    # Define Project Plan
    define_project_plan(project_title, scorecard_type, client_category, product_type)

# Function to create business plan content
def create_business_plan(project_title, scorecard_type, client_category, product_type):
    with st.expander("Create Business Plan"):
        st.success("...je te remettrai le texte plus tard")
        objectives = st.text_input("Quels sont les objectifs organisationnels derrière le développement du scorecard ?")
        internal_or_external = st.text_input("Est-ce que vous allez développer la grille en interne ou recourir à un cabinet d’analytics ?")
        
        # Button to get assistant's response
        if st.button("Get Assistant's Response", key="business_plan_response"):
            with st.spinner("Demande en cours..."):
                assistant_query(project_title, scorecard_type, client_category, product_type, objectives, internal_or_external)
                log_interaction(f"Business Plan Query: {objectives}, {internal_or_external}", assistant_query(project_title, scorecard_type, client_category, product_type, objectives, internal_or_external))


# Function to define project plan content
def define_project_plan(project_title, scorecard_type, client_category, product_type):
    with st.expander("Define Project Plan"):
        st.success("...je te remettrai le texte plus tard")
        team_members = st.text_input("Pour chaque, quels sont les membres de l’équipe projet ?")
        timeline = st.text_input("Quel est votre Timeline pour implémenter le projet ?")
        
        # Button to get assistant's response
        if st.button("Get Assistant's Response", key="project_plan_response"):
            with st.spinner("Demande en cours..."):
                assistant_query(project_title, scorecard_type, client_category, product_type, team_members, timeline)
                log_interaction(f"Project Plan Query: {team_members}, {timeline}", assistant_query(project_title, scorecard_type, client_category, product_type, team_members, timeline))

# Function to process OpenAI requests
def process_openai_request(prompt):
    log_interaction("user", prompt)
    client = openai.OpenAI(api_key=openai_api_key)
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
        instructions="Please assist the user with their query."
    )

    while run.status not in ["completed", "failed"]:
        sleep(1)  # Sleep for a while before checking again
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

    if run.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=thread.id)

        # Extract and format the text content from the latest message
        response_text = ""
        for message in messages.data:
            if message.role == 'assistant':
                for content_block in message.content:
                    if content_block.type == 'text':
                        clean_text = re.sub(r'【[^】]*†source】', '', content_block.text.value)
                        response_text += clean_text.strip() + "\n"

        # Append the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Display the assistant's response
        with st.chat_message("assistant"):
            st.markdown(response_text)
    else:
        st.error("The assistant could not complete the request.")

# Function to trigger the assistant's response
def assistant_query(project_title, scorecard_type, client_category, product_type, *args):
    content = f"Project Title: {project_title}\nScorecard Type: {scorecard_type}\nClient Category: {client_category}\nProduct Type: {product_type}"
    for arg in args:
        content += f"\n{arg}"
    process_openai_request(content)

# Function to display previous chat messages
def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Page navigation logic
sidebar_navigation()
if st.session_state.current_page == "home":
    home_page()
elif st.session_state.current_page == "new_project":
    new_project_page()
elif st.session_state.current_page == "data_quality":
    data_quality_page()
elif st.session_state.current_page == "project_parameters":
    define_project_parameters()  
elif st.session_state.current_page == "data_availability":
    data_availability_page()    

