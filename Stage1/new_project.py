import streamlit as st

def new_project_page():
    st.header("Créer un Nouveau Projet 🆕")
    
    # Fields for the new project
    project_id = st.text_input("ID-Project (automatique)", value="Auto-incrementé", disabled=True)
    project_title = st.text_input("Intitulé du Projet *", "")
    scorecard_type = st.selectbox("Scorecard Type *", ["Score d’octroi", "Score de comportement", "Rating"])
    client_category = st.selectbox("Catégorie de clientèle *", ["Particuliers", "TPE", "PME/GE", "Professionnels"])
    product_type = st.selectbox("Type de produits *", ["Consommation", "Immobilier", "Découverts", "Crédits de fonctionnement", "Crédit d’investissement"])
    
    # Text area with predefined text
    st.text_area("Instructions", 
                  "Scorecard development projects do not start with the acquisition of data. Intelligent scorecard development requires proper planning before any analytical work can start. This includes identifying the reason or objective for the project, identifying the key participants in the development and implementation of the scorecards, and assigning tasks to these individuals so that everyone is aware of what is required from them.",
                  height=200)

    # Widgets for creating business and project plans
    if st.button("Create Business Plan"):
        st.success("...je te remettrai le texte plus tard")
        st.text_input("Quels sont les objectifs organisationnels derrière le développement du scorecard ?")
        st.text_input("Est-ce que vous allez développer la grille en interne ou recourir à un cabinet d’analytics ?")

    if st.button("Define Project Plan"):
        st.success("...je te remettrai le texte plus tard")
        st.text_input("Pour chaque, quels sont les membres de l’équipe projet ?")
        st.text_input("Quel est votre Timeline pour implémenter le projet ?")
