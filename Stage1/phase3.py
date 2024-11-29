import pandas as pd
import streamlit as st
import openai
import yaml
import mysql.connector
from io import BytesIO
import json
from time import sleep
from plotly.subplots import make_subplots
from sklearn.model_selection import learning_curve
import re
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from logging_utils import log_interaction
from phase4 import DataQualityAgent, VariableSelectionAgent, UnivariateAnalysisAgent, CorrelationAnalysisAgent, SegmentationAgent, ModelingAgent,PreprocessingAgent

# Load configuration
def load_config():
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config["OPENAI_API_KEY"], config["ASSISTANT_ID"]

# Database functions
def get_db_connection(db_name=None):
    config = {
        'user': 'root',
        'password': '',
        'host': 'localhost'
    }
    
    if db_name:
        config['database'] = db_name
    
    return mysql.connector.connect(**config)

def create_database_if_not_exists(db_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    conn.close()

def store_project_parameters(params):
    db_name = 'parametres_du_projet'
    create_database_if_not_exists(db_name)
    conn = get_db_connection(db_name=db_name)
    cursor = conn.cursor()
    
    table_creation_query = """
    CREATE TABLE IF NOT EXISTS project_parameters (
        id INT AUTO_INCREMENT PRIMARY KEY,
        exclusions TEXT,
        performance_windows TEXT,
        definition_bad TEXT,
        definition_good_indeterminate TEXT,
        data_availability_quality TEXT,
        segmentation TEXT,
        methodology TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    cursor.execute(table_creation_query)
    
    insert_query = """
    INSERT INTO project_parameters (
        exclusions, performance_windows, definition_bad,
        definition_good_indeterminate, data_availability_quality,
        segmentation, methodology
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, (
        params["Exclusions"],
        params["Performance and Sample Windows"],
        params["Definition of 'Bad'"],
        params["Definition of 'Good' and 'Indeterminate'"],
        params["Data Availability and Quality"],
        params["Segmentation"],
        params["Methodology"]
    ))

    conn.commit()
    cursor.close()
    conn.close()

def store_data_availability(data):
    db_name = 'parametres_du_projet'
    create_database_if_not_exists(db_name)
    conn = get_db_connection(db_name=db_name)
    cursor = conn.cursor()

    table_creation_query = """
    CREATE TABLE IF NOT EXISTS data_availability (
        id INT AUTO_INCREMENT PRIMARY KEY,
        data JSON,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    cursor.execute(table_creation_query)

    insert_query = "INSERT INTO data_availability (data) VALUES (%s)"
    cursor.execute(insert_query, (json.dumps(data),))

    conn.commit()
    cursor.close()
    conn.close()

def data_quality_page():
    st.markdown(
        "<h1 style='color:#2C3E50;text-align:center;'>🔍 Données disponibles et leur qualité 📊</h1>", 
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="background-color:#ECF0F1;padding:10px;border-radius:10px;">
        <h3>Introduction</h3>
        <p>This stage is likely the longest and most labor-intensive phase of scorecard development. It is designed to determine whether scorecard development is feasible and to set high-level parameters for the project. The parameters include exclusions, target definition, sample window, and performance window.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown(
        "<h2 style='color:#2980B9;'>📋 Données disponibles et leur qualité</h2>", 
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="background-color:#F7F9F9;padding:10px;border-radius:10px;">
        <p>Data availability, quality, and quantity are crucial for scorecard development:</p>
        <ul>
            <li>✅ Reliable and clean data is needed, with a minimum number of "good" and "bad" accounts.</li>
            <li>📊 For application scorecards, about 2,000 "bad" and 2,000 "good" accounts should be randomly selected.</li>
            <li>🛠 Behavior scorecards use accounts that were current or at a certain delinquency status.</li>
            <li>🔍 Internal data reliability must be assessed; some data types are more susceptible to misrepresentation.</li>
            <li>🔗 External data from sources like credit bureaus may supplement internal data.</li>
            <li>🕒 The time frame for data extracts aligns with performance and sample window definitions.</li>
        </ul>
        </div>
        """, 
        unsafe_allow_html=True
    )

    if "scorecard_type" not in st.session_state:
        st.error("🚨 Type de scorecard non trouvé. Veuillez sélectionner un type de scorecard dans l'onglet 'Nouveau Projet'.")
        return
    
    scorecard_type = st.session_state["scorecard_type"]

    if scorecard_type == "Score d’octroi" or scorecard_type == "Score de comportement":
        st.markdown(
            "<h3 style='color:#16A085;'>📑 Données à collecter pour le score d'octroi ou le score de comportement</h3>", 
            unsafe_allow_html=True
        )
        st.markdown(
            """
            - 👥 **Données Socio démographiques du client** : Age, situation matrimoniale, Catégorie socio professionnelle, etc.
            - 📈 **Données de comportement interne** : Historique du compte (flux créditeurs/Débiteurs sur X mois), taux d'utilisation du découvert, Historique d'impayés, Contentieux, etc.
            - 💼 **Données au niveau du prêt** : Type de produit, avec ou sans Garantie, etc.
            """)
    elif scorecard_type == "Rating":
        st.markdown(
            "<h3 style='color:#8E44AD;'>📑 Données à collecter pour le modèle de Rating</h3>", 
            unsafe_allow_html=True
        )
        st.markdown(
            """
            - 📊 **Données Financières** : Bilan, compte de produit et de charges, etc.
            - 📝 **Données Qualitatives** : Situation du secteur d'activité, Qualité du Management, etc.
            """)

    if st.button("💾 Exporter les types de données à collecter", key="export_data_button"):
        data_to_export = {}
        if scorecard_type == "Score d’octroi" or scorecard_type == "Score de comportement":
            data_to_export = {
                "Données": [
                    "Données Socio démographiques du client",
                    "Données de comportement interne",
                    "Données au niveau du prêt"
                ],
                "Types de données": [
                    "Age, situation matrimoniale, Catégorie socio professionnelle, etc.",
                    "Historique du compte (flux créditeurs/Débiteurs sur X mois), taux d'utilisation du découvert, Historique d'impayés, Contentieux, etc.",
                    "Type de produit, avec ou sans Garantie, etc."
                ]
            }
        elif scorecard_type == "Rating":
            data_to_export = {
                "Données": [
                    "Données Financières",
                    "Données Qualitatives"
                ],
                "Types de données": [
                    "Bilan, compte de produit et de charges, etc.",
                    "Situation du secteur d'activité, Qualité du Management, etc."
                ]
            }

        df = pd.DataFrame(data_to_export)

        @st.cache_data
        def convert_df_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            return output.getvalue()

        excel_data = convert_df_to_excel(df)
        st.download_button(
            label="📥 Télécharger les données",
            data=excel_data,
            file_name=f"donnees_{scorecard_type}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    openai_api_key, assistant_id = load_config()
    openai.api_key = openai_api_key

    client = openai.OpenAI(api_key=openai_api_key)

    st.markdown(
        "<h3 style='color:#2C3E50;'>🤖 Vous avez d'autres questions n'hésitez pas à les poser</h3>", 
        unsafe_allow_html=True
    )

    with st.expander("❓ Poser une question à l'assistant"):
        chat_input = st.text_input("Que recherchez-vous ?")
        if chat_input:
            with st.chat_message("user"):
                st.markdown(chat_input)
            st.session_state.messages.append({"role": "user", "content": chat_input})

            # Log user interaction
            log_interaction("user", chat_input)

            client = openai.OpenAI(api_key=openai_api_key)
            thread = client.beta.threads.create()
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=chat_input
            )
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id,
                instructions="Please assist the user with their question."
            )

            # Log assistant interaction
            log_interaction("assistant", run.result)
            st.session_state.messages.append({"role": "assistant", "content": run.result})
            st.chat_message("assistant").markdown(run.result)
    if st.form("Aller à la définition des paramètres du projet"):
        st.session_state.current_page = "project_parameters"  


    # Bouton pour exécuter les analyses de qualité des données
    if st.button("🔍 Aller à la définition des paramètres du projet"):
        with st.spinner("🔄 Analyse en cours..."):
            quality_analysis_result = DataQualityAgent.analyze()  # Appel à ton agent de qualité des données
            variable_selection_result = VariableSelectionAgent.select()  # Appel à l'agent de sélection de variables

            # Afficher les résultats
            st.subheader("Résultats de l'analyse de qualité des données")
            st.write(quality_analysis_result)

            st.subheader("Résultats de la sélection des variables")
            st.write(variable_selection_result)

            # Log les résultats
            log_interaction("Quality Analysis", quality_analysis_result)
            log_interaction("Variable Selection", variable_selection_result)              

def define_project_parameters():
    openai_api_key, assistant_id = load_config()
    openai.api_key = openai_api_key
    
    st.title("📊 Définition des paramètres du projet")


    
     # Define parameter details
    parameter_info = {
        "Exclusions": """
        Certain types of accounts need to be excluded from the development sample. This includes accounts with abnormal performance (e.g., frauds), accounts adjudicated using non-score-dependent criteria, and accounts from geographic areas or markets where the company no longer operates.
        """,
        "Performance and Sample Windows": """
        Scorecards are developed based on the assumption that future performance will reflect past performance. This involves gathering data for accounts opened during a specific time frame and monitoring their performance over another specific length of time to determine if they were good or bad.
        """,
        "Definition of 'Bad'": """
        This involves setting a clear and interpretable definition of what constitutes a "bad" account. The definition should be in line with the product or purpose for which the scorecard is being built and should be easily trackable.
        """,
        "Definition of 'Good' and 'Indeterminate'": """
        Similarly, defining what constitutes a "good" account is essential. Indeterminate accounts are those that do not conclusively fall into either the "good" or "bad" categories and should be carefully considered to avoid misclassification.
        """,
        "Data Availability and Quality": """
        Reliable and clean data is crucial for scorecard development. This includes ensuring that there is sufficient good-quality internal and external data to proceed with the development.
        """,
        "Segmentation": """
        This involves dividing the data into meaningful segments to improve the predictive power of the scorecard. Segmentation can be based on various factors such as demographics, product types, or geographic regions.
        """,
        "Methodology": """
        The methodology for developing the scorecard should be clearly defined. This includes the statistical techniques to be used, the process for handling missing values and outliers, and the criteria for selecting predictor variables.
        """
    }

    # Widgets and Expanders for each parameter
    parameters = [
        "Exclusions",
        "Performance and Sample Windows",
        "Definition of 'Bad'",
        "Definition of 'Good' and 'Indeterminate'",
        "Data Availability and Quality",
        "Segmentation",
        "Methodology"
    ]
    emojis=["🔎",
            "📅",
            "🚫",
            "✅"
            ,"📊"
            ,"🔢"
            ,"📈"
            ]

    parameter_values = {}

    for param, emo in zip(parameters, emojis):
        with st.container():
            with st.expander(f"**{emo} {param}**"):
                st.markdown(parameter_info[param])
            parameter_values[param] = st.text_input(f"Enter {param}", key=param)
    
    
    
    # Button to store project parameters
    if st.button("💾 Enregistrer les Paramètres du Projet"):
        store_project_parameters(parameters)
        st.success("✅ Paramètres enregistrés avec succès.")

    # Button to get assistant's response for each parameter
    def get_assistant_response(content, parameter_name):
        with st.spinner(f"🔄 Demande en cours pour {parameter_name}..."):
            client = openai.OpenAI(api_key=openai_api_key)
            thread = client.beta.threads.create()
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"{parameter_name}: {content}"
            )
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id,
                instructions="Please assist the user with their query."
            )
            
            while run.status not in ["completed", "failed"]:
                sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            if run.status == 'completed':
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                response_text = ""
                for message in messages.data:
                    if message.role == 'assistant':
                        for content_block in message.content:
                            if content_block.type == 'text':
                                clean_text = re.sub(r'【[^】]*†source】', '', content_block.text.value)
                                response_text += clean_text.strip() + "\n"
                log_interaction(content, response_text)  # Log the interaction
                return response_text
            else:
                st.error(f"❌ L'assistant n'a pas pu compléter la demande pour {parameter_name}.")
                return ""

    # Chat area for additional questions
    with st.expander("💬 Poser une question à l'assistant"):
        chat_input = st.text_input("Que recherchez-vous ?")
        if chat_input:
            with st.chat_message("user"):
                st.markdown(chat_input)
            
            response = get_assistant_response(chat_input, "chat")
            if response:
                with st.chat_message("assistant"):
                    st.markdown(response)

def data_availability_page():
    openai_api_key, assistant_id = load_config()
    openai.api_key = openai_api_key

    st.title("📁 Données disponibles")

    # Step 1: Input for data availability
    st.subheader("🔢 Saisir la liste des données disponibles")
    data_input = st.text_area("Liste des données disponibles (ex: Nom du champ, Type de champ, Description du champ)")

    # Step 2: File uploader for data dictionary
    uploaded_file = st.file_uploader("📥 Charger un fichier CSV ou Excel", type=["csv", "xlsx"])

    # Handling uploaded file
    df = None  # Initialize df
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Format de fichier non supporté. Veuillez charger un fichier CSV ou Excel.")
                return
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            return
    elif data_input:
        try:
            data_lines = [line.strip() for line in data_input.split('\n') if line.strip()]
            data = [line.split(',') for line in data_lines]
            df = pd.DataFrame(data, columns=["Nom du champ", "Type de champ", "Description du champ"])
        except Exception as e:
            st.error(f"Erreur lors du traitement des données saisies : {e}")
            return
    else:
        st.warning("Veuillez entrer des données ou charger un fichier.")
        return

    # Step 3: Display the data in a table
    st.subheader("📊 Synthèse des données")
    st.write(df)

    # Step 4: Allow user to interact and update the table
    with st.expander("✏️ Modifier le tableau"):
        st.write("Choisissez une action pour modifier le tableau :")
        
        action = st.radio("Action :", ["Ajouter une colonne", "Supprimer une colonne"])
        
        if action == "Ajouter une colonne":
            new_column_name = st.text_input("Nom de la nouvelle colonne")
            if new_column_name:
                df[new_column_name] = None  # Ajoute une colonne avec des valeurs par défaut
                st.write(f"✅ Colonne '{new_column_name}' ajoutée.")
                st.write(df)
        
        elif action == "Supprimer une colonne":
            columns_to_remove = st.multiselect("Sélectionnez les colonnes à supprimer", df.columns.tolist())
            if columns_to_remove:
                df = df.drop(columns=columns_to_remove)
                st.write(f"❌ Colonnes supprimées : {', '.join(columns_to_remove)}.")
                st.write(df)

    # Step 4: Data Quality Assessment
    st.subheader("🔍 Évaluation de la qualité des données")
    dq_agent = DataQualityAgent(df)
    dq_results = dq_agent.assess_quality()
    st.write("Nos recommendation sur l'évaluation de la qualité des données :", dq_results["advice"])
    st.write("Valeurs manquantes :", dq_results["missing_values"])
    st.write("Types de données :", dq_results["data_types"])
    st.write("Valeurs aberrantes :", dq_results["outliers"])

    # Step 5: User choice for further actions based on data quality advice
    if st.button("Choisissez des actions pour la qualité des données"):
        action = st.selectbox("Choisissez une action basée sur les conseils :", 
                          ["Gérer les valeurs manquantes", "Supprimer les valeurs aberrantes", "Vérifier les types de données"])
        st.write(f"Vous avez choisi de : {action}")

    # Step 6: Preprocessing Step - Handle Missing Values
    st.subheader("🔧 Prétraitement des données")
    preprocessing_agent = PreprocessingAgent(df)

    missing_value_method = st.selectbox("Choisissez une méthode pour gérer les valeurs manquantes", 
                                    ["Aucune action", "Suppression", "Imputation par la moyenne", "Imputation par la médiane", "Imputation par la mode"])
    if missing_value_method != "Aucune action":
        try:
            df = preprocessing_agent.handle_missing_values(missing_value_method)
            st.write(f"✅ Méthode de gestion des valeurs manquantes appliquée : {missing_value_method}")
        except Exception as e:
            st.error(f"Erreur lors de la gestion des valeurs manquantes : {e}")

    # Step 8: Encoding Categorical Data
    encoding_method = st.selectbox("Choisissez une méthode pour l'encodage des variables catégoriques", 
                               ["Aucune action", "Encodage one-hot", "Encodage ordinal"])
    if encoding_method != "Aucune action":
        try:
            df = preprocessing_agent.encode_categorical_data(method=encoding_method)
            st.write(f"✅ Méthode d'encodage appliquée : {encoding_method}")
        except Exception as e:
            st.error(f"Erreur lors de l'encodage des variables catégoriques : {e}")

    st.write("Données après prétraitement :")
    st.write(df)

    # Step 7: Univariate Analysis
    st.subheader("📊 Analyse Univariée")
    univariate_agent = UnivariateAnalysisAgent(df)
    univariate_results = univariate_agent.perform_analysis()  # Exécutez l'analyse univariée
    st.write("Résultats de l'analyse univariée :", univariate_results)  # Affichez les résultats

    # Step 8: Scale Features (Normalization)
    st.subheader("⚖️ Normalisation des données")
    scaling_method = st.selectbox("Choisissez une méthode de normalisation des données", 
                                   ["Aucune action", "Normalisation min-max", "Standardisation (Z-score)"])
    if scaling_method != "Aucune action":
        try:
            df = preprocessing_agent.scale_features(method=scaling_method)
            st.write(f"✅ Méthode de normalisation appliquée : {scaling_method}")
        except Exception as e:
            st.error(f"Erreur lors de la normalisation des données : {e}")

    # Step 9: Proceed to Variable Selection only after data preprocessing
    st.subheader("🔍 Sélection des variables")
    target_variable = st.selectbox("Sélectionnez la variable cible", df.columns)
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    vs_agent = VariableSelectionAgent(X, y, dq_results)
    vs_results = vs_agent.select_variables()
    if "error" in vs_results:
        st.write("Erreur dans la sélection des variables :", vs_results["error"])
        st.write("Résultats précédents :", vs_results["previous_results"])
    else:
        st.write("Nos recommendation sur la sélection des variables :", vs_results["advice"])
        st.write("Fonctionnalités sélectionnées :", df.columns[vs_results["selected_features"]].tolist())

    # Step 10: Assess Correlation Implications
    st.subheader("🔍 Analyse de corrélation")
    ca_agent = CorrelationAnalysisAgent(df)
    ca_advice = ca_agent.analyze_correlation()
    st.write("Nos recommendation sur l'analyse de corrélation :", ca_advice)
    # Visualize the correlation matrix
    ca_agent.visualize_correlation()  # Use the visualization function
    
    # Display correlated variables for user input
    ca_agent.display_correlated_variables()


    # Step 11: Handle Correlated Variables
    action_choice = st.radio("Que souhaitez-vous faire avec les variables corrélées ?", 
                         options=["Appliquer PCA", "Supprimer les variables corrélées"])

    if action_choice == "Appliquer PCA":
        st.write("Recherche du meilleur nombre de composantes pour PCA...")
        data_pca = ca_agent.apply_pca_with_grid_search(X, y)
    
        # Afficher les données réduites en dimension après PCA
        st.write("Données après réduction de dimension avec PCA optimisé :")
        st.write(pd.DataFrame(data_pca))

    elif action_choice == "Supprimer les variables corrélées":
        # Get the correlated variables
        correlated_pairs = ca_agent.display_correlated_variables()

        if correlated_pairs:
            # Flatten the list of variable pairs for multiselect
            all_correlated_vars = list(set([var for pair in correlated_pairs for var in pair]))

            # Allow user to select which variables to remove
            variables_to_remove = st.multiselect("Choisissez les variables à supprimer", all_correlated_vars)
        
            if st.button("Confirmer suppression des variables sélectionnées"):
                if variables_to_remove:
                    # Suppression des variables sélectionnées
                    df = df.drop(columns=variables_to_remove)
                    st.write(f"Variables supprimées : {', '.join(variables_to_remove)}")
                else:
                    st.write("Aucune variable n'a été sélectionnée pour suppression.")
        else:
            st.write("Aucune variable fortement corrélée trouvée.")

    # Step 12: Data Splitting
    #st.subheader("📊 Division des données")
    X_train, X_test, y_train, y_test = train_test_split(
        X.iloc[:, vs_results["selected_features"]], y, test_size=0.3, random_state=42
    )

    # Step 13: Model Selection and Training
    st.subheader("🔍 Modélisation")
    model_agent = ModelingAgent(X_train, y_train, X_test, y_test)

    results, recommendation = model_agent.train_and_evaluate()

    # Convert results to a DataFrame for better display
    results_df = pd.DataFrame(results).T  # Transpose for better layout

    # Display the modeling results as a pretty table
    st.write("📊 Résultats de la modélisation :", results_df.style.background_gradient(cmap='Blues'))

    # Display confusion matrices
    st.subheader("📊 Matrices de confusion")
    model_agent.display_confusion_matrices()
    # Visualize the metrics using bar plots
    st.subheader("📈 Visualisation des performances des modèles")
    # Create a bar plot with Plotly
    fig = go.Figure()

    # Add bars for Accuracy
    fig.add_trace(go.Bar(
        x=results_df.index,
        y=results_df['Accuracy'],
        name='Accuracy',
        marker_color='rgb(55, 83, 109)'
    ))

    # Add bars for F1-Score
    fig.add_trace(go.Bar(
        x=results_df.index,
        y=results_df['F1-Score'],
        name='F1-Score',
        marker_color='rgb(26, 118, 255)'
    ))

    # Add bars for ROC AUC
    fig.add_trace(go.Bar(
        x=results_df.index,
        y=results_df['ROC AUC'],
        name='ROC AUC',
        marker_color='rgb(50, 171, 96)'
    ))

    # Update layout for aesthetics
    fig.update_layout(
        title="Comparaison des performances des modèles",
        xaxis=dict(
            title="Modèles",
            tickmode="array",
            tickvals=results_df.index,
            ticktext=results_df.index
        ),
        yaxis=dict(
            title="Score",
            range=[0, 1]
        ),
        barmode='group',  # Group bars by metric
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        legend=dict(
            x=0.02,
            y=1.05,
            bgcolor='rgba(255,255,255,0)',
            bordercolor='rgba(255,255,255,0)'
        )
    )

    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)


    # Display final comparison table (if implemented)
    st.subheader("📊 Table de comparaison finale")
    model_agent.display_comparison_table()




    # Display the recommendation from OpenAI
    st.write("📝 Recommandation d'OpenAI pour le choix du modèle :", recommendation)

    # Debugging: Print the recommendation string
    st.write(f"Contenu de la recommandation: {recommendation}")

    # List of model names from the dictionary
    model_names = list(model_agent.models.keys())  
    recommended_model_name = None

    # Ensure the recommended model name matches the format used in the self.models dictionary
    try:
        # Search for the model name in the recommendation string
        for name in model_names:
            if name.lower() in recommendation.lower():
                recommended_model_name = name
                break
        
        if recommended_model_name:
            st.write(f"Modèle recommandé: {recommended_model_name}")  # Display the extracted model name

            # Check if the recommended model exists in the models dictionary
            if recommended_model_name in model_agent.models:
                recommended_model = model_agent.models[recommended_model_name]  # Retrieve the model object

                # Performance for training and testing sets
                train_accuracy = recommended_model.score(X_train, y_train)
                test_accuracy = recommended_model.score(X_test, y_test)

                # Plotting performance curves for training and testing accuracy
                st.subheader("📈 Performance du modèle recommandé")

                fig = make_subplots(rows=1, cols=2, subplot_titles=("Précision du modèle", "Courbe d'apprentissage"))

                # Bar plot for training and test accuracy
                fig.add_trace(go.Bar(
                    x=['Entraînement', 'Test'],
                    y=[train_accuracy, test_accuracy],
                    text=[f'{train_accuracy:.2f}', f'{test_accuracy:.2f}'],  # Display accuracy values
                    textposition='auto',
                    marker_color=['#4C72B0', '#DD8452'],  # Custom colors
                    name="Précision"
                ), row=1, col=1)

                # Compute learning curve
                train_sizes, train_scores, test_scores = learning_curve(recommended_model, X_train, y_train, cv=5, n_jobs=-1)

                # Calculate the mean and std for training and test scores
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)

                # Learning curve plot (Training and Test scores over different training sizes)
                fig.add_trace(go.Scatter(
                    x=train_sizes,
                    y=train_scores_mean,
                    mode='lines+markers',
                    name="Score Entraînement",
                    line=dict(color='#4C72B0'),
                    error_y=dict(type='data', array=train_scores_std, visible=True)  # Add error bars
                ), row=1, col=2)

                fig.add_trace(go.Scatter(
                    x=train_sizes,
                    y=test_scores_mean,
                    mode='lines+markers',
                    name="Score Validation",
                    line=dict(color='#DD8452'),
                    error_y=dict(type='data', array=test_scores_std, visible=True)  # Add error bars
                ), row=1, col=2)

                # Update layout for aesthetics
                fig.update_layout(
                    title=f"Performance et courbe d'apprentissage du modèle: {recommended_model_name}",
                    title_font=dict(size=20, family='Arial', color='black'),
                    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    height=500,
                    xaxis_title="Jeu de données",
                    yaxis_title="Précision",
                    yaxis=dict(range=[0, 1]),
                    template="plotly_white",
                    bargap=0.4,
                    showlegend=True
                )

                # Set labels and titles for subplots
                fig.update_xaxes(title_text="Jeu de données", row=1, col=1)
                fig.update_yaxes(title_text="Précision", row=1, col=1)

                fig.update_xaxes(title_text="Taille des données d'entraînement", row=1, col=2)
                fig.update_yaxes(title_text="Score", row=1, col=2)

                # Display the interactive plotly figure in Streamlit
                st.plotly_chart(fig)

                # Show the final accuracies
                st.write(f"Précision Entraînement: {train_accuracy:.4f}")
                st.write(f"Précision Test: {test_accuracy:.4f}")
            else:
                st.error(f"Le modèle recommandé '{recommended_model_name}' n'est pas disponible dans la liste des modèles.")
        else:
            st.error("Aucun modèle recommandé trouvé dans la réponse d'OpenAI.")
    except Exception as e:
        st.error(f"Erreur lors de l'extraction ou de l'affichage du modèle recommandé: {str(e)}")


        


    



    



