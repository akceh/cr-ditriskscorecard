import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix




# Load configuration
def load_config():
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config["OPENAI_API_KEY"], config["ASSISTANT_ID"]

def query_openai(prompt):
    # Load API key from config
    openai_api_key, _ = load_config()

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Use the client to create a completion
    response = client.chat.completions.create(
        model="gpt-4",  # Use the appropriate model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150  # Adjust max tokens as needed
    )

    # Extract the content from the response properly
    return response.choices[0].message.content



# 1. Data Quality Assessment
class DataQualityAgent:
    def __init__(self, data):
        self.data = data
        self.results = {}

    def assess_quality(self):
        prompt = f"I have a dataset with {self.data.shape[1]} columns and {self.data.shape[0]} rows. What should I check for in terms of data quality?"
        advice = query_openai(prompt)
        missing_values = self.data.isnull().sum()
        data_types = self.data.dtypes
        
        # Outlier Detection for numeric data
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            outliers = numeric_data[(numeric_data - numeric_data.mean()).abs() > 3 * numeric_data.std()]
        else:
            outliers = None

        self.results = {
            "advice": advice,
            "missing_values": missing_values,
            "data_types": data_types,
            "outliers": outliers
        }
        return self.results

# 2. Preprocessing Agent
class PreprocessingAgent:
    def __init__(self, data):
        self.data = data

    def handle_missing_values(self, method):
        if method == "Imputation par la moyenne":
            self.data.fillna(self.data.mean(numeric_only=True), inplace=True)
        elif method == "Imputation par la médiane":
            self.data.fillna(self.data.median(numeric_only=True), inplace=True)
        elif method == "Imputation par la mode":
            self.data.fillna(self.data.mode().iloc[0], inplace=True)
        elif method == "Suppression":
            self.data.dropna(inplace=True)
        return self.data

    def encode_categorical_data(self, method="None"):
        if method == "Encodage one-hot":
            # Apply one-hot encoding to categorical columns
            self.data = pd.get_dummies(self.data, drop_first=True)
        elif method == "Encodage ordinal":
            # Apply label encoding to categorical columns
            label_encoders = {}
            for column in self.data.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                self.data[column] = le.fit_transform(self.data[column])
                label_encoders[column] = le
        # Return the updated DataFrame after encoding
        return self.data
    
    def scale_features(self, method):
        # Use self.data instead of self.dataframe
        if method == "Normalisation min-max":
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(self.data)
            return pd.DataFrame(scaled_data, columns=self.data.columns)
        elif method == "Standardisation (Z-score)":
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.data)
            return pd.DataFrame(scaled_data, columns=self.data.columns)
        else:
            return self.data  # Return the original DataFrame if no action is taken
    

# 3. Univariate Analysis
class UnivariateAnalysisAgent:
    def __init__(self, data):
        self.data = data
        self.numeric_data = data.select_dtypes(include='number')
        self.categorical_data = data.select_dtypes(include='object')

    def perform_analysis(self):
        # Demande de conseils à OpenAI
        prompt = f"I have {self.numeric_data.shape[1]} numerical and {self.categorical_data.shape[1]} categorical variables. What are the best visualizations for univariate analysis?"
        advice = query_openai(prompt)
        
        # Effectuer l'analyse des variables numériques et catégorielles
        self.analyze_numerical()  # Assurez-vous que ces méthodes affichent les graphiques
        self.analyze_categorical()
        
        # Retournez les conseils d'OpenAI
        return advice  # Cela ne devrait pas affecter l'affichage des graphiques

    def analyze_numerical(self):
        st.subheader("🔍 Univariate Analysis of Numerical Variables")
        for column in self.numeric_data.columns:
            st.write(f"### {column}")
            
            # Histogram
            st.write("Creating histogram...")  # Cette ligne peut être facultative
            fig, ax = plt.subplots()
            sns.histplot(self.numeric_data[column], kde=True, ax=ax, color="skyblue")
            ax.set_title(f"Histogram of {column}")
            st.pyplot(fig)  # Assurez-vous que cela est appelé après avoir créé le graphique
            
            # Box plot
            st.write("Creating box plot...")
            fig, ax = plt.subplots()
            sns.boxplot(x=self.numeric_data[column], ax=ax, color="lightgreen")
            ax.set_title(f"Boxplot of {column}")
            st.pyplot(fig)
            
            # Density plot
            st.write("Creating density plot...")
            fig, ax = plt.subplots()
            sns.kdeplot(self.numeric_data[column], shade=True, ax=ax, color="orange")
            ax.set_title(f"Density Plot of {column}")
            st.pyplot(fig)

    def analyze_categorical(self):
        st.subheader("🔍 Univariate Analysis of Categorical Variables")
        for column in self.categorical_data.columns:
            st.write(f"### {column}")

            # Bar plot
            st.write("Creating bar plot...")
            fig = px.bar(self.data[column].value_counts().reset_index(),
                         x='index', y=column,
                         labels={'index': column, column: 'Count'},
                         title=f"Bar Plot of {column}",
                         color_discrete_sequence=["#1f77b4"])
            st.plotly_chart(fig)  # Utilisez st.plotly_chart pour les graphiques Plotly
            
            # Count plot
            st.write("Creating count plot...")
            fig, ax = plt.subplots()
            sns.countplot(x=self.categorical_data[column], ax=ax, palette="Set2")
            ax.set_title(f"Count Plot of {column}")
            plt.xticks(rotation=45)
            st.pyplot(fig)

# 4. Correlation Analysis
class CorrelationAnalysisAgent:
    def __init__(self, data):
        self.data = data

    def analyze_correlation(self):
        prompt = f"I have computed a correlation matrix for my dataset with {self.data.shape[1]} variables. How do I interpret the results?"
        advice = query_openai(prompt)
        return advice

    def visualize_correlation(self):
        """Visualize the correlation matrix using Plotly in Streamlit."""
        correlation_matrix = self.data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='Viridis',
            colorbar=dict(title='Correlation Coefficient'),
            zmin=-1, zmax=1
        ))

        fig.update_layout(title='Correlation Matrix Heatmap', xaxis_title='Variables', yaxis_title='Variables')

        # Displaying the plot using Streamlit's plotly chart function
        st.plotly_chart(fig, use_container_width=True)

    def display_correlated_variables(self):
        """Return the list of correlated variable pairs without dropping them immediately."""
        correlation_matrix = self.data.corr().abs()
        correlated_pairs = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1)).stack()
        highly_correlated = correlated_pairs[correlated_pairs > 0.8]  # Adjust threshold as needed

        if not highly_correlated.empty:
            correlated_vars = []
            st.write("Variables fortement corrélées :")
            for (var1, var2), correlation in highly_correlated.items():
                st.write(f"{var1} et {var2}: {correlation:.2f}")
                correlated_vars.append((var1, var2))  # Store the pairs instead of immediately dropping variables
            return correlated_vars
        else:
            st.write("Aucune variable fortement corrélée trouvée.")
            return None




    def apply_pca_with_grid_search(self, X, y):
        """Appliquer GridSearch pour trouver le meilleur n_components pour PCA."""
        # Pipeline avec PCA et un classifieur (par exemple RandomForest)
        pipe = Pipeline([
            ('pca', PCA()),
            ('clf', RandomForestClassifier(random_state=42))
        ])
        
        # Définir la grille de recherche pour les n_components de PCA
        param_grid = {
            'pca__n_components': [2, 5, 10, 20, 30, 50]  # On peut ajuster ces valeurs en fonction du dataset
        }
        
        # GridSearch pour trouver la meilleure valeur de n_components
        grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')  # Utilisation de la précision comme critère
        grid_search.fit(X, y)

        # Meilleur nombre de composantes
        best_n_components = grid_search.best_params_['pca__n_components']
        st.write(f"Meilleur nombre de composantes pour l'ACP : {best_n_components}")
        
        # Appliquer l'ACP avec le meilleur n_components
        pca = PCA(n_components=best_n_components)
        X_pca = pca.fit_transform(X)

        return X_pca
       

# 5. Feature Selection
class VariableSelectionAgent:
    def __init__(self, X, y, previous_results):
        self.X = X
        self.y = y
        self.previous_results = previous_results
        self.results = {}

    def select_variables(self):
        if "error" in self.previous_results:
            return {"error": "Previous agent encountered an error.", "previous_results": self.previous_results}

        prompt = f"I have {self.X.shape[1]} features and a target variable. What are the best methods for feature selection in credit scoring?"
        
        # Query OpenAI for advice on feature selection methods.
        advice = query_openai(prompt)

        model = RandomForestClassifier()
        model.fit(self.X, self.y)
        
        selector = SelectFromModel(model, prefit=True)
        
        selected_features_indices = selector.get_support(indices=True)

        # Convert indices to a list for better readability.
        selected_features_list = selected_features_indices.tolist()

        self.results = {
            "advice": advice,
            "selected_features": selected_features_list  # Return as list for easier handling.
        }
        
        return self.results

# 6. Optional: Segmentation
class SegmentationAgent:
    def __init__(self, data, n_clusters):
        self.data = data
        self.n_clusters = n_clusters

    def segment_data(self):
        
       prompt = f"I have a dataset with {self.data.shape[1]} features. What are the best practices for segmenting the data into {self.n_clusters} clusters?"
        
       # Query OpenAI for segmentation advice.
       advice = query_openai(prompt)

       kmeans = KMeans(n_clusters=self.n_clusters)
       
       # Fit KMeans clustering.
       self.data['Cluster'] = kmeans.fit_predict(self.data)

       return advice, self.data

class ModelingAgent:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}
        self.recommendation_response_text = ""

        # Assurer qu'il n'y a pas de NaN avant de modéliser
        if X_train.isnull().values.any() or X_test.isnull().values.any():
            raise ValueError("Les données contiennent des valeurs manquantes. Assurez-vous que le prétraitement est correctement appliqué.")

        # Initialisation des modèles
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),  # Désactivation de l'encoder XGBoost
            "Random Forest": RandomForestClassifier(),
            "Neural Network": MLPClassifier(max_iter=1000)
        }

    def train_and_evaluate(self):
        for model_name, model in self.models.items():
            try:
                # Entraînement du modèle
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                # Calcul des métriques
                accuracy = accuracy_score(self.y_test, y_pred)
                f1_score_value = f1_score(self.y_test, y_pred, average='weighted')

                # Pour ROC AUC, vérifier si le modèle peut prédire des probabilités
                if hasattr(model, "predict_proba"):
                    roc_auc_value = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])
                else:
                    roc_auc_value = None  # ROC AUC non applicable

                # Enregistrement des résultats pour chaque modèle
                self.results[model_name] = {
                    "Accuracy": accuracy,
                    "F1-Score": f1_score_value,
                    "ROC AUC": roc_auc_value if roc_auc_value is not None else "Non applicable"
                }

            except Exception as e:
                # Affichage de l'erreur dans l'interface Streamlit
                st.error(f"Erreur lors de la modélisation avec {model_name}: {str(e)}")
        
        # Demander à OpenAI de choisir le meilleur modèle (hypothétique)
        prompt_results_summary = f"Based on the following modeling results: {self.results}, which model do you recommend for a credit risk scorecard?"
        self.recommendation_response_text = query_openai(prompt_results_summary)

        return self.results, self.recommendation_response_text
    
    def display_confusion_matrices(self):
        for model_name, model in self.models.items():  # Utilisez items() pour récupérer à la fois le nom et l'objet du modèle
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)

            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f'Confusion Matrix for {model_name}')  # Utilisez model_name ici pour l'affichage correct du nom
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()


    def display_comparison_table(self):
        """Display comparison table of all models."""
        st.subheader("🔍 Comparaison des modèles")
        comparison_df = pd.DataFrame(self.results).T  # Transpose to get models as rows
        st.write(comparison_df)


    def get_model_history(self, model, X, y):
        """Get training/testing history for a model."""
        # Assuming you're tracking accuracy over epochs
        history = {"epoch": [], "accuracy": []}
        for epoch in range(1, 11):  # Example: 10 epochs
            model.fit(X, y)  # Fit the model (in practice, you would use validation data)
            accuracy = model.score(X, y)
            history["epoch"].append(epoch)
            history["accuracy"].append(accuracy)
        return history

    def get_train_history(self, model_name):
        """Retrieve training history for a specific model."""
        return self.train_histories.get(model_name, None)

    def get_test_history(self, model_name):
        """Retrieve testing history for a specific model."""
        return self.test_histories.get(model_name, None)
    





