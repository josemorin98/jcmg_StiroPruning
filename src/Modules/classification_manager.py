import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

class ClassificationManager:
    """
    Clase para generar modelos de clasificación basados en clusters obtenidos
    del clustering y entrenar clasificadores para predecir la pertenencia a clusters.
    """
    
    def __init__(self, random_state=42):
        """
        Inicializa el gestor de clasificación.
        
        Parámetros:
            random_state (int): Semilla para reproducibilidad.
        """
        self.random_state = random_state
        self.classifiers = {
            'random_forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
            'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'svm': SVC(random_state=random_state, probability=True),
            'knn': KNeighborsClassifier(n_neighbors=5)
        }
        self.trained_models = {}
        self.best_model = None
        self.best_score = 0
        
    def _save_time(self, metodo, tiempo, modelos_dir):
        """Guarda el tiempo de ejecución en un CSV."""
        os.makedirs(modelos_dir, exist_ok=True)
        csv_path = os.path.join(modelos_dir, "times_classification.csv")
        df = pd.DataFrame([{
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metodo": metodo,
            "tiempo_segundos": round(tiempo, 2)
        }])
        if os.path.exists(csv_path):
            df_ant = pd.read_csv(csv_path)
            df = pd.concat([df_ant, df], ignore_index=True)
        df.to_csv(csv_path, index=False)
    
    def load_clustering_data(self, embeddings_path, labels_path=None, clustering_manager_dir=None, method="bayesian"):
        """
        Carga los embeddings y las etiquetas de clustering.
        
        Parámetros:
            embeddings_path (str): Ruta al archivo de embeddings (.npy).
            labels_path (str): Ruta al archivo de etiquetas (.npy o .csv) (opcional).
            clustering_manager_dir (str): Directorio con modelos de clustering (opcional).
            method (str): Método de clustering usado ('bayesian', 'random', 'grid').
            
        Retorna:
            embeddings (np.ndarray): Embeddings cargados.
            labels (np.ndarray): Etiquetas de clustering.
        """
        # Cargar embeddings
        if embeddings_path.endswith('.npy'):
            embeddings = np.load(embeddings_path)
        else:
            raise ValueError("El archivo de embeddings debe ser .npy")
            
        # Cargar etiquetas
        labels = None
        if labels_path:
            if labels_path.endswith('.npy'):
                labels = np.load(labels_path)
                if labels.ndim > 1:  # Si incluye embeddings + labels
                    labels = labels[:, -1]  # Última columna son las etiquetas
            elif labels_path.endswith('.csv'):
                df = pd.read_csv(labels_path)
                labels = df['label'].values
        elif clustering_manager_dir:
            # Intentar cargar desde el directorio de clustering
            label_file = os.path.join(clustering_manager_dir, f"embeddings_originales_labeled_{method}.csv")
            if os.path.exists(label_file):
                df = pd.read_csv(label_file)
                labels = df['label'].values
            else:
                raise FileNotFoundError(f"No se encontraron etiquetas en {label_file}")
        else:
            raise ValueError("Debe proporcionar labels_path o clustering_manager_dir")
            
        return embeddings, labels
    
    def prepare_classification_data(self, embeddings, labels, remove_noise=True, test_size=0.2):
        """
        Prepara los datos para clasificación.
        
        Parámetros:
            embeddings (np.ndarray): Embeddings originales.
            labels (np.ndarray): Etiquetas de clustering.
            remove_noise (bool): Si eliminar puntos etiquetados como ruido (-1).
            test_size (float): Proporción de datos para test.
            
        Retorna:
            X_train, X_test, y_train, y_test: Datos divididos para entrenamiento y prueba.
        """
        # Remover ruido si se solicita
        if remove_noise:
            mask = labels != -1
            embeddings = embeddings[mask]
            labels = labels[mask]
            
        # Verificar que hay suficientes clases
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError(f"Se necesitan al menos 2 clases para clasificación. Se encontraron: {len(unique_labels)}")
            
        print(f"Datos preparados: {len(embeddings)} muestras, {len(unique_labels)} clases")
        print(f"Distribución de clases: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, 
            random_state=self.random_state, stratify=labels
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_classifiers(self, X_train, y_train, cv_folds=5):
        """
        Entrena múltiples clasificadores y evalúa su rendimiento con validación cruzada.
        
        Parámetros:
            X_train (np.ndarray): Datos de entrenamiento.
            y_train (np.ndarray): Etiquetas de entrenamiento.
            cv_folds (int): Número de folds para validación cruzada.
            
        Retorna:
            results (dict): Resultados de validación cruzada por clasificador.
        """
        start_time = time.time()
        results = {}
        
        for name, classifier in self.classifiers.items():
            print(f"Entrenando {name}...")
            
            # Validación cruzada
            cv_scores = cross_val_score(classifier, X_train, y_train, cv=cv_folds)
            
            # Entrenar en todos los datos de entrenamiento
            classifier.fit(X_train, y_train)
            self.trained_models[name] = classifier
            
            # Guardar resultados
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Actualizar mejor modelo
            if cv_scores.mean() > self.best_score:
                self.best_score = cv_scores.mean()
                self.best_model = name
                
        print(f"\nMejor modelo: {self.best_model} (Score: {self.best_score:.4f})")
        
        tiempo = time.time() - start_time
        print(f"Tiempo de entrenamiento: {tiempo:.2f} segundos")
        
        return results
    
    def predict_new_data(self, model_name, new_embeddings):
        """
        Predice las etiquetas para nuevos datos usando un modelo entrenado.
        
        Parámetros:
            model_name (str): Nombre del modelo a usar.
            new_embeddings (np.ndarray): Nuevos embeddings para clasificar.
            
        Retorna:
            predictions (np.ndarray): Predicciones del modelo.
            probabilities (np.ndarray): Probabilidades de cada clase (si disponible).
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} no encontrado")
            
        model = self.trained_models[model_name]
        predictions = model.predict(new_embeddings)
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(new_embeddings)
            
        return predictions, probabilities
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evalúa un modelo entrenado en el conjunto de prueba.
        
        Parámetros:
            model_name (str): Nombre del modelo a evaluar.
            X_test (np.ndarray): Datos de prueba.
            y_test (np.ndarray): Etiquetas de prueba.
            
        Retorna:
            metrics (dict): Métricas de evaluación.
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} no encontrado")
            
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nEvaluación del modelo: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred))
        
        metrics = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics
    
    def save_models(self, modelos_dir, model_suffix=""):
        """
        Guarda todos los modelos entrenados.
        
        Parámetros:
            modelos_dir (str): Directorio donde guardar los modelos.
            model_suffix (str): Sufijo para los archivos de modelo.
        """
        os.makedirs(modelos_dir, exist_ok=True)
        
        for name, model in self.trained_models.items():
            filename = f"classifier_{name}{model_suffix}.pkl"
            filepath = os.path.join(modelos_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"Modelo {name} guardado en {filepath}")
            
        # Guardar información del mejor modelo
        best_info = {
            'best_model': self.best_model,
            'best_score': self.best_score,
            'random_state': self.random_state
        }
        
        info_path = os.path.join(modelos_dir, f"classification_info{model_suffix}.pkl")
        with open(info_path, 'wb') as f:
            pickle.dump(best_info, f)
        print(f"Información de clasificación guardada en {info_path}")
    
    def load_model(self, model_path):
        """
        Carga un modelo de clasificación guardado.
        
        Parámetros:
            model_path (str): Ruta al archivo del modelo.
            
        Retorna:
            model: Modelo de clasificación cargado.
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
