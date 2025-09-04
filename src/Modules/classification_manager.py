import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import time
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
# import xgboost as xgb
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
            # 'random_forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
            # 'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
            # 'svm': SVC(random_state=random_state, probability=True),
            # 'knn': KNeighborsClassifier(n_neighbors=5),
            # "xgboost": XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='mlogloss'),
            "mlp": MLPClassifier(random_state=random_state, max_iter=100)
        }
        self.trained_models = {}
        self.best_model = None
        self.best_score = 0
        self.labels = None
        self.embeddings = None
        self.embeddings_path = None
        
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
    
    def load_data(self, embeddings_path):
        """
        Carga los embeddings y las etiquetas de clustering.
        
        NOTA: Este método asume que las etiquetas están en un archivo de embeddings
        por lo tanto la clase etiqueta se debe colocar hasta el final con el nombre de "label".
        
        Parámetros:
            embeddings_path (str): Ruta al archivo de embeddings (.npy).
    
        Retorna:
            embeddings (np.ndarray): Embeddings cargados.
            labels (np.ndarray): Etiquetas de clustering.
        """
        # Cargar embeddings
        # print(f"Cargando.... {embeddings_path}")
        if embeddings_path.endswith('.npy'):
            embeddings = np.load(embeddings_path)
            # Convertir a DataFrame para manejar etiquetas
            embeddings = pd.DataFrame(embeddings)  
            # Asignar nombres de columnas
            embeddings.columns = [f"{i}" for i in range(embeddings.shape[1])]
            # Las etiquetas colocarlas con el nombre label
            embeddings.columns[-1] = 'label'
            
        elif embeddings_path.endswith('.csv'):
            embeddings = pd.read_csv(embeddings_path)  # Convertir a numpy array
        else:
            raise ValueError("El archivo de embeddings debe ser .npy o .csv")
        
        print(f"Embeddings cargados desde con forma {embeddings.shape}")
        print(f"Columna etiqueta al final {embeddings.columns[-1]}")
        
        
        labels = None
        NAME_COLUMN_LABEL = "label"
        if NAME_COLUMN_LABEL in embeddings.columns:
            labels = embeddings[NAME_COLUMN_LABEL].values
            # Separar embeddings de etiquetas
            embeddings = embeddings.drop(columns=[NAME_COLUMN_LABEL]).values
        else:
            raise FileNotFoundError(f"No se encontraron etiquetas en {embeddings_path}")
        
        # SIGUIENTE ACTUALIZACIÓN:
        # Si embeddings incluye las etiquetas, separarlas
        
               
        # # Si embeddings incluye las etiquetas, separarlas
        # if embeddings.ndim > 1 and embeddings_path.endswith('.npy'):
        #     if labels is None:  # Solo separar si no hemos cargado las etiquetas por separado
        #         labels = embeddings[:, -1]  # Última columna son las etiquetas
        #         embeddings = embeddings[:, :-1]  # Todas las columnas excepto la última
        self.labels = labels  
        self.embeddings = embeddings 
        self.embeddings_path = embeddings_path
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
            # y_train = np.where(y_train == -1, 999, y_train)
            # Validación cruzada
            # cv_scores = cross_val_score(classifier, X_train, y_train, cv=cv_folds)

            
            # Entrenar en todos los datos de entrenamiento
            classifier.fit(X_train, y_train)
            self.trained_models[name] = classifier
            
            # Guardar resultados
            # results[name] = {
            #     'cv_mean': cv_scores.mean(),
            #     'cv_std': cv_scores.std(),
            #     'cv_scores': cv_scores
            # }
            
            # print(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Actualizar mejor modelo
            # if cv_scores.mean() > self.best_score:
            #     self.best_score = cv_scores.mean()
            #     self.best_model = name
            self.best_model = name
            # Guardar resultados en CSV
            # results_df = pd.DataFrame([{
            #     'modelo': name,
            #     'cv_mean': cv_scores.mean(),
            #     'cv_std': cv_scores.std(),
            #     'cv_scores': cv_scores.tolist()
            # }])
            # csv_path = os.path.join("model_results.csv")
            # if os.path.exists(csv_path):
            #     prev_df = pd.read_csv(csv_path)
            #     results_df = pd.concat([prev_df, results_df], ignore_index=True)
            # # results_df.to_csv(csv_path, index=False)

            # Calcular R1 (macro recall)
            # y_pred = classifier.predict(X_train)
            # r1 = recall_score(y_train, y_pred, average='macro')
            # results[name]['r1'] = r1
            # print(f"{name} - R1: {r1:.4f}")

            # accuracy = accuracy_score(y_train, y_pred)
            # results[name]['accuracy'] = accuracy
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
