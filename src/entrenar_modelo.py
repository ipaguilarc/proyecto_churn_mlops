from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

TRAIN_DATA = DATA_DIR / "train.csv"
MODEL_FILE = MODELS_DIR / "modelo_churn.pkl"
MODEL_FILE_LOGREG = MODELS_DIR / "modelo_churn_logreg.pkl"
MODEL_FILE_RF = MODELS_DIR / "modelo_churn_rf.pkl"

def entrenar_modelo():
    """
    Entrena dos modelo  de clasificación para predecir churn.
    - Logistic Regression
    - Random Forest
    """

    if not TRAIN_DATA.exists():
        raise FileNotFoundError(
            "No se encontró data/train.csv. Primero ejecuta src/preparar_datos.py"
        )

    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(TRAIN_DATA)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    modelo_logreg = Pipeline(
        steps=[
            ("escalado", StandardScaler()),
             ("clasificador", LogisticRegression(max_iter=500))
        ]
    )

    modelo_logreg.fit(X, y)
    
    joblib.dump(modelo_logreg, MODEL_FILE_LOGREG)
    print("Modelo entrenado correctamente.")
    print(f"Modelo Logistic Regression guardado en: {MODEL_FILE_LOGREG}")

 # Modelo 2: Random Forest
    modelo_rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
    modelo_rf.fit(X, y)
    joblib.dump(modelo_rf, MODEL_FILE_RF)
    print("Modelo entrenado correctamente.")
    print(f"Modelo Random Forest guardado en: {MODEL_FILE_RF}")

if __name__ == "__main__":
    entrenar_modelo()
