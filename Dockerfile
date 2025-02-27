# Utiliser une image Python comme base
FROM python:3.8

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier tous les fichiers du projet dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

# Exposer le port sur lequel l'application FastAPI tourne
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

