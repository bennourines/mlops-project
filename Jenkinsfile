pipeline {
    agent any

    stages {
        stage('Start MLflow Server') {
            steps {
                sh 'mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &'
                sh 'sleep 10' // Wait for the server to start
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'make install'
            }
        }

        stage('Run Unit Tests') {
            steps {
                sh 'make unit-test'
            }
        }

        stage('Run Functional Tests') {
            steps {
                sh 'make functional-test'
            }
        }

        stage('Data Pipeline') {
            steps {
                sh 'make data'
            }
        }

        stage('Training Pipeline') {
            steps {
                sh 'make train'
            }
        }

        stage('Evaluation Pipeline') {
            steps {
                sh 'make evaluate'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'make docker-build'
            }
        }

        stage('Push Docker Image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_HUB_USER', passwordVariable: 'DOCKER_HUB_PASSWORD')]) {
                    sh 'docker login -u $DOCKER_HUB_USER -p $DOCKER_HUB_PASSWORD'
                    sh 'make docker-push'
                }
            }
        }

        stage('Deploy') {
            steps {
                sh 'make docker-run'
            }
        }
    }

    post {
        success {
            emailext body: 'The pipeline completed successfully.', subject: 'Pipeline Success', to: 'bennourines00@gmail.com'
        }
        failure {
            emailext body: 'The pipeline failed.', subject: 'Pipeline Failure', to: 'bennourines00@gmail.com'
        }
    }
}
