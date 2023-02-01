pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                withCredentials([string(credentialsId: 'AML_CLIENT_SECRET', variable: 'CLIENT_SECRET')]) {
                    sh '/Users/sandeepmaurya/opt/anaconda3/bin/python3 src/diabetes_classification/pr_job.py'
                }
            }
        }
    }
}
