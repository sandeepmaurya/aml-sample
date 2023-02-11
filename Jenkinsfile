pipeline {
    agent any
    stages {
        stage('PR Check') {
            when {
                    branch 'PR-*'
            }

            steps {
                withCredentials([string(credentialsId: 'AML_CLIENT_SECRET', variable: 'CLIENT_SECRET')]) {
                    sh '/Users/sandeepmaurya/opt/anaconda3/bin/python3 diabetes_classification/pr_push.py'
                }
            }
        }
        stage('Main Build') {
            when {
                    branch 'main'
            }

            steps {
                withCredentials([string(credentialsId: 'AML_CLIENT_SECRET', variable: 'CLIENT_SECRET')]) {
                    sh '/Users/sandeepmaurya/opt/anaconda3/bin/python3 diabetes_classification/main_push.py'
                }
            }
        }
    }
}
