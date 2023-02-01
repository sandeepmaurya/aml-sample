pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                withCredentials([string(credentialsId: 'AML_CLIENT_SECRET', variable: 'CLIENT_SECRET')]) {
                    sh 'python pr_job.py'
                }
            }
        }
    }
}