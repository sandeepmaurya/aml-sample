from mlflow.pyfunc import PythonModel, PythonModelContext


class ModelWrapper(PythonModel):
    def __init__(self, model):
        self._model = model

    def predict(self, context: PythonModelContext, data):
        return self._model.predict(data)
