Make Differentially Private Prediction
======================================

API Function
------------

.. code-block:: python

   from flask import Flask, request, jsonify
   import numpy as np

   app = Flask(__name__)

   @app.route('/predict', methods=['POST'])
   '''
   Make differentially private prediction of 
   a given model and a given privacy budget 
   (passed through the .JSON file from the 
   API call below), then return the prediction 
   output in .JSON format.
   '''
   def predict():
      par = request.get_json()
      dp_pred_vec, _ = make_prediction(
         par, has_label=False
      )
      dp_pred_label = np.argmax(dp_pred_vec, axis=1)

      return jsonify(
         {
            'predicted_label': dp_pred_label,
            'probability_vec': dp_pred_vec
         }
      )


API Call
--------

   **Method**: ``PUT``

   **API URL**: ``/predict``

   **Description**: Make differentially private prediction of a given model based on the privacy budget for the given datast ``demo_data/prediction.csv``

   **Request body**: A .JSON file containing the model name and the privacy budget

   **Example of the request body**:

      .. code-block:: json

         {
            "model_name": "string",
            "epsilon": "float"
         }

   **Responses**: Default HTTP response code; A .JSON file containing the predicted label and a probability vector

   **Example of the responses**:

      .. code-block:: json

         {
            "predicted_label": "int32",
            "probability_vec": "float[]"
         }

   **Example Python code calling API**:

      .. code-block:: python
         
         import requests
         API_URL = 'http://localhost:5000'

         response = requests.post(
            '{}/predict'.format(API_URL),
            json={
               'model_name':'purchase-2',
               'epsilon':10000
            }
         )
        response.json()

   **Example of returned JSON**:

   .. code-block:: json

      {
         "predicted_label": "1",
         "probability_vec": "[0.1, 0.9]"
      }