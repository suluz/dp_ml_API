Compare Test Accuracy of DP Predictions and Non-Private Predictions
===================================================================

API Function
------------

.. code-block:: python

   from flask import Flask, request, jsonify
   import numpy as np

   app = Flask(__name__)

   @app.route('/compare', methods=['POST'])
   '''
   Compare test accuracy of differentially 
   private prediction and non-private prediction 
   of a given model and a given privacy budget 
   (passed through the .JSON file from the 
   API call below), then return the prediction 
   output in .JSON format.
   '''
   def compare():
      par = request.get_json()
      dp_pred_vec, _ = make_prediction(
         par, has_label=True
      )
      dp_pred_label = np.argmax(dp_pred_vec, axis=1)

      '''Read test_acc_ from 
      /outputs/par['model_name']/out.txt'''

      return jsonify(
         {
            'trained_models': str(par['model_name']), 
            'privacy_budget': float(par['epsilon']), 
            'dp_test_accuracy': dp_test_acc,
            'non_private_test_accuracy': test_acc_
         }
      )


API Call
--------

   **Method**: ``PUT``

   **API URL**: ``/compare``

   **Description**: Compare test accuracy of differentially private prediction and non-private prediction.

   **Request body**: A .JSON file containing the model name and the privacy budget

   **Example of the request body**:

      .. code-block:: json

         {
            "model_name": "string",
            "epsilon": "float"
         }

   **Responses**: Default HTTP response code; A .JSON file containing the test accuracy of differentially private prediction and non-private prediction

   **Example of the responses**:

      .. code-block:: json

         {
            "trained_model": "string",
            "privacy_budget": "float",
            "dp_test_accuracy": "float",
            "non_private_test_accuracy": "float"
         }

   **Example Python code calling API**:

      .. code-block:: python
         
         import requests
         API_URL = 'http://localhost:5000'

         response = requests.post(
            '{}/compare'.format(API_URL),
            json={
               'model_name':'purchase-100',
               'epsilon':10000
            }
         )
        response.json()

   **Example of returned JSON**:

      .. code-block:: json

         {
            "trained_model": "purchase-100",
            "privacy_budget": "10000",
            "dp_test_accuracy": "0.866",
            "non_private_test_accuracy": "0.866"
         }