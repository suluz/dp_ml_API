Show Model Performance
======================

API Function
------------

.. code-block:: python

   from flask import Flask, request, jsonify

   app = Flask(__name__)

   @app.route('/show', methods=['POST'])
   '''
   Show the training and test accuracy 
   of given trained models (passed through 
   the .JSON file from the API call below)
   '''
   def show():
      requested_model = request.get_json()
      '''Read training_acc_ and test_acc_ from 
      /outputs/requested_model/out.txt'''

      return jsonify(
         {
            'model_name': requested_model,
            'training_acc': training_acc_,
            'test_acc': test_acc_
         }
      )


API Call
--------

   **Method**: ``PUT``

   **API URL**: ``/show``

   **Description**: Show the training performance of a given model

   **Request body**: A .JSON file containing the quiried model names

   **Example of the request body**:
   
      .. code-block:: json

         {
            "model_name": "string"
         }

   **Responses**: Default HTTP response code; A .JSON file containing the training and test accuracy of a given model

   **Example of the responses**:

      .. code-block:: json

         {
            "model_name": "string",
            "training_acc": "float",
            "test_acc": "float"
         }

   **Example Python code calling API**:

      .. code-block:: python
         
         import requests
         API_URL = 'http://localhost:5000'

         response = requests.post(
            '{}/show'.format(API_URL), 
            json=[
               {'model_name':'purchase-100'},
               {'model_name':'purchase-50'}
            ]
         )
         response.json()

   **Example of returned JSON**:

      .. code-block:: json

         [
            {
               "model_name": "purchase-100",
               "training_acc": "0.96",
               "test_acc": "0.81"
            },
            {
               "model_name": "purchase-50",
               "training_acc": "0.97",
               "test_acc": "0.85"
            }
         ]