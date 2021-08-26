Check Available Models
======================

API Function
------------

.. code-block:: python

   from flask import Flask, jsonify

   app = Flask(__name__)

   @app.route('/check', methods=['GET'])
   def check():
      '''
      Get the trained model list, then return 
      it in .JSON format.
      '''
      return jsonify({'trained_models': model_list_})


API Call
--------

   **Method**: ``GET``

   **API URL**: ``/check``

   **Description**: Check the trained models available for prediction

   **Request body**: None

   **Resonses**: Default HTTP response code; A .JSON file containing a list of available models

   **Example of the responses**:

      .. code-block:: json

         {
            "trained_models": "string[]"
         }
   
   **Example Python code calling API**:

      .. code-block:: python

         import requests
         API_URL = 'http://localhost:5000'

         response = requests.get(
            '{}/check'.format(API_URL), 
            headers=headers
         )
         response.json()

   **Example of returned JSON**:

      .. code-block:: json

         {
            "trained_models": "[purchase-100, purchase-50]"
         }