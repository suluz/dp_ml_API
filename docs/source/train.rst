Train a Non-private Model
=========================

API Function
------------

.. code-block:: python

   from flask import Flask, request

   app = Flask(__name__)

   @app.route('/train', methods=['POST'])
   '''
   Train a non-private model from 
   the hyperparameters (passed through 
   the .JSON file from the API call below), 
   then save the trained model to 
   /saved_models.
   '''
   def train():
      hyper_par = request.get_json()
      train_model(hyper_par)

      return '', 201

.. **Description**: Train a non-private model from the hyperparameters (passed through the .JSON file from the API call below), then save the trained model to ``/saved_models``.

API Call
--------

   **Method**: ``PUT``

   **API URL**: ``/train``

   **Description**: Train a non-private model with given hyperparameters on the given dataset ``demo_data/train.csv``

   **Request body**: A .JSON file containing hyperparameters to a neural network.

   **Example of the request body**:
   
      .. code-block:: json

         {
            "training_set": "string", 
            "n_train": "int32",
            "n_test": "int32",
            "n_labels": "int32",
            "n_hidden_layers": "int32",
            "n_hidden_neurons": "int32",
            "n_epochs": "int32",
            "n_batch": "int32",
            "learning_rate": "float",
            "l2_reg": "float",
            "activation": "string"
         }

   **Resonses**: Default HTTP response code

   **Example Python code calling API**:

      .. code-block:: python
         
         import requests
         API_URL = 'http://localhost:5000'
         
         response = requests.post(
            '{}/train'.format(API_URL), 
            json={
                'training_set':'purchase', 
                'n_train':10000,
                'n_test':10000,
                'n_labels':100,
                'n_hidden_layers':1,
                'n_hidden_neurons':128,
                'n_epochs':100,
                'n_batch':100,
                'learning_rate':0.001,
                'l2_reg':0.001,
                'activation':'tanh'
            }
        )
        response.status_code