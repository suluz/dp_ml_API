Upload Data for Prediction
==========================

API Function
------------

.. code-block:: python

   from flask import Flask, request

   app = Flask(__name__)

   @app.route('/upload/predict/<filename>', methods=['POST'])
   def upload_predict(filename):
      '''
      Save filename from request.data to the target
      directory /datasets/pred_set.csv
      '''
      return '', 201

.. **Description**: Save ``filename`` (taking from the API call below) to the target directory ``/datasets/pred_set.csv``.

API Call
--------

**Method**: ``PUT``

**API URL**: ``/upload/predict``

**Description**: Upload a file for prediction

**Request body**: A .CSV file containing data points for *prediction*, where each line represents a data point, all the columns are the features of the data points (normalisation is not required; label is not required).

**Example of the request body**: ``demo_data/prediction.csv`` (refer to :ref:`Files and Directories <org>` section)

**Resonses**: Default HTTP response code

**Example Python code calling API**:

   .. code-block:: python

      import requests
      API_URL = 'http://localhost:5000'
      RAWDATA_DIR = '/home/hub62/Documents/dpaip_api/demo_data'

      f = open(os.path.join(RAWDATA_DIR, 'prediction.csv'), 'r')
      content = f.read()

      response = requests.post(
         '{}/upload/predict/prediction.csv'.format(API_URL), 
         headers=headers, 
         data=content
      )
      response.status_code