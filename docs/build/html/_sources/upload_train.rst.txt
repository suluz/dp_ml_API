Upload Data for Training
========================

API Function
------------

.. code-block:: python

   from flask import Flask, request

   app = Flask(__name__)

   @app.route('/upload/train/<filename>', methods=['POST'])
   def upload_train(filename):
      '''
      Save filename from request.data to the target
      directory /datasets/raw_set.csv
      '''
      return '', 201

.. **Description**: Save ``filename`` (taking from the API call below) to the target directory ``/datasets/raw_set.csv``.

API Call
--------

**Method**: ``PUT``

**API URL**: ``/upload/train``

**Description**: Upload a file for training

**Request body**: A .CSV file containing data points for *both training and test*, where each line represents a data point, the first column is the label (in non-negative integer) and the rest columns are the features of the data points (normalisation is not required).

**Example of the request body**: ``demo_data/train.csv`` (refer to :ref:`Files and Directories <org>` section)

**Resonses**: Default HTTP response code

**Example Python code calling API**:

   .. code-block:: python

      import requests
      API_URL = 'http://localhost:5000'
      RAWDATA_DIR = '/home/hub62/Documents/dpaip_api/demo_data'

      f = open(os.path.join(RAWDATA_DIR, 'train.csv'), 'r')
      content = f.read()

      response = requests.post(
         '{}/upload/train/train.csv'.format(API_URL), 
         headers=headers, 
         data=content
      )
      response.status_code
