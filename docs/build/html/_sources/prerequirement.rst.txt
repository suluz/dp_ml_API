Before you start
==================================

Installation
------------

This API server requires `Python 3 installation <https://www.python.org/downloads/>`_ and some `Python packages installation <https://packaging.python.org/tutorials/installing-packages//>`_, such as *Flask*, *Tensorflow Keras*, *NumPy* and *scikit-learn*. The following example shows the required package installation via *Unix/MaxOS termianl*. 

.. code-block:: console

   $ pip install flask
   $ pip install tensorflow
   $ pip install numpy
   $ pip install sklearn

.. note::
   
   All the examples in this documentation are based on a Unix/MaxOS system.


.. _org:

Files and Directories
---------------------

Our APIs files and directories are organised as follow.

.. code-block:: text

   └── dpaip_api
       ├── datasets
       │   ├── pred_set.csv 
       │   └── raw_set.csv
       ├── demo_data
       │   ├── prediction.csv
       │   └── train.csv
       ├── docs
       │   ├── build
       │   │   └── html
       │   │       └── index.html
       ├── outputs
       │   ├── trained_model_1
       │   │   ├── attack_loss.pdf
       │   │   └── out.txt
       │   └── trained_model_n
       ├── package_data_io
       ├── package_dp_model
       ├── package_train_model
       ├── saved_models
       │   ├── trained_model_1
       │   │
       │   └── trained_model_n
       ├── upload_files
       ├── api.py
       ├── api_calls.py
       ├── web_app.html
       └── README.rst

The purpose of each file/directory is:

``/datasets/*.csv``
   A folder stores the .csv files used during the training and prediction process.

``/demo_data/*.csv``
   A folder stores the .csv files for the API demonstration, i.e., those files are the ones for uploading to the API server.

``/docs/build/html/index.html``
   A .html file for a static API documentation. Online documentation is available after running the server (refer to :ref:`Online Documentation <online_doc>` section).
   
``/outputs/trained_model_1/*``
   Files of the training outputs of the ``trained_model_1``

``/package_*/``
   Folders store the Python source code to implement the differentially private neural networks.

``/saved_models/trained_model_1/``
   A folder stores the model details of the ``trained_model_1``

``/upload_files``
   A folder temporally stores the uploaded files.

``/api.py``
   A back-end Python file handles all the APIs calls.

``api_calls.py``
   A Python file provides demostration of making API calls with Python code. Run it as

   .. code-block:: console
   
      $ python3 api_calls.py --call=STRING
   
   ``STRING = /upload/train``: call ``/upload/train``

   ``STRING = /upload/predict``: call ``/upload/predict``

   ``STRING = /check``: call ``/check``

   ``STRING = /show``: call ``/show``

   ``STRING = /train``: call ``/train``

   ``STRING = /predict``: call ``/predict``

   ``STRING = /compare``: call ``/compare``

``web_app.html``
   A .html file for the front-end of the web application of the APIs.

