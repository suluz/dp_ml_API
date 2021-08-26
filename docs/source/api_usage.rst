API Usage
=========

Start the Server
----------------

Once you have installed all the dependencies, change the hard-coded directories in ``api.py`` and ``api_calls.py`` to your local directories, then start your API server by running the ``api.py``.

.. code-block:: console

   $ python3 api.py

.. note::
   * You should restart the server when making any changes in ``api.py``
   * The following sections are all relied on running the APIs server.

.. _online_doc:

Online Documentation
--------------------

You can also access this documentation by typing ``localhost:5000/help`` in your web browser.


Web Application
---------------

You can access the web-based application of the *DPNN APIs* by typing ``localhost:5000/web`` in your web browser. In the web application, you can
   * **upload** datasets for training and/or prediction;

   * **train** a non-private model with the uploaded files and the required configurations for the model hyperparameters; 
  
   * **check** the trained models list;

   * **show** the training performance of a given trained model;

   * make *differentially private* **prediction** of given data samples by a selected trained model and a selected privacy budget;

   * **compare** the test accuracy of predictions made by the non-private model and the differentially private neural network.


API Functions and API Calls
---------------------------

We provide seven API calls. The details of the APIs (*API functions* and *API calls*) are:

.. toctree::
   :maxdepth: 1

   upload_train
   upload_pred
   train
   check
   show
   predict
   compare

