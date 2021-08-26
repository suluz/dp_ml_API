.. differentially private neural networks documentation master file, created by
   sphinx-quickstart on Wed Aug 18 14:17:19 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Differentially Private Neural Networks API Documentation
========================================================

**Differentially Private Neural Networks (DPNN) API** is a production of **Data Privacy in AI Platforms, risks quantifications and defence apparatus (DPAIP)** Project, which was funded by the **Defence Science & Technology (DST) Group** through the Next Generation Technologies Fund.

**DPAIP** project aims to 

   * Quantify information leakage in terms of the extent to which parameters of the learned model and the original dataset can be inferred when machine learning is provided as a service;
   * Develop solutions for privacy-preserving Machine Learning-as-a-Service where data subjects are guaranteed that exposure of the model will not result in plausible inference of their data and data owners are guaranteed that proprietary datasets and models remain private.

In line with the aims of the **DPAIP** project, **DPNN API** provides a *black-box* *privacy-preserving* *Machine Learning-as-a-Service (MLaaS)* platform, protecting the privacy of the training dataset against `Membership Inference Attacks <https://bdtechtalks.com/2021/04/23/machine-learning-membership-inference-attacks/>`_. With DPNN APIs, you can make `Differentially Private <https://en.wikipedia.org/wiki/Differential_privacy>`_ predictions by a trained *non-private* model. Technical details of the APIs can be found in our :download:`research paper </paper.pdf>`.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   prerequirement
   api_usage




.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
