Quickstart
===============
To get started clone the LOS Estimator repository and setup a virtual environment for its dependencies

.. code:: bash

    $ git clone git@git.rwth-aachen.de:jrc-combine/los-estimator.git
    $ cd los-estimator
    $ python -m venv .venv
    $ .\.venv\Scripts\activate
    $ pip install -r requirements.txt

Synthetic Example
-----------------
Included in the package is an example script :file:`examples/synthetic_example.py` which demonstrates how to use the LOS Estimator for a synthetic dataset. You can run this script directly after setting up the environment:

.. code:: bash

    $ python examples/synthetic_example.py

This script will generate synthetic data, run the LOS estimation, and visualize the results.
The configuration is pulled from :file:`examples/synthetic_example.toml`.
You can also modify the parameters in the script to experiment with different configurations.

The script generates a synthetic dataset, by simulating a time series of ICU admissions and applying a lognormal LoS distribution to create LoS occupancy data.
It then performs an estimation of the LoS distribution from the generated data and visualizes the results.

Results
-------

In :file:`examples/synthetic_example/results`, you can find the results of the run.
In :file:`run.log` the log of the run is stored.
In :file:`run_configurations.toml`, the final used configuration including all changes to the default configuration is saved.


Metrics
^^^^^^^
In :file:`/metrics`, the calculated metrics can be found in csv files and plots.

Visualization
^^^^^^^^^^^^^
In :file:`/visualization`, various visualizations of the estimation results are provided.

Animation
^^^^^^^^^
In :file:`/animation`, the trained models are visualized for each time step of the rolling window training. In :file:`examples/animation/combined_video.gif`, the animation is combined into a gif.

.. image:: ../img/animation.gif
   :alt: Animation Gif
   :align: center
   :width: 1000px

Model data
^^^^^^^^^^
In :file:`/models_data`, the trained model parameters for each time step of the rolling window training are saved in csv files. Also the python pickle files of the trained models are saved here.
