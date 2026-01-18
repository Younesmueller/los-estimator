Estimation Run
======================
This example demonstrates how to use the LoS Estimator with real-world data.
We will use a sample dataset containing ICU admissions and occupancy data.

A prepared script is provided in :file:`run_analysis.py` that performs the estimation using the provided data.


Data Preparation
----------------
The package provides a copy of preprocessed ICU data from Germany that can be used without further preprocessing.

Manual preprocessing (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If updated data is desired, the data can be preprocessed using the provided preprocessing script:

    
The RKI provides COVID-19 related data including ICU admissions and occupancy for Germany.
You can download the data from the `RKI Intensivregister <https://github.com/robert-koch-institut/Intensivkapazitaeten_und_COVID-19-Intensivbettenbelegung_in_Deutschland/blob/main/Intensivregister_Bundeslaender_Kapazitaeten.csv>`_.
Place the downloaded CSV file in the :file:`los_estimator/data/preprocessing` directory.

Then call the script :file:`los_estimator/data/preprocessing/__init__.py` to preprocess the data and generate the required input files for the LoS Estimator.

.. code:: bash

    $ python los_estimator/data/preprocessing/__init__.py

The :file:`default_config.toml` is configured to access the preprocessed data files directly.


Performing the Estimation
-------------------------
To perform the estimation using the real data, run the provided script :file:`run_analysis.py`:

.. code:: bash

    $ python run_analysis.py

By default the flag `less_windows` is set to true, reducing the number of rolling windows to 3 for faster execution during testing.
For a full run set the flag to false in the script.

Performing the Estimation via CLI
----------------------------------
Alternatively, run the estimator directly from the command line:

.. code:: bash

    $ python -m los_estimator --config_file los_estimator/default_config.toml

See :doc:`../usage/cli_usage` for full CLI documentation including argument reference and configuration options.

Results
-------
The results of the estimation will be saved in the :file:`results` directory.
Further information on the results structure can be found in the :doc:`/usage/output_format` guide.


Reloading Results for Re-Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To reload and re-visualize results without re-fitting:

.. code:: python

    from los_estimator.estimation_run import LosEstimationRun

    run = LosEstimationRun.load_run("results/<run_folder>")
    run.visualize_results()  # Re-create plots
    run.animate_results()     # Re-create animations

Modify visualization settings in config before reloading to test different styles.
