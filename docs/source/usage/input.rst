Input Format
============

Configuration
-------------
The configuration for the LOS Estimator is provided in a TOML file.
The default configuration can be found at :file:`los_estimator/default_config.toml`.
In  you can find the documentation of all configuration options.

Data files
----------

LoS data
^^^^^^^^
The los estimator expects a csv file containing the input data with the following columns:

* date: Date in YYYY-MM-DD format
* admissions: Number of new ICU admissions on that date
* occupancy: Number of ICU beds occupied on that date

Init parameters
^^^^^^^^^^^^^^^

Optionally, initial paramter values can be provided in a csv file with the following columns:

* distro: Name of the LoS distribution (e.g., lognormal, gamma, etc.)
* params: List of parameters in brackets, seperated by spaces

Sample LoS distribution
^^^^^^^^^^^^^^^^^^^^^^^

Optionally, a sample LoS distribution file can be provided. A sample ditribution is provided with the package. In csv format with the following columns:

* day: Length of stay in days
* probability: Discharge probability for that day



