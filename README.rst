LoS Estimator
=============


The LoS Estimator is a tool for estimating Length of Stay (LoS) distributions in healthcare settings based on admission and occupancy data. It utilizes statistical fitting techniques to derive LoS distributions that can help in resource planning and management.

It is based on the work presented in the following paper:
XXX

The basic assumption is, that the individual length of stay for a patient follows a certain probability distribution (e.g., lognormal, gamma, etc.). By convolving the admission time series with the discharge probabilities derived from the LoS distribution, we can model the expected occupancy over time. The estimator fits the parameters of the chosen LoS distribution to minimize the difference between the modeled and observed occupancy data.
Mathematically, the occupancy can be calculated from the admissions by performing a convolution with the LoS distribution.
Thus the estimation of LoS distributions can be formulated as a deconvolution problem.

The exact distribution function, that best describes the length of stay is not known a priori and has to be estimated from the data.
Also the length of stay distribution may change over time due to various factors (e.g., changes in treatment protocols, patient demographics, etc.).
Thus the LoS Estimator employs a rolling window approach, where the estimation is performed on overlapping time windows of the data.

An example for a such a training process is shown in the animation below, where the estimated LoS distribution is updated for each time step of the rolling window training.

.. image:: img/animation.gif
   :alt: Animation Gif
   :align: center
   :width: 1000px


