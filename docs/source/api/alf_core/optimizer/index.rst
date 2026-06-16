Optimizer
=========

The ``Optimizer`` orchestrates the active learning loop by coordinating acquisition functions and search
strategies. It selects the most promising candidates for evaluation through acquisition scoring and
manages the candidate pool exploration. The optimizer components include the main ``Optimizer`` class,
acquisition functions for scoring candidates, and search strategies for defining candidate pools.

.. toctree::
   :maxdepth: 1

   Optimizer <optimizer/index>
   Acquisition Function <acquisition_function/index>
   Search <search/index>
