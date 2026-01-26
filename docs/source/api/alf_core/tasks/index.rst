Tasks
=====

The task classes define different experiment types in ALF. The ``DesignTask`` runs a multi-round
active learning optimization loop, the ``SupervisedTask`` trains and evaluates models on fixed data
splits, and the ``ZeroShotTask`` evaluates pre-trained models without additional training. All tasks
inherit from ``BaseTask`` which provides the common interface.

.. toctree::
   :maxdepth: 2

   Base Task <base_task/index>
   Design Task <design_task/index>
   Supervised Task <supervised_task/index>
   Zero-Shot Task <zeroshot_task/index>
