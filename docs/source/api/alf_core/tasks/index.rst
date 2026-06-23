Tasks
=====

The task classes define different experiment types in ALF. The :class:`~alf_core.tasks.design_task.DesignTask` runs a multi-round
active learning optimization loop, the :class:`~alf_core.tasks.supervised_task.SupervisedTask` trains and evaluates models on fixed data
splits, and the :class:`~alf_core.tasks.zeroshot_task.ZeroShotTask` evaluates pre-trained models without additional training. All tasks
inherit from :class:`~alf_core.tasks.base_task.BaseTask` which provides the common interface.

.. toctree::
   :maxdepth: 1

   Base Task <base_task/index>
   Design Task <design_task/index>
   Supervised Task <supervised_task/index>
   Zero-Shot Task <zeroshot_task/index>
