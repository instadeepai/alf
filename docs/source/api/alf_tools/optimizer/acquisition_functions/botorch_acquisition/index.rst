BoTorch Acquisition Wrapper
===========================

:class:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition.BoTorchAcquisition` is a generic wrapper that exposes BoTorch's analytic and
Monte Carlo acquisition functions (qEI, qLogEI, qNEI, qUCB, and their analytic
counterparts) through the ALF :class:`~alf_core.optimizer.acquisition_function.AcquisitionFunction` interface, so strategies can
be swapped via a single ``acquisition_type`` argument.

It operates in two modes: when the search function provides candidates, it scores
the discrete pool; when the candidate list is empty (e.g. with :class:`~alf_tools.optimizer.search.botorch_continuous_search.BotorchContinuousSearch`),
it optimises the acquisition function directly in continuous space via BoTorch's
``optimize_acqf``, which requires the ``bounds`` parameter. Monte Carlo variants are
configured with a :class:`~alf_tools.optimizer.acquisition_functions.botorch_samplers.BoTorchMCSampler`; scipy optimiser behaviour is controlled via
:class:`~alf_tools.optimizer.acquisition_functions.botorch_acquisition.BoTorchAcquisitionOptConfig`.

Batch acquisition (``batch_size`` > 1) requires a surrogate with a joint posterior — a
native BoTorch model or an ALF model exposing a trained ``botorch_model``. Marginal-only
``predict()``-based surrogates can only supply a diagonal posterior, so requesting
``batch_size`` > 1 with one raises a ``ValueError``; score them one point at a time, or use
ALF's native :class:`~alf_tools.optimizer.acquisition_functions.core_set.CoreSet`/:class:`~alf_tools.optimizer.acquisition_functions.thompson_sampling.ThompsonSampling` acquisitions for batch selection.

.. automodule:: alf_tools.optimizer.acquisition_functions.botorch_acquisition
   :members:
   :show-inheritance:
   :undoc-members:
