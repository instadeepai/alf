BoTorch Acquisition Wrapper
===========================

``BoTorchAcquisition`` is a generic wrapper that exposes BoTorch's analytic and
Monte Carlo acquisition functions (qEI, qLogEI, qNEI, qUCB, and their analytic
counterparts) through the ALF ``AcquisitionFunction`` interface, so strategies can
be swapped via a single ``acquisition_type`` argument.

It operates in two modes: when the search function provides candidates, it scores
the discrete pool; when the candidate list is empty (e.g. with ``ContinuousSearch``),
it optimises the acquisition function directly in continuous space via BoTorch's
``optimize_acqf``, which requires the ``bounds`` parameter. Monte Carlo variants are
configured with a ``BoTorchMCSampler``; scipy optimiser behaviour is controlled via
``BoTorchAcquisitionOptConfig``.

.. automodule:: alf_tools.optimizer.acquisition_functions.botorch_acquisition
   :members:
   :show-inheritance:
   :undoc-members:
