# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demo script showing how to use the new BoTorch integration components.

This script demonstrates:
1. Using BoTorchGPModel instead of the standard GPModel
2. Using BoTorchMCSampler for configuring sampling strategies
3. Using BoTorchAcquisition for easy switching between acquisition functions
4. Using ContinuousSearch for continuous optimization

Run with:
    python examples/botorch_integration_demo.py
"""

import numpy as np
import torch
from alf_core import Candidate, LabelledCandidates, Optimizer, Surrogate

# Import BoTorch components
from alf_tools.models import BoTorchGPModel
from alf_tools.optimizer.acquisition_functions import BoTorchAcquisition
from alf_tools.optimizer.search import BoTorchMCSampler, ContinuousSearch


def branin_function(x: torch.Tensor) -> torch.Tensor:
    """Branin test function (2D, 3 global minima).

    Global minimum: f(x*) = 0.397887 at
        x* = (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)
    """
    x1, x2 = x[:, 0], x[:, 1]
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)

    term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * torch.cos(x1)
    term3 = s

    return -(term1 + term2 + term3)  # Negative for maximization


def generate_initial_data(n_samples: int = 20) -> LabelledCandidates:
    """Generate initial random samples from Branin function."""
    # Branin bounds: x1 in [-5, 10], x2 in [0, 15]
    # Normalized to [0, 1]
    X = torch.rand(n_samples, 2)
    X[:, 0] = X[:, 0] * 15 - 5  # Scale to [-5, 10]
    X[:, 1] = X[:, 1] * 15  # Scale to [0, 15]

    y = branin_function(X).numpy()

    candidates = [Candidate(data=x.numpy(), modality="tabular") for x in X]

    return LabelledCandidates(candidates=candidates, labels=y)


def demo_botorch_gp_model():
    """Demo 1: Using BoTorchGPModel."""
    print("\n" + "=" * 60)
    print("DEMO 1: BoTorchGPModel")
    print("=" * 60)

    # Generate training data
    train_data = generate_initial_data(n_samples=20)
    print(f"Generated {len(train_data)} training samples")
    print(f"Best value: {train_data.labels.max():.4f}")

    # Create BoTorch GP model
    model = BoTorchGPModel(
        normalize_inputs=False,
        standardize_outputs=True,
        num_iterations=100,
    )

    # Train model
    print("\nTraining BoTorchGPModel...")
    model.train(train_data)

    # Get training metrics
    metrics = model.get_training_summary_metrics()
    print(f"Training metrics: {metrics}")

    # Make predictions
    test_data = generate_initial_data(n_samples=5)
    predictions = model.predict(test_data.candidates)
    print(f"\nPredictions on {len(test_data)} test points:")
    print(f"  Means: {predictions.means}")
    print(f"  Std devs: {np.sqrt(predictions.variances)}")


def demo_mc_sampler():
    """Demo 2: Using BoTorchMCSampler."""
    print("\n" + "=" * 60)
    print("DEMO 2: BoTorchMCSampler")
    print("=" * 60)

    # Create different sampler configurations
    sobol_sampler = BoTorchMCSampler(
        sampler_type="sobol",
        num_samples=512,
        seed=42,
    )
    print(f"Sobol QMC Sampler: {sobol_sampler}")

    iid_sampler = BoTorchMCSampler(
        sampler_type="iid",
        num_samples=1024,
        seed=42,
    )
    print(f"IID Sampler: {iid_sampler}")

    # Get the actual BoTorch sampler objects
    sobol = sobol_sampler.get_sampler()
    iid = iid_sampler.get_sampler()
    print(f"\nCreated samplers: {type(sobol).__name__}, {type(iid).__name__}")


def demo_botorch_acquisition():
    """Demo 3: Switching between acquisition functions."""
    print("\n" + "=" * 60)
    print("DEMO 3: BoTorchAcquisition - Easy Switching")
    print("=" * 60)

    # Branin bounds: x1 in [-5, 10], x2 in [0, 15]
    bounds = [[-5.0, 10.0], [0.0, 15.0]]

    # Create sampler
    sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=512)

    # Create different acquisition functions with the same interface
    acq_qei = BoTorchAcquisition(
        acquisition_type="qEI",
        sampler=sampler,
        bounds=bounds,
        batch_size=3,
    )
    print(f"Created qEI acquisition: {acq_qei.acquisition_type}")

    acq_qucb = BoTorchAcquisition(
        acquisition_type="qUCB",
        sampler=sampler,
        bounds=bounds,
        batch_size=3,
        beta=0.2,  # Exploration parameter
    )
    print(f"Created qUCB acquisition: {acq_qucb.acquisition_type}")

    acq_qnei = BoTorchAcquisition(
        acquisition_type="qNEI",
        sampler=sampler,
        bounds=bounds,
        batch_size=3,
    )
    print(f"Created qNEI acquisition: {acq_qnei.acquisition_type}")

    print("\nEasy to switch: just change 'acquisition_type' parameter!")


def demo_full_optimization():
    """Demo 4: Full optimization loop with all components."""
    print("\n" + "=" * 60)
    print("DEMO 4: Full Optimization with BoTorch Components")
    print("=" * 60)

    # Setup
    train_data = generate_initial_data(n_samples=15)
    print(f"Initial data: {len(train_data)} samples")
    print(f"Initial best: {train_data.labels.max():.4f}")

    # Create components
    model = BoTorchGPModel()
    surrogate = Surrogate(model)

    sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=512)

    # Try with qEI
    print("\n--- Using qEI ---")
    acq_fn = BoTorchAcquisition(
        acquisition_type="qEI",
        sampler=sampler,
        bounds=[[-5.0, 10.0], [0.0, 15.0]],
        batch_size=5,
        num_restarts=10,
    )

    optimizer = Optimizer(
        acquisition_fn=acq_fn,
        search_fn=ContinuousSearch(),
    )

    # Train surrogate
    surrogate.fit(train_data, train_data)
    print("Trained surrogate model")

    # Note: For a full optimization loop, you would need to integrate with
    # TaskState, which requires the full ALF task setup. This demo shows
    # the component creation and configuration.

    print("\nComponents created successfully!")
    print("To run a full optimization, use these components with:")
    print("  - DesignTask.setup() to create TaskState")
    print("  - DesignTask.run() to execute the optimization loop")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("BoTorch Integration Demo")
    print("=" * 60)
    print("\nThis demo shows the new BoTorch integration features:")
    print("1. BoTorchGPModel - Modern GP with better defaults")
    print("2. BoTorchMCSampler - Configurable Monte Carlo sampling")
    print("3. BoTorchAcquisition - Easy switching between acquisition functions")
    print("4. ContinuousSearch - For continuous optimization")

    demo_botorch_gp_model()
    demo_mc_sampler()
    demo_botorch_acquisition()
    demo_full_optimization()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
