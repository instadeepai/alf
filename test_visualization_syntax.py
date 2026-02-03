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

"""Test script to validate visualization code syntax"""

import sys

code = """
# Create a dense grid for plotting
x1 = np.linspace(bounds[0, 0].item(), bounds[1, 0].item(), 100)
x2 = np.linspace(bounds[0, 1].item(), bounds[1, 1].item(), 100)
X1, X2 = np.meshgrid(x1, x2)
grid_points = np.column_stack([X1.ravel(), X2.ravel()])

# Evaluate true function on grid
true_values = dataset.test_function(torch.tensor(grid_points, dtype=torch.float32))
true_values = true_values.numpy().reshape(X1.shape)

# Get surrogate predictions
gp_model.model.eval()
with torch.no_grad():
    # Prepare grid for GP
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    # Get predictions
    predictions = gp_model.model(grid_tensor)
    mean_pred = predictions.mean.numpy().reshape(X1.shape)
    var_pred = predictions.variance.numpy().reshape(X1.shape)
    std_pred = np.sqrt(var_pred)

# Get training data
train_X = state.dataset.train_dataset.designs
train_y = state.dataset.train_dataset.labels

# Load all acquired points across rounds
acquired_X = []
acquired_y = []
initial_train_size = dataset_config.train_ratio * dataset_config.n_initial_samples  # Initial training set size

for round_num in range(1, 21):  # 20 acquisition rounds
    acq_file = save_path / f"acq_round_{round_num}.csv"
    if acq_file.exists():
        acq_df = pd.read_csv(acq_file)
        # The CSV has features and labels
        for col in acq_df.columns:
            if 'feature' in col.lower():
                feat_data = acq_df[col].values
                if acquired_X == []:
                    acquired_X = [feat_data]
                else:
                    # Stack horizontally to build feature matrix
                    pass
        acquired_y.extend(acq_df['label'].values)

# Convert lists to arrays
if len(acquired_X) > 0:
    # Reconstruct acquired_X from train_X (last N points where N = len(acquired_y))
    n_acquired = len(acquired_y)
    acquired_X = train_X[-n_acquired:]
    acquired_X_np = acquired_X
    acquired_y_np = np.array(acquired_y)
else:
    acquired_X_np = np.array([])
    acquired_y_np = np.array([])

# Initial training data (first points)
initial_size = len(train_X) - len(acquired_y_np) if len(acquired_y_np) > 0 else len(train_X)
initial_X = train_X[:initial_size]
initial_y = train_y[:initial_size]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. True Function
ax1 = axes[0, 0]
contour1 = ax1.contourf(X1, X2, true_values, levels=30, cmap='viridis')
ax1.set_title('True Branin Function', fontsize=14, fontweight='bold')
ax1.set_xlabel('x1', fontsize=12)
ax1.set_ylabel('x2', fontsize=12)
plt.colorbar(contour1, ax=ax1, label='Function Value')
# Mark the known optima
optima_x1 = [-np.pi, np.pi, 9.42478]
optima_x2 = [12.275, 2.275, 2.475]
ax1.scatter(optima_x1, optima_x2, c='red', s=200, marker='*',
           edgecolors='white', linewidths=2, label='True Optima', zorder=5)
ax1.legend(fontsize=10)

# 2. Surrogate Model (GP Mean)
ax2 = axes[0, 1]
contour2 = ax2.contourf(X1, X2, mean_pred, levels=30, cmap='viridis')
ax2.set_title('Learned Surrogate Model (GP Mean)', fontsize=14, fontweight='bold')
ax2.set_xlabel('x1', fontsize=12)
ax2.set_ylabel('x2', fontsize=12)
plt.colorbar(contour2, ax=ax2, label='Predicted Value')
# Plot training points
if len(initial_X) > 0:
    ax2.scatter(initial_X[:, 0], initial_X[:, 1], c='white', s=30,
               marker='o', edgecolors='black', linewidths=1,
               label=f'Initial Training ({len(initial_X)})', alpha=0.7, zorder=4)
if len(acquired_X_np) > 0:
    ax2.scatter(acquired_X_np[:, 0], acquired_X_np[:, 1], c='red', s=80,
               marker='X', edgecolors='white', linewidths=2,
               label=f'Acquired Points ({len(acquired_X_np)})', zorder=5)
ax2.legend(fontsize=10)

# 3. Surrogate Model Uncertainty (GP Std Dev)
ax3 = axes[1, 0]
contour3 = ax3.contourf(X1, X2, std_pred, levels=30, cmap='plasma')
ax3.set_title('Surrogate Model Uncertainty (GP Std Dev)', fontsize=14, fontweight='bold')
ax3.set_xlabel('x1', fontsize=12)
ax3.set_ylabel('x2', fontsize=12)
plt.colorbar(contour3, ax=ax3, label='Uncertainty (σ)')
# Plot training points
if len(initial_X) > 0:
    ax3.scatter(initial_X[:, 0], initial_X[:, 1], c='white', s=30,
               marker='o', edgecolors='black', linewidths=1,
               label=f'Initial Training ({len(initial_X)})', alpha=0.7, zorder=4)
if len(acquired_X_np) > 0:
    ax3.scatter(acquired_X_np[:, 0], acquired_X_np[:, 1], c='yellow', s=80,
               marker='X', edgecolors='black', linewidths=2,
               label=f'Acquired Points ({len(acquired_X_np)})', zorder=5)
ax3.legend(fontsize=10)

# 4. All Training Data with Values
ax4 = axes[1, 1]
contour4 = ax4.contourf(X1, X2, true_values, levels=30, cmap='viridis', alpha=0.3)
ax4.set_title('Training Data Evolution', fontsize=14, fontweight='bold')
ax4.set_xlabel('x1', fontsize=12)
ax4.set_ylabel('x2', fontsize=12)
# Plot initial training data
if len(initial_X) > 0:
    scatter1 = ax4.scatter(initial_X[:, 0], initial_X[:, 1], c=initial_y, s=50,
                          cmap='coolwarm', marker='o', edgecolors='black', linewidths=1,
                          label=f'Initial Training ({len(initial_X)})', alpha=0.8, zorder=3,
                          vmin=train_y.min(), vmax=train_y.max())
# Plot acquired points with their actual values
if len(acquired_X_np) > 0:
    scatter2 = ax4.scatter(acquired_X_np[:, 0], acquired_X_np[:, 1], c=acquired_y_np, s=100,
                          cmap='coolwarm', marker='X', edgecolors='white', linewidths=2,
                          label=f'Acquired Points ({len(acquired_X_np)})', alpha=1.0, zorder=4,
                          vmin=train_y.min(), vmax=train_y.max())
# Add colorbar for the scatter points
cbar = plt.colorbar(scatter2 if len(acquired_X_np) > 0 else scatter1, ax=ax4, label='Observed Value')
# Mark best point found
best_idx = np.argmax(train_y)
ax4.scatter(train_X[best_idx, 0], train_X[best_idx, 1], c='lime', s=300,
           marker='*', edgecolors='black', linewidths=2,
           label=f'Best Found ({train_y[best_idx]:.3f})', zorder=5)
ax4.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.show()

print(f"\\n{'='*60}")
print(f"Optimization Summary")
print(f"{'='*60}")
print(f"True optimum:         {dataset.true_optimum:.6f}")
print(f"Best value found:     {train_y.max():.6f}")
print(f"Final regret:         {dataset.true_optimum - train_y.max():.6f}")
print(f"Initial best:         {initial_y.max():.6f}")
print(f"Improvement:          {train_y.max() - initial_y.max():.6f}")
print(f"Total evaluations:    {len(train_X)}")
print(f"Initial training:     {len(initial_X)}")
print(f"Acquired points:      {len(acquired_X_np)}")
print(f"{'='*60}")
"""

# Try to compile the code
try:
    compile(code, "<string>", "exec")
    print("✓ Syntax check passed! The visualization code is syntactically correct.")
    sys.exit(0)
except SyntaxError as e:
    print("✗ Syntax error found:")
    print(f"  Line {e.lineno}: {e.msg}")
    print(f"  Text: {e.text}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
