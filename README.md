# Boltzmann 1D Solver

This repository is a small kinetic-theory codebase centered on three channel-flow examples:

- `couette`
- `poiseuille`
- `heat-conduction`

The active public cases now use the same steady channel BGK implementation in:

- `Boltzmann_1D_Solver/solver/BGKChannel1D.h`

An additional explicit `1D_x-3D_v` full-Boltzmann comparison solver is available in:

- `Boltzmann_1D_Solver/solver/FullBoltzmann1D3V.h`

The executable is:

- `Boltzmann1DBGKSolver`

## Build

From the repository root:

```bash
cmake -S . -B build
cmake --build build
```

If OpenMP is unavailable, the project still builds and runs in serial mode through:

- `Boltzmann_1D_Solver/utils/OpenMpCompat.h`

## Run

```bash
./build/Boltzmann1DBGKSolver couette
./build/Boltzmann1DBGKSolver poiseuille
./build/Boltzmann1DBGKSolver heat-conduction
```

The default run with no argument is `couette`.

Full-Boltzmann comparison runs:

```bash
./build/Boltzmann1DBGKSolver full-couette
./build/Boltzmann1DBGKSolver full-poiseuille
./build/Boltzmann1DBGKSolver full-heat-conduction
```

Generated output folders:

- `output/bgk_channel/couette`
- `output/bgk_channel/poiseuille`
- `output/bgk_channel/heat_conduction`
- `output/full_boltzmann_1d3v/couette`
- `output/full_boltzmann_1d3v/poiseuille`
- `output/full_boltzmann_1d3v/heat_conduction`

Useful generated plots:

- `channel_profile.png`
- `distribution_contour.png`
- `output/three_case_convergence.png`

##  Solver

It uses:

- one wall-normal spatial coordinate `y`
- an even 3D molecular velocity grid `(v_x, v_y, v_z)` so no exact trapped `v_y = 0` ordinate is used
- complete-accommodation diffuse walls with zero-normal-mass-flux normalization
- BGK relaxation toward a local shifted Maxwellian
- an optional streamwise body force for Poiseuille flow, discretized as a velocity-space derivative
- unequal wall temperatures for heat conduction

The implemented equation is a channel BGK approximation, not the full Boltzmann collision integral:

```text
v_y df/dy = (M[f] - f) / tau + forcing
```

The solver performs directional sweeps through the channel:

- particles with positive `v_y` sweep from the lower wall upward
- particles with negative `v_y` sweep from the upper wall downward
- incoming wall values are built from wall Maxwellians scaled by the outgoing flux, so each wall has zero net normal mass flux
- interior values relax toward a local Maxwellian computed from density, streamwise velocity, and temperature
- Poiseuille forcing uses the explicit velocity-space source term `-a_x df/dv_x`, not only a shifted equilibrium

## Full-Boltzmann Comparison Solver

The full-Boltzmann comparison solver is an explicit `1D_x-3D_v` kinetic model.

It uses:

- one spatial coordinate across the channel
- a 3D velocity grid
- upwind transport in physical space
- diffuse wall boundary inputs
- a discrete hard-sphere collision operator

Axis convention note:

- both the BGK channel solver and the full-Boltzmann comparison solver use `y` as the wall-normal direction and `x` as the flow direction in the default Couette/Poiseuille setups
- because of that, the main Couette/Poiseuille flow profile is written to `bulk_vx.txt`

The collision operator is a direct discrete approximation to gain/loss collisions over velocity pairs and a small set of scattering directions. It is much more expensive than BGK, so the default full-Boltzmann runs use smaller grids and shorter times.

The full-Boltzmann outputs are intended for qualitative comparison with BGK, not for high-resolution benchmark validation.

## Three Public Cases

### Couette

Couette flow is driven by moving walls.

Implementation:

- lower wall has positive streamwise velocity
- upper wall has negative streamwise velocity
- wall temperatures are equal
- no body force is applied

Main output quantity:

- BGK solver: streamwise velocity `bulk_vx.txt`
- full-Boltzmann comparison solver: flow-direction velocity `bulk_vx.txt`

### Poiseuille

Poiseuille flow is driven by a streamwise body force.

Implementation:

- both walls are stationary
- wall temperatures are equal
- `body_force_x` enters through the discrete force term `-a_x df/dv_x`

Main output quantity:

- BGK solver: streamwise velocity `bulk_vx.txt`
- full-Boltzmann comparison solver: flow-direction velocity `bulk_vx.txt`

### Heat Conduction

Heat conduction is driven by different wall temperatures.

Implementation:

- both walls are stationary
- lower and upper wall temperatures differ
- no body force is applied

Main output quantity:

- temperature `temperature.txt`

## Channel Output Files

Each case writes files such as:

- `y_cells.txt`: wall-normal grid
- `density.txt`: density profile
- `bulk_vx.txt`: streamwise velocity profile for both BGK and full-Boltzmann runs
- `bulk_vy.txt`: transverse velocity profile; this is typically near zero in the default channel cases
- `temperature.txt`: temperature profile
- `velocity_axis.txt`: velocity-grid axis used for contours
- `distribution_left.txt`: lower-wall velocity-space slice
- `distribution_center.txt`: center-channel velocity-space slice
- `distribution_right.txt`: upper-wall velocity-space slice
- `convergence_history.txt`: steady-solver residual by iteration

## Plotting 

```bash
python3 scripts/plot_bgk_channel.py output/bgk_channel/couette
python3 scripts/plot_distribution_contour.py output/bgk_channel/couette
python3 scripts/plot_three_case_convergence.py
python3 scripts/plot_bgk_full_comparison.py couette output/bgk_channel/couette output/full_boltzmann_1d3v/couette
```
## Current Limitations

This is a compact educational/research code, not a validated production solver.

Current limitations:

- the main channel solver uses a BGK approximation, not the full collision operator `Q(f,f)`
- the full-Boltzmann solver is coarse and explicit, so it is useful for comparison but not a validated production solver
- the channel discretization is simplified
- results are qualitative and should be validated before making physical claims
- velocity-space contour plots are diagnostic visualizations, not exact reproductions of published benchmark figures
