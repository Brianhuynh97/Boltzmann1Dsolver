# Boltzmann 1D Solver

This repository contains a simple 1D BGK flow solver built around one steady normal-flow setup.

The active solver implementation is:

- `Boltzmann_1D_Solver/solver/BGKChannel1D.h`

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
./build/Boltzmann1DBGKSolver
```

Optional output directory:

```bash
./build/Boltzmann1DBGKSolver output/bgk_channel
```

Generated output folder:

- `output/bgk_channel`

Useful generated plots:

- `channel_profile.png`
- `distribution_f.png`: plot of `f(x,v)`

## Solver

It uses:

- one spatial coordinate `x`
- one velocity coordinate `v`
- complete-accommodation diffuse walls with zero-normal-mass-flux normalization
- BGK relaxation toward a local Maxwellian
- different left and right wall temperatures to drive a simple flow profile

The implemented equation is a channel BGK approximation, not the full Boltzmann collision integral:

```text
v df/dx = (M[f] - f) / tau
```

The solver performs directional sweeps through the 1D domain:

- a finite-volume discrete velocity method for the transport term
- a first-order Godunov upwind flux in physical space
- explicit Euler time integration for the transient marching step
- particles with positive `v` sweep from the left wall to the right
- particles with negative `v` sweep from the right wall to the left
- incoming wall values are built from wall Maxwellians scaled by the outgoing flux, so each wall has zero net normal mass flux
- interior values relax toward a local Maxwellian computed from density, bulk velocity, and temperature

## Full-Boltzmann Comparison Solver

The full-Boltzmann comparison solver is an explicit `1D_x-3D_v` kinetic model.

It uses:

- one spatial coordinate across the domain
- a 3D velocity grid
- upwind transport in physical space
- diffuse wall boundary inputs
- a discrete hard-sphere collision operator

The collision operator is a direct discrete approximation to gain/loss collisions over velocity pairs and a small set of scattering directions. It is much more expensive than BGK, so the default full-Boltzmann runs use smaller grids and shorter times.

The full-Boltzmann outputs are intended for qualitative comparison with BGK, not for high-resolution benchmark validation.

## Output Files

The solver writes files such as:

- `x_cells.txt`: spatial grid
- `y_cells.txt`: legacy alias for the same grid
- `density.txt`: density profile
- `bulk_vx.txt`: bulk velocity profile
- `bulk_vy.txt`: legacy zero transverse-velocity output
- `temperature.txt`: temperature profile
- `velocity_axis.txt`: velocity-grid axis
- `distribution_left.txt`: left-wall velocity slice
- `distribution_center.txt`: center velocity slice
- `distribution_right.txt`: right-wall velocity slice
- `distribution_y_v.txt`: full `f(x,v)` array
- `convergence_history.txt`: steady-solver residual by iteration

## Plotting

```bash
python3 scripts/plot_bgk_channel.py output/bgk_channel
python3 scripts/plot_distribution_x_v.py output/bgk_channel
```

## Current Limitations

- The main solver uses a BGK approximation, not the full collision operator `Q(f,f)`
- The channel discretization is simplified
