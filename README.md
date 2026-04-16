# Boltzmann 1D Solver

This repository is a small kinetic-theory codebase with several solver paths that grew over time.

The code currently contains:

- a legacy `1D_x-1D_v` BGK solver
- an experimental `1D_x-3D_v` discrete-velocity full-collision solver
- an experimental steady channel BGK solver used for Couette-style wall-driven tests

The project builds as a single executable:

- `Boltzmann1DBGKSolver`

## Repo layout

- `Boltzmann_1D_Solver/solver`
  solver headers, problem definitions, and `main.cpp`
- `Boltzmann_1D_Solver/interface`
  file-output helpers for macro/state data
- `Boltzmann_1D_Solver/utils`
  math helpers, RK methods, OpenMP compatibility layer
- `scripts`
  Python plotting scripts for generated outputs
- `output`
  generated result files and plots

## Solver paths

### 1. Legacy BGK solver

Main files:

- `Boltzmann_1D_Solver/solver/BoltzmannEq1DSolvers.h`
- `Boltzmann_1D_Solver/solver/BoltzmannEq1DProblems.h`

This path solves a `1D_x-1D_v` BGK-type kinetic problem and writes:

- density
- velocity
- temperature
- heat flux
- optional full kinetic state

Legacy example setups include:

- Sod / Riemann-type problems
- density jump problems
- emitting wall
- evaporating wall

Implementation details:

- state representation:
  - `Destribution1dState<T>` stores two velocity-dependent fields, `h` and `g`
  - each field is stored as `std::vector<std::vector<T>>` on a uniform `(x, xi)` grid
- problem container:
  - `Bgk1dProblemData<T>` stores `N_x`, `N_xi`, domain bounds, `dt`, `t_end`, initial distribution `f_1`, cell temperatures `Ts`, and rarefaction parameter `delta`
- macroscopic recovery:
  - `bgk1dMacroparameters(...)` computes `n`, `u_1`, `T`, and `q_1`
  - velocity moments are evaluated with numerical quadrature through `ApproxIntFunc`
- time stepping:
  - `bgk1dMethod<...>(...)` advances the solution with the RK methods defined in `utils/rkMethods.h`
  - the current code uses first-order or Simpson-style velocity integration depending on the chosen case
- transport/collision structure:
  - transport is discretized on a uniform 1D spatial mesh
  - the collision term is BGK relaxation toward a Maxwellian-type equilibrium
  - this path does not evaluate the full Boltzmann collision integral `Q(f,f)`
- file output:
  - `interface/MacroparametersFileInterface.h` writes macro fields and optional full state histories
  - typical files are `n.txt`, `u_1.txt`, `T.txt`, `q_1.txt`, `t_levels.txt`, and optionally `h.txt`, `g.txt`

### 2. Experimental full-Boltzmann path

Main file:

- `Boltzmann_1D_Solver/solver/FullBoltzmann1D3V.h`

This path uses:

- one physical dimension
- three velocity dimensions
- a discrete hard-sphere-like collision update

It is useful for experimentation and visualization, but it is not a validated production Boltzmann solver.

Implemented example modes:

- `poiseuille`
- `heat-conduction`
- `uniform-equilibrium`

Outputs go under:

- `output/full_boltzmann_1d3v/...`

Implementation details:

- velocity grid:
  - `VelocityGrid3D<T>` constructs a uniform Cartesian velocity grid with `n_v^3` nodes
  - each velocity node stores `(v_x, v_y, v_z)` and a uniform cell volume
- state representation:
  - `DistributionState1D3V<T>` stores `f(x, v_x, v_y, v_z)` in a flattened 1D array
  - indexing is by spatial cell and flattened velocity index
- macro recovery:
  - `computeMacroState(...)` integrates the distribution over velocity space to recover:
    - density
    - bulk velocity components
    - temperature
- collision model:
  - `HardSphereCollisionModel<T>` defines a coarse hard-sphere-like binary collision update
  - the current implementation uses a small discrete set of scattering directions and fixed angular weights
  - post-collision values are remapped back to the grid with trilinear-style deposition
- transport and forcing:
  - spatial transport is 1D upwind transport in `x`
  - diffuse wall boundary data can be imposed on the left and right boundaries
  - a body-force term is available in the `y` direction for Poiseuille-style runs
- time stepping:
  - `fullBoltzmannMethod<...>(...)` advances the state with the RK utilities from `rkMethods.h`
  - negative values are clipped after each step
- output:
  - `MacroOutput1D3V<T>` writes `density.txt`, `bulk_vx.txt`, `bulk_vy.txt`, `temperature.txt`, and `t_levels.txt`
  - distribution slices are also written for contour plotting:
    - `velocity_axis.txt`
    - `distribution_left.txt`
    - `distribution_center.txt`
    - `distribution_right.txt`

Current limitations:

- collision quadrature is coarse
- conservation and benchmark validation are incomplete
- this path is useful for experiments and visualization, not for claiming a validated full Boltzmann solver

### 3. Experimental steady channel BGK Couette path

Main file:

- `Boltzmann_1D_Solver/solver/LinearizedBGKChannel1D.h`

This path was added to better match wall-driven channel-flow figures. It is a steady wall-normal sweep around a global Maxwellian and is currently used by the `couette` mode.

Outputs go under:

- `output/linearized_bgk_channel/couette`

Important limitation:

- this is still an approximation and does not yet reproduce published Couette figures exactly

Implementation details:

- geometry:
  - this path is a wall-normal channel solve in a single spatial coordinate `y`
  - it is aimed at steady Couette-style wall-driven flow
- state representation:
  - `ChannelState<T>` stores one distribution value per `(y, v_x, v_y, v_z)` node
  - `CouetteProblemData<T>` stores wall speeds, wall temperature, relaxation time `tau`, iteration count, tolerance, and the velocity grid
- equilibrium and perturbation model:
  - the current implementation is organized around a global Maxwellian plus a linearized Couette perturbation
  - helper routines build:
    - equilibrium distributions
    - linearized wall perturbations
    - complete-accommodation-style incoming wall values
- steady solve:
  - `solveCouetteSteadyState(...)` performs a sweep-based steady update rather than explicit time marching
  - positive and negative wall-normal velocities are handled in opposite sweep directions
  - boundary values are applied from the wall model and interior values are propagated through the channel
- diagnostics and output:
  - `computeMacroState(...)` recovers density, streamwise velocity, and temperature profiles
  - `writeCouetteOutput(...)` writes:
    - `y_cells.txt`
    - `density.txt`
    - `bulk_vx.txt`
    - `temperature.txt`
    - distribution slices for lower wall, center, and upper wall

Current limitations:

- the solver is still a simplified linearized BGK channel approximation
- it is not yet a literal reproduction of the target paper’s discretization
- the distribution contours are closer to the paper than before, but still not exact

## Build

From the repository root:

```bash
cmake -S . -B build
cmake --build build
```

If OpenMP is unavailable, the project still builds and runs in serial mode through:

- `Boltzmann_1D_Solver/utils/OpenMpCompat.h`

## Run

### Couette channel mode

```bash
./build/Boltzmann1DBGKSolver couette
```

This uses the steady channel BGK path and writes:

- `output/linearized_bgk_channel/couette/channel_profile.png`
- `output/linearized_bgk_channel/couette/distribution_contour.png`

### Full-Boltzmann Poiseuille mode

```bash
./build/Boltzmann1DBGKSolver poiseuille
```

Outputs:

- `output/full_boltzmann_1d3v/poiseuille`

### Full-Boltzmann heat-conduction mode

```bash
./build/Boltzmann1DBGKSolver heat-conduction
```

Outputs:

- `output/full_boltzmann_1d3v/heat_conduction`

### Full-Boltzmann uniform-equilibrium mode

```bash
./build/Boltzmann1DBGKSolver uniform-equilibrium
```

Outputs:

- `output/full_boltzmann_1d3v/uniform_equilibrium`

### Default run

If you run the executable with no arguments, it currently defaults to:

- `couette`

## Plotting scripts

Available scripts:

- `scripts/plot_full_boltzmann_shock_tube.py`
  summary macro plots for full-Boltzmann output folders
- `scripts/plot_macro_heatmap.py`
  space-time heat maps for macro fields
- `scripts/plot_distribution_contour.py`
  velocity-space contour plots from saved distribution slices
- `scripts/plot_linearized_bgk_channel.py`
  channel profile plots for the steady Couette path

Most solver modes already call the relevant plotting scripts automatically from `main.cpp`.

## Supporting implementation

Utility code used across the repo:

- `Boltzmann_1D_Solver/utils/rkMethods.h`
  explicit RK steppers such as Euler, Heun, and SSPRK-style methods
- `Boltzmann_1D_Solver/utils/Integration.h`
  numerical integration helpers used for velocity moments
- `Boltzmann_1D_Solver/utils/VectorOperations.h`
  vector arithmetic helpers and printing utilities used by the legacy solver
- `Boltzmann_1D_Solver/utils/OpenMpCompat.h`
  compatibility wrapper so the project can compile without a native `omp.h`
- `Boltzmann_1D_Solver/interface/MacroparametersFileInterface.h`
  output interface for the legacy BGK solver
- `Boltzmann_1D_Solver/interface/HyperbolicFileInterface.h`
  input helper for the older hyperbolic/advection side of the codebase

The executable entry point is:

- `Boltzmann_1D_Solver/solver/main.cpp`

`main.cpp` currently:

- configures OpenMP thread usage
- selects a scenario from the command line
- runs one solver path
- invokes Python plotting scripts for the generated output folders

## Current status

What is reliable:

- cross-platform CMake build
- legacy BGK solver path
- text-file output and plotting workflow
- basic Couette / Poiseuille / heat-conduction experiments

What is still experimental:

- the `1D_x-3D_v` full-collision solver
- the steady channel BGK Couette reproduction effort
- agreement with published reference figures

## Scope statement

The safest description of this repo is:

`A kinetic solver project with a legacy 1D BGK code, an experimental discrete-velocity full-Boltzmann path, and an experimental steady BGK channel solver for wall-driven tests.`
