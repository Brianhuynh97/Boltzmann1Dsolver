#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ExplicitEuler.h"

namespace bgk_channel
{

    template <std::floating_point T>
    inline constexpr T kPi = static_cast<T>(3.14159265358979323846);

    enum class VelocityBoundaryCondition
    {
        DirichletZero,
        NeumannZeroGradient
    };

    template <std::floating_point T>
    struct VelocityGrid1D
    {
        int n_v{};
        T v_max{};
        T dv{};
        std::vector<T> axis;

        VelocityGrid1D() = default;

        VelocityGrid1D(int n_v_, T v_max_)
            : n_v(n_v_), v_max(v_max_), dv((T(2) * v_max_) / T(n_v_))
        {
            assert(n_v > 1);
            axis.resize(n_v);
            for (int i = 0; i < n_v; ++i)
            {
                axis[i] = -v_max + (T(i) + T(0.5)) * dv;
            }
        }

        [[nodiscard]] int size() const
        {
            return n_v;
        }

        [[nodiscard]] T cellVolume() const
        {
            return dv;
        }
    };

    template <std::floating_point T>
    struct ChannelState
    {
        int n_x{};
        int n_v{};
        std::vector<T> values;

        ChannelState() = default;

        ChannelState(int n_x_, int n_v_)
            : n_x(n_x_), n_v(n_v_), values(static_cast<std::size_t>(n_x_) * static_cast<std::size_t>(n_v_), T{})
        {
        }

        [[nodiscard]] T &at(int x_i, int v_i)
        {
            return values[static_cast<std::size_t>(x_i) * static_cast<std::size_t>(n_v) + static_cast<std::size_t>(v_i)];
        }

        [[nodiscard]] const T &at(int x_i, int v_i) const
        {
            return values[static_cast<std::size_t>(x_i) * static_cast<std::size_t>(n_v) + static_cast<std::size_t>(v_i)];
        }
    };

    template <std::floating_point T>
    struct MacroState
    {
        std::vector<T> density;
        std::vector<T> bulk_v;
        std::vector<T> temperature;
    };

    template <std::floating_point T>
    struct ChannelProblemData
    {
        int n_x{};
        T x_min{};
        T x_max{};
        VelocityGrid1D<T> velocity_grid;
        T density{T(1)};
        T particle_mass{T(1)};
        T body_force{T(0)};
        T left_wall_temperature{T(1)};
        T right_wall_temperature{T(1)};
        T boltzmann_over_mass{T(0.5)};
        VelocityBoundaryCondition vmin_boundary{VelocityBoundaryCondition::DirichletZero};
        VelocityBoundaryCondition vmax_boundary{VelocityBoundaryCondition::DirichletZero};
        T collision_frequency{T(10)};
        T cfl{T(0.5)};
        T final_time{T(-1)};
        int snapshot_interval{10};
        int max_iterations{250};
        T tolerance{T(1e-8)};
    };

    template <std::floating_point T>
    inline void applyVelocityBoundary(
        ChannelState<T> &state,
        VelocityBoundaryCondition vmin_boundary,
        VelocityBoundaryCondition vmax_boundary)
    {
        if (state.n_v < 2)
        {
            return;
        }

        for (int x_i = 0; x_i < state.n_x; ++x_i)
        {
            if (vmin_boundary == VelocityBoundaryCondition::DirichletZero)
            {
                state.at(x_i, 0) = T(0);
            }
            else
            {
                state.at(x_i, 0) = state.at(x_i, 1);
            }

            if (vmax_boundary == VelocityBoundaryCondition::DirichletZero)
            {
                state.at(x_i, state.n_v - 1) = T(0);
            }
            else
            {
                state.at(x_i, state.n_v - 1) = state.at(x_i, state.n_v - 2);
            }
        }
    }

    template <std::floating_point T>
    inline std::vector<T> maxwellianCell(
        const VelocityGrid1D<T> &grid,
        T density,
        T particle_mass,
        T bulk_velocity,
        T temperature,
        T boltzmann_over_mass)
    {
        std::vector<T> values(grid.size());
        const T safe_temperature = std::max(temperature, T(1e-12));
        const T thermal_variance = T(2) * boltzmann_over_mass * safe_temperature;
        const T prefactor = (density / std::max(particle_mass, T(1e-12))) /
                            std::sqrt(T(2) * kPi<T> * boltzmann_over_mass * safe_temperature);

        for (int v_i = 0; v_i < grid.size(); ++v_i)
        {
            const T c = grid.axis[static_cast<std::size_t>(v_i)] - bulk_velocity;
            values[static_cast<std::size_t>(v_i)] = prefactor * std::exp(-(c * c) / thermal_variance);
        }

        return values;
    }

    template <std::floating_point T>
    inline T completeAccommodationLeftWallValue(
        const VelocityGrid1D<T> &grid,
        const std::vector<T> &wall_distribution,
        const ChannelState<T> &state,
        int interior_x,
        int v_i)
    {
        const T c = grid.axis[static_cast<std::size_t>(v_i)];
        if (c > T(0))
        {
            T outgoing_flux{};
            T incoming_wall_flux{};
            for (int k = 0; k < grid.size(); ++k)
            {
                const T wall_c = grid.axis[static_cast<std::size_t>(k)];
                if (wall_c < T(0))
                {
                    outgoing_flux += -wall_c * state.at(interior_x, k);
                }
                else if (wall_c > T(0))
                {
                    incoming_wall_flux += wall_c * wall_distribution[static_cast<std::size_t>(k)];
                }
            }

            const T scale = outgoing_flux / std::max(incoming_wall_flux, T(1e-30));
            return scale * wall_distribution[static_cast<std::size_t>(v_i)];
        }
        return state.at(interior_x, v_i);
    }

    template <std::floating_point T>
    inline T completeAccommodationRightWallValue(
        const VelocityGrid1D<T> &grid,
        const std::vector<T> &wall_distribution,
        const ChannelState<T> &state,
        int interior_x,
        int v_i)
    {
        const T c = grid.axis[static_cast<std::size_t>(v_i)];
        if (c < T(0))
        {
            T outgoing_flux{};
            T incoming_wall_flux{};
            for (int k = 0; k < grid.size(); ++k)
            {
                const T wall_c = grid.axis[static_cast<std::size_t>(k)];
                if (wall_c > T(0))
                {
                    outgoing_flux += wall_c * state.at(interior_x, k);
                }
                else if (wall_c < T(0))
                {
                    incoming_wall_flux += -wall_c * wall_distribution[static_cast<std::size_t>(k)];
                }
            }

            const T scale = outgoing_flux / std::max(incoming_wall_flux, T(1e-30));
            return scale * wall_distribution[static_cast<std::size_t>(v_i)];
        }
        return state.at(interior_x, v_i);
    }

    template <std::floating_point T>
    inline MacroState<T> computeMacroState(
        const ChannelState<T> &state,
        const VelocityGrid1D<T> &grid,
        T particle_mass,
        T boltzmann_over_mass)
    {
        MacroState<T> macro;
        macro.density.resize(state.n_x);
        macro.bulk_v.resize(state.n_x);
        macro.temperature.resize(state.n_x);

        const T dv = grid.cellVolume();
        for (int x_i = 0; x_i < state.n_x; ++x_i)
        {
            T mass_density{};
            T momentum_density{};
            T total_energy_density{};

            for (int v_i = 0; v_i < state.n_v; ++v_i)
            {
                const T f = state.at(x_i, v_i);
                const T c = grid.axis[static_cast<std::size_t>(v_i)];
                mass_density += particle_mass * f * dv;
                momentum_density += particle_mass * c * f * dv;
                total_energy_density += T(0.5) * particle_mass * c * c * f * dv;
            }

            const T safe_density = std::max(mass_density, T(1e-12));
            const T bulk_v = momentum_density / safe_density;
            const T temperature = std::max(
                (T(2) * total_energy_density / safe_density - bulk_v * bulk_v) / std::max(boltzmann_over_mass, T(1e-12)),
                T(1e-6));
            const T density = mass_density;

            macro.density[x_i] = density;
            macro.bulk_v[x_i] = bulk_v;
            macro.temperature[x_i] = temperature;
        }

        return macro;
    }

    template <std::floating_point T>
    inline void writeChannelOutput(
        const std::filesystem::path &folder_path,
        const ChannelProblemData<T> &data,
        const ChannelState<T> &state)
    {
        std::filesystem::create_directories(folder_path);

        const MacroState<T> macro = computeMacroState(state, data.velocity_grid, data.particle_mass, data.boltzmann_over_mass);

        std::ofstream x_output(folder_path / "x_cells.txt");
        const T dx = (data.x_max - data.x_min) / T(data.n_x - 1);
        std::ofstream density_output(folder_path / "density.txt");
        std::ofstream velocity_output(folder_path / "bulk_vx.txt");
        std::ofstream temperature_output(folder_path / "temperature.txt");

        for (int x_i = 0; x_i < data.n_x; ++x_i)
        {
            const T x = data.x_min + T(x_i) * dx;
            x_output << x << '\n';
            density_output << macro.density[x_i] << '\n';
            velocity_output << macro.bulk_v[x_i] << '\n';
            temperature_output << macro.temperature[x_i] << '\n';
        }

        std::ofstream axis_output(folder_path / "velocity_axis.txt");
        for (int i = 0; i < data.velocity_grid.n_v; ++i)
        {
            if (i > 0)
            {
                axis_output << ' ';
            }
            axis_output << data.velocity_grid.axis[i];
        }
        axis_output << '\n';

        std::ofstream distribution_output(folder_path / "distribution_f.txt");
        for (int x_i = 0; x_i < data.n_x; ++x_i)
        {
            for (int v_i = 0; v_i < data.velocity_grid.n_v; ++v_i)
            {
                if (v_i > 0)
                {
                    distribution_output << ' ';
                }
                distribution_output << state.at(x_i, v_i);
            }
            distribution_output << '\n';
        }
    }

    template <std::floating_point T>
    inline void writeDistributionSnapshot(
        const std::filesystem::path &snapshot_folder,
        const ChannelProblemData<T> &data,
        const ChannelState<T> &state,
        int frame_index,
        T time_value)
    {
        std::filesystem::create_directories(snapshot_folder);

        {
            std::ofstream time_output(snapshot_folder / "snapshot_times.txt", frame_index == 0 ? std::ios::out : std::ios::app);
            time_output << frame_index << ' ' << time_value << '\n';
        }

        std::ostringstream file_name;
        file_name << "distribution_f_" << std::setw(4) << std::setfill('0') << frame_index << ".txt";
        std::ofstream snapshot_output(snapshot_folder / file_name.str());
        for (int x_i = 0; x_i < data.n_x; ++x_i)
        {
            for (int v_i = 0; v_i < data.velocity_grid.n_v; ++v_i)
            {
                if (v_i > 0)
                {
                    snapshot_output << ' ';
                }
                snapshot_output << state.at(x_i, v_i);
            }
            snapshot_output << '\n';
        }
    }

    template <std::floating_point T>
    [[nodiscard]] inline T velocityDerivativeUpwind(
        const ChannelState<T> &state,
        const VelocityGrid1D<T> &grid,
        T acceleration,
        int x_i,
        int v_i)
    {
        if (state.n_v < 2)
        {
            return T(0);
        }

        if (acceleration > T(0))
        {
            if (v_i == 0)
            {
                return (state.at(x_i, 1) - state.at(x_i, 0)) / grid.dv;
            }
            return (state.at(x_i, v_i) - state.at(x_i, v_i - 1)) / grid.dv;
        }

        if (acceleration < T(0))
        {
            if (v_i == state.n_v - 1)
            {
                return (state.at(x_i, state.n_v - 1) - state.at(x_i, state.n_v - 2)) / grid.dv;
            }
            return (state.at(x_i, v_i + 1) - state.at(x_i, v_i)) / grid.dv;
        }

        return T(0);
    }

    template <std::floating_point T>
    inline ChannelState<T> solveSteadyChannelBGK(
        const ChannelProblemData<T> &data,
        const std::filesystem::path *snapshot_folder = nullptr)
    {
        // Reduced 1D_x-1D_c BGK model:
        //   df/dt + c df/dx + (b/m) df/dc = -nu (f - f_eq).
        const T reference_temperature = T(0.5) * (data.left_wall_temperature + data.right_wall_temperature);
        const auto reference_equilibrium = maxwellianCell(
            data.velocity_grid,
            data.density,
            data.particle_mass,
            T(0),
            reference_temperature,
            data.boltzmann_over_mass);
        const auto left_wall_distribution = maxwellianCell(
            data.velocity_grid,
            data.density,
            data.particle_mass,
            T(0),
            data.left_wall_temperature,
            data.boltzmann_over_mass);
        const auto right_wall_distribution = maxwellianCell(
            data.velocity_grid,
            data.density,
            data.particle_mass,
            T(0),
            data.right_wall_temperature,
            data.boltzmann_over_mass);

        ChannelState<T> state(data.n_x, data.velocity_grid.size());
        const T dx = (data.x_max - data.x_min) / T(data.n_x - 1);
        const T acceleration = data.body_force / std::max(data.particle_mass, T(1e-12));
        const T transport_dt = data.cfl * dx / std::max(data.velocity_grid.v_max, T(1e-12));
        const T velocity_dt =
            std::abs(acceleration) > T(1e-12)
                ? data.cfl * data.velocity_grid.dv / std::abs(acceleration)
                : std::numeric_limits<T>::max();
        const T collision_dt =
            data.collision_frequency > T(1e-12)
                ? T(0.5) / data.collision_frequency
                : std::numeric_limits<T>::max();
        // The explicit time step resolves x-transport, c-transport, and BGK relaxation:
        //   df/dt + c df/dx + (b/m) df/dc = -nu (f - f_eq).
        // Snapshot times recorded for the GIF are multiples of this dt.
        const T stability_dt = std::min({transport_dt, velocity_dt, collision_dt});
        const int time_limited_iterations =
            data.final_time > T(0)
                ? std::max(1, static_cast<int>(std::ceil(data.final_time / stability_dt)))
                : data.max_iterations;
        const int iteration_limit =
            data.final_time > T(0)
                ? std::min(data.max_iterations, time_limited_iterations)
                : data.max_iterations;

        for (int x_i = 0; x_i < data.n_x; ++x_i)
        {
            for (int v_i = 0; v_i < data.velocity_grid.size(); ++v_i)
            {
                state.at(x_i, v_i) = reference_equilibrium[v_i];
            }
        }
        // Enforce the selected velocity-space boundary conditions on the truncated domain.
        applyVelocityBoundary(state, data.vmin_boundary, data.vmax_boundary);

        int frame_index = 0;
        if (snapshot_folder != nullptr && data.snapshot_interval > 0)
        {
            writeDistributionSnapshot(*snapshot_folder, data, state, frame_index, T(0));
            ++frame_index;
        }

        for (int iter = 0; iter < iteration_limit; ++iter)
        {
            const MacroState<T> macro = computeMacroState(state, data.velocity_grid, data.particle_mass, data.boltzmann_over_mass);
            std::vector<std::vector<T>> local_equilibria(data.n_x);
            for (int x_i = 0; x_i < data.n_x; ++x_i)
            {
                local_equilibria[x_i] = maxwellianCell(
                    data.velocity_grid,
                    macro.density[x_i],
                    data.particle_mass,
                    macro.bulk_v[x_i],
                    macro.temperature[x_i],
                    data.boltzmann_over_mass);
            }

            ChannelState<T> next(data.n_x, data.velocity_grid.size());
            for (int v_i = 0; v_i < data.velocity_grid.size(); ++v_i)
            {
                const T c = data.velocity_grid.axis[static_cast<std::size_t>(v_i)];

                if (c > T(1e-12))
                {
                    T inflow_value = left_wall_distribution[static_cast<std::size_t>(v_i)];
                    for (int x_i = 0; x_i < data.n_x; ++x_i)
                    {
                        const T f = state.at(x_i, v_i);
                        const T x_upwind = (f - inflow_value) / dx;
                        const T c_upwind = velocityDerivativeUpwind(state, data.velocity_grid, acceleration, x_i, v_i);
                        const T rhs =
                            -c * x_upwind
                            -acceleration * c_upwind
                            -data.collision_frequency * (f - local_equilibria[x_i][v_i]);
                        next.at(x_i, v_i) = time_integration::stepFromSlope(f, stability_dt, rhs);
                        inflow_value = f;
                    }
                }
                else if (c < -T(1e-12))
                {
                    T inflow_value = right_wall_distribution[static_cast<std::size_t>(v_i)];
                    for (int x_i = data.n_x - 1; x_i >= 0; --x_i)
                    {
                        const T f = state.at(x_i, v_i);
                        const T x_upwind = (inflow_value - f) / dx;
                        const T c_upwind = velocityDerivativeUpwind(state, data.velocity_grid, acceleration, x_i, v_i);
                        const T rhs =
                            -c * x_upwind
                            -acceleration * c_upwind
                            -data.collision_frequency * (f - local_equilibria[x_i][v_i]);
                        next.at(x_i, v_i) = time_integration::stepFromSlope(f, stability_dt, rhs);
                        inflow_value = f;
                    }
                }
                else
                {
                    for (int x_i = 0; x_i < data.n_x; ++x_i)
                    {
                        const T f = state.at(x_i, v_i);
                        const T c_upwind = velocityDerivativeUpwind(state, data.velocity_grid, acceleration, x_i, v_i);
                        const T rhs =
                            -acceleration * c_upwind
                            -data.collision_frequency * (f - local_equilibria[x_i][v_i]);
                        next.at(x_i, v_i) = time_integration::stepFromSlope(f, stability_dt, rhs);
                    }
                }
            }

            T change_norm_squared{};
            T previous_norm_squared{};
            for (std::size_t i = 0; i < state.values.size(); ++i)
            {
                T candidate = next.values[i];
                if (!std::isfinite(candidate) || candidate < T(0))
                {
                    candidate = state.values[i];
                }

                const T change = candidate - state.values[i];
                change_norm_squared += change * change;
                previous_norm_squared += state.values[i] * state.values[i];
                next.values[i] = candidate;
            }

            // Explicit velocity-space boundary condition on the computational cutoffs.
            applyVelocityBoundary(next, data.vmin_boundary, data.vmax_boundary);

            const T residual = std::sqrt(change_norm_squared / std::max(previous_norm_squared, T(1e-30)));
            state = std::move(next);

            if (snapshot_folder != nullptr &&
                data.snapshot_interval > 0 &&
                (((iter + 1) % data.snapshot_interval) == 0 || residual < data.tolerance || iter + 1 == iteration_limit))
            {
                writeDistributionSnapshot(*snapshot_folder, data, state, frame_index, T(iter + 1) * stability_dt);
                ++frame_index;
            }

            if (residual < data.tolerance && (data.final_time <= T(0) || T(iter + 1) * stability_dt >= data.final_time))
            {
                break;
            }
        }

        return state;
    }

    template <std::floating_point T>
    inline ChannelState<T> solveSteadyChannelBGK(const ChannelProblemData<T> &data)
    {
        return solveSteadyChannelBGK(data, nullptr);
    }

    template <std::floating_point T>
    inline ChannelProblemData<T> normalFlowProblem(
        int n_x,
        int n_v,
        T x_min,
        T x_max,
        T v_max,
        T density,
        T left_wall_temperature,
        T right_wall_temperature,
        int max_iterations,
        T tolerance)
    {
        ChannelProblemData<T> data;
        data.n_x = n_x;
        data.x_min = x_min;
        data.x_max = x_max;
        data.velocity_grid = VelocityGrid1D<T>(n_v, v_max);
        data.density = density;
        data.body_force = T(0);
        data.left_wall_temperature = left_wall_temperature;
        data.right_wall_temperature = right_wall_temperature;
        data.vmin_boundary = VelocityBoundaryCondition::NeumannZeroGradient;
        data.vmax_boundary = VelocityBoundaryCondition::NeumannZeroGradient;
        data.collision_frequency = T(10);
        data.max_iterations = max_iterations;
        data.tolerance = tolerance;
        return data;
    }

}
