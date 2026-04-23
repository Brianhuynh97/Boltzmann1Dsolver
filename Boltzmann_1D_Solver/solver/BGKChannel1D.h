#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "AdditionalMath.h"

namespace bgk_channel
{

    template <std::floating_point T>
    struct Vec3
    {
        T x{};
        T y{};
        T z{};
    };

    template <std::floating_point T>
    inline Vec3<T> operator-(const Vec3<T> &lhs, const Vec3<T> &rhs)
    {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }

    template <std::floating_point T>
    inline T dot(const Vec3<T> &lhs, const Vec3<T> &rhs)
    {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }

    template <std::floating_point T>
    struct VelocityGrid3D
    {
        int n_v{};
        T v_max{};
        T dv{};
        std::vector<T> axis;
        std::vector<Vec3<T>> velocities;

        VelocityGrid3D() = default;

        VelocityGrid3D(int n_v_, T v_max_)
            : n_v(n_v_), v_max(v_max_), dv((T(2) * v_max_) / T(n_v_))
        {
            assert(n_v > 1);
            axis.resize(n_v);
            for (int i = 0; i < n_v; ++i)
            {
                axis[i] = -v_max + (T(i) + T(0.5)) * dv;
            }

            velocities.reserve(size());
            for (int ix = 0; ix < n_v; ++ix)
            {
                for (int iy = 0; iy < n_v; ++iy)
                {
                    for (int iz = 0; iz < n_v; ++iz)
                    {
                        velocities.push_back({axis[ix], axis[iy], axis[iz]});
                    }
                }
            }
        }

        [[nodiscard]] int size() const
        {
            return n_v * n_v * n_v;
        }

        [[nodiscard]] T cellVolume() const
        {
            return dv * dv * dv;
        }

        [[nodiscard]] int flattenIndex(int ix, int iy, int iz) const
        {
            return ix * n_v * n_v + iy * n_v + iz;
        }
    };

    template <std::floating_point T>
    struct ChannelState
    {
        int n_y{};
        int n_v{};
        std::vector<T> values;

        ChannelState() = default;

        ChannelState(int n_y_, int n_v_)
            : n_y(n_y_), n_v(n_v_), values(static_cast<std::size_t>(n_y_) * static_cast<std::size_t>(n_v_), T{})
        {
        }

        [[nodiscard]] T &at(int y_i, int v_i)
        {
            return values[static_cast<std::size_t>(y_i) * static_cast<std::size_t>(n_v) + static_cast<std::size_t>(v_i)];
        }

        [[nodiscard]] const T &at(int y_i, int v_i) const
        {
            return values[static_cast<std::size_t>(y_i) * static_cast<std::size_t>(n_v) + static_cast<std::size_t>(v_i)];
        }
    };

    template <std::floating_point T>
    struct MacroState
    {
        std::vector<T> density;
        std::vector<T> bulk_vx;
        std::vector<T> temperature;
    };

    template <std::floating_point T>
    struct ChannelProblemData
    {
        int n_y{};
        T y_min{};
        T y_max{};
        VelocityGrid3D<T> velocity_grid;
        T reference_density{T(1)};
        T reference_temperature{T(1)};
        T lower_wall_density{T(1)};
        T upper_wall_density{T(1)};
        T lower_wall_temperature{T(1)};
        T upper_wall_temperature{T(1)};
        T lower_wall_bulk_vx{T(0)};
        T upper_wall_bulk_vx{T(0)};
        T body_force_x{T(0)};
        T tau{T(0.5)};
        T relaxation{T(0.03)};
        int max_iterations{250};
        T tolerance{T(1e-8)};
    };

    template <std::floating_point T>
    inline std::vector<T> maxwellianCell(
        const VelocityGrid3D<T> &grid,
        T density,
        const Vec3<T> &bulk_velocity,
        T temperature)
    {
        std::vector<T> values(grid.size());
        const T prefactor = density / std::pow(T(D_PI) * temperature, T(1.5));

        for (int v_i = 0; v_i < grid.size(); ++v_i)
        {
            const Vec3<T> c = grid.velocities[v_i] - bulk_velocity;
            values[v_i] = prefactor * std::exp(-dot(c, c) / temperature);
        }

        return values;
    }

    template <std::floating_point T>
    inline std::vector<T> shiftedMaxwellianDistribution(
        const VelocityGrid3D<T> &grid,
        T density,
        T temperature,
        T bulk_vx)
    {
        return maxwellianCell(grid, density, Vec3<T>{bulk_vx, T(0), T(0)}, temperature);
    }

    template <std::floating_point T>
    inline std::vector<T> equilibriumDistribution(
        const VelocityGrid3D<T> &grid,
        T density,
        T temperature)
    {
        return maxwellianCell(grid, density, Vec3<T>{T(0), T(0), T(0)}, temperature);
    }

    template <std::floating_point T>
    inline T completeAccommodationLowerWallValue(
        const VelocityGrid3D<T> &grid,
        const std::vector<T> &wall_distribution,
        const ChannelState<T> &state,
        int interior_y,
        int v_i)
    {
        const T cy = grid.velocities[v_i].y;
        if (cy > T(0))
        {
            T outgoing_flux{};
            T incoming_wall_flux{};
            for (int k = 0; k < grid.size(); ++k)
            {
                const T wall_cy = grid.velocities[k].y;
                if (wall_cy < T(0))
                {
                    outgoing_flux += -wall_cy * state.at(interior_y, k);
                }
                else if (wall_cy > T(0))
                {
                    incoming_wall_flux += wall_cy * wall_distribution[k];
                }
            }

            const T scale = outgoing_flux / std::max(incoming_wall_flux, T(1e-30));
            return scale * wall_distribution[v_i];
        }
        return state.at(interior_y, v_i);
    }

    template <std::floating_point T>
    inline T completeAccommodationUpperWallValue(
        const VelocityGrid3D<T> &grid,
        const std::vector<T> &wall_distribution,
        const ChannelState<T> &state,
        int interior_y,
        int v_i)
    {
        const T cy = grid.velocities[v_i].y;
        if (cy < T(0))
        {
            T outgoing_flux{};
            T incoming_wall_flux{};
            for (int k = 0; k < grid.size(); ++k)
            {
                const T wall_cy = grid.velocities[k].y;
                if (wall_cy > T(0))
                {
                    outgoing_flux += wall_cy * state.at(interior_y, k);
                }
                else if (wall_cy < T(0))
                {
                    incoming_wall_flux += -wall_cy * wall_distribution[k];
                }
            }

            const T scale = outgoing_flux / std::max(incoming_wall_flux, T(1e-30));
            return scale * wall_distribution[v_i];
        }
        return state.at(interior_y, v_i);
    }

    template <std::floating_point T>
    inline MacroState<T> computeMacroState(const ChannelState<T> &state, const VelocityGrid3D<T> &grid)
    {
        MacroState<T> macro;
        macro.density.resize(state.n_y);
        macro.bulk_vx.resize(state.n_y);
        macro.temperature.resize(state.n_y);

        const T dv3 = grid.cellVolume();
        for (int y_i = 0; y_i < state.n_y; ++y_i)
        {
            T density{};
            T momentum_x{};
            T energy{};

            for (int v_i = 0; v_i < state.n_v; ++v_i)
            {
                const T f = state.at(y_i, v_i);
                const Vec3<T> &v = grid.velocities[v_i];
                density += f * dv3;
                momentum_x += v.x * f * dv3;
                energy += dot(v, v) * f * dv3;
            }

            const T safe_density = std::max(density, T(1e-12));
            const T bulk_vx = momentum_x / safe_density;
            const T temperature = std::max(T(2) * (energy / safe_density - bulk_vx * bulk_vx) / T(3), T(1e-6));

            macro.density[y_i] = density;
            macro.bulk_vx[y_i] = bulk_vx;
            macro.temperature[y_i] = temperature;
        }

        return macro;
    }

    template <std::floating_point T>
    inline T velocityForceSource(
        const VelocityGrid3D<T> &grid,
        const ChannelState<T> &state,
        int y_i,
        int v_i,
        T force_x)
    {
        if (std::abs(force_x) < T(1e-14))
        {
            return T(0);
        }

        const int n_v = grid.n_v;
        const int ix = v_i / (n_v * n_v);
        const int rem = v_i % (n_v * n_v);
        const int iy = rem / n_v;
        const int iz = rem % n_v;

        T derivative{};
        if (force_x > T(0))
        {
            if (ix == 0)
            {
                derivative = T(0);
            }
            else
            {
                const int left = grid.flattenIndex(ix - 1, iy, iz);
                derivative = (state.at(y_i, v_i) - state.at(y_i, left)) / grid.dv;
            }
        }
        else
        {
            if (ix == n_v - 1)
            {
                derivative = T(0);
            }
            else
            {
                const int right = grid.flattenIndex(ix + 1, iy, iz);
                derivative = (state.at(y_i, right) - state.at(y_i, v_i)) / grid.dv;
            }
        }

        return -force_x * derivative;
    }

    template <std::floating_point T>
    inline void writeChannelOutput(
        const std::filesystem::path &folder_path,
        const ChannelProblemData<T> &data,
        const ChannelState<T> &state)
    {
        std::filesystem::create_directories(folder_path);

        const MacroState<T> macro = computeMacroState(state, data.velocity_grid);
        const T dy = (data.y_max - data.y_min) / T(data.n_y - 1);

        std::ofstream y_output(folder_path / "y_cells.txt");
        std::ofstream density_output(folder_path / "density.txt");
        std::ofstream velocity_output(folder_path / "bulk_vx.txt");
        std::ofstream velocity_y_output(folder_path / "bulk_vy.txt");
        std::ofstream temperature_output(folder_path / "temperature.txt");
        std::ofstream wall_temperature_output(folder_path / "wall_temperature.txt");
        wall_temperature_output << data.lower_wall_temperature << '\n'
                                << data.upper_wall_temperature << '\n';

        for (int y_i = 0; y_i < data.n_y; ++y_i)
        {
            const T y = data.y_min + T(y_i) * dy;
            y_output << y << '\n';
            density_output << macro.density[y_i] << '\n';
            velocity_output << macro.bulk_vx[y_i] << '\n';
            velocity_y_output << T(0) << '\n';
            temperature_output << macro.temperature[y_i] << '\n';
        }

        std::ofstream labels_output(folder_path / "distribution_labels.txt");
        labels_output << "Lower wall\nCenter\nUpper wall\n";

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

        std::ofstream time_output(folder_path / "distribution_time.txt");
        time_output << "steady\n";

        const int iz = data.velocity_grid.n_v / 2;
        auto write_slice = [&](const char *file_name, auto value_provider)
        {
            std::ofstream slice_output(folder_path / file_name);
            for (int iy = 0; iy < data.velocity_grid.n_v; ++iy)
            {
                for (int ix = 0; ix < data.velocity_grid.n_v; ++ix)
                {
                    if (ix > 0)
                    {
                        slice_output << ' ';
                    }
                    const int flat_v = data.velocity_grid.flattenIndex(ix, iy, iz);
                    slice_output << value_provider(flat_v);
                }
                slice_output << '\n';
            }
        };

        write_slice("distribution_left.txt", [&](int flat_v)
                    { return state.at(0, flat_v); });
        write_slice("distribution_center.txt", [&](int flat_v)
                    { return state.at(data.n_y / 2, flat_v); });
        write_slice("distribution_right.txt", [&](int flat_v)
                    { return state.at(data.n_y - 1, flat_v); });
    }

    template <std::floating_point T>
    inline ChannelState<T> solveSteadyChannelBGK(
        const ChannelProblemData<T> &data,
        std::vector<T> &convergence_history)
    {
        convergence_history.clear();

        const auto reference_equilibrium = equilibriumDistribution(
            data.velocity_grid,
            data.reference_density,
            data.reference_temperature);
        const auto lower_wall_distribution = shiftedMaxwellianDistribution(
            data.velocity_grid,
            data.lower_wall_density,
            data.lower_wall_temperature,
            data.lower_wall_bulk_vx);
        const auto upper_wall_distribution = shiftedMaxwellianDistribution(
            data.velocity_grid,
            data.upper_wall_density,
            data.upper_wall_temperature,
            data.upper_wall_bulk_vx);

        ChannelState<T> state(data.n_y, data.velocity_grid.size());
        const T dy = (data.y_max - data.y_min) / T(data.n_y - 1);

        for (int y_i = 0; y_i < data.n_y; ++y_i)
        {
            for (int v_i = 0; v_i < data.velocity_grid.size(); ++v_i)
            {
                state.at(y_i, v_i) = reference_equilibrium[v_i];
            }
        }

        for (int iter = 0; iter < data.max_iterations; ++iter)
        {
            const MacroState<T> macro = computeMacroState(state, data.velocity_grid);
            std::vector<std::vector<T>> local_equilibria(data.n_y);
            for (int y_i = 0; y_i < data.n_y; ++y_i)
            {
                local_equilibria[y_i] = shiftedMaxwellianDistribution(
                    data.velocity_grid,
                    macro.density[y_i],
                    macro.temperature[y_i],
                    macro.bulk_vx[y_i]);
            }

            ChannelState<T> next(data.n_y, data.velocity_grid.size());

            for (int v_i = 0; v_i < data.velocity_grid.size(); ++v_i)
            {
                const Vec3<T> &velocity = data.velocity_grid.velocities[v_i];
                const T cy = velocity.y;

                if (cy > T(1e-12))
                {
                    next.at(0, v_i) = completeAccommodationLowerWallValue(
                        data.velocity_grid,
                        lower_wall_distribution,
                        state,
                        1,
                        v_i);
                    for (int y_i = 1; y_i < data.n_y; ++y_i)
                    {
                        const T alpha = dy / (data.tau * cy);
                        const T source = velocityForceSource(data.velocity_grid, state, y_i, v_i, data.body_force_x);
                        next.at(y_i, v_i) = std::max((next.at(y_i - 1, v_i) + alpha * local_equilibria[y_i][v_i] + (dy / cy) * source) / (T(1) + alpha), T(0));
                    }
                }
                else if (cy < -T(1e-12))
                {
                    next.at(data.n_y - 1, v_i) = completeAccommodationUpperWallValue(
                        data.velocity_grid,
                        upper_wall_distribution,
                        state,
                        data.n_y - 2,
                        v_i);
                    for (int y_i = data.n_y - 2; y_i >= 0; --y_i)
                    {
                        const T alpha = dy / (data.tau * (-cy));
                        const T source = velocityForceSource(data.velocity_grid, state, y_i, v_i, data.body_force_x);
                        next.at(y_i, v_i) = std::max((next.at(y_i + 1, v_i) + alpha * local_equilibria[y_i][v_i] + (dy / (-cy)) * source) / (T(1) + alpha), T(0));
                    }
                }
                else
                {
                    for (int y_i = 0; y_i < data.n_y; ++y_i)
                    {
                        const T source = velocityForceSource(data.velocity_grid, state, y_i, v_i, data.body_force_x);
                        next.at(y_i, v_i) = std::max(local_equilibria[y_i][v_i] + data.tau * source, T(0));
                    }
                }
            }

            T change_norm_squared{};
            T previous_norm_squared{};
            for (std::size_t i = 0; i < state.values.size(); ++i)
            {
                T candidate = (T(1) - data.relaxation) * state.values[i] + data.relaxation * next.values[i];
                if (!std::isfinite(candidate) || candidate < T(0))
                {
                    candidate = state.values[i];
                }

                const T change = candidate - state.values[i];
                change_norm_squared += change * change;
                previous_norm_squared += state.values[i] * state.values[i];
                next.values[i] = candidate;
            }

            const T residual = std::sqrt(change_norm_squared / std::max(previous_norm_squared, T(1e-30)));
            convergence_history.push_back(residual);
            state = std::move(next);
            if (residual < data.tolerance)
            {
                break;
            }
        }

        return state;
    }

    template <std::floating_point T>
    inline ChannelState<T> solveSteadyChannelBGK(const ChannelProblemData<T> &data)
    {
        std::vector<T> unused_convergence_history;
        return solveSteadyChannelBGK(data, unused_convergence_history);
    }

    template <std::floating_point T>
    inline ChannelProblemData<T> couetteProblem(
        int n_y,
        int n_v,
        T y_min,
        T y_max,
        T v_max,
        T density,
        T temperature,
        T wall_speed,
        T tau,
        int max_iterations,
        T tolerance)
    {
        ChannelProblemData<T> data;
        data.n_y = n_y;
        data.y_min = y_min;
        data.y_max = y_max;
        data.velocity_grid = VelocityGrid3D<T>(n_v, v_max);
        data.reference_density = density;
        data.reference_temperature = temperature;
        data.lower_wall_density = density;
        data.upper_wall_density = density;
        data.lower_wall_temperature = temperature;
        data.upper_wall_temperature = temperature;
        data.lower_wall_bulk_vx = T(0.5) * wall_speed;
        data.upper_wall_bulk_vx = -T(0.5) * wall_speed;
        data.tau = tau;
        data.max_iterations = max_iterations;
        data.tolerance = tolerance;
        return data;
    }

    template <std::floating_point T>
    inline ChannelProblemData<T> poiseuilleProblem(
        int n_y,
        int n_v,
        T y_min,
        T y_max,
        T v_max,
        T density,
        T temperature,
        T body_force_x,
        T tau,
        int max_iterations,
        T tolerance)
    {
        ChannelProblemData<T> data;
        data.n_y = n_y;
        data.y_min = y_min;
        data.y_max = y_max;
        data.velocity_grid = VelocityGrid3D<T>(n_v, v_max);
        data.reference_density = density;
        data.reference_temperature = temperature;
        data.lower_wall_density = density;
        data.upper_wall_density = density;
        data.lower_wall_temperature = temperature;
        data.upper_wall_temperature = temperature;
        data.body_force_x = body_force_x;
        data.tau = tau;
        data.max_iterations = max_iterations;
        data.tolerance = tolerance;
        return data;
    }

    template <std::floating_point T>
    inline ChannelProblemData<T> heatConductionProblem(
        int n_y,
        int n_v,
        T y_min,
        T y_max,
        T v_max,
        T density,
        T lower_wall_temperature,
        T upper_wall_temperature,
        T tau,
        int max_iterations,
        T tolerance)
    {
        ChannelProblemData<T> data;
        data.n_y = n_y;
        data.y_min = y_min;
        data.y_max = y_max;
        data.velocity_grid = VelocityGrid3D<T>(n_v, v_max);
        data.reference_density = density;
        data.reference_temperature = T(0.5) * (lower_wall_temperature + upper_wall_temperature);
        data.lower_wall_density = density;
        data.upper_wall_density = density;
        data.lower_wall_temperature = lower_wall_temperature;
        data.upper_wall_temperature = upper_wall_temperature;
        data.tau = tau;
        data.max_iterations = max_iterations;
        data.tolerance = tolerance;
        return data;
    }

}
