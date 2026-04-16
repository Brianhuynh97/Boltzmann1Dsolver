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
#include <utility>
#include <vector>

#include "AdditionalMath.h"
#include "OpenMpCompat.h"
#include "rkMethods.h"

namespace full_boltzmann_1d3v {

template <std::floating_point T>
struct Vec3 {
    T x{};
    T y{};
    T z{};
};

template <std::floating_point T>
inline Vec3<T> operator+(const Vec3<T>& lhs, const Vec3<T>& rhs) {
    return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
}

template <std::floating_point T>
inline Vec3<T> operator-(const Vec3<T>& lhs, const Vec3<T>& rhs) {
    return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
}

template <std::floating_point T>
inline Vec3<T> operator*(T factor, const Vec3<T>& value) {
    return { factor * value.x, factor * value.y, factor * value.z };
}

template <std::floating_point T>
inline T dot(const Vec3<T>& lhs, const Vec3<T>& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

template <std::floating_point T>
inline T norm(const Vec3<T>& value) {
    return std::sqrt(dot(value, value));
}

template <std::floating_point T>
struct VelocityGrid3D {
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
        for (int i = 0; i < n_v; ++i) {
            axis[i] = -v_max + (T(i) + T(0.5)) * dv;
        }

        velocities.reserve(size());
        for (int ix = 0; ix < n_v; ++ix) {
            for (int iy = 0; iy < n_v; ++iy) {
                for (int iz = 0; iz < n_v; ++iz) {
                    velocities.push_back({ axis[ix], axis[iy], axis[iz] });
                }
            }
        }
    }

    [[nodiscard]] int size() const {
        return n_v * n_v * n_v;
    }

    [[nodiscard]] T cellVolume() const {
        return dv * dv * dv;
    }

    [[nodiscard]] int flattenIndex(int ix, int iy, int iz) const {
        return ix * n_v * n_v + iy * n_v + iz;
    }

    [[nodiscard]] std::array<int, 3> unflattenIndex(int flat) const {
        const int ix = flat / (n_v * n_v);
        const int rem = flat % (n_v * n_v);
        const int iy = rem / n_v;
        const int iz = rem % n_v;
        return { ix, iy, iz };
    }
};

template <std::floating_point T>
struct MacroState1D3V {
    std::vector<T> density;
    std::vector<T> bulk_vx;
    std::vector<T> bulk_vy;
    std::vector<T> bulk_vz;
    std::vector<T> temperature;
};

template <std::floating_point T>
struct DistributionState1D3V {
    int n_x{};
    int n_v{};
    std::vector<T> values;

    DistributionState1D3V() = default;

    DistributionState1D3V(int n_x_, int n_v_)
        : n_x(n_x_), n_v(n_v_), values(static_cast<std::size_t>(n_x_) * static_cast<std::size_t>(n_v_), T{})
    {}

    [[nodiscard]] T& at(int x_i, int v_i) {
        return values[static_cast<std::size_t>(x_i) * static_cast<std::size_t>(n_v) + static_cast<std::size_t>(v_i)];
    }

    [[nodiscard]] const T& at(int x_i, int v_i) const {
        return values[static_cast<std::size_t>(x_i) * static_cast<std::size_t>(n_v) + static_cast<std::size_t>(v_i)];
    }
};

template <std::floating_point T>
inline DistributionState1D3V<T> operator+(const DistributionState1D3V<T>& lhs, const DistributionState1D3V<T>& rhs) {
    assert(lhs.n_x == rhs.n_x);
    assert(lhs.n_v == rhs.n_v);

    DistributionState1D3V<T> result(lhs.n_x, lhs.n_v);
    for (std::size_t i = 0; i < lhs.values.size(); ++i) {
        result.values[i] = lhs.values[i] + rhs.values[i];
    }
    return result;
}

template <std::floating_point T>
inline DistributionState1D3V<T> operator*(T factor, const DistributionState1D3V<T>& state) {
    DistributionState1D3V<T> result(state.n_x, state.n_v);
    for (std::size_t i = 0; i < state.values.size(); ++i) {
        result.values[i] = factor * state.values[i];
    }
    return result;
}

template <std::floating_point T>
struct HardSphereCollisionModel {
    T cross_section;
    std::vector<Vec3<T>> directions;
    std::vector<T> direction_weights;

    explicit HardSphereCollisionModel(T cross_section_ = T(1))
        : cross_section(cross_section_)
    {
        directions = {
            { T(1), T(0), T(0) }, { T(-1), T(0), T(0) },
            { T(0), T(1), T(0) }, { T(0), T(-1), T(0) },
            { T(0), T(0), T(1) }, { T(0), T(0), T(-1) }
        };

        const T angular_weight = T(4) * T(D_PI) / T(directions.size());
        direction_weights.assign(directions.size(), angular_weight);
    }
};

template <std::floating_point T>
struct DiffuseWall1D3V {
    bool enabled{ false };
    T density{ T(1) };
    T temperature{ T(1) };
    Vec3<T> velocity{};
    std::vector<T> incoming_maxwellian;
};

template <std::floating_point T>
struct ProblemData1D3V {
    int n_x{};
    T x_left{};
    T x_right{};
    T dt{};
    T t_end{};
    VelocityGrid3D<T> velocity_grid;
    DistributionState1D3V<T> initial_state;
    HardSphereCollisionModel<T> collision_model;
    DiffuseWall1D3V<T> left_wall;
    DiffuseWall1D3V<T> right_wall;
    Vec3<T> body_force{};
};

template <std::floating_point T>
class MacroOutput1D3V {
private:
    std::filesystem::path folder_path;
    std::ofstream time_output;
    std::ofstream density_output;
    std::ofstream bulk_vx_output;
    std::ofstream bulk_vy_output;
    std::ofstream temperature_output;

public:
    explicit MacroOutput1D3V(std::filesystem::path folder_path_)
        : folder_path(std::move(folder_path_))
    {
        std::filesystem::create_directories(folder_path);
        time_output.open(folder_path / "t_levels.txt");
        density_output.open(folder_path / "density.txt");
        bulk_vx_output.open(folder_path / "bulk_vx.txt");
        bulk_vy_output.open(folder_path / "bulk_vy.txt");
        temperature_output.open(folder_path / "temperature.txt");
    }

    void print(T t, const MacroState1D3V<T>& macro) {
        time_output << t << '\n';

        for (std::size_t i = 0; i < macro.density.size(); ++i) {
            if (i > 0) {
                density_output << ' ';
                bulk_vx_output << ' ';
                bulk_vy_output << ' ';
                temperature_output << ' ';
            }

            density_output << macro.density[i];
            bulk_vx_output << macro.bulk_vx[i];
            bulk_vy_output << macro.bulk_vy[i];
            temperature_output << macro.temperature[i];
        }

        density_output << '\n';
        bulk_vx_output << '\n';
        bulk_vy_output << '\n';
        temperature_output << '\n';
    }

    void close() {
        time_output.close();
        density_output.close();
        bulk_vx_output.close();
        bulk_vy_output.close();
        temperature_output.close();
    }

    [[nodiscard]] const std::filesystem::path& outputFolder() const {
        return folder_path;
    }
};

template <std::floating_point T>
inline void writeDistributionSlices(
    const std::filesystem::path& folder_path,
    const DistributionState1D3V<T>& state,
    const VelocityGrid3D<T>& grid,
    T time
) {
    std::filesystem::create_directories(folder_path);

    {
        std::ofstream time_output(folder_path / "distribution_time.txt");
        time_output << time << '\n';
    }

    {
        std::ofstream axis_output(folder_path / "velocity_axis.txt");
        for (int i = 0; i < grid.n_v; ++i) {
            if (i > 0) {
                axis_output << ' ';
            }
            axis_output << grid.axis[i];
        }
        axis_output << '\n';
    }

    const int iz = grid.n_v / 2;
    const std::array<std::pair<const char*, int>, 3> slices = {{
        { "distribution_left.txt", 0 },
        { "distribution_center.txt", state.n_x / 2 },
        { "distribution_right.txt", state.n_x - 1 }
    }};

    for (const auto& [file_name, x_i] : slices) {
        std::ofstream slice_output(folder_path / file_name);
        for (int iy = 0; iy < grid.n_v; ++iy) {
            for (int ix = 0; ix < grid.n_v; ++ix) {
                if (ix > 0) {
                    slice_output << ' ';
                }
                const int flat_v = grid.flattenIndex(ix, iy, iz);
                slice_output << state.at(x_i, flat_v);
            }
            slice_output << '\n';
        }
    }
}

template <std::floating_point T>
inline std::vector<T> maxwellianCell(
    const VelocityGrid3D<T>& grid,
    T density,
    const Vec3<T>& bulk_velocity,
    T temperature
);

template <std::floating_point T>
inline void initializeWallMaxwellian(DiffuseWall1D3V<T>& wall, const VelocityGrid3D<T>& grid) {
    if (!wall.enabled) {
        return;
    }
    wall.incoming_maxwellian = maxwellianCell(grid, wall.density, wall.velocity, wall.temperature);
}

template <std::floating_point T>
inline int nearestAxisIndex(const VelocityGrid3D<T>& grid, T value) {
    const T shifted = (value + grid.v_max) / grid.dv;
    const int idx = static_cast<int>(std::floor(shifted));
    return std::clamp(idx, 0, grid.n_v - 1);
}

template <std::floating_point T>
inline void depositToVelocityCell(
    const VelocityGrid3D<T>& grid,
    std::vector<T>& destination,
    const Vec3<T>& velocity,
    T amount
) {
    const auto clamp_index = [&grid](int idx) {
        return std::clamp(idx, 0, grid.n_v - 1);
    };

    const auto normalized_coordinate = [&grid](T value) {
        return (value + grid.v_max) / grid.dv - T(0.5);
    };

    const T sx = normalized_coordinate(velocity.x);
    const T sy = normalized_coordinate(velocity.y);
    const T sz = normalized_coordinate(velocity.z);

    const int ix0 = clamp_index(static_cast<int>(std::floor(sx)));
    const int iy0 = clamp_index(static_cast<int>(std::floor(sy)));
    const int iz0 = clamp_index(static_cast<int>(std::floor(sz)));

    const int ix1 = clamp_index(ix0 + 1);
    const int iy1 = clamp_index(iy0 + 1);
    const int iz1 = clamp_index(iz0 + 1);

    const T tx = std::clamp(sx - std::floor(sx), T(0), T(1));
    const T ty = std::clamp(sy - std::floor(sy), T(0), T(1));
    const T tz = std::clamp(sz - std::floor(sz), T(0), T(1));

    const std::array<int, 2> xs = { ix0, ix1 };
    const std::array<int, 2> ys = { iy0, iy1 };
    const std::array<int, 2> zs = { iz0, iz1 };
    const std::array<T, 2> wx = { T(1) - tx, tx };
    const std::array<T, 2> wy = { T(1) - ty, ty };
    const std::array<T, 2> wz = { T(1) - tz, tz };

    for (int ix_i = 0; ix_i < 2; ++ix_i) {
        for (int iy_i = 0; iy_i < 2; ++iy_i) {
            for (int iz_i = 0; iz_i < 2; ++iz_i) {
                const int ix = xs[ix_i];
                const int iy = ys[iy_i];
                const int iz = zs[iz_i];
                const int flat = ix * grid.n_v * grid.n_v + iy * grid.n_v + iz;
                destination[flat] += amount * wx[ix_i] * wy[iy_i] * wz[iz_i];
            }
        }
    }
}

template <std::floating_point T>
inline MacroState1D3V<T> computeMacroState(
    const DistributionState1D3V<T>& state,
    const VelocityGrid3D<T>& grid
) {
    MacroState1D3V<T> macro;
    macro.density.resize(state.n_x);
    macro.bulk_vx.resize(state.n_x);
    macro.bulk_vy.resize(state.n_x);
    macro.bulk_vz.resize(state.n_x);
    macro.temperature.resize(state.n_x);

    const T dv3 = grid.cellVolume();

#pragma omp parallel for
    for (int x_i = 0; x_i < state.n_x; ++x_i) {
        T density{};
        Vec3<T> momentum{};
        T energy{};

        for (int v_i = 0; v_i < state.n_v; ++v_i) {
            const T f = state.at(x_i, v_i);
            const Vec3<T>& v = grid.velocities[v_i];

            density += f * dv3;
            momentum.x += v.x * f * dv3;
            momentum.y += v.y * f * dv3;
            momentum.z += v.z * f * dv3;
            energy += dot(v, v) * f * dv3;
        }

        const T safe_density = std::max(density, T(1e-12));
        const Vec3<T> bulk = { momentum.x / safe_density, momentum.y / safe_density, momentum.z / safe_density };
        const T bulk_energy = dot(bulk, bulk);
        const T temperature = std::max((energy / safe_density - bulk_energy) / T(3), T(1e-8));

        macro.density[x_i] = density;
        macro.bulk_vx[x_i] = bulk.x;
        macro.bulk_vy[x_i] = bulk.y;
        macro.bulk_vz[x_i] = bulk.z;
        macro.temperature[x_i] = temperature;
    }

    return macro;
}

template <std::floating_point T>
inline std::vector<T> maxwellianCell(
    const VelocityGrid3D<T>& grid,
    T density,
    const Vec3<T>& bulk_velocity,
    T temperature
) {
    std::vector<T> values(grid.size());
    const T prefactor = density / std::pow(T(D_PI) * temperature, T(1.5));

    for (int v_i = 0; v_i < grid.size(); ++v_i) {
        const Vec3<T> c = grid.velocities[v_i] - bulk_velocity;
        values[v_i] = prefactor * std::exp(-dot(c, c) / temperature);
    }

    return values;
}

template <std::floating_point T>
inline ProblemData1D3V<T> shockTubeProblem(
    int n_x,
    int n_v,
    T x_left,
    T x_right,
    T v_max,
    T dt,
    T t_end,
    T left_density,
    T left_temperature,
    T right_density,
    T right_temperature,
    T cross_section
) {
    ProblemData1D3V<T> data;
    data.n_x = n_x;
    data.x_left = x_left;
    data.x_right = x_right;
    data.dt = dt;
    data.t_end = t_end;
    data.velocity_grid = VelocityGrid3D<T>(n_v, v_max);
    data.initial_state = DistributionState1D3V<T>(n_x, data.velocity_grid.size());
    data.collision_model = HardSphereCollisionModel<T>(cross_section);

    const Vec3<T> zero_velocity{};
    const auto left_maxwellian = maxwellianCell(data.velocity_grid, left_density, zero_velocity, left_temperature);
    const auto right_maxwellian = maxwellianCell(data.velocity_grid, right_density, zero_velocity, right_temperature);

    for (int x_i = 0; x_i < n_x; ++x_i) {
        const bool is_left = x_i < (n_x / 2);
        const auto& cell_values = is_left ? left_maxwellian : right_maxwellian;
        for (int v_i = 0; v_i < data.velocity_grid.size(); ++v_i) {
            data.initial_state.at(x_i, v_i) = cell_values[v_i];
        }
    }

    return data;
}

template <std::floating_point T>
inline ProblemData1D3V<T> parabolicDensityProblem(
    int n_x,
    int n_v,
    T x_left,
    T x_right,
    T v_max,
    T dt,
    T t_end,
    T density_min,
    T curvature,
    T temperature,
    T cross_section
) {
    ProblemData1D3V<T> data;
    data.n_x = n_x;
    data.x_left = x_left;
    data.x_right = x_right;
    data.dt = dt;
    data.t_end = t_end;
    data.velocity_grid = VelocityGrid3D<T>(n_v, v_max);
    data.initial_state = DistributionState1D3V<T>(n_x, data.velocity_grid.size());
    data.collision_model = HardSphereCollisionModel<T>(cross_section);

    const T dx = (x_right - x_left) / T(n_x);
    const T x_center = T(0.5) * (x_left + x_right);
    const Vec3<T> zero_velocity{};

    for (int x_i = 0; x_i < n_x; ++x_i) {
        const T x = x_left + (T(x_i) + T(0.5)) * dx;
        const T shifted_x = x - x_center;
        const T density = density_min + curvature * shifted_x * shifted_x;
        const auto cell_values = maxwellianCell(data.velocity_grid, density, zero_velocity, temperature);

        for (int v_i = 0; v_i < data.velocity_grid.size(); ++v_i) {
            data.initial_state.at(x_i, v_i) = cell_values[v_i];
        }
    }

    return data;
}

template <std::floating_point T>
inline ProblemData1D3V<T> uniformEquilibriumProblem(
    int n_x,
    int n_v,
    T x_left,
    T x_right,
    T v_max,
    T dt,
    T t_end,
    T density,
    T temperature,
    T cross_section
) {
    ProblemData1D3V<T> data;
    data.n_x = n_x;
    data.x_left = x_left;
    data.x_right = x_right;
    data.dt = dt;
    data.t_end = t_end;
    data.velocity_grid = VelocityGrid3D<T>(n_v, v_max);
    data.initial_state = DistributionState1D3V<T>(n_x, data.velocity_grid.size());
    data.collision_model = HardSphereCollisionModel<T>(cross_section);

    const Vec3<T> zero_velocity{};
    const auto cell_values = maxwellianCell(data.velocity_grid, density, zero_velocity, temperature);

    for (int x_i = 0; x_i < n_x; ++x_i) {
        for (int v_i = 0; v_i < data.velocity_grid.size(); ++v_i) {
            data.initial_state.at(x_i, v_i) = cell_values[v_i];
        }
    }

    return data;
}

template <std::floating_point T>
inline ProblemData1D3V<T> couetteFlowProblem(
    int n_x,
    int n_v,
    T x_left,
    T x_right,
    T v_max,
    T dt,
    T t_end,
    T wall_density,
    T wall_temperature,
    T wall_speed_y,
    T cross_section
) {
    ProblemData1D3V<T> data;
    data.n_x = n_x;
    data.x_left = x_left;
    data.x_right = x_right;
    data.dt = dt;
    data.t_end = t_end;
    data.velocity_grid = VelocityGrid3D<T>(n_v, v_max);
    data.initial_state = DistributionState1D3V<T>(n_x, data.velocity_grid.size());
    data.collision_model = HardSphereCollisionModel<T>(cross_section);

    data.left_wall = { true, wall_density, wall_temperature, { T(0), -wall_speed_y, T(0) }, {} };
    data.right_wall = { true, wall_density, wall_temperature, { T(0), wall_speed_y, T(0) }, {} };
    initializeWallMaxwellian(data.left_wall, data.velocity_grid);
    initializeWallMaxwellian(data.right_wall, data.velocity_grid);

    const T dx = (x_right - x_left) / T(n_x);
    for (int x_i = 0; x_i < n_x; ++x_i) {
        const T x = x_left + (T(x_i) + T(0.5)) * dx;
        const T alpha = (x - x_left) / (x_right - x_left);
        const Vec3<T> bulk_velocity{ T(0), (T(2) * alpha - T(1)) * wall_speed_y, T(0) };
        const auto cell_values = maxwellianCell(data.velocity_grid, wall_density, bulk_velocity, wall_temperature);
        for (int v_i = 0; v_i < data.velocity_grid.size(); ++v_i) {
            data.initial_state.at(x_i, v_i) = cell_values[v_i];
        }
    }

    return data;
}

template <std::floating_point T>
inline ProblemData1D3V<T> poiseuilleFlowProblem(
    int n_x,
    int n_v,
    T x_left,
    T x_right,
    T v_max,
    T dt,
    T t_end,
    T density,
    T wall_temperature,
    T body_force_y,
    T cross_section
) {
    auto data = uniformEquilibriumProblem<T>(
        n_x, n_v, x_left, x_right, v_max, dt, t_end, density, wall_temperature, cross_section
    );

    data.left_wall = { true, density, wall_temperature, { T(0), T(0), T(0) }, {} };
    data.right_wall = { true, density, wall_temperature, { T(0), T(0), T(0) }, {} };
    data.body_force = { T(0), body_force_y, T(0) };
    initializeWallMaxwellian(data.left_wall, data.velocity_grid);
    initializeWallMaxwellian(data.right_wall, data.velocity_grid);

    return data;
}

template <std::floating_point T>
inline ProblemData1D3V<T> heatConductionProblem(
    int n_x,
    int n_v,
    T x_left,
    T x_right,
    T v_max,
    T dt,
    T t_end,
    T wall_density,
    T left_wall_temperature,
    T right_wall_temperature,
    T cross_section
) {
    ProblemData1D3V<T> data;
    data.n_x = n_x;
    data.x_left = x_left;
    data.x_right = x_right;
    data.dt = dt;
    data.t_end = t_end;
    data.velocity_grid = VelocityGrid3D<T>(n_v, v_max);
    data.initial_state = DistributionState1D3V<T>(n_x, data.velocity_grid.size());
    data.collision_model = HardSphereCollisionModel<T>(cross_section);

    data.left_wall = { true, wall_density, left_wall_temperature, { T(0), T(0), T(0) }, {} };
    data.right_wall = { true, wall_density, right_wall_temperature, { T(0), T(0), T(0) }, {} };
    initializeWallMaxwellian(data.left_wall, data.velocity_grid);
    initializeWallMaxwellian(data.right_wall, data.velocity_grid);

    const T dx = (x_right - x_left) / T(n_x);
    for (int x_i = 0; x_i < n_x; ++x_i) {
        const T x = x_left + (T(x_i) + T(0.5)) * dx;
        const T alpha = (x - x_left) / (x_right - x_left);
        const T temperature = (T(1) - alpha) * left_wall_temperature + alpha * right_wall_temperature;
        const auto cell_values = maxwellianCell(data.velocity_grid, wall_density, { T(0), T(0), T(0) }, temperature);
        for (int v_i = 0; v_i < data.velocity_grid.size(); ++v_i) {
            data.initial_state.at(x_i, v_i) = cell_values[v_i];
        }
    }

    return data;
}

template <std::floating_point T>
inline void addCollisionContribution(
    const VelocityGrid3D<T>& grid,
    const HardSphereCollisionModel<T>& collision_model,
    const DistributionState1D3V<T>& state,
    int x_i,
    std::vector<T>& cell_rhs
) {
    const T dv3 = grid.cellVolume();
    const T phase_weight = dv3 * dv3;

    for (int a = 0; a < grid.size(); ++a) {
        const T f_a = state.at(x_i, a);
        if (f_a <= T(0)) {
            continue;
        }

        const Vec3<T>& v = grid.velocities[a];

        for (int b = 0; b < grid.size(); ++b) {
            const T f_b = state.at(x_i, b);
            if (f_b <= T(0)) {
                continue;
            }

            const Vec3<T>& v_star = grid.velocities[b];
            const Vec3<T> g = v - v_star;
            const T g_norm = norm(g);
            if (g_norm <= T(1e-12)) {
                continue;
            }

            for (std::size_t dir_i = 0; dir_i < collision_model.directions.size(); ++dir_i) {
                const Vec3<T>& sigma = collision_model.directions[dir_i];
                const T rate = T(0.5) * collision_model.cross_section * g_norm * f_a * f_b * phase_weight * collision_model.direction_weights[dir_i];

                const Vec3<T> center = T(0.5) * (v + v_star);
                const Vec3<T> post_v = center + T(0.5) * g_norm * sigma;
                const Vec3<T> post_v_star = center - T(0.5) * g_norm * sigma;

                cell_rhs[a] -= rate;
                cell_rhs[b] -= rate;

                depositToVelocityCell(grid, cell_rhs, post_v, rate);
                depositToVelocityCell(grid, cell_rhs, post_v_star, rate);
            }
        }
    }
}

template <std::floating_point T>
inline T boundaryIncomingValue(
    const DiffuseWall1D3V<T>& wall,
    const DistributionState1D3V<T>& state,
    int boundary_cell,
    int v_i
) {
    if (wall.enabled && !wall.incoming_maxwellian.empty()) {
        return wall.incoming_maxwellian[static_cast<std::size_t>(v_i)];
    }
    return state.at(boundary_cell, v_i);
}

template <std::floating_point T>
inline DistributionState1D3V<T> fullBoltzmannRightHandSide(
    const ProblemData1D3V<T>& data,
    const DistributionState1D3V<T>& state
) {
    DistributionState1D3V<T> rhs(state.n_x, state.n_v);
    const T dx = (data.x_right - data.x_left) / T(data.n_x);

#pragma omp parallel for
    for (int x_i = 0; x_i < state.n_x; ++x_i) {
        std::vector<T> cell_rhs(static_cast<std::size_t>(state.n_v), T{});

        addCollisionContribution(data.velocity_grid, data.collision_model, state, x_i, cell_rhs);

        for (int v_i = 0; v_i < state.n_v; ++v_i) {
            const T vx = data.velocity_grid.velocities[v_i].x;
            const int left_cell = (x_i == 0) ? 0 : x_i - 1;
            const int right_cell = (x_i == state.n_x - 1) ? state.n_x - 1 : x_i + 1;

            T left_neighbor = state.at(left_cell, v_i);
            T right_neighbor = state.at(right_cell, v_i);

            if (x_i == 0 && vx > T(0)) {
                left_neighbor = boundaryIncomingValue(data.left_wall, state, 0, v_i);
            }
            if (x_i == state.n_x - 1 && vx < T(0)) {
                right_neighbor = boundaryIncomingValue(data.right_wall, state, state.n_x - 1, v_i);
            }

            const T left_state = (vx >= T(0)) ? left_neighbor : state.at(x_i, v_i);
            const T right_state = (vx >= T(0)) ? state.at(x_i, v_i) : right_neighbor;

            const T left_flux = vx * left_state;
            const T right_flux = vx * right_state;

            T value_rhs = -(right_flux - left_flux) / dx + cell_rhs[v_i];

            const auto [ix, iy, iz] = data.velocity_grid.unflattenIndex(v_i);
            if (std::abs(data.body_force.y) > T(0)) {
                const int prev_iy = (iy == 0) ? 0 : iy - 1;
                const int next_iy = (iy == data.velocity_grid.n_v - 1) ? data.velocity_grid.n_v - 1 : iy + 1;
                const int prev_v = data.velocity_grid.flattenIndex(ix, prev_iy, iz);
                const int next_v = data.velocity_grid.flattenIndex(ix, next_iy, iz);

                const T force_term = (data.body_force.y > T(0))
                    ? -data.body_force.y * (state.at(x_i, v_i) - state.at(x_i, prev_v)) / data.velocity_grid.dv
                    : -data.body_force.y * (state.at(x_i, next_v) - state.at(x_i, v_i)) / data.velocity_grid.dv;
                value_rhs += force_term;
            }

            rhs.at(x_i, v_i) = value_rhs;
        }
    }

    return rhs;
}

template <typename RkMethod, std::floating_point T, typename OutputRule>
inline void fullBoltzmannMethod(
    const ProblemData1D3V<T>& data,
    MacroOutput1D3V<T>& output,
    const OutputRule& output_rule
) {
    DistributionState1D3V<T> state = data.initial_state;
    T final_time = T{};

    for (T t{}; t < data.t_end; t += data.dt) {
        if (output_rule(t)) {
            output.print(t, computeMacroState(state, data.velocity_grid));
        }

        const T step = std::min(data.dt, data.t_end - t);
        state = RkMethod::stepY(
            [&data](const DistributionState1D3V<T>& current_state) {
                return fullBoltzmannRightHandSide(data, current_state);
            },
            state,
            step
        );

        for (T& value : state.values) {
            value = std::max(value, T(0));
        }

        final_time = t + step;
    }

    output.print(final_time, computeMacroState(state, data.velocity_grid));
    writeDistributionSlices(output.outputFolder(), state, data.velocity_grid, final_time);
}

} // namespace full_boltzmann_1d3v
