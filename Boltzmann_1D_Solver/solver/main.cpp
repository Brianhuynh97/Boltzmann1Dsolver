#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "BoltzmannEq1DProblems.h"
#include "BoltzmannEq1DSolvers.h"
#include "FullBoltzmann1D3V.h"
#include "LinearizedBGKChannel1D.h"
#include "MacroparametersFileInterface.h"
#include "Mkt.h"
#include "OpenMpCompat.h"
#include "rkMethods.h"

namespace {

constexpr int kRequestedThreadCount = 20;

enum class Scenario {
    FullBoltzmannCouette,
    FullBoltzmannPoiseuille,
    FullBoltzmannHeatConduction,
    FullBoltzmannUniformEquilibrium,
    Sod,
    Density10,
    StudentSpringDensity,
    FreeMoleculeDensity,
    EmittingWall,
    EvaporatingWall
};

template <std::floating_point T>
auto everyNthStepOutputRule(T dt, int stepInterval) {
    return [dt, stepInterval](T t) {
        return stepInterval > 0 && static_cast<int>(std::lround(t / dt)) % stepInterval == 0;
    };
}

void plotAndOpenFullBoltzmannResults(const std::filesystem::path& outputDir) {
    const std::string summaryPlotCommand =
        "python3 scripts/plot_full_boltzmann_shock_tube.py \"" + outputDir.string() + "\"";

    const int summaryPlotStatus = std::system(summaryPlotCommand.c_str());
    if (summaryPlotStatus != 0) {
        std::cerr << "summary plot generation failed with status " << summaryPlotStatus << '\n';
        return;
    }

    const std::string distributionPlotCommand =
        "python3 scripts/plot_distribution_contour.py \"" + outputDir.string() + "\"";
    const int distributionPlotStatus = std::system(distributionPlotCommand.c_str());
    if (distributionPlotStatus != 0) {
        std::cerr << "distribution contour generation failed with status " << distributionPlotStatus << '\n';
    }

    const std::filesystem::path plotPath = outputDir / "summary_plot.png";

#if defined(__APPLE__)
    const std::string openCommand = "open \"" + plotPath.string() + "\"";
    const int openStatus = std::system(openCommand.c_str());
    if (openStatus != 0) {
        std::cerr << "plot was generated but could not be opened automatically: " << plotPath << '\n';
    }
#else
    std::cout << "plot saved to " << plotPath << '\n';
#endif
}

void plotAndOpenChannelCouetteResults(const std::filesystem::path& outputDir) {
    const std::string profilePlotCommand =
        "python3 scripts/plot_linearized_bgk_channel.py \"" + outputDir.string() + "\"";
    const int profilePlotStatus = std::system(profilePlotCommand.c_str());
    if (profilePlotStatus != 0) {
        std::cerr << "channel profile generation failed with status " << profilePlotStatus << '\n';
    }

    const std::string distributionPlotCommand =
        "python3 scripts/plot_distribution_contour.py \"" + outputDir.string() + "\"";
    const int distributionPlotStatus = std::system(distributionPlotCommand.c_str());
    if (distributionPlotStatus != 0) {
        std::cerr << "distribution contour generation failed with status " << distributionPlotStatus << '\n';
    }

    const std::filesystem::path plotPath = outputDir / "channel_profile.png";

#if defined(__APPLE__)
    const std::string openCommand = "open \"" + plotPath.string() + "\"";
    const int openStatus = std::system(openCommand.c_str());
    if (openStatus != 0) {
        std::cerr << "plot was generated but could not be opened automatically: " << plotPath << '\n';
    }
#else
    std::cout << "plot saved to " << plotPath << '\n';
#endif
}

void configureOpenMp() {
    const int availableThreads = omp_get_max_threads();
    omp_set_num_threads(std::min(kRequestedThreadCount, availableThreads));
    std::cout << "num_threads: " << omp_get_max_threads() << '\n';
}

void runStudentSpringDensityTest() {
    constexpr double tEnd = 5.0;
    constexpr double dt = 0.001;

    const auto data = densityRiemannProblem<double>(1.0, 2.0, 1.0, 150, 5.0, 45, 4.0, dt, tEnd, 100.0);
    Full1dStateOutput output("output/bgk1d/testDensityMultiple");

    bgk1dMethod<ExplicitEulerRK>(data, Cell1stOrderInt, output, everyNthStepOutputRule(dt, 100));
}

void runFreeMoleculeDensityTest() {
    constexpr double tEnd = 10.0;
    constexpr double dt = 0.001;

    const auto data = densityRiemannProblem<double>(1.0, 2.0, 1.0, 150, 5.0, 224, 4.0, dt, tEnd, 0.0, 1.0);
    Full1dStateOutput output("output/bgk1d/testFreeMoleculeDensityShifted");

    bgk1dMethod<ExplicitEulerRK>(data, Cell1stOrderInt, output, everyNthStepOutputRule(dt, 100));
}

void runEmittingWallTest() {
    constexpr double tEnd = 10.0;
    constexpr double dt = 0.001;

    const auto data = emittingWallProblem<double>(
        1.0, 1.0,
        1.5, 2.0,
        400, 12.5,
        50, 6.0,
        dt, tEnd,
        100.0
    );

    const auto outputRule = everyNthStepOutputRule(dt, 100);

    Full1dStateOutput outputBall("output/bgk1d/testEmittingWallBall");
    bgk1dMethod<ExplicitEulerRK>(data, Cell1stOrderInt, outputBall, outputRule, ballMoleculeViscosityRule);

    Full1dStateOutput outputMaxwell("output/bgk1d/testEmittingWallMaxwell");
    bgk1dMethod<ExplicitEulerRK>(data, Cell1stOrderInt, outputMaxwell, outputRule, maxwellMoleculeViscosityRule);
}

void runEvaporatingWallTest() {
    constexpr double tEnd = 5.0;
    constexpr double dt = 0.0001;

    const int stepCount = static_cast<int>(std::lround((tEnd + 2 * dt) / dt));
    const auto outputRule = everyNthStepOutputRule(dt, stepCount / 5);

    const auto data = evaporatingWallProblem<double>(
        1.0, 1.0,
        2.0, 10.0,
        300, 15.0,
        100, 9.0,
        dt, tEnd,
        knToDelta(0.01)
    );

    Full1dStateOutput output("output/bgk1d/testEvaporatingWallMaxwell4");
    bgk1dMethod<ExplicitEulerRK>(data, simpsonInt, output, outputRule, maxwellMoleculeViscosityRule);
}

void runDensity10Test() {
    const double tEnd = 0.2 * std::sqrt(2.0);
    constexpr double dt = 0.00002;

    const int stepCount = static_cast<int>(std::lround(tEnd / dt));
    const auto outputRule = everyNthStepOutputRule(dt, stepCount / 100);

    const auto data = densityRiemannProblem<double>(
        1.0, 1.0, 0.125,
        150 * 5, 0.0, 1.0, 45 * 8, 8.0,
        dt, tEnd,
        10000.0
    );

    Full1dStateOutput output("output/bgk1d/testDensity10Multiple");
    bgk1dMethod<ExplicitEulerRK>(data, simpsonInt, output, outputRule);
}

void runSodTest() {
    const double tEnd = 0.2 * std::sqrt(2.0);
    constexpr double dt = 0.00002;

    const int stepCount = static_cast<int>(std::lround(tEnd / dt));
    const auto outputRule = everyNthStepOutputRule(dt, stepCount / 100);

    const auto data = densityTemperatureRiemannProblem<double>(
        1.0, 1.0,
        0.8, 0.125,
        150 * 3, 0.0, 1.0, 45 * 4, 8.0,
        dt, tEnd,
        10000.0
    );

    Full1dStateOutput output("output/bgk1d/SOD_1");
    bgk1dMethod<ExplicitEulerRK>(data, simpsonInt, output, outputRule);
}

void runFullBoltzmannUniformEquilibriumTest(
    int n_x = 33,
    const std::filesystem::path& outputDir = "output/full_boltzmann_1d3v/uniform_equilibrium",
    bool generatePlot = true
) {
    using namespace full_boltzmann_1d3v;

    constexpr double dt = 0.0025;
    constexpr double tEnd = 0.02;

    const auto data = uniformEquilibriumProblem<double>(
        n_x,
        5,      // n_v per axis => 125 velocity nodes
        0.0, 1.0,
        4.0,
        dt, tEnd,
        1.0,    // uniform density
        0.8,    // uniform temperature
        0.02
    );

    MacroOutput1D3V<double> output(outputDir);
    fullBoltzmannMethod<ExplicitEulerRK>(data, output, everyNthStepOutputRule(dt, 1));
    output.close();
    if (generatePlot) {
        plotAndOpenFullBoltzmannResults(outputDir);
    }
}

void runFullBoltzmannCouetteTest() {
    using namespace linearized_bgk_channel;

    CouetteProblemData<double> data;
    data.n_y = 41;
    data.y_min = 0.0;
    data.y_max = 1.0;
    data.velocity_grid = VelocityGrid3D<double>(11, 3.5);
    data.wall_density = 1.0;
    data.wall_temperature = 1.0;
    data.wall_speed = 0.9;
    data.tau = 0.5;
    data.max_iterations = 120;
    data.tolerance = 1e-7;

    const ChannelState<double> state = solveCouetteSteadyState(data);
    const std::filesystem::path outputDir = "output/linearized_bgk_channel/couette";
    writeCouetteOutput(outputDir, data, state);
    plotAndOpenChannelCouetteResults(outputDir);
}

void runFullBoltzmannPoiseuilleTest(
    int n_x = 33,
    const std::filesystem::path& outputDir = "output/full_boltzmann_1d3v/poiseuille",
    bool generatePlot = true
) {
    using namespace full_boltzmann_1d3v;

    constexpr double dt = 0.0025;
    constexpr double tEnd = 0.03;

    const auto data = poiseuilleFlowProblem<double>(
        n_x,
        5,
        0.0, 1.0,
        4.0,
        dt, tEnd,
        1.0,
        0.8,
        0.12,
        0.02
    );

    MacroOutput1D3V<double> output(outputDir);
    fullBoltzmannMethod<ExplicitEulerRK>(data, output, everyNthStepOutputRule(dt, 1));
    output.close();
    if (generatePlot) {
        plotAndOpenFullBoltzmannResults(outputDir);
    }
}

void runFullBoltzmannHeatConductionTest() {
    using namespace full_boltzmann_1d3v;

    constexpr double dt = 0.0025;
    constexpr double tEnd = 0.03;

    const auto data = heatConductionProblem<double>(
        33,
        5,
        0.0, 1.0,
        4.0,
        dt, tEnd,
        1.0,
        0.5,
        1.1,
        0.02
    );

    const std::filesystem::path outputDir = "output/full_boltzmann_1d3v/heat_conduction";
    MacroOutput1D3V<double> output(outputDir);
    fullBoltzmannMethod<ExplicitEulerRK>(data, output, everyNthStepOutputRule(dt, 1));
    output.close();
    plotAndOpenFullBoltzmannResults(outputDir);
}

void runFullBoltzmannBoundaryScenario(Scenario scenario) {
    switch (scenario) {
    case Scenario::FullBoltzmannCouette:
        runFullBoltzmannCouetteTest();
        break;
    case Scenario::FullBoltzmannPoiseuille:
        runFullBoltzmannPoiseuilleTest();
        break;
    case Scenario::FullBoltzmannHeatConduction:
        runFullBoltzmannHeatConductionTest();
        break;
    case Scenario::FullBoltzmannUniformEquilibrium:
        runFullBoltzmannUniformEquilibriumTest();
        break;
    default:
        break;
    }
}

void runScenario(Scenario scenario) {
    switch (scenario) {
    case Scenario::FullBoltzmannCouette:
    case Scenario::FullBoltzmannPoiseuille:
    case Scenario::FullBoltzmannHeatConduction:
    case Scenario::FullBoltzmannUniformEquilibrium:
        runFullBoltzmannBoundaryScenario(scenario);
        break;
    case Scenario::Sod:
        runSodTest();
        break;
    case Scenario::Density10:
        runDensity10Test();
        break;
    case Scenario::StudentSpringDensity:
        runStudentSpringDensityTest();
        break;
    case Scenario::FreeMoleculeDensity:
        runFreeMoleculeDensityTest();
        break;
    case Scenario::EmittingWall:
        runEmittingWallTest();
        break;
    case Scenario::EvaporatingWall:
        runEvaporatingWallTest();
        break;
    }
}

} // namespace

int main(int argc, char* argv[]) {
    configureOpenMp();

    if (argc >= 2) {
        const std::string_view mode(argv[1]);
        if (mode == "couette") {
            runFullBoltzmannCouetteTest();
            return EXIT_SUCCESS;
        }
        if (mode == "poiseuille") {
            int n_x = 33;
            std::filesystem::path outputDir = "output/full_boltzmann_1d3v/poiseuille";
            bool generatePlot = true;

            if (argc >= 3) {
                n_x = std::stoi(argv[2]);
            }
            if (argc >= 4) {
                outputDir = argv[3];
            }
            if (argc >= 5 && std::string_view(argv[4]) == "--no-plot") {
                generatePlot = false;
            }

            runFullBoltzmannPoiseuilleTest(n_x, outputDir, generatePlot);
            return EXIT_SUCCESS;
        }
        if (mode == "heat-conduction") {
            runFullBoltzmannHeatConductionTest();
            return EXIT_SUCCESS;
        }
        if (mode == "uniform-equilibrium") {
            int n_x = 33;
            std::filesystem::path outputDir = "output/full_boltzmann_1d3v/uniform_equilibrium";
            bool generatePlot = true;

            if (argc >= 3) {
                n_x = std::stoi(argv[2]);
            }
            if (argc >= 4) {
                outputDir = argv[3];
            }
            if (argc >= 5 && std::string_view(argv[4]) == "--no-plot") {
                generatePlot = false;
            }

            runFullBoltzmannUniformEquilibriumTest(n_x, outputDir, generatePlot);
            return EXIT_SUCCESS;
        }
    }

    runScenario(Scenario::FullBoltzmannCouette);
    return EXIT_SUCCESS;
}
