#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "BGKChannel1D.h"
#include "FullBoltzmann1D3V.h"
#include "OpenMpCompat.h"
#include "rkMethods.h"

namespace {

constexpr int kRequestedThreadCount = 20;

enum class Scenario {
    Couette,
    Poiseuille,
    HeatConduction,
    FullCouette,
    FullPoiseuille,
    FullHeatConduction
};

template <std::floating_point T>
auto everyNthStepOutputRule(T dt, int stepInterval) {
    return [dt, stepInterval](T t) {
        return stepInterval > 0 && static_cast<int>(t / dt + T(0.5)) % stepInterval == 0;
    };
}

void writeConvergenceHistory(
    const std::filesystem::path& outputDir,
    const std::vector<double>& convergenceHistory
) {
    std::ofstream convergenceOutput(outputDir / "convergence_history.txt");
    for (std::size_t i = 0; i < convergenceHistory.size(); ++i) {
        convergenceOutput << i << ' ' << convergenceHistory[i] << '\n';
    }
}

void plotAndOpenChannelResults(const std::filesystem::path& outputDir) {
    const std::string profilePlotCommand =
        "python3 scripts/plot_bgk_channel.py \"" + outputDir.string() + "\"";
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

void plotFullBoltzmannResults(
    const std::string& caseName,
    const std::filesystem::path& fullOutputDir
) {
    const std::filesystem::path bgkOutputDir = std::filesystem::path("output/bgk_channel") / caseName;
    const std::string command =
        "python3 scripts/plot_bgk_full_comparison.py \"" + caseName + "\" \"" +
        bgkOutputDir.string() + "\" \"" + fullOutputDir.string() + "\"";

    const int status = std::system(command.c_str());
    if (status != 0) {
        std::cerr << "BGK/full comparison plot generation failed with status " << status << '\n';
    }
}

void configureOpenMp() {
    const int availableThreads = omp_get_max_threads();
    omp_set_num_threads(std::min(kRequestedThreadCount, availableThreads));
    std::cout << "num_threads: " << omp_get_max_threads() << '\n';
}

void runCouetteCase(
    const std::filesystem::path& outputDir = "output/bgk_channel/couette",
    bool generatePlot = true
) {
    using namespace bgk_channel;

    const auto data = couetteProblem<double>(
        41,
        16,
        0.0,
        1.0,
        4.0,
        1.0,
        1.0,
        0.9,
        0.5,
        240,
        1e-7
    );

    std::vector<double> convergenceHistory;
    const ChannelState<double> state = solveSteadyChannelBGK(data, convergenceHistory);
    writeChannelOutput(outputDir, data, state);
    writeConvergenceHistory(outputDir, convergenceHistory);

    if (generatePlot) {
        plotAndOpenChannelResults(outputDir);
    }
}

void runPoiseuilleCase(
    int n_y = 41,
    const std::filesystem::path& outputDir = "output/bgk_channel/poiseuille",
    bool generatePlot = true
) {
    using namespace bgk_channel;

    const auto data = poiseuilleProblem<double>(
        n_y,
        16,
        0.0,
        1.0,
        4.0,
        1.0,
        1.0,
        0.35,
        0.5,
        240,
        1e-7
    );

    std::vector<double> convergenceHistory;
    const ChannelState<double> state = solveSteadyChannelBGK(data, convergenceHistory);
    writeChannelOutput(outputDir, data, state);
    writeConvergenceHistory(outputDir, convergenceHistory);

    if (generatePlot) {
        plotAndOpenChannelResults(outputDir);
    }
}

void runHeatConductionCase(
    const std::filesystem::path& outputDir = "output/bgk_channel/heat_conduction",
    bool generatePlot = true
) {
    using namespace bgk_channel;

    const auto data = heatConductionProblem<double>(
        41,
        16,
        0.0,
        1.0,
        4.0,
        1.0,
        0.5,
        1.1,
        0.5,
        240,
        1e-7
    );

    std::vector<double> convergenceHistory;
    const ChannelState<double> state = solveSteadyChannelBGK(data, convergenceHistory);
    writeChannelOutput(outputDir, data, state);
    writeConvergenceHistory(outputDir, convergenceHistory);

    if (generatePlot) {
        plotAndOpenChannelResults(outputDir);
    }
}

void runFullCouetteCase(
    const std::filesystem::path& outputDir = "output/full_boltzmann_1d3v/couette",
    bool generatePlot = true
) {
    using namespace full_boltzmann_1d3v;

    constexpr double dt = 0.002;
    constexpr double tEnd = 0.04;
    const auto data = couetteFlowProblem<double>(
        31,
        6,
        0.0,
        1.0,
        4.0,
        dt,
        tEnd,
        1.0,
        1.0,
        0.9,
        0.01
    );

    MacroOutput1D3V<double> output(outputDir);
    fullBoltzmannMethod<ExplicitEulerRK>(data, output, everyNthStepOutputRule(dt, 5));
    output.close();

    if (generatePlot) {
        runCouetteCase("output/bgk_channel/couette", false);
        plotFullBoltzmannResults("couette", outputDir);
    }
}

void runFullPoiseuilleCase(
    const std::filesystem::path& outputDir = "output/full_boltzmann_1d3v/poiseuille",
    bool generatePlot = true
) {
    using namespace full_boltzmann_1d3v;

    constexpr double dt = 0.002;
    constexpr double tEnd = 0.04;
    const auto data = poiseuilleFlowProblem<double>(
        31,
        6,
        0.0,
        1.0,
        4.0,
        dt,
        tEnd,
        1.0,
        1.0,
        0.12,
        0.01
    );

    MacroOutput1D3V<double> output(outputDir);
    fullBoltzmannMethod<ExplicitEulerRK>(data, output, everyNthStepOutputRule(dt, 5));
    output.close();

    if (generatePlot) {
        runPoiseuilleCase(41, "output/bgk_channel/poiseuille", false);
        plotFullBoltzmannResults("poiseuille", outputDir);
    }
}

void runFullHeatConductionCase(
    const std::filesystem::path& outputDir = "output/full_boltzmann_1d3v/heat_conduction",
    bool generatePlot = true
) {
    using namespace full_boltzmann_1d3v;

    constexpr double dt = 0.002;
    constexpr double tEnd = 0.04;
    const auto data = heatConductionProblem<double>(
        31,
        6,
        0.0,
        1.0,
        4.0,
        dt,
        tEnd,
        1.0,
        0.5,
        1.1,
        0.01
    );

    MacroOutput1D3V<double> output(outputDir);
    fullBoltzmannMethod<ExplicitEulerRK>(data, output, everyNthStepOutputRule(dt, 5));
    output.close();

    if (generatePlot) {
        runHeatConductionCase("output/bgk_channel/heat_conduction", false);
        plotFullBoltzmannResults("heat_conduction", outputDir);
    }
}

void runScenario(Scenario scenario) {
    switch (scenario) {
    case Scenario::Couette:
        runCouetteCase();
        break;
    case Scenario::Poiseuille:
        runPoiseuilleCase();
        break;
    case Scenario::HeatConduction:
        runHeatConductionCase();
        break;
    case Scenario::FullCouette:
        runFullCouetteCase();
        break;
    case Scenario::FullPoiseuille:
        runFullPoiseuilleCase();
        break;
    case Scenario::FullHeatConduction:
        runFullHeatConductionCase();
        break;
    }
}

} // namespace

int main(int argc, char* argv[]) {
    configureOpenMp();

    if (argc >= 2) {
        const std::string_view mode(argv[1]);
        if (mode == "couette") {
            std::filesystem::path outputDir = "output/bgk_channel/couette";
            bool generatePlot = true;

            if (argc >= 3) {
                outputDir = argv[2];
            }
            if (argc >= 4 && std::string_view(argv[3]) == "--no-plot") {
                generatePlot = false;
            }

            runCouetteCase(outputDir, generatePlot);
            return EXIT_SUCCESS;
        }
        if (mode == "poiseuille") {
            int n_y = 41;
            std::filesystem::path outputDir = "output/bgk_channel/poiseuille";
            bool generatePlot = true;

            if (argc >= 3) {
                n_y = std::stoi(argv[2]);
            }
            if (argc >= 4) {
                outputDir = argv[3];
            }
            if (argc >= 5 && std::string_view(argv[4]) == "--no-plot") {
                generatePlot = false;
            }

            runPoiseuilleCase(n_y, outputDir, generatePlot);
            return EXIT_SUCCESS;
        }
        if (mode == "heat-conduction") {
            std::filesystem::path outputDir = "output/bgk_channel/heat_conduction";
            bool generatePlot = true;

            if (argc >= 3) {
                outputDir = argv[2];
            }
            if (argc >= 4 && std::string_view(argv[3]) == "--no-plot") {
                generatePlot = false;
            }

            runHeatConductionCase(outputDir, generatePlot);
            return EXIT_SUCCESS;
        }
        if (mode == "full-couette") {
            std::filesystem::path outputDir = "output/full_boltzmann_1d3v/couette";
            bool generatePlot = true;

            if (argc >= 3) {
                outputDir = argv[2];
            }
            if (argc >= 4 && std::string_view(argv[3]) == "--no-plot") {
                generatePlot = false;
            }

            runFullCouetteCase(outputDir, generatePlot);
            return EXIT_SUCCESS;
        }
        if (mode == "full-poiseuille") {
            std::filesystem::path outputDir = "output/full_boltzmann_1d3v/poiseuille";
            bool generatePlot = true;

            if (argc >= 3) {
                outputDir = argv[2];
            }
            if (argc >= 4 && std::string_view(argv[3]) == "--no-plot") {
                generatePlot = false;
            }

            runFullPoiseuilleCase(outputDir, generatePlot);
            return EXIT_SUCCESS;
        }
        if (mode == "full-heat-conduction") {
            std::filesystem::path outputDir = "output/full_boltzmann_1d3v/heat_conduction";
            bool generatePlot = true;

            if (argc >= 3) {
                outputDir = argv[2];
            }
            if (argc >= 4 && std::string_view(argv[3]) == "--no-plot") {
                generatePlot = false;
            }

            runFullHeatConductionCase(outputDir, generatePlot);
            return EXIT_SUCCESS;
        }
    }

    runScenario(Scenario::Couette);
    return EXIT_SUCCESS;
}
