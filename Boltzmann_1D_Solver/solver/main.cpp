#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "BGKChannel1D.h"
#include "OpenMpCompat.h"

namespace
{

    constexpr int kRequestedThreadCount = 20;

    void generateDistributionPlot(const std::filesystem::path &outputDir)
    {
        const std::string plotCommand =
            "python3 scripts/plot_distribution_x_v.py \"" + outputDir.string() + "\"";
        const int plotStatus = std::system(plotCommand.c_str());
        if (plotStatus != 0)
        {
            std::cerr << "distribution plot generation failed with status " << plotStatus << '\n';
        }
    }

    void generateDistributionAnimation(const std::filesystem::path &outputDir)
    {
        const std::string animationCommand =
            "python3 scripts/animate_distribution_x_v.py \"" + outputDir.string() + "\"";
        const int animationStatus = std::system(animationCommand.c_str());
        if (animationStatus != 0)
        {
            std::cerr << "distribution animation generation failed with status " << animationStatus << '\n';
        }
    }

    void configureOpenMp()
    {
        const int availableThreads = omp_get_max_threads();
        omp_set_num_threads(std::min(kRequestedThreadCount, availableThreads));
        std::cout << "num_threads: " << omp_get_max_threads() << '\n';
    }

    void runNormalFlow(
        const std::filesystem::path &outputDir = "output/bgk_channel",
        bool generatePlot = true)
    {
        using namespace bgk_channel;

        // User-chosen computational domain and resolution for the reduced BGK model:
        //   x in [xL, xR] = [0.0, 1.0]
        //   c in [cmin, cmax] = [-5.0, 5.0]
        // The first two integers are the numbers of x-cells and velocity cells.
        auto data = normalFlowProblem<double>(
            81,
            64,
            0.0,
            1.0,
            5.0,
            1.0,
            0.7,
            1.3,
            400,
            1e-7);

        // Positive body force adds a visible right-going drift so the distribution motion
        // is easier to see in the x-c plots and animation.
        data.body_force = 0.4;
        // Animate over a longer window so the drift is visibly separated between frames.
        data.final_time = 8.0;
        data.snapshot_interval = 20;

        std::filesystem::remove_all(outputDir);
        const std::filesystem::path snapshotDir = outputDir / "snapshots";

        const ChannelState<double> state = solveSteadyChannelBGK(data, &snapshotDir);
        writeChannelOutput(outputDir, data, state);

        if (generatePlot)
        {
            generateDistributionPlot(outputDir);
            generateDistributionAnimation(outputDir);
        }
    }

}

int main(int argc, char *argv[])
{
    configureOpenMp();

    std::filesystem::path outputDir = "output/bgk_channel";
    bool generatePlot = true;

    if (argc >= 2)
    {
        outputDir = argv[1];
    }
    if (argc >= 3 && std::string(argv[2]) == "--no-plot")
    {
        generatePlot = false;
    }

    runNormalFlow(outputDir, generatePlot);
    return EXIT_SUCCESS;
}
