#pragma once

#include <concepts>

namespace time_integration
{

    template <std::floating_point T, typename Y, typename F>
    inline Y step(Y y, T dt, F rhs)
    {
        return y + dt * rhs(y);
    }

    template <std::floating_point T, typename Y>
    inline Y stepFromSlope(Y y, T dt, Y slope)
    {
        return y + dt * slope;
    }

}
