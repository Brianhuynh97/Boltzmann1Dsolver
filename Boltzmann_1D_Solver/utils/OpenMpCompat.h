#pragma once

#if defined(BOLTZMANN_HAS_OPENMP) || defined(_OPENMP)
#include <omp.h>
#else
inline void omp_set_num_threads(int) {}

inline int omp_get_max_threads() {
    return 1;
}
#endif
