#pragma once

#ifndef _SharkCsvm_h_
#define _SharkCsvm_h_

// =================================================================================================

#include "CoreTypes/Export/CompilerDefines.h"

// =================================================================================================

#if defined(MCompiler_VisualCPP)
  #pragma MAddLibrary("Shark")
  #pragma MAddLibrary("BoostSerialization")
  #pragma MAddLibrary("BoostSystem")

  #if defined(MRelease)
    #pragma MAddLibrary("OpenBLAS")
  #endif

  #pragma warning (push)
  #pragma warning (disable: 4800 4702 4297 4267 4244 4189 4100)

  // avoid #pragma library for boost libs
  #define BOOST_ALL_NO_LIB

#elif defined(MCompiler_Clang)
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  #pragma clang diagnostic ignored "-Wshift-count-overflow"
#endif

// =================================================================================================

#if defined(MCompiler_VisualCPP) && defined(MRelease)
  #define SHARK_USE_OPENMP 1
  #define SHARK_USE_CBLAS 1
#endif

#include <shark/Data/Dataset.h>
#include <shark/Data/Csv.h>

#include <shark/Models/LinearModel.h>
#include <shark/Models/Normalizer.h>

#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/MonomialKernel.h>
#include <shark/Models/Kernels/PolynomialKernel.h>

#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitInterval.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>

#include <shark/Algorithms/JaakkolaHeuristic.h>

#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

// =================================================================================================

#if defined(MCompiler_VisualCPP)
  #pragma warning(pop)

#elif defined(MCompiler_Clang)
  #pragma clang diagnostic pop
#endif


#endif // _SharkCsvm_h_

