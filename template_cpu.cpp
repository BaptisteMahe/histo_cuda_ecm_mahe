#include <stdio.h>
/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////

#include "common.h"

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param nsteps     nombre de pas de calcul
//! @param outpi      output of the computed value
////////////////////////////////////////////////////////////////////////////////
void
computeGold(const unsigned int nsteps)
{
    unsigned int i;
    DTYPE step, sum_ref = 0.0;
    step = (1.0)/((DTYPE)nsteps);

    for (i = 0; i < nsteps; ++i) {
      DTYPE x = ((DTYPE)i+0.5)*step;
      sum_ref += 1.0 / (1.0 + x * x);
    }

    printf("Pi ref CPU : %.10lf \n", 4.0 * step * sum_ref) ;

}
