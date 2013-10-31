/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
 // OpenCL Kernel Function for element by element vector addition
__kernel void VectorAdd(__global const float* a, __global const float* b, __global float* c, int iNumElements, __global uint *indexDst)
{
    // get index into global data array
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
 //   if (iGID >= iNumElements)
 //   {   
 //       return; 
 //   }
    
    // add the vector elements
    //c[iGID] = a[iGID] + b[iGID];
    c[iGID] = 12.0f;
    
    if  (iGID % 3 == 0) 
    {
	indexDst[iGID] = iGID/3;  
	indexDst[iGID+1] = iGID/3;  
	indexDst[iGID+2] = iGID/3;  
    }
}
