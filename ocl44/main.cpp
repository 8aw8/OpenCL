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

// *********************************************************************
// oclVectorAdd Notes:
//
// A simple OpenCL API demo application that implements
// element by element vector addition between 2 float arrays.
//
// Runs computations with OpenCL on the GPU device and then checks results
// against basic host CPU/C++ computation.
//
// Uses some 'shr' and 'ocl' functions from oclUtils and shrUtils libraries for
// compactness, but these are NOT required libs for OpenCL developement in general.
// *********************************************************************

// common SDK header for standard utilities and system libs
#include "CL/cl.h"
//#include <oclUtils.h>
#include <shrQATest.h>
#include <textfile.h>


// Name of the file with the source code for the computation kernel
// *********************************************************************
const char* cSourceFile = "VectorAdd.cl";

// Host buffers for demo
// *********************************************************************
//void *srcA, *srcB, *dst;        // Host buffers for OpenCL test
cl_float *srcA, *srcB, *dst;
cl_int *index_dst;
void* Golden;
// Host buffer for host golden processing cross check

clock_t interval;

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_mem cmDevSrcA;               // OpenCL device source buffer A
cl_mem cmDevSrcB;               // OpenCL device source buffer B
cl_mem cmDevDst;                // OpenCL device destination buffer
cl_mem cmIndexDst;
size_t szGlobalWorkSize;        // 1D var for Total # of work items
size_t szLocalWorkSize;		    // 1D var for # of work items in the work group
size_t szParmDataBytes;			// Byte size of context information
size_t szKernelLength;			// Byte size of kernel code
cl_int ciErr1, ciErr2;			// Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
const char* cExecutableName = NULL;
char* cSourceCL = NULL;         // Buffer to hold source for compilation


// demo config vars
//int iNumElements = 11444777;	// Length of float arrays to process (odd # for illustration)
int iNumElements =   33334;	// Length of float arrays to process (odd # for illustration)
bool bNoPrompt = false;

size_t shrRoundUp_(int group_size, int global_size)
{
    int r = global_size % group_size;
    if(r == 0)
    {
        return global_size;
    } else
    {
        return global_size + group_size - r;
    }
}

void shrFillArray_(float* pfData, int iSize)
{
    int i;
    const float fScale = 1.0f / (float)RAND_MAX;
    for (i = 0; i < iSize; ++i)
    {
        //pfData[i] = fScale * rand();
        pfData[i] = 2;

    }
}

void uintFillArray_(int* pfData, int iSize)
{
    for (int i = 0; i < iSize; ++i)
    {
        //pfData[i] = fScale * rand();
        pfData[i] = 8;

    }
}

/*

void getSelfPath() {
    char path[PATH_MAX + 1] = { 0 };

#if defined(__linux__)

    ssize_t len = readlink("/proc/self/exe", path, PATH_MAX);
    path[len]   = 0;
    char* p = strrchr(path, '/');
    if(p) *(p + 1)  = 0;
    else path[0]    = 0;

    m_pathmain  = path;

#elif defined(__MACOSX__)

    CFBundleRef mainBundle  = CFBundleGetMainBundle();
    CFURLRef executable     = CFBundleCopyExecutableURL(mainBundle);
    CFStringRef string      = CFURLCopyFileSystemPath(executable, kCFURLPOSIXPathStyle);
    CFStringGetCString(string, path, PATH_MAX, kCFStringEncodingUTF8);
    CFRelease(string);
    CFRelease(executable);
    CFRelease(mainBundle);

    // go up two levels
    m_pathmain  = path;
    size_t i    = m_pathmain.find_last_of('/');
    m_pathmain.erase(i, m_pathmain.length() - i);
    i   = m_pathmain.find_last_of('/');
    m_pathmain.erase(i, m_pathmain.length() - i);
    m_pathmain  += "/Resources/";

#else

    GetModuleFileName(NULL, path, PATH_MAX);
    char* p = strrchr(path, '\\');
    if(p) *(p + 1)  = 0;
    else path[0]    = 0;

    m_pathmain  = path;

#endif
}

*/

// Forward Declarations
// *********************************************************************
void Cleanup (int argc, char **argv, int iExitCode);

// Main function
// *********************************************************************
int main(int argc, char **argv)
{
  //  shrQAStart(argc, argv);

    // start logs
    printf("%s Starting...\n\n# of float elements per Array \t= %i\n", argv[0], iNumElements);

    // set and log Global and Local work size dimensions
    //szLocalWorkSize = 256;
    szLocalWorkSize = 128;
    szGlobalWorkSize = shrRoundUp_((int)szLocalWorkSize, iNumElements);  // rounded up to the nearest multiple of the LocalWorkSize
    printf("Global Work Size \t\t= %d\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n",
           (int)szGlobalWorkSize, (int)szLocalWorkSize, (int)(szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize));

    // Allocate and initialize host arrays
    printf( "Allocate and Init Host Mem...\n");
       srcA = new cl_float[szGlobalWorkSize];  //srcA = (void *)malloc(sizeof(cl_float) * szGlobalWorkSize);
       srcB = new cl_float[szGlobalWorkSize];  //srcB = (void *)malloc(sizeof(cl_float) * szGlobalWorkSize);
        dst = new cl_float[szGlobalWorkSize];  //dst = (void *)malloc(sizeof(cl_float) * szGlobalWorkSize);
  index_dst = new cl_int[szGlobalWorkSize];

    shrFillArray_((float*)srcA, iNumElements);
    shrFillArray_((float*)srcB, iNumElements);
    uintFillArray_((cl_int*)index_dst, iNumElements);

    //Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);

    printf("clGetPlatformID...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    //Get the devices
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
    printf("clGetDeviceIDs...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    //Create the context
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
    printf("clCreateContext...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
    printf("clCreateCommandQueue...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    interval = clock();
    printf("%f", (double)interval/CLOCKS_PER_SEC);


    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    cmDevSrcA = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErr1);
    cmDevSrcB = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmDevDst = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErr2);
    ciErr1 |= ciErr2;    
    cmIndexDst = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_int) * szGlobalWorkSize, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    printf("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Read the OpenCL kernel in from source file

    char *cSourceCL = textFileRead("/home/aw/data/projects/ocl44/VectorAdd.cl");

    printf("oclLoadProgSource \n %s \n", cSourceCL);

    szKernelLength = strlen(cSourceCL);

    printf("szKernelLength = %d \n", (int)szKernelLength);

      // Create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErr1);
    printf("clCreateProgramWithSource...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Build the program with 'mad' Optimization option
    #ifdef MAC
        char* flags = "-cl-fast-relaxed-math -DMAC";
    #else
        {
            char flags__[23] =  "-cl-fast-relaxed-math\n";
            char* flags;
            strcpy(flags,flags__);
        }
    #endif

        ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    printf("clBuildProgram...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Create the kernel
    ckKernel = clCreateKernel(cpProgram, "VectorAdd", &ciErr1);
    printf("clCreateKernel (VectorAdd)...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Set the Argument values
    ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevSrcA);
    ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&cmDevSrcB);
    ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmDevDst);
    ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_int), (void*)&iNumElements);
    ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_mem), (void*)&cmIndexDst);
    printf("clSetKernelArg 0 - 3...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcA, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize, srcA, 0, NULL, NULL);
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcB, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize, srcB, 0, NULL, NULL);
    printf("clEnqueueWriteBuffer (SrcA and SrcB)...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }


    // Launch kernel
    ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
    printf("clEnqueueNDRangeKernel (VectorAdd)...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Synchronous/blocking read of results, and check accumulated errors
    ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmDevDst, CL_TRUE, 0, sizeof(cl_float) * szGlobalWorkSize, dst, 0, NULL, NULL);
    ciErr1|= clEnqueueReadBuffer(cqCommandQueue, cmIndexDst, CL_TRUE, 0, sizeof(cl_int) * szGlobalWorkSize, index_dst, 0, NULL, NULL);


    printf("clEnqueueReadBuffer (Dst)...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    //--------------------------------------------------------

    // Compute and compare results for golden-host and report errors and pass/fail
    printf("Comparing against Host/C++ computation...\n\n");

    bool bMatch = true;

    for (int i = 0; i < 512; ++i)
    {
    //  printf("dst[%i]=%f ", i, dst[i]);
        printf("index_dst[%i]=%d ", i, index_dst[i]);
        
    }
    // Cleanup and leave
    Cleanup (argc, argv, (bMatch == true) ? EXIT_SUCCESS : EXIT_FAILURE);
}

void Cleanup (int argc, char **argv, int iExitCode)
{
    // Cleanup allocated objects
    printf("Starting Cleanup...\n\n");
    if(cPathAndName)free(cPathAndName);   
    if(ckKernel)clReleaseKernel(ckKernel);
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cmDevSrcA)clReleaseMemObject(cmDevSrcA);
    if(cmDevSrcB)clReleaseMemObject(cmDevSrcB);
    if(cmDevDst)clReleaseMemObject(cmDevDst);


    printf("sizeof(__cl_float4) = %d \n", sizeof(__cl_float4));
    printf("sizeof(float) = %d \n", sizeof(float));

    // Free host memory
    delete [] srcA;  //free(srcA);
    delete [] srcB;  //free(srcB);
    delete [] dst;  // free (dst);  
    delete [] index_dst;
}

