#define CL_TARGET_OPENCL_VERSION 300

#include <stdio.h>
#include <stdlib.h>
#include "helper_cwk.h"

int main(int argc, char **argv)
{
    // Initialisation
    cl_device_id device;
    cl_context context = simpleOpenContext_GPU(&device);

    cl_int status;
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &status);

    int N, M;
    getCmdLineArgs(argc, argv, &N, &M);

    float
        *gradients = (float *)malloc(N * sizeof(float)),
        *inputs = (float *)malloc(M * sizeof(float)),
        *weights = (float *)malloc(N * M * sizeof(float));
    initialiseArrays(gradients, inputs, weights, N, M);

    // Create device buffers
    cl_mem d_gradients = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), gradients, &status);
      if (status != CL_SUCCESS)
    {
        printf("Error creating device buffer for gradients: %d\n", status);
        exit(EXIT_FAILURE);
    }
    cl_mem d_inputs = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M * sizeof(float), inputs, &status);
       if (status != CL_SUCCESS)
    {
        printf("Error creating device buffer for inputs: %d\n", status);
        exit(EXIT_FAILURE);
    }
    cl_mem d_weights = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * M * sizeof(float), weights, &status);
    if (status != CL_SUCCESS)
    {
        printf("Error creating device buffer for weights: %d\n", status);
        exit(EXIT_FAILURE);
    }

    // Compile kernel
    cl_kernel kernel = compileKernelFromFile("cwk3.cl", "updateWeights", context, device);

    // Set kernel arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_gradients);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_inputs);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_weights);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &N);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &M);
    if (status != CL_SUCCESS)
    {
        printf("Error setting kernel arguments: %d\n", status);        exit(EXIT_FAILURE);
    }

    // Query device capabilities
    size_t maxWorkGroupSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);


    // Set global and local work sizes
    size_t globalWorkSize[2] = {N, M};
    size_t localWorkSize[2] = {16, 16};
    if (localWorkSize[0] * localWorkSize[1] > maxWorkGroupSize)
    {
        localWorkSize[0] = maxWorkGroupSize;
        localWorkSize[1] = 1;
    }


    // Execute kernel
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if (status != CL_SUCCESS)
    {
        printf("Error executing kernel: %d\n", status);
        exit(EXIT_FAILURE);
    }

    // Read back results
    status = clEnqueueReadBuffer(queue, d_weights, CL_TRUE, 0, N * M * sizeof(float), weights, 0, NULL, NULL);
        if (status != CL_SUCCESS)
    {
        printf("Error reading back results: %d\n", status);
        exit(EXIT_FAILURE);    }

    // Output result and clean up
    displayWeights(weights, N, M);

    free(gradients);
    free(inputs);
    free(weights);

    clReleaseMemObject(d_gradients);
    clReleaseMemObject(d_inputs);
    clReleaseMemObject(d_weights);

    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return EXIT_SUCCESS;
}