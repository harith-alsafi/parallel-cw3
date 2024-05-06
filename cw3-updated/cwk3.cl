// Implement the kernel (or kernels) for coursework 3 in this file.

__kernel void updateWeights(__global const float *gradients, __global const float *inputs, __global float *weights, int N, int M)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < N && j < M)
    {
        weights[i * M + j] += gradients[i] * inputs[j];
    }
}