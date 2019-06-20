#include <stdio.h>

__global__ void spmv_csr_kernel(unsigned int dim, unsigned int *csrRowPtr, 
    unsigned int *csrColIdx, float *csrData, float *inVector, 
    float *outVector) {
    // INSERT KERNEL CODE HERE
    int rowIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if(rowIdx<dim){
        float dotP = 0.0f;
        for(int i=csrRowPtr[rowIdx]; i<csrRowPtr[rowIdx+1]; i++){
            dotP += csrData[i]*inVector[csrColIdx[i]];
        }
        outVector[rowIdx] = dotP;
    }
}

__global__ void spmv_jds_kernel(unsigned int dim, unsigned int *jdsRowPerm, 
    unsigned int *jdsRowNNZ, unsigned int *jdsColStartIdx, 
    unsigned int *jdsColIdx, float *jdsData, float* inVector,
    float *outVector) {
    // INSERT KERNEL CODE HERE
    int rowIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if(rowIdx<dim){
        float dotP = 0.0f;
        for(int i=0; i<jdsRowNNZ[rowIdx]; i++){
            dotP += jdsData[rowIdx+jdsColStartIdx[i]]*inVector[jdsColIdx[rowIdx+jdsColStartIdx[i]]];
        }
        outVector[jdsRowPerm[rowIdx]] = dotP; 
    }
}

void spmv_csr(unsigned int dim, unsigned int *csrRowPtr, unsigned int *csrColIdx, 
    float *csrData, float *inVector, float *outVector) {
    // INSERT CODE HERE
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    spmv_csr_kernel<<<ceil(dim/(float)maxThreadsPerBlock), maxThreadsPerBlock>>>(dim, csrRowPtr,
         csrColIdx, csrData, inVector, outVector);
}

void spmv_jds(unsigned int dim, unsigned int *jdsRowPerm, unsigned int *jdsRowNNZ, 
    unsigned int *jdsColStartIdx, unsigned int *jdsColIdx, float *jdsData, 
    float* inVector, float *outVector) {
    // INSERT CODE HERE
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    spmv_jds_kernel<<<ceil(dim/(float)maxThreadsPerBlock), maxThreadsPerBlock>>>(dim, jdsRowPerm,
         jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData, inVector, outVector);
}






