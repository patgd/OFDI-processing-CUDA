#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cufft.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <thread>
#include <chrono>


int gnMode = -1;
int gnRawLineLength;
int gnRawNumberLines;
int gnCalibrationNumberLines;
int gnProcessNumberLines;
int gnProcessedNumberLines;
int gnPerpendicular;
int gnAllocationStatus = 0;
int gnMidLength;

float* gpfRawCalibration;
float* gpfProcessCalibration;
size_t gnProcessCalibrationPitch;

// reference
float* gpfReferenceEven;
float* gpfReferenceOdd;

// fft
cufftComplex* gpcProcessDepthProfile;
size_t gnProcessDepthProfilePitch;
cufftHandle gchForward;

// calibration mask
int gnCalibrationStart;
int gnCalibrationStop;
int gnCalibrationRound;
float* gpfCalibrationMask;

// reverse fft
cufftComplex* gpcProcessSpectrum;
size_t gnProcessSpectrumPitch;
cufftHandle gchReverse;

// phase
float* gpfProcessPhase;
size_t gnProcessPhasePitch;

// unwrap
float gfPiEps = (float)(acos(-1.0) - 1.0e-30);
float gf2Pi = (float)(2.0 * acos(-1.0));

// linear fit and interpolation
float* gpfLeftPhase;
float* gpfRightPhase;
float* gpfKLineCoefficients;
float* gpfProcessK;
size_t gnKPitch;
int* gpnProcessIndex;
size_t gnIndexPitch;
int* gpnProcessAssigned;
size_t gnAssignedPitch;
int gnKMode;
float* gpfProcessSpectrumK;
size_t gnSpectrumKPitch;

float* gpfProcessOCT;
size_t gnProcessOCTPitch;
cufftComplex* gpcProcessedOCT;

// dispersion mask
int gnDispersionStart;
int gnDispersionStop;
int gnDispersionRound;
float* gpfDispersionMask;

// dispersion correction
cufftComplex* gpcDispersionCorrection;
cufftHandle gchForwardComplex;
cufftComplex* gpcProcessKCorrected;
size_t gnKCorrectedPitch;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

extern "C" {
    __declspec(dllexport) int getDeviceCount(int* nNumberDevices) {
        // check for GPU
        int nDevices = -1;
        int nRet = cudaGetDeviceCount(&nDevices);
        if (nRet == cudaSuccess) {
            *(nNumberDevices) = nDevices;
        }	// if (nRet
        return nRet;
    }	// int getDeviceCount
}	// extern "C"

extern "C" {
    __declspec(dllexport) int getDeviceName(int nDeviceNumber, char* strDeviceName) {
        // check for GPU
        cudaDeviceProp currentDevice;
        int nRet = cudaGetDeviceProperties(&currentDevice, nDeviceNumber);
        if (nRet == cudaSuccess) {
            sprintf(strDeviceName, "%s (%d SMs, %d b/s, %d t/b, %d t/s, %d shared kB, %d GB)", 
                currentDevice.name, 
                currentDevice.multiProcessorCount, 
                currentDevice.maxBlocksPerMultiProcessor, 
                currentDevice.maxThreadsPerBlock, 
                currentDevice.maxThreadsPerMultiProcessor, 
                currentDevice.sharedMemPerBlock / 1024,
                currentDevice.totalGlobalMem / 1024 / 1024 / 1024);

        }	// if (nRet
        return nRet;
    }	// int getDeviceName
}	// extern "C"

extern "C" {
    __declspec(dllexport) int cleanup() {
        // free memory allocations
        if (gnAllocationStatus == 1) {
            gpuErrchk(cudaFreeHost(gpfRawCalibration));
            gpuErrchk(cudaFree(gpfProcessCalibration));
            gpuErrchk(cudaFree(gpfReferenceEven));
            gpuErrchk(cudaFree(gpfReferenceOdd));
            gpuErrchk(cudaFree(gpcProcessDepthProfile));
            cufftDestroy(gchForward);
            gpuErrchk(cudaFree(gpfCalibrationMask));
            gpuErrchk(cudaFree(gpcProcessSpectrum));
            cufftDestroy(gchReverse);
            gpuErrchk(cudaFree(gpfProcessPhase));
            cudaFree(gpfLeftPhase);
            cudaFree(gpfRightPhase);
            cudaFree(gpfKLineCoefficients);
            cudaFree(gpfProcessK);
            cudaFree(gpnProcessIndex);
            cudaFree(gpnProcessAssigned);
            cudaFree(gpfProcessSpectrumK);
            cudaFree(gpfProcessOCT);
            cudaFreeHost(gpcProcessedOCT);
            gpuErrchk(cudaFree(gpfDispersionMask));
            gpuErrchk(cudaFree(gpcDispersionCorrection));
            cufftDestroy(gchForwardComplex);
            cudaFree(gpcProcessKCorrected);

            gnAllocationStatus = 0;
        }   // if (gnAllocationStatus
        return -1;
    }   // __declspec
}   // extern

extern "C" {
    __declspec(dllexport) int initialize(int nMode, int nRawLineLength, int nRawNumberLines, int nProcessNumberLines, int nProcessedNumberLines) {

        cleanup();

        // copy parameters to global parameters
        gnMode = nMode;
        gnRawLineLength = nRawLineLength;
        gnRawNumberLines = nRawNumberLines;
        gnProcessNumberLines = nProcessNumberLines;
        gnProcessedNumberLines = nProcessedNumberLines;

        // allocate memory
        gnPerpendicular = 0;
        switch (gnMode) {
        case 0: // SD-OCT
            gnPerpendicular = 0;
            gnCalibrationNumberLines = 1;
            break;
        case 1: // PS SD-OCT
            gnPerpendicular = 1;
            gnCalibrationNumberLines = gnRawNumberLines;
            break;
        case 2: // line field
            gnPerpendicular = 0;
            gnCalibrationNumberLines = 1;
            break;
        case 3: // OFDI
            gnPerpendicular = 0;
            gnCalibrationNumberLines = gnRawNumberLines;
            break;
        case 4: // PS OFDI
            gnPerpendicular = 1;
            gnCalibrationNumberLines = gnRawNumberLines;
            break;
        }   // switch (gnMode

        gpuErrchk(cudaMallocHost((void**)&gpfRawCalibration, (gnRawLineLength * gnCalibrationNumberLines) * sizeof(float)));
        gpuErrchk(cudaMallocPitch((void**)&gpfProcessCalibration, &gnProcessCalibrationPitch, gnRawLineLength * sizeof(float), gnProcessNumberLines >> 1));

        gpuErrchk(cudaMalloc((void**)&gpfReferenceEven, gnRawLineLength * sizeof(float)));
        gpuErrchk(cudaMalloc((void**)&gpfReferenceOdd, gnRawLineLength * sizeof(float)));

        gnMidLength = (int)(gnRawLineLength / 2 + 1);
        gpuErrchk(cudaMallocPitch((void**)&gpcProcessDepthProfile, &gnProcessDepthProfilePitch, gnRawLineLength * sizeof(cufftComplex), gnProcessNumberLines >> 1));
        int nRank = 1;
        int pn[] = { gnRawLineLength };
        int nIStride = 1, nOStride = 1;
        int nIDist = gnProcessCalibrationPitch / sizeof(float);
        int nODist = gnProcessDepthProfilePitch / sizeof(cufftComplex);
        int pnINEmbed[] = { 0 };
        int pnONEmbed[] = { 0 };
        int nBatch = gnProcessNumberLines >> 1;
        cufftPlanMany(&gchForward, nRank, pn, pnINEmbed, nIStride, nIDist, pnONEmbed, nOStride, nODist, CUFFT_R2C, nBatch);

        gpuErrchk(cudaMalloc((void**)&gpfCalibrationMask, gnRawLineLength * sizeof(float)));

        gpuErrchk(cudaMallocPitch((void**)&gpcProcessSpectrum, &gnProcessSpectrumPitch, gnRawLineLength * sizeof(cufftComplex), gnProcessNumberLines >> 1));
        nIDist = gnProcessDepthProfilePitch / sizeof(cufftComplex);
        nODist = gnProcessSpectrumPitch / sizeof(cufftComplex);
        cufftPlanMany(&gchReverse, nRank, pn, pnINEmbed, nIStride, nIDist, pnONEmbed, nOStride, nODist, CUFFT_C2C, nBatch);

        gpuErrchk(cudaMallocPitch((void**)&gpfProcessPhase, &gnProcessPhasePitch, gnRawLineLength * sizeof(float), gnProcessNumberLines >> 1));

        cudaMalloc((void**)&gpfLeftPhase, sizeof(float));
        cudaMalloc((void**)&gpfRightPhase, sizeof(float));
        cudaMalloc((void**)&gpfKLineCoefficients, 2 * sizeof(float));
        gpuErrchk(cudaMallocPitch((void**)&gpfProcessK, &gnKPitch, gnRawLineLength * sizeof(float), gnProcessNumberLines >> 1));
        gpuErrchk(cudaMallocPitch((void**)&gpnProcessIndex, &gnIndexPitch, gnRawLineLength * sizeof(int), gnProcessNumberLines >> 1));
        gpuErrchk(cudaMallocPitch((void**)&gpnProcessAssigned, &gnAssignedPitch, gnRawLineLength * sizeof(int), gnProcessNumberLines >> 1));

        gpuErrchk(cudaMallocPitch((void**)&gpfProcessSpectrumK, &gnSpectrumKPitch, gnRawLineLength * sizeof(float), gnProcessNumberLines >> 1));

        gpuErrchk(cudaMallocPitch((void**)&gpfProcessOCT, &gnProcessOCTPitch, gnRawLineLength * sizeof(float), gnProcessNumberLines >> 1));
        gpuErrchk(cudaMallocHost((void**)&gpcProcessedOCT, (gnMidLength * gnProcessedNumberLines) * sizeof(cufftComplex)));

        gpuErrchk(cudaMalloc((void**)&gpfDispersionMask, gnRawLineLength * sizeof(float)));
        gpuErrchk(cudaMalloc((void**)&gpcDispersionCorrection, gnRawLineLength * sizeof(cufftComplex)));
        gpuErrchk(cudaMallocPitch((void**)&gpcProcessKCorrected, &gnKCorrectedPitch, gnRawLineLength * sizeof(cufftComplex), gnProcessNumberLines >> 1));

        nIDist = gnKCorrectedPitch / sizeof(cufftComplex);
        cufftPlanMany(&gchForwardComplex, nRank, pn, pnINEmbed, nIStride, nIDist, pnONEmbed, nOStride, nODist, CUFFT_C2C, nBatch);

        gnAllocationStatus = 1;

        return -1;
    }   // int initialize
}   // extern

int readPSSDOCTFile(short** pnBuffer) {
    // read data from file
    std::ifstream fRawBinary("pdH.bin");
    // get size of file and move back to beginning
    fRawBinary.seekg(0, std::ios_base::end);
    std::size_t nSize = fRawBinary.tellg();
    fRawBinary.seekg(0, std::ios_base::beg);
    // allocate space for the array
    *pnBuffer = (short*)malloc(nSize);
    // read data
    fRawBinary.read((char*)(*pnBuffer), nSize);
    // close file
    fRawBinary.close();

    return -1;
}

extern "C" {
    __declspec(dllexport) int getDataPSSDOCT(void* pnIMAQParallel, void* pnIMAQPerpendicular) {


        int nAline, nPoint, nLocation;

        // copy to host memory
        for (nAline = 0; nAline < gnCalibrationNumberLines; nAline++)
            for (nPoint = 0; nPoint < gnRawLineLength; nPoint++) {
                nLocation = nAline * gnRawLineLength + nPoint;
        //                gpfRawCalibrationParallel[nLocation] = (short)v[nLocation];
            }   // for (nPoint


        return -1;
    }   // int getData
}   // extern

__global__ void calculateMean(float* pfMatrix, float* pfMean, int nNumberLines, int nLineLength) {
    __shared__ float pfSum[1024];

    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    float fSum = 0.0;
    int nLine;
    int nNumber = nNumberLines / blockDim.y;
    int nPosition = threadIdx.y * nNumber * nLineLength + nPoint;
    for (nLine = 0; nLine < nNumber; nLine++) {
        fSum += pfMatrix[nPosition];
        nPosition += nLineLength;
    }   // for (int nLine
    pfSum[threadIdx.x * blockDim.y + threadIdx.y] = fSum;

    __syncthreads();
    if (threadIdx.y == 0) {
        fSum = 0;
        nPosition = threadIdx.x * blockDim.y;
        for (nLine = 0; nLine < blockDim.y; nLine++) {
            fSum += pfSum[nPosition];
            nPosition++;
        }   // for (nLine
        pfMean[nPoint] = fSum / nNumberLines;
    }
}   // void calculateMean

__global__ void subtractMean(float* pfMatrix, float* pfMean, int nNumberLines, int nLineLength) {
    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    float fMean = pfMean[nPoint];
    int nLine;
    int nNumber = nNumberLines / blockDim.y;
    int nPosition = threadIdx.y * nNumber * nLineLength + nPoint;
    for (nLine = 0; nLine < nNumber; nLine++) {
        pfMatrix[nPosition] -= fMean;
        nPosition += nLineLength;
    }   // for (int nLine
}   // void subtractMean

__global__ void calculateMask(float* pfMask, int nLength, int nStart, int nStop, int nRound) {
    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    pfMask[nPoint] = 0.0;
    if (nPoint < nLength) {
        if (nPoint >= nStart - nRound)
            if (nPoint < nStart)
                pfMask[nPoint] = sin(0.5*nPoint);
            else
                if (nPoint < nStop)
                    pfMask[nPoint] = 1.0;
                else
                    if (nPoint < nStop + nRound)
                        pfMask[nPoint] = sin(0.5*nPoint);
    }   // if (nPoint
}   // void calculateMask

__global__ void applyMask(cufftComplex* pcMatrix, float* pfMask, int nNumberLines, int nLineLength) {
    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    float fMask = pfMask[nPoint];
    int nLine;
    int nNumber = nNumberLines / blockDim.y;
    int nPosition = threadIdx.y * nNumber * nLineLength + nPoint;
    for (nLine = 0; nLine < nNumber; nLine++) {
        pcMatrix[nPosition].x *= fMask;
        pcMatrix[nPosition].y *= fMask;
        nPosition += nLineLength;
    }   // for (int nLine
}   // void subtractMean

__global__ void calculatePhase(cufftComplex* pcMatrix, float* pfPhase, int nNumberLines, int nLineLength) {
    int nPosition = (blockIdx.y * blockDim.y + threadIdx.y) * nLineLength + (blockIdx.x * blockDim.x + threadIdx.x);
    pfPhase[nPosition] = atan2(pcMatrix[nPosition].y, pcMatrix[nPosition].x);
}   // void calculatePhase

__global__ void unwrapPhase(float* pfPhase, int nNumberLines, int nLineLength, float fPiEps, float f2Pi) {
    __shared__ float pfUnwrappedEnds[2048];
    __shared__ int pn2pi[1024];
    
    int nLineNumber = blockIdx.x * blockDim.y + threadIdx.y;
    int nNumberPoints = nLineLength / blockDim.x;
    int nStartPoint = nLineNumber * nLineLength + threadIdx.x * nNumberPoints;
    int nStopPoint = nStartPoint + nNumberPoints;

    pfUnwrappedEnds[2 * (threadIdx.y * blockDim.x + threadIdx.x)] = pfPhase[nStartPoint];
    int nPoint = nStartPoint;
    float fOldPhase = pfPhase[nPoint];
    float fNewPhase;
    float fDeltaPhase;
    int n2Pi = 0;
    nPoint++;
    while (nPoint < nStopPoint) {
        fNewPhase = pfPhase[nPoint];
        fDeltaPhase = fNewPhase - fOldPhase;
        fOldPhase = fNewPhase;

        if (fDeltaPhase < -fPiEps)
            n2Pi++;
        if (fDeltaPhase > fPiEps)
            n2Pi--;

        pfPhase[nPoint] = fNewPhase + n2Pi * f2Pi;
        nPoint++;
    }   // while (nPoint
    nPoint--;
    pfUnwrappedEnds[2 * (threadIdx.y * blockDim.x + threadIdx.x) + 1] = pfPhase[nPoint];

    __syncthreads();

    if (threadIdx.x == 0) {
        int nSection = threadIdx.y * blockDim.x;
        int nEnd = 2 * nSection + 1;
        int nStart = nEnd + 1;
        pn2pi[nSection] = 0;
        for (nPoint = 1; nPoint < blockDim.y; nPoint++) {
            fDeltaPhase = pfUnwrappedEnds[nStart] - pfUnwrappedEnds[nEnd];
            pn2pi[nSection + 1] = pn2pi[nSection];
            nStart += 2;
            nEnd += 2;
            nSection++;
            if (fDeltaPhase < -fPiEps)
                pn2pi[nSection]++;
            if (fDeltaPhase > fPiEps)
                pn2pi[nSection]--;
        }   // for (nPoint
    }   // if (threadIdx.x

    __syncthreads();

    fDeltaPhase = f2Pi * (pn2pi[threadIdx.y * blockDim.x + threadIdx.x]);
    nPoint = nStartPoint + 1;
    while (nPoint < nStopPoint) {
        pfPhase[nPoint] += fDeltaPhase;
        nPoint++;
    }
}   // void unwrapPhase

__global__ void matchPhase(float* pfPhase, int nNumberLines, int nLineLength, float f2Pi) {
    __shared__ float pfOffset[1024];

    int nLineNumber = blockIdx.x * blockDim.y + threadIdx.y;
    int nNumberPoints = nLineLength / blockDim.x;
    int nStartPoint = nLineNumber * nLineLength + threadIdx.x * nNumberPoints;
    int nStopPoint = nStartPoint + nNumberPoints;

    if (threadIdx.x == 0)
        pfOffset[threadIdx.y] = f2Pi * roundf(pfPhase[nLineNumber * nLineLength + (nLineLength >> 1)] / f2Pi);

    __syncthreads();

    float fOffset = pfOffset[threadIdx.y];
    for (int nPoint = nStartPoint; nPoint < nStopPoint; nPoint++)
        pfPhase[nPoint] -= fOffset;
}   // void matchPhase

__global__ void getPhaseLimits(float* pfPhase, int nNumberLines, int nLineLength, int nLeft, int nRight, float *pfLeft, float *pfRight) {
    __shared__ float pfSum[1024];

    int nLinesInSection = nNumberLines / blockDim.x;
    int nStartingLine = threadIdx.x * nLinesInSection;
    int nPoint = nLeft;
    if (blockIdx.x == 1)
        nPoint = nRight;
    nPoint += nStartingLine * nLineLength;
    int nLine;
    
    float fSum = 0.0;
    for (nLine = 0; nLine < nLinesInSection; nLine++) {
        fSum += pfPhase[nPoint];
        nPoint += nLineLength;
    }   // for (int nLine
    pfSum[threadIdx.x] = fSum;

    __syncthreads();

    if (threadIdx.x == 0) {
        fSum = 0;
        for (nLine = 0; nLine < blockDim.x; nLine++)
            fSum += pfSum[nLine];
        if (blockIdx.x == 0)
            *pfLeft = fSum / nNumberLines;
        else
            *pfRight = fSum / nNumberLines;
    }   // if (threadIdx.x
}

__global__ void calculateK(float* pfPhase, float* pfK, int* pnIndex, int* pnAssigned, int nNumberLines, int nLineLength, float* pfLineParameters, int nLeft, int nRight, float* pfLeft, float* pfRight, int nMode) {

    // calculate slope and offset
    switch (nMode) {
    case 1:
        pfLineParameters[0] = (pfRight[0] - pfLeft[0]) / ((float) (nRight - nLeft));
        pfLineParameters[1] = - ((nLineLength >> 1) + nLineLength) * pfLineParameters[0];
        break;
    case 2:
        break;
    }   // switch (nMode

    float fSlope = pfLineParameters[0];
    float fOffset = pfLineParameters[1];

    int nLine = blockIdx.x * blockDim.y + threadIdx.y;
    int nNumberPoints = nLineLength / blockDim.x;
    int nOffset1 = threadIdx.x * nNumberPoints;
    int nOffset2 = nLine * nLineLength;
    int nIndex = nOffset2 + nOffset1;
    int nX;
    for (int nPoint = 0; nPoint < nNumberPoints; nPoint++) {
        pfK[nIndex] = (pfPhase[nIndex] - fOffset) / fSlope;
        nX = ceilf(pfK[nIndex]) - nLineLength + nOffset1;
        if ((nX >= 0) && (nX < nLineLength)) {
            pnIndex[nOffset2 + nX] = nIndex - nOffset2;
            pnAssigned[nOffset2 + nX] = 1;
        }   //  if ((nX
        nIndex++;
    }   // for (int nPoint
}   // void calculateK

__global__ void cleanIndex(float* pfK, int* pnIndex, int* pnAssigned, int nNumberLines, int nLineLength) {
    int nLine = blockIdx.x * blockDim.y + threadIdx.y;
    int nNumberPoints = nLineLength / blockDim.x;
    int nLineOffset = nLine * nLineLength;
    int nPointOffset = threadIdx.x * nNumberPoints;

    // find first non-assigned element
    bool bKeepSearching = true;
    int nCurrentPoint = nLineOffset + nPointOffset;
    int nEndOfSection = nCurrentPoint + nNumberPoints;
    bKeepSearching = (pnAssigned[nCurrentPoint] == 0);
    while (bKeepSearching) {
        nCurrentPoint++;
        if (nCurrentPoint < nEndOfSection)
            bKeepSearching = (pnAssigned[nCurrentPoint] == 0);
        else
            bKeepSearching = false;
    }

    if (nCurrentPoint != nEndOfSection) {
        // if (thread == 0) track backwards
        if (threadIdx.x == 0) {
            int nBackwardPoint = nCurrentPoint - 1;
            while ((pfK[nBackwardPoint] > nLineLength) && (nBackwardPoint > (nLineOffset + 1)))
                nBackwardPoint--;
            int nSearchK = nBackwardPoint - 1 - nLineOffset;
            for (int nPoint = nLineOffset; nPoint < nCurrentPoint; nPoint++) {
                pnAssigned[nPoint] = 1;
                pnIndex[nPoint] = nSearchK;
            }   // for (int nPoint
        }   // if (threadIdx.x

        // once complete, track forward
        int nEndOfLine = nLineOffset + nLineLength;
        int nLastIndex = pnIndex[nCurrentPoint];
        bKeepSearching = true;
        while (bKeepSearching) {
            if (nCurrentPoint < nEndOfSection) {
                if (pnAssigned[nCurrentPoint] == 0)
                    pnIndex[nCurrentPoint] = nLastIndex;
                else {
                    nLastIndex = pnIndex[nCurrentPoint];
                    if (nLastIndex > nEndOfLine - 3) {
                        nLastIndex = nEndOfLine - 3;
                        pnIndex[nCurrentPoint] = nLastIndex;
                    }
                }
                nCurrentPoint++;
            }
            else {
                if (nCurrentPoint < nEndOfLine)
                    if (pnAssigned[nCurrentPoint] = 0) {
                        pnIndex[nCurrentPoint] = nLastIndex;
                        nCurrentPoint++;
                    }
                    else
                        bKeepSearching = false;
                else
                    bKeepSearching = false;
            }   // if (nCurrentPoint
        }   // while (bKeepSearching
    }   // if (nCurrentPoint

}

__global__ void interpCubicSpline(float* pfK, int* pnIndex, float* pfSpectrum, float* pfInterpSpectrum, int nNumberLines, int nLineLength) {
    int nPosition = (blockIdx.y * blockDim.y + threadIdx.y) * nLineLength + (blockIdx.x * blockDim.x + threadIdx.x);
    int nIndex = pnIndex[nPosition];
    float fk1_1 = pfK[nIndex];
    float fk1_2 = fk1_1 * fk1_1;
    float fk1_3 = fk1_2 * fk1_1;
    float fS1 = pfSpectrum[nIndex];
    nIndex++;
    float fk2_1 = pfK[nIndex];
    float fk2_2 = fk2_1 * fk2_1;
    float fk2_3 = fk2_2 * fk2_1;
    float fS2 = pfSpectrum[nIndex];
    nIndex++;
    float fk3_1 = pfK[nIndex];
    float fk3_2 = fk3_1 * fk3_1;
    float fk3_3 = fk3_2 * fk3_1;
    float fS3 = pfSpectrum[nIndex];
    nIndex++;
    float fk4_1 = pfK[nIndex];
    float fk4_2 = fk4_1 * fk4_1;
    float fk4_3 = fk4_2 * fk4_1;
    float fS4 = pfSpectrum[nIndex];

    float f0 = (fk1_3 * fk2_2 * fk3_1       + fk2_3 * fk3_2 * fk4_1       + fk3_3 * fk4_2 * fk1_1       + fk4_3 * fk1_2 * fk2_1      ) - (fk1_3 * fk4_2 * fk3_1       + fk2_3 * fk1_2 * fk4_1       + fk3_3 * fk2_2 * fk1_1       + fk4_3 * fk3_2 * fk2_1      );
    float f1 = (fS1   * fk2_2 * fk3_1       + fS2   * fk3_2 * fk4_1       + fS3   * fk4_2 * fk1_1       + fS4   * fk1_2 * fk2_1      ) - (fS1   * fk4_2 * fk3_1       + fS2   * fk1_2 * fk4_1       + fS3   * fk2_2 * fk1_1       + fS4   * fk3_2 * fk2_1      );
    float f2 = (fk1_3 * fS2   * fk3_1       + fk2_3 * fS3   * fk4_1       + fk3_3 * fS4   * fk1_1       + fk4_3 * fS1   * fk2_1      ) - (fk1_3 * fS4   * fk3_1       + fk2_3 * fS1   * fk4_1       + fk3_3 * fS2   * fk1_1       + fk4_3 * fS3   * fk2_1      );
    float f3 = (fk1_3 * fk2_2 * fS3         + fk2_3 * fk3_2 * fS4         + fk3_3 * fk4_2 * fS1         + fk4_3 * fk1_2 * fS2        ) - (fk1_3 * fk4_2 * fS3         + fk2_3 * fk1_2 * fS4         + fk3_3 * fk2_2 * fS1         + fk4_3 * fk3_2 * fS2        );
    float f4 = (fk1_3 * fk2_2 * fk3_1 * fS4 + fk2_3 * fk3_2 * fk4_1 * fS1 + fk3_3 * fk4_2 * fk1_1 * fS2 + fk4_3 * fk1_2 * fk2_1 * fS3) - (fk1_3 * fk4_2 * fk3_1 * fS2 + fk2_3 * fk1_2 * fk4_1 * fS3 + fk3_3 * fk2_2 * fk1_1 * fS4 + fk4_3 * fk3_2 * fk2_1 * fS1);

    float fK = (blockIdx.x * blockDim.x + threadIdx.x) + nLineLength;
    pfInterpSpectrum[nPosition] = (((f1 / f0) * fK + (f2 / f0)) * fK + (f3 / f0)) * fK + (f4 / f0);
}

__global__ void calculateDispersionCorrection(float* pfPhase, cufftComplex* pcCorrection) {
    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    pcCorrection[nPoint].x = cosf(pfPhase[nPoint]);
    pcCorrection[nPoint].y = -sinf(pfPhase[nPoint]);
}

__global__ void applyDispersionCorrection(float* pfMatrix, cufftComplex* pcCorrection, cufftComplex* pcMatrix, int nNumberLines, int nLineLength) {
    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    cufftComplex cCorrection = pcCorrection[nPoint];
    float fOriginal;
    int nLine;
    int nNumber = nNumberLines / blockDim.y;
    int nPosition = threadIdx.y * nNumber * nLineLength + nPoint;
    for (nLine = 0; nLine < nNumber; nLine++) {
        fOriginal = pfMatrix[nPosition];
        pcMatrix[nPosition].x = fOriginal * cCorrection.x;
        pcMatrix[nPosition].y = fOriginal * cCorrection.y;
        nPosition += nLineLength;
    }   // for (int nLine
}   // void subtractMean

extern "C" {
    __declspec(dllexport) __cdecl void copyFromHostToDevice_nOCTcuda(){
        cudaMemcpy2D(
            gpfProcessCalibration, 
            gnProcessCalibrationPitch, 
            gpfRawCalibration + (nAline + 0) * gnRawLineLength, 
            2 * gnProcessCalibrationPitch, 
            gnProcessCalibrationPitch, 
            nNumberLinesInChunk >> 1, 
            cudaMemcpyHostToDevice);
    }
}

extern "C" {
    __declspec(dllexport) __cdecl void calculateReferenceArrays_nOCTcuda(){
        
        //declare d3Threads-analogue object globally
        #region calculate reference
        d3Threads.x = 128;
        d3Threads.y = 1024 / d3Threads.x;
        d3Threads.z = 1;
        d3Blocks.x = gnProcessNumberLines / d3Threads.x;
        d3Blocks.y = 1;
        d3Blocks.z = 1;
        calculateMean<<<d3Blocks, d3Threads>>>(gpfProcessCalibration, gpfReferenceEven, nNumberLinesInChunk >> 1, gnRawLineLength);
        //gpuErrchk(cudaPeekAtLastError());
        #endregion // calculate reference
        
        #region subtract reference
        d3Threads.x = 32;
        d3Threads.y = 1024 / d3Threads.x;
        d3Threads.z = 1;
        d3Blocks.x = gnProcessNumberLines / d3Threads.x;
        d3Blocks.y = 1;
        d3Blocks.z = 1;
        subtractMean<<<d3Blocks, d3Threads>>>(gpfProcessCalibration, gpfReferenceEven, nNumberLinesInChunk >> 1, gnRawLineLength);
        //gpuErrchk(cudaPeekAtLastError());
        #endregion // subtract reference
        
    }
}