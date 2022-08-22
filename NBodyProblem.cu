#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <iomanip>

using namespace std;

typedef double  T;

#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#define G 6.67e-11f
#define eps 1e-5f
#define BS 32

//__device__ const bool writeTrajFiles = true;

std::random_device rd;
std::mt19937 gen(rd());

#define MAX_VAL 1e13
std::uniform_real_distribution<> distribution(0, MAX_VAL);

__host__ T fRand(T fMin, T fMax)
{
    T f = distribution(gen) / MAX_VAL;
    return fMin + f * (fMax - fMin);
}

struct point
{
    T mass; 
    T coord[3];
    T vel[3];

__device__ __host__ point& operator=(const point& p)
    {
        mass = p.mass;
        for (int i = 0; i < 3; ++i)
        {
            coord[i] = p.coord[i];
            vel[i] = p.vel[i];
        }
        return *this;
    }
};

ostream &operator<<(std::ostream &out, const point &point)
{
    out << point.coord[0] << " " << point.coord[1] << " " << point.coord[2];
    return out;
}

__host__ int getNum(const char *const fileName)
{
    int n;
    ifstream ifile;
    ifile.open(fileName);
    if (!ifile.is_open())
    {
        cerr << " Error : file with settings is not open !\n";
        return -1;
    }
    ifile >> n;
    ifile.close();
    return n;
}

__host__ int readFile(const int n, const string fileName, point *points)
{
    ifstream ifile;
    ifile.open(fileName);
    string str;
    if (!ifile.is_open())
    {
        cerr << " Error : file with settings is not open !\n";
        return -1;
    }
    getline(ifile, str);
    for (int i = 0; i < n; ++i)
    {
        ifile >> points[i].mass;
        for (int j = 0; j < 3; ++j)
            ifile >> points[i].coord[j];

        for (int j = 0; j < 3; ++j)
            ifile >> points[i].vel[j];
    }
    ifile.close();
    return 0;
}

__host__ int createRandomFile(const int n, const string fileName)
{
    ofstream ofile;
    ofile.open(fileName);
    string str;
    if (!ofile.is_open())
    {
        cerr << " Error : file with settings is not open !\n";
        return -1;
    }
    ofile << n << endl;
    for (int i = 0; i < n; ++i)
    {
        ofile << fRand(9e9, 10e9) << " "; //mass
        for (int j = 0; j < 3; ++j)
            ofile << fRand(-3, 3) << " "; //coord
        for (int j = 0; j < 3; ++j)
            ofile << fRand(-0.3, 0.3) << " "; //vel
        ofile << endl;
    }
    ofile.close();
    return 0;
}

__device__ T *f(const point &p, T *accel, T *result)
{
    //y = {rx,ry,rz,vx,vy,vz}
        result[0] = p.vel[0];
        result[1] = p.vel[1];
        result[2] = p.vel[2];
        result[3] = accel[0];
        result[4] = accel[1];
        result[5] = accel[2];

        return result;
}


__device__ T dnorm(const T *vec1, const T *vec2)
{
    T sum = 0;
    for (int i = 0; i < 3; ++i)
    {
        sum += (vec1[i] - vec2[i])*(vec1[i] - vec2[i]);
    }
    return  sqrtf(sum);//__fsqrt_rn(sum);
}

__device__ void calculateAccelerations(int *N, point * points, T *accel, int id)
{
        point p = points[id];
        for (int dim = 0; dim < 3; ++dim)
        {
            T a = 0;
            for (int i = 0; i < (*N); ++i)
            {
                T k = dnorm(p.coord, points[i].coord);
                T denominator = k*k*k;
                a += points[i].mass * (p.coord[dim] - points[i].coord[dim]) / (MAX(denominator, eps));
            }

            accel[dim] =  -G * a;
        }
        
}

__global__ void simulate(int *N, point* points)
{
    int myid = blockIdx.x * blockDim.x + threadIdx.x;

    const T tau = 1e-3f;
const T tmax=tau;
    T* k1 = new T[6];
    T* k2 = new T[6];
    
    T* accel = new T[3];

    point ps1;
    point p = points[myid];

    const T tau05 = tau / 2;
    const T tlim = tmax - tau05;

    
    if (myid< (*N)) {

    for (T t = 0; t < tlim; t += tau)
    {
        calculateAccelerations(N, points, accel, myid);
        
        f(p, accel, k1);

            ps1 = p;

            for (int j = 0; j < 3; ++j) {                   
                p.coord[j] = ps1.coord[j] + tau05 * k1[j];
                p.vel[j] = ps1.vel[j] + tau05 * k1[3 + j];
            }

            points[myid]=p;

            __syncthreads();
  

        calculateAccelerations(N, points, accel, myid);

        f(p, accel, k2);

            for (int j = 0; j < 3; ++j) {
                p.coord[j] = ps1.coord[j] + tau * k2[j];
                p.vel[j] = ps1.vel[j] + tau * k2[3 + j];
            }

            points[myid]=p;

            __syncthreads();
    }

}
    delete[] accel;    
    delete[] k1;
    delete[] k2;
}

__host__ T getError(const int N, const point* points)
{
    T t = 0;
    point p, my_p;
    
    T error = 0, max_error =0;
    for (int i = 0; i < N; ++i)
    {
        ifstream ifile;
        ifile.open("traj" + to_string(i + 1) + ".txt");
        int count = 0;
        while (count < 200)
        {
            ifile >> t >> p.coord[0] >> p.coord[1] >> p.coord[2];
            ++count;
        }
        my_p = points[i];
        ifile >> t >> p.coord[0] >> p.coord[1] >> p.coord[2];
        error = MAX(MAX(fabs(p.coord[0] - my_p.coord[0]), fabs(p.coord[1] - my_p.coord[1])), fabs(p.coord[2] - my_p.coord[2]));
        if (error>max_error)
        {
        max_error = error;
        }
        
        ifile.close();
    }
    return max_error;
}

int main()
{
    const bool createNewFile = false;

    if (createNewFile)
    {
        createRandomFile(20000, "Nbody.txt");
    }

    const char *filename = "Nbody.txt"; //4body.txt or Nbody.txt

    int N = getNum(filename);
    int *N_dev;
    cudaMalloc((void**)&N_dev,sizeof(int));
    cudaMemcpy(N_dev,&N,sizeof(int),cudaMemcpyHostToDevice);
    point *points = new point[N];

    readFile(N, filename, points);

    point *GPUpoints;
    cudaMalloc((void**)&GPUpoints,N*sizeof(point));
    cudaMemcpy(GPUpoints,points,N*sizeof(point),cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    if (N < BS)
    {
        simulate<<<1, N>>>(N_dev, GPUpoints);
    } 
    else
    {
        int blocks=N/BS + (N%BS!=0);
        simulate<<<blocks,BS>>>(N_dev, GPUpoints);
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);

    printf("Time spent by GPU: %.2f ms\n",elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaMemcpy(points,GPUpoints,N*sizeof(point),cudaMemcpyHostToDevice);
    if (filename == "4body.txt")
    {
        cout << "error = " << getError(4,points);
    }


    cudaFree(GPUpoints);
    cudaFree(N_dev);
    delete[]points;

    return 0;   
}