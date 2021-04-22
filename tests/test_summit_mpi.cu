/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <vector>
#include "mpi.h"

__global__ void set_mem(double* buffer, int size, double value) {
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < size;
       n += gridDim.x * blockDim.x) {
    buffer[n] = value;
  }
}

int
main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cout << "rank is: " << rank << ", total size is: " << size << std::endl;

  int n_devices;
  cudaGetDeviceCount(&n_devices);
  if (n_devices <= 0) {
    std::cerr << "No usable Cuda device found!!" << std::endl;
    exit(1);
  } else {
    std::cout << "Found " << n_devices << " Cuda devices!" << std::endl;
  }
  // set device number
  int dev_id = rank % n_devices;
  std::cout << "Rank " << rank << " is on device #" << dev_id << std::endl;
  cudaSetDevice(dev_id);

  // Prepare the buffer for communication
  double* dev_send_buffer;
  double* dev_recv_buffer;
  
  int N = 1000000;
  cudaMalloc(&dev_send_buffer, N * sizeof(double));
  cudaMalloc(&dev_recv_buffer, N * sizeof(double));

  set_mem<<<512, 512>>>(dev_send_buffer, N, (double)rank);
  cudaDeviceSynchronize();

  // Send the buffer to the next rank
  int dst = (rank + 1) % size;
  int src = (rank - 1 + size) % size;
  std::cout << "Sending to rank " << dst << ", receiving from rank " << src << std::endl;

  for (int n = 0; n < 100; n++) {
    size_t free_mem, total_mem;
    MPI_Status status;
    MPI_Sendrecv(dev_send_buffer, N, MPI_DOUBLE, dst, 0,
                dev_recv_buffer, N, MPI_DOUBLE, src, 0,
                MPI_COMM_WORLD, &status);
    cudaMemGetInfo( &free_mem, &total_mem );
    std::cout << "GPU memory: free=" << free_mem/1.0e9 << "GiB, total=" << total_mem/1.0e9 << "GiB" << std::endl;
  }

  std::vector<double> host_recv_buffer(N);
  cudaMemcpy(host_recv_buffer.data(), dev_recv_buffer, N * sizeof(double), cudaMemcpyDeviceToHost);
  std::cout << "value is " << host_recv_buffer[N - 1] << " on rank " << rank << std::endl;
  
  cudaFree(dev_send_buffer);
  cudaFree(dev_recv_buffer);

  MPI_Finalize();
  
  return 0;
}
