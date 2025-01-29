#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CHECK_CUDA(call) \
{ \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (err num=" << err << ")" << std::endl; \
    exit(1); \
  } \
}

struct Page {
  int ID;
  std::vector<int> incoming_ids;
  int size_incoming_ids;
  int num_in_pages;
  int num_out_pages;
  double page_rank;
};

// Kernel: Compute new PR values given old PR values
__global__ void AddPagesPrKernel(
  int num_pages,
  const int *incoming_id_starts,
  const int *incoming_id_list,
  const float *old_pr,
  const double *out_link_cnts_rcp,
  float *new_pr)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_pages) {
    int start = incoming_id_starts[i];
    int end = incoming_id_starts[i+1];
    double sum = 0.0;
    for (int j = start; j < end; j++){
      int in_id = incoming_id_list[j];
      sum += old_pr[in_id] * out_link_cnts_rcp[in_id];
    }
    new_pr[i] = (float)sum;
  }
}

// Kernel: Add effect of dangling pages
__global__ void AddDanglingPagesPrKernel(int num_pages, float *pr, double val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_pages) {
    pr[i] += (float)val;
  }
}

// Kernel: Add random jumps
__global__ void AddRandomJumpsPrKernel(int num_pages, float *pr, double damping_factor, double val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_pages) {
    pr[i] = pr[i] * (float)damping_factor + (float)val;
  }
}

// Kernel: Sum dangling pages' PR (partial sums)
__global__ void SumDanglingKernel(const int *dangling_pages, int num_dangling, const float *old_pr, double *partial_sums) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double s = 0.0;
  if (i < num_dangling) {
    s = old_pr[dangling_pages[i]];
  }
  __shared__ double shmem[256]; 
  int tid = threadIdx.x;
  shmem[tid] = s;
  __syncthreads();

  for (int stride = blockDim.x/2; stride>0; stride/=2) {
    if (tid < stride) {
      shmem[tid] += shmem[tid+stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[blockIdx.x] = shmem[0];
  }
}

// Simple CPU reduction for partial sums
double reduceOnCPU(const std::vector<double> &partial) {
  double sum=0.0;
  for (auto v: partial) sum+=v;
  return sum;
}

int main(int argc, char** argv){
  char *inputFilename = NULL;
  int numIterations = 80;
  int opt=0;

  // Parse command-line arguments
  while ((opt = getopt(argc, argv, "f:i:")) != -1) {
    switch (opt) {
      case 'f':
        inputFilename = optarg;
        break;
      case 'i':
        numIterations = atoi(optarg);
        break;
      case '?':
      default:
        break;
    }
  }

  if (inputFilename == NULL) {
    std::cerr << "Please provide an input file with -f" << std::endl;
    return 1;
  }

  FILE *fid = fopen(inputFilename, "r");
  if (fid == NULL) {
    std::cerr << "Error opening data file: " << inputFilename << std::endl;
    return 1;
  }

  std::map<int,Page> input_pages;
  std::map<int,int> lookup, rev_lookup;

  int from_idx, to_idx;
  int num_pages=0;
  
  // Read edge list: format "from to"
  while (!feof(fid)) {
    if (fscanf(fid,"%d %d", &from_idx,&to_idx)==2) {
      if (!input_pages.count(from_idx)) {
        input_pages[from_idx]=Page();
        input_pages[from_idx].num_in_pages=0;
        input_pages[from_idx].num_out_pages=0;
        lookup[num_pages]=from_idx; rev_lookup[from_idx]=num_pages;
        num_pages++;
      }
      if (!input_pages.count(to_idx)) {
        input_pages[to_idx]=Page();
        input_pages[to_idx].num_in_pages=0;
        input_pages[to_idx].num_out_pages=0;
        lookup[num_pages]=to_idx; rev_lookup[to_idx]=num_pages;
        num_pages++;
      }
      input_pages[from_idx].num_out_pages++;
      input_pages[to_idx].num_in_pages++;
      input_pages[to_idx].incoming_ids.push_back(from_idx);
    }
  }
  fclose(fid);

  if (num_pages == 0) {
    std::cerr << "No edges read from the input file. Check file format." << std::endl;
    return 1;
  }

  // Initialize data structures
  std::vector<int> out_link_cnts(num_pages);
  std::vector<double> out_link_cnts_rcp(num_pages);
  std::vector<Page> pages(num_pages);

  for (int i=0; i<num_pages; i++){
    int idx = lookup[i];
    pages[i].ID = idx;
    pages[i].incoming_ids = input_pages[idx].incoming_ids;
    pages[i].num_in_pages = input_pages[idx].num_in_pages;
    pages[i].num_out_pages = input_pages[idx].num_out_pages;
    pages[i].size_incoming_ids = (int)pages[i].incoming_ids.size();
    out_link_cnts[i] = pages[i].num_out_pages;
    out_link_cnts_rcp[i] = (pages[i].num_out_pages > 0) ? (1.0/pages[i].num_out_pages):0.0;
  }

  // Flatten incoming edges
  std::vector<int> incoming_id_starts(num_pages+1,0);
  {
    int counter=0;
    for (int i=0; i<num_pages; i++){
      incoming_id_starts[i]=counter;
      counter += pages[i].size_incoming_ids;
    }
    incoming_id_starts[num_pages]=counter;
  }
  std::vector<int> incoming_id_list(incoming_id_starts[num_pages]);
  for (int i=0; i<num_pages; i++){
    int start = incoming_id_starts[i];
    for (int j=0; j<pages[i].size_incoming_ids; j++){
      incoming_id_list[start+j]= rev_lookup[pages[i].incoming_ids[j]];
    }
  }

  // Identify dangling pages
  std::vector<int> dangling_pages;
  for (int i=0; i<num_pages; i++){
    if (out_link_cnts[i]==0) dangling_pages.push_back(i);
  }

  // Initialize PageRank vector
  std::vector<float> pr(num_pages, 1.0f/num_pages);
  std::vector<float> old_pr(num_pages, 0.0f);
  for (int i=0; i<10 && i<num_pages; i++){
    std::cout << "Page " << i << " PR: " << pr[i] << std::endl;
  }
  double damping_factor=0.85;
  double random_val = (1.0 - damping_factor)/num_pages;

  // Allocate GPU memory
  float *d_old_pr, *d_new_pr;
  double *d_out_link_cnts_rcp;
  int *d_incoming_id_starts, *d_incoming_id_list, *d_dangling_pages = NULL;

  CHECK_CUDA(cudaMalloc(&d_old_pr,sizeof(float)*num_pages));
  CHECK_CUDA(cudaMalloc(&d_new_pr,sizeof(float)*num_pages));
  CHECK_CUDA(cudaMalloc(&d_out_link_cnts_rcp,sizeof(double)*num_pages));
  CHECK_CUDA(cudaMalloc(&d_incoming_id_starts,sizeof(int)*(num_pages+1)));
  CHECK_CUDA(cudaMalloc(&d_incoming_id_list,sizeof(int)*incoming_id_list.size()));
  if (!dangling_pages.empty()) {
    CHECK_CUDA(cudaMalloc(&d_dangling_pages,sizeof(int)*dangling_pages.size()));
  }

  CHECK_CUDA(cudaMemcpy(d_out_link_cnts_rcp,out_link_cnts_rcp.data(),sizeof(double)*num_pages,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_incoming_id_starts,incoming_id_starts.data(),sizeof(int)*(num_pages+1),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_incoming_id_list,incoming_id_list.data(),sizeof(int)*incoming_id_list.size(),cudaMemcpyHostToDevice));
  if (!dangling_pages.empty()) {
    CHECK_CUDA(cudaMemcpy(d_dangling_pages,dangling_pages.data(),sizeof(int)*dangling_pages.size(),cudaMemcpyHostToDevice));
  }

  int blockSize=256;
  int gridSize=(num_pages+blockSize-1)/blockSize;

  // Create CUDA events for timing
  cudaEvent_t start_event, stop_event;
  CHECK_CUDA(cudaEventCreate(&start_event));
  CHECK_CUDA(cudaEventCreate(&stop_event));

  // Record start
  CHECK_CUDA(cudaEventRecord(start_event, 0));

  for (int iter=0; iter<numIterations; iter++){
    // old_pr = pr
    CHECK_CUDA(cudaMemcpy(d_old_pr,pr.data(),sizeof(float)*num_pages,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_new_pr,pr.data(),sizeof(float)*num_pages,cudaMemcpyHostToDevice));

    // Compute new PR from old PR
    AddPagesPrKernel<<<gridSize,blockSize>>>(num_pages, d_incoming_id_starts, d_incoming_id_list, d_old_pr, d_out_link_cnts_rcp, d_new_pr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Handle dangling pages
    if (!dangling_pages.empty()) {
      int num_dangling=(int)dangling_pages.size();
      int dgrid = (num_dangling+blockSize-1)/blockSize;

      double *d_partial_sums;
      CHECK_CUDA(cudaMalloc(&d_partial_sums,sizeof(double)*dgrid));

      SumDanglingKernel<<<dgrid,blockSize>>>(d_dangling_pages, num_dangling, d_old_pr, d_partial_sums);
      CHECK_CUDA(cudaDeviceSynchronize());

      std::vector<double> partial(dgrid);
      CHECK_CUDA(cudaMemcpy(partial.data(),d_partial_sums,sizeof(double)*dgrid,cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaFree(d_partial_sums));

      double dp_sum = reduceOnCPU(partial);
      double val = dp_sum/num_pages;

      AddDanglingPagesPrKernel<<<gridSize,blockSize>>>(num_pages, d_new_pr, val);
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Add Random Jumps
    AddRandomJumpsPrKernel<<<gridSize,blockSize>>>(num_pages, d_new_pr, damping_factor, random_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back to pr for next iteration
    CHECK_CUDA(cudaMemcpy(pr.data(),d_new_pr,sizeof(float)*num_pages,cudaMemcpyDeviceToHost));
  }

  // Record stop
  CHECK_CUDA(cudaEventRecord(stop_event, 0));
  CHECK_CUDA(cudaEventSynchronize(stop_event));

  float milliseconds = 0;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

  // Cleanup GPU memory
  cudaFree(d_old_pr);
  cudaFree(d_new_pr);
  cudaFree(d_out_link_cnts_rcp);
  cudaFree(d_incoming_id_starts);
  cudaFree(d_incoming_id_list);
  if (d_dangling_pages != NULL) cudaFree(d_dangling_pages);

  // Destroy events
  CHECK_CUDA(cudaEventDestroy(start_event));
  CHECK_CUDA(cudaEventDestroy(stop_event));

  // Print first 10 ranks
  for (int i=0; i<10 && i<num_pages; i++){
    std::cout << "Page " << i << " PR: " << pr[i] << std::endl;
  }

  // Print computation time
  std::cout << "Total computation time for " << numIterations << " iterations: " << milliseconds << " ms" << std::endl;

  return 0;
}
