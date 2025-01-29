#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <map>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <chrono>
#include <mpi.h>
#include <cstdlib>
#include <unistd.h>
#include "immintrin.h"
using namespace std;

//Data structure to hold metadata for webpages
struct Page {
  int ID;
  vector <int> incoming_ids;
  int size_incoming_ids;
  int num_in_pages;
  int num_out_pages;
  double page_rank;
};

//Helper function to make a vector of all nodes with no outlinks
std::vector<int> ExploreDanglingPages(std::vector<int> &out_link_cnts) {
	std::vector<int> dangling_pages;
	for (int i = 0; i < out_link_cnts.size(); i++) {
		if (out_link_cnts[i] == 0) {
			dangling_pages.push_back(i);
		}
	}
	return dangling_pages;
}

//Helper function to initialize page rank
std::vector<float> InitPr(int page_cnt) {
	std::vector<float> pr;
	pr.reserve(page_cnt);
  float init_pr = 1.0/page_cnt;
	for (int i = 0; i < page_cnt; i++) {
		pr.push_back(init_pr);
	}
	return pr;
}

//First step of main computation 
void AddPagesPr(
		std::vector<Page> &pages,
		std::vector<double> out_link_cnts_rcp,
		std::vector<float> &old_pr,
		std::vector<float> &new_pr,
    int start_index,
    int end_index,
    int procID) {

  //Handling remainder nodes 
  if (start_index == end_index) {
    	double sum = 0;
    int  num_incoming = pages[start_index].size_incoming_ids;
    for (int j = 0; j < num_incoming; j++){
      int in_id1 = pages[start_index].incoming_ids[j];
      sum = sum +  (old_pr[in_id1] * out_link_cnts_rcp[in_id1]);
    }
    new_pr[start_index] = sum;
  }

	for (int i = start_index; i < end_index; i++) {
    //Loop unrolling
		double sum = 0;
    int  num_incoming = pages[i].size_incoming_ids;
    for (int j = 0 ; j < (num_incoming/4)*4; j+= 4){
      int in_id1 = pages[i].incoming_ids[j]; 
      int in_id2 = pages[i].incoming_ids[j+1]; 
      int in_id3 = pages[i].incoming_ids[j + 2]; 
      int in_id4 = pages[i].incoming_ids[j+3]; 
      sum = sum +  (old_pr[in_id1] * out_link_cnts_rcp[in_id1]) + 
        (old_pr[in_id2] * out_link_cnts_rcp[in_id2]) + (old_pr[in_id3] * out_link_cnts_rcp[in_id3]) 
        +  (old_pr[in_id4] * out_link_cnts_rcp[in_id4]);
    }
    for (int j = (num_incoming/4)*4; j < num_incoming; j++){
      int in_id1 = pages[i].incoming_ids[j];
      sum = sum +  (old_pr[in_id1] * out_link_cnts_rcp[in_id1]);
    }
    new_pr[i] = sum;
	}  
}

//Helper function to add effect of dangling pages
void AddDanglingPagesPr(
		std::vector<int> &dangling_pages,
		std::vector<float> &old_pr,
		std::vector<float> &new_pr) {
	
  double sum = 0;
  int dangling_pages_size = dangling_pages.size();
  for (int i = 0; i < dangling_pages_size; i++) {
    sum += old_pr[dangling_pages[i]];
  }
  int new_pr_size = new_pr.size();
  double val = sum/new_pr_size;

  for (int i = 0; i < new_pr_size; i++) {
    new_pr[i] += val;
  }
}

//Helper function to calculate random surfer
void AddRandomJumpsPr(
  double damping_factor,
  std::vector<float> &new_pr) {
  int new_pr_size = new_pr.size();
  double val = (1- damping_factor) * 1.0 /new_pr_size;

  for (int i = 0; i < new_pr_size; i++) {
		new_pr[i] = new_pr[i] * damping_factor + val; 
	}
}

//Helper function to test convergence 
double L1Norm(
		const std::vector<double> a,
		const std::vector<double> b) {
	double sum = 0;
	for (int i = 0; i < a.size(); i++) {
		sum += std::abs(a[i] - b[i]);
	}
	return sum;
}

int main(int argc, char** argv){
  using namespace std::chrono;
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> dsec;

  int procID;
  int nproc;
  int span, startIndex, endIndex, remainder;
  int num_threads, numIterations;
  int opt = 0;
  MPI_Init(&argc, &argv);
  char *inputFilename = NULL;

  MPI_Comm_rank(MPI_COMM_WORLD, &procID);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  
  if (procID == 0){
    cout << "Total procs: " << nproc << endl;
  }

  do {
    opt = getopt(argc, argv, "f:n:i:");
    switch (opt) {
    case 'f':
      inputFilename = optarg;
      break;

    case 'n':
      num_threads = atoi(optarg);
      break;

    case 'i':
      numIterations = atoi(optarg);
      break;

    case -1:
      break;

    default:
      break;
    }
  } while (opt != -1);


    map <int, Page> input_pages;
    map <int, int> lookup;
    map<int, int > rev_lookup;

    std::vector<Page> pages;
    std::vector<int> out_link_cnts;
    std::vector<double> out_link_cnts_rcp;

    int from_idx, to_idx;
    FILE *fid;
    fid = fopen(inputFilename, "r");
    if (fid == NULL){
      printf("Error opening data file\n");
   }

  int num_pages = 0;
  int from_idx_pos, to_idx_pos;;
  auto load_start = Clock::now();
  double load_time = 0;

  //Read edgelist into map to handle incomplete indices 
  while (!feof(fid)) {
    if (fscanf(fid,"%d %d\n", &from_idx,&to_idx)) {
      if (!input_pages.count(from_idx)) {
        input_pages[from_idx] = Page();
        input_pages[from_idx].num_in_pages = 0;
        input_pages[from_idx].num_out_pages = 0;
        input_pages[from_idx].page_rank = 0;
        lookup[num_pages] = from_idx;
        rev_lookup[from_idx] = num_pages;
        num_pages++;
      }
      if (!input_pages.count(to_idx)) {
        input_pages[to_idx] = Page();
        input_pages[to_idx].num_in_pages = 0;
        input_pages[to_idx].num_out_pages = 0;
        input_pages[to_idx].page_rank = 0;
        lookup[num_pages] = to_idx;
        rev_lookup[to_idx] = num_pages;
        num_pages++;
      }
      input_pages[from_idx].num_out_pages++;
      input_pages[to_idx].num_in_pages++;
      input_pages[to_idx].incoming_ids.push_back(from_idx);
    }
   }


  cout << "Num pages: " << num_pages<< endl;
  out_link_cnts.reserve(num_pages);
  out_link_cnts_rcp.reserve(num_pages);
  
  //Initialize vector containing outgoing links 
  int idx;
  for (int i=0; i<num_pages;i++){
    idx = lookup[i];  
    out_link_cnts.push_back(input_pages[idx].num_out_pages);
    if (input_pages[idx].num_out_pages != 0) {
      out_link_cnts_rcp.push_back(1.0/input_pages[idx].num_out_pages);  
    } else {
      out_link_cnts_rcp.push_back(0);
    }
  }

  //Initialize vector of structs containing Page data strucutre
  int vec_idx;
  for (int i=0; i<num_pages;i++){
    pages.push_back(Page());
    idx = lookup[i];
    pages[i].num_in_pages = input_pages[idx].num_in_pages;
    pages[i].ID = idx;
    for (int j=0;j<input_pages[idx].incoming_ids.size();j++){
      vec_idx = rev_lookup[input_pages[idx].incoming_ids[j]];
      pages[i].incoming_ids.push_back(vec_idx);
    } 
    pages[i].size_incoming_ids = pages[i].incoming_ids.size(); 
  }

  load_time += duration_cast<dsec>(Clock::now() - load_start).count();
  printf("Graph data loaded in: %lf.\n", load_time);

  //Initialize Dangling Pages, Page Rank
  std::vector<int> dangling_pages = ExploreDanglingPages(out_link_cnts);
  std::vector<float> pr = InitPr(num_pages);
	std::vector<float> old_pr(num_pages);
  if (procID == 0) {
    std::cout << "First 10 PageRanks:" << std::endl;
    for (int i=0; i<10 && i<num_pages; i++){
      std::cout << "Page " << i << " PR: " << pr[i] << std::endl;
    }
  }
  //Partition workload 
  int reminder_cnt = num_pages % nproc;
	int reminder_begin = num_pages - reminder_cnt;
	int reminder_end = num_pages - 1;
  int chunk_size = num_pages/nproc;
  int start_index = procID * chunk_size;
  int end_index = start_index + chunk_size;

  auto compute_start = Clock::now();
  double compute_time = 0;
  double broadcast_time = 0;
  double other_compute = 0;

  for (int iter = 0; iter < 80; iter++){

    //Broadcast page rank data to each node 
    MPI_Bcast(pr.data(), num_pages, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
    //Make a copy to calculate dangling pages later
    if (procID == 0) {
			std::copy(pr.begin(), pr.end(), old_pr.begin());
		}

		AddPagesPr(pages, out_link_cnts_rcp, old_pr, pr, start_index, end_index, procID);
    
    //Each node gathers updated page rank 
    MPI_Allgather(&(pr[start_index]), chunk_size, MPI_FLOAT, &(pr[0]), chunk_size, MPI_FLOAT, MPI_COMM_WORLD);
  
  //Compute remainder at root
  if (procID == 0){
    AddPagesPr(pages, out_link_cnts_rcp, old_pr, pr, reminder_begin, reminder_end, procID);
    AddDanglingPagesPr(dangling_pages, old_pr, pr);
		AddRandomJumpsPr(0.85, pr); 
    }
  }

  compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
  //printf("Computation Time: %lf.\n", compute_time);
   //Print first 10 PageRanks and total computation time on root
  // Print first 10 PageRanks and total computation time on root
  if (procID == 0) {
    std::cout << "First 10 PageRanks:" << std::endl;
    for (int i=0; i<10 && i<num_pages; i++){
      std::cout << "Page " << i << " PR: " << pr[i] << std::endl;
    }
    std::cout << "Total computation time for 80 iterations: " << compute_time << " seconds" << std::endl;
  }
  
  MPI_Finalize();
 
}
