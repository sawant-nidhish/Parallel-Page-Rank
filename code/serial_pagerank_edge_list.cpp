#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <chrono>
#include <algorithm>

using namespace std;

// Data structure to hold metadata for webpages
struct Page {
  int ID;
  std::vector<int> incoming_ids;
  int size_incoming_ids;
  int num_in_pages;
  int num_out_pages;
  double page_rank;
};

// Helper function to make a vector of all nodes with no outlinks
std::vector<int> ExploreDanglingPages(std::vector<int> &out_link_cnts) {
  std::vector<int> dangling_pages;
  for (int i = 0; i < (int)out_link_cnts.size(); i++) {
    if (out_link_cnts[i] == 0) {
      dangling_pages.push_back(i);
    }
  }
  return dangling_pages;
}

// Helper function to initialize page rank
std::vector<float> InitPr(int page_cnt) {
  std::vector<float> pr;
  pr.reserve(page_cnt);
  float init_pr = 1.0f/page_cnt;
  for (int i = 0; i < page_cnt; i++) {
    pr.push_back(init_pr);
  }
  return pr;
}

// First step of main computation
void AddPagesPr(
    std::vector<Page> &pages,
    std::vector<double> &out_link_cnts_rcp,
    std::vector<float> &old_pr,
    std::vector<float> &new_pr)
{
  int num_pages = (int)pages.size();
  for (int i = 0; i < num_pages; i++) {
    double sum = 0.0;
    int  num_incoming = pages[i].size_incoming_ids;
    // Loop unrolling to speed up summation
    for (int j = 0 ; j < (num_incoming/4)*4; j+=4){
      int in_id1 = pages[i].incoming_ids[j]; 
      int in_id2 = pages[i].incoming_ids[j+1]; 
      int in_id3 = pages[i].incoming_ids[j+2]; 
      int in_id4 = pages[i].incoming_ids[j+3]; 
      sum += (old_pr[in_id1] * out_link_cnts_rcp[in_id1]) +
             (old_pr[in_id2] * out_link_cnts_rcp[in_id2]) +
             (old_pr[in_id3] * out_link_cnts_rcp[in_id3]) +
             (old_pr[in_id4] * out_link_cnts_rcp[in_id4]);
    }
    for (int j = (num_incoming/4)*4; j < num_incoming; j++){
      int in_id1 = pages[i].incoming_ids[j];
      sum += (old_pr[in_id1] * out_link_cnts_rcp[in_id1]);
    }
    new_pr[i] = (float)sum;
  }
}

// Helper function to add effect of dangling pages
void AddDanglingPagesPr(
    std::vector<int> &dangling_pages,
    std::vector<float> &old_pr,
    std::vector<float> &new_pr) {

  double sum = 0.0;
  int dangling_pages_size = (int)dangling_pages.size();
  for (int i = 0; i < dangling_pages_size; i++) {
    sum += old_pr[dangling_pages[i]];
  }
  int new_pr_size = (int)new_pr.size();
  double val = sum/new_pr_size;

  for (int i = 0; i < new_pr_size; i++) {
    new_pr[i] += (float)val;
  }
}

// Helper function to calculate random surfer
void AddRandomJumpsPr(
  double damping_factor,
  std::vector<float> &new_pr) {
  int new_pr_size = (int)new_pr.size();
  double val = (1 - damping_factor) / new_pr_size;

  for (int i = 0; i < new_pr_size; i++) {
    new_pr[i] = (float)(new_pr[i]*damping_factor + val); 
  }
}

int main(int argc, char** argv){
  using namespace std::chrono;
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> dsec;

  int numIterations = 80;
  char *inputFilename = NULL;
  int opt = 0;

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

  std::map<int, Page> input_pages;
  std::map<int,int> lookup;
  std::map<int,int> rev_lookup;

  FILE *fid;
  fid = fopen(inputFilename, "r");
  if (fid == NULL){
    std::cerr << "Error opening data file: " << inputFilename << std::endl;
    return 1;
  }

  int from_idx, to_idx;
  int num_pages = 0;
  auto load_start = Clock::now();

  // Read edgelist: "from to"
  while (!feof(fid)) {
    if (fscanf(fid,"%d %d\n", &from_idx,&to_idx) == 2) {
      if (!input_pages.count(from_idx)) {
        input_pages[from_idx] = Page();
        input_pages[from_idx].num_in_pages=0;
        input_pages[from_idx].num_out_pages=0;
        input_pages[from_idx].page_rank=0;
        lookup[num_pages]=from_idx; rev_lookup[from_idx]=num_pages;
        num_pages++;
      }
      if (!input_pages.count(to_idx)) {
        input_pages[to_idx]=Page();
        input_pages[to_idx].num_in_pages=0;
        input_pages[to_idx].num_out_pages=0;
        input_pages[to_idx].page_rank=0;
        lookup[num_pages]=to_idx; rev_lookup[to_idx]=num_pages;
        num_pages++;
      }
      input_pages[from_idx].num_out_pages++;
      input_pages[to_idx].num_in_pages++;
      input_pages[to_idx].incoming_ids.push_back(from_idx);
    }
  }
  fclose(fid);

  std::cout << "Num pages: " << num_pages << std::endl;

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

  double load_time = duration_cast<dsec>(Clock::now() - load_start).count();
  std::cout << "Graph data loaded in: " << load_time << " seconds." << std::endl;

  // Initialize Dangling Pages, Page Rank
  std::vector<int> dangling_pages = ExploreDanglingPages(out_link_cnts);
  std::vector<float> pr = InitPr(num_pages);
  std::vector<float> old_pr(num_pages);

  // Print initial PR for the first 10 pages
  std::cout << "Initial first 10 PageRanks:" << std::endl;
  for (int i=0; i<10 && i<num_pages; i++){
    std::cout << "Page " << i << " PR: " << pr[i] << std::endl;
  }

  double damping_factor = 0.85;
  
  // Start computation timer
  auto compute_start = Clock::now();

  for (int iter = 0; iter < numIterations; iter++){
    // Save old PR
    std::copy(pr.begin(), pr.end(), old_pr.begin());

    // Compute new PR from old PR
    AddPagesPr(pages, out_link_cnts_rcp, old_pr, pr);

    // Add dangling pages
    AddDanglingPagesPr(dangling_pages, old_pr, pr);

    // Add random jumps
    AddRandomJumpsPr(damping_factor, pr);
  }

  double compute_time = duration_cast<dsec>(Clock::now() - compute_start).count();

  // Print final PR for the first 10 pages
  std::cout << "Final first 10 PageRanks after " << numIterations << " iterations:" << std::endl;
  for (int i=0; i<10 && i<num_pages; i++){
    std::cout << "Page " << i << " PR: " << pr[i] << std::endl;
  }
  std::cout << "Total computation time for " << numIterations << " iterations: " << compute_time << " seconds" << std::endl;

  return 0;
}
