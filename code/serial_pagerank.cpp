#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

int main() {
    using namespace std::chrono;
    

    float d = 0.85f;

    const char filename[] = "./Wiki-Vote.txt";
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 1;
    }

    int node1, node2;
    std::vector<std::pair<int,int>> edges;
    std::map<int,int> node_map; // Maps original node ID to compressed ID

    // Read all edges and collect unique nodes
    while (fscanf(fp, "%d %d", &node1, &node2) == 2) {
        edges.push_back({node1, node2});
        // Insert nodes into map if not already present
        if (node_map.find(node1) == node_map.end()) {
            node_map[node1] = -1; // placeholder
        }
        if (node_map.find(node2) == node_map.end()) {
            node_map[node2] = -1; // placeholder
        }
    }
    fclose(fp);

    if (node_map.empty()) {
        std::cerr << "No edges were read from the input file." << std::endl;
        return 1;
    }

    // Assign new IDs from 0 to n-1
    int new_id = 0;
    for (auto &kv : node_map) {
        kv.second = new_id++;
    }
    int n = (int)node_map.size();
    std::cout << "\nNumber of nodes = " << n << "\n";

    // Now we have n unique nodes. We must re-map edges to new IDs.
    for (auto &e : edges) {
        e.first = node_map[e.first];
        e.second = node_map[e.second];
    }
    auto begin = high_resolution_clock::now();
    // Allocate adjacency matrix
    std::vector<std::vector<float>> a(n, std::vector<float>(n, 0.0f));
    std::vector<int> out_link(n, 0);

    // Process edges to build adjacency matrix
    for (auto &e : edges) {
        a[e.first][e.second] = 1.0f;
        out_link[e.first]++;
    }

    // Handle dangling nodes and normalize
    for (int i = 0; i < n; i++) {
        if (out_link[i] == 0) {
            for (int j = 0; j < n; j++) {
                a[i][j] = 1.0f / n;
            }
        } else {
            for (int j = 0; j < n; j++) {
                if (a[i][j] != 0.0f) {
                    a[i][j] /= out_link[i];
                }
            }
        }
    }

    // Create transpose matrix
    std::vector<std::vector<float>> at(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            at[j][i] = a[i][j];
        }
    }

    std::vector<float> p(n, 1.0f/n);
    std::vector<float> p_new(n, 0.0f);

    float tolerance = 0.0000001f;
    int iter = 0;

    // PageRank iterations
    for (iter = 0; iter < 100; iter++) {
        for (int i = 0; i < n; i++) {
            p_new[i] = 0.0f;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                p_new[i] += at[i][j] * p[j];
            }
        }
        for (int i = 0; i < n; i++) {
            p_new[i] = d * p_new[i] + (1.0f - d) / n;
        }
        float error = 0.0f;
        for (int i = 0; i < n; i++) {
            error += std::fabs(p_new[i] - p[i]);
        }
        if (error < tolerance) {
            break;
        }
        p = p_new;
    }

    auto end = high_resolution_clock::now();
    double time_spent = duration_cast<duration<double>>(end - begin).count();

    std::cout << "Number of iterations to converge: " << iter << "\n\n";
    std::cout << "Final PageRank values:\n[";
    for (int i = 0; i < std::min(n, 10); i++) {
        std::cout << p[i];
        if (i != std::min(n,10) - 1) std::cout << ", ";
    }
    std::cout << "]\n\nTime spent: " << time_spent << " seconds.\n";

    return 0;
}
