#pragma once
#include <vector>
#include <thrust/device_vector.h>

#include "common/vec_math.h"

struct SimulatorParams;
struct SolverBase;

struct SewingData {
    int start_idx; // Start index of stitches
    int count; // Count of stitches
    float angle; // Dihedral angel of sewing line
    float compress; // Compress lenght of sewing line
};

struct SimulatorParams {
    int nb_all_vertices;
    int nb_all_edges;
    int nb_all_triangles;
    int nb_all_stitches;
    int nb_all_cloth_vertices;
    int nb_all_cloth_edges;
    int nb_all_cloth_triangles;
	float cloth_edge_mean_length;
};
class Simulator{
private:
    Simulator();
    Simulator(const Simulator&) = delete;
    Simulator& operator=(const Simulator&) = delete;
    ~Simulator();
    
public:
    static Simulator& instance(){
        static Simulator s{};
        return s;
    }
    void init(const std::vector<float>& vertices,const std::vector<float>& vertices_sim,const std::vector<int>&edges,
        const std::vector<int>&triangles,
        const std::vector<int>& object_types,
        const std::vector<float>& mass,
        const std::vector<Mat4>& world_matrixs,
        const std::vector<int>& _vertices_size,
        const std::vector<int>& triangle_index_offsets,
        const std::vector<SewingData>& sewings,
        const std::vector<int2>& stitches,
        int nb_all_cloth_v,int nb_all_cloth_e,int nb_all_cloth_f);
    void update(float h);
    void copy_vertices(float*,bool world_space);
    const SimulatorParams* get_params() const;
    void reset();

    int add_pick_triangle(int mesh_index,int tri_index, float3 position);
    void update_pick_triangle(int index, float3 position);
    void remove_pick_triangle(int index);
    void clear_pick_triangle();

    // return picker index
    int add_picker(float3 position);
    // return picked triangle index or -1
    void update_picker(int index, float3 position);
    void remove_picker(int index);
    void clear_picker();
private:
    SolverBase* m_solver;
    
};

