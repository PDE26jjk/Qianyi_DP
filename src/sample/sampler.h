#include <vector>
struct Params
{
	float radius;
	float one_grid_length;
	int grid_size;
	int max_size;
	int n;
};

class Sampler
{

    public:
	Params params;

	// Device pointers
	unsigned char* d_grid_status = nullptr;
	int* d_grid_point = nullptr;
	float2* d_final = nullptr;
	int* d_grid_multi_point = nullptr; // flattened 3D array
	unsigned char* d_grid_multi_point_size = nullptr;
	float2* d_force = nullptr;
	unsigned char* d_valid_status = nullptr;

	int* d_nb_points = nullptr;
	float current_radius_scaled = FLT_MAX;

	// RNG states
	curandState* d_rng_states = nullptr;

	// Helper device memory for inputs
	float2* d_input_points = nullptr;
	int* d_input_next = nullptr;
    Sampler();
	~Sampler();
    void set_radius(float _radius);

    std::vector<float2> sample(
		const std::vector<float2>& boundary_points,
		const std::vector<int>& next_point,
		float raw_radius,
		float f1, int t1, float f2, int t2
	);
};