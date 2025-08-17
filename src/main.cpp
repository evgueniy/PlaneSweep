#include "../kernels/main.cuh"
#include "cam_params.hpp"
#include "constants.hpp"
#include "graph.h"
#include <chrono>
#include <thread>
#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
//#include <math>
#include <string>
#include <fstream> 

#define SHRT_MAX 32767
/*
Function to store cost cube because it takes too long to compare results
*/
void store_cost_cube(const std::vector<cv::Mat>& cost_cube, const std::string& filename) {
	std::ofstream outFile(filename, std::ios::binary);
	if (!outFile) {
		printf("Error: Cannot open file for writing: %s\n", filename.c_str());
		return;
	}

	size_t planes = cost_cube.size();
	int height = cost_cube[0].rows;
	int width = cost_cube[0].cols;

	outFile.write(reinterpret_cast<const char*>(&planes), sizeof(planes));
	outFile.write(reinterpret_cast<const char*>(&height), sizeof(height));
	outFile.write(reinterpret_cast<const char*>(&width), sizeof(width));

	for (const auto& mat : cost_cube) {
		outFile.write(reinterpret_cast<const char*>(mat.data), mat.total() * mat.elemSize());
	}
	printf("Cost cube saved to %s\n", filename.c_str());
}


std::vector<cam> read_cams(std::string const &folder)
{
	// Init parameters
	std::vector<params<double>> cam_params_vector = get_cam_params();

	// Init cameras
	std::vector<cam> cam_array(cam_params_vector.size());
	for (int i = 0; i < cam_params_vector.size(); i++)
	{
		// Name
		std::string name = folder + "/v" + std::to_string(i) + ".png";

		// Read PNG file
		cv::Mat im_rgb = cv::imread(name);
		cv::Mat im_yuv;
		const int width = im_rgb.cols;
		const int height = im_rgb.rows;

		// Convert to YUV420
		cv::cvtColor(im_rgb, im_yuv, cv::COLOR_BGR2YUV_I420);
		const int size = width * height * 1.5; // YUV 420
		cv::Mat Y(height, width, CV_8UC1, im_yuv.data);
		uint8_t* uv_ptr = im_yuv.data + width * height;
		cv::Mat U(height / 2, width / 2, CV_8UC1, uv_ptr);
		uint8_t* v_ptr = uv_ptr + (width / 2) * (height / 2);
		cv::Mat V(height / 2, width / 2, CV_8UC1, v_ptr);
		/*std::cout
			<< "dims=" << im_yuv.dims
			<< " rows=" << im_yuv.rows
			<< " cols=" << im_yuv.cols
			<< " total=" << im_yuv.total()
			<< " ch=" << im_yuv.channels()
			<< " elemBytes=" << im_yuv.elemSize()
			<< " elemBytes1=" << im_yuv.elemSize1()
			<< " continuous=" << (im_yuv.isContinuous() ? "yes" : "no")
			<< " step0=" << im_yuv.step[0]
			<< std::endl;*/
		std::vector<cv::Mat> YUV = {Y.clone(),U.clone(),V.clone()};

		//cv::split(im_yuv, YUV);
		std::cout
			<< "dims=" << YUV.size()
			<< " elem=" << YUV.at(0).size()
			<< std::endl;
		// Params
		cam_array.at(i) = cam(name, width, height, size, YUV, cam_params_vector.at(i));
	}

	return cam_array;

	// cv::Mat U(height / 2, width / 2, CV_8UC1, cam_array.at(0).image.data() + (int)(width * height * 1.25));
	// cv::namedWindow("im", cv::WINDOW_NORMAL);
	// cv::imshow("im", U);
	// cv::waitKey(0);
}

std::vector<cv::Mat> sweeping_plane(cam const ref, std::vector<cam> const &cam_vector, int window = 3)
{
	// Initialization to MAX value
	// std::vector<float> cost_cube(ref.width * ref.height * ZPlanes, 255.f);
	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int i = 0; i < cost_cube.size(); ++i)
	{
		cost_cube[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
	}

	// For each camera in the setup (reference is skipped)
	for (auto &cam : cam_vector)
	{
		if (cam.name == ref.name)
			continue;

		std::cout << "Cam: " << cam.name << std::endl;
		// For each pixel and candidate: (i) calculate projection index, (ii) calculate cost against reference, (iii) store minimum cost
		for (int zi = 0; zi < ZPlanes; zi++)
		{
			std::cout << "Plane " << zi << std::endl;
			for (int y = 0; y < ref.height; y++)
			{
				for (int x = 0; x < ref.width; x++)
				{
					// (i) calculate projection index

					// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
					double z = ZNear * ZFar / (ZNear + (((double)zi / (double)ZPlanes) * (ZFar - ZNear)));

					// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
					double X_ref = (ref.p.K_inv[0] * x + ref.p.K_inv[1] * y + ref.p.K_inv[2]) * z;
					double Y_ref = (ref.p.K_inv[3] * x + ref.p.K_inv[4] * y + ref.p.K_inv[5]) * z;
					double Z_ref = (ref.p.K_inv[6] * x + ref.p.K_inv[7] * y + ref.p.K_inv[8]) * z;

					// 3D in ref camera coordinates to 3D world
					double X = ref.p.R_inv[0] * X_ref + ref.p.R_inv[1] * Y_ref + ref.p.R_inv[2] * Z_ref - ref.p.t_inv[0];
					double Y = ref.p.R_inv[3] * X_ref + ref.p.R_inv[4] * Y_ref + ref.p.R_inv[5] * Z_ref - ref.p.t_inv[1];
					double Z = ref.p.R_inv[6] * X_ref + ref.p.R_inv[7] * Y_ref + ref.p.R_inv[8] * Z_ref - ref.p.t_inv[2];

					// 3D world to projected camera 3D coordinates
					double X_proj = cam.p.R[0] * X + cam.p.R[1] * Y + cam.p.R[2] * Z - cam.p.t[0];
					double Y_proj = cam.p.R[3] * X + cam.p.R[4] * Y + cam.p.R[5] * Z - cam.p.t[1];
					double Z_proj = cam.p.R[6] * X + cam.p.R[7] * Y + cam.p.R[8] * Z - cam.p.t[2];

					// Projected camera 3D coordinates to projected camera 2D coordinates
					double x_proj = (cam.p.K[0] * X_proj / Z_proj + cam.p.K[1] * Y_proj / Z_proj + cam.p.K[2]);
					double y_proj = (cam.p.K[3] * X_proj / Z_proj + cam.p.K[4] * Y_proj / Z_proj + cam.p.K[5]);
					double z_proj = Z_proj;

					x_proj = x_proj < 0 || x_proj >= cam.width ? 0 : roundf(x_proj);
					y_proj = y_proj < 0 || y_proj >= cam.height ? 0 : roundf(y_proj);

					// (ii) calculate cost against reference
					// Calculating cost in a window
					float cost = 0.0f;
					float cc = 0.0f;
					for (int k = -window / 2; k <= window / 2; k++)
					{
						for (int l = -window / 2; l <= window / 2; l++)
						{
							if (x + l < 0 || x + l >= ref.width)
								continue;
							if (y + k < 0 || y + k >= ref.height)
								continue;
							if (x_proj + l < 0 || x_proj + l >= cam.width)
								continue;
							if (y_proj + k < 0 || y_proj + k >= cam.height)
								continue;

							// Y
							/*if (zi == 0 && x == 0 && y == 0) {
								printf("Host Ref value at coordinates Y: %d - X: %d = %d\n", y+k, x+l, ref.YUV[0].at<uint8_t>(y + k, x + l));
								printf("Host Cam value at coordinates Y: %d - X: %d = %d\n", (int)y_proj + k, (int)x_proj + l, cam.YUV[0].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
							}*/
							cost += fabs(ref.YUV[0].at<uint8_t>(y + k, x + l) - cam.YUV[0].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
							// U
							// cost += fabs(ref.YUV[1].at<uint8_t >(y + k, x + l) - cam.YUV[1].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
							// V
							// cost += fabs(ref.YUV[2].at<uint8_t >(y + k, x + l) - cam.YUV[2].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
							cc += 1.0f;
						}
					}
					cost /= cc;

					//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
					// only the minimum cost for all the cameras is stored
					cost_cube[zi].at<float>(y, x) = fminf(cost_cube[zi].at<float>(y, x), cost);
				}
			}
		}
	}

	// Visualize costs
	// for (int zi = 0; zi < ZPlanes; zi++)
	// {
	// 	std::cout << "plane " << zi << std::endl;
	// 	cv::namedWindow("Cost", cv::WINDOW_NORMAL);
	// 	cv::imshow("Cost", cost_cube.at(zi) / 255.f);
	// 	cv::waitKey(0);
	// }
	return cost_cube;
}


cv::Mat find_min(std::vector<cv::Mat> const &cost_cube)
{
	const int zPlanes = cost_cube.size();
	const int height = cost_cube[0].size().height;
	const int width = cost_cube[0].size().width;

	cv::Mat ret(height, width, CV_32FC1, 255.);
	cv::Mat depth(height, width, CV_8U, 255);

	for (int zi = 0; zi < zPlanes; zi++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				if (cost_cube[zi].at<float>(y, x) < ret.at<float>(y, x))
				{
					ret.at<float>(y, x) = cost_cube[zi].at<float>(y, x);
					depth.at<u_char>(y, x) = zi;
				}
			}
		}
	}

	return depth;
}

/*The next two function are used to perform the graph cut on the results
DO NOT MODIFY THOSE FUNCTIONS - DO NOT TRY TO IMPLEMENT THEM ON THE GPU*/
void depth_estimation_by_graph_cut_sWeight_add_nodes(Graph& g, std::vector<Graph::node_id>& nodes, cv::Size destPixel, cv::Size sourcePixel, cv::Size imgSize, std::vector<double> m_aiEdgeCost, cv::Mat1w labels, int label, double cost_cur) {
	const int idxSourcePixel = sourcePixel.height * imgSize.width + sourcePixel.width;
	const int idxDestPixel = destPixel.height * imgSize.width + destPixel.width;
	const double cost_cur_temp = cost_cur;

	if (labels(sourcePixel.height, sourcePixel.width) != labels(destPixel.height, destPixel.width)) {
		//add a new node and add edge between it and the adjacent nodes
		Graph::node_id tmp_node = g.add_node();
		const double cost_temp = m_aiEdgeCost[std::abs(labels(destPixel.height, destPixel.width) - label)];
		g.set_tweights(tmp_node, 0, m_aiEdgeCost[std::abs(labels(sourcePixel.height, sourcePixel.width) - labels(destPixel.height, destPixel.width))]);
		g.add_edge(nodes[idxSourcePixel], tmp_node, cost_cur_temp, cost_cur_temp);
		g.add_edge(tmp_node, nodes[idxDestPixel], cost_temp, cost_temp);
	}
	else //only add an edge between two nodes
		g.add_edge(nodes[idxSourcePixel], nodes[idxDestPixel], cost_cur_temp, cost_cur_temp);
}
//DO NOT TRY TO IMPLEMENT THIS FUNCTION ON THE GPU
cv::Mat depth_estimation_by_graph_cut_sWeight(std::vector<cv::Mat> const& cost_cube) {
	//DO NOT TRY TO IMPLEMENT THIS FUNCTION ON THE GPU

	const int zPlanes = cost_cube.size();
	const int height = cost_cube[0].size().height;
	const int width = cost_cube[0].size().width;

	//To store the depth values assigned to each pixels, start with 0
	cv::Mat1w labels = cv::Mat::zeros(height, width, CV_16U); 
	//store the cost for a label
	std::vector<double> m_aiEdgeCost;
	double smoothing_lambda = 1.0;
	m_aiEdgeCost.resize(zPlanes);
	for (int i = 0; i < zPlanes; ++i)
		m_aiEdgeCost[i] = smoothing_lambda * i;

	for (int source = 0; source < zPlanes; ++source) {
		printf("depth layer %i \n", source);
		Graph g;
		std::vector<Graph::node_id> nodes(height * width, nullptr);

		//Putting the weights for the connection to the source and the sink for each nodes
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				//indice global du pixel
				const int pp = r * width + c;
				nodes[pp] = g.add_node();
				const ushort label = labels(r, c);
				if (label == source)
					g.set_tweights(nodes[pp], cost_cube[source].at<float>(r, c), SHRT_MAX);
				else
					g.set_tweights(nodes[pp], cost_cube[source].at<float>(r, c), cost_cube[label].at<float>(r, c));
			}
		}

		
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				const double cost_curr = m_aiEdgeCost[std::abs(labels(j, i) - source)];

				//create an edge between the adjacent nodes, may add an additional node on this edge if the previously calculated labels are different
				if (i != width - 1) {
					depth_estimation_by_graph_cut_sWeight_add_nodes(g, nodes, cv::Size(i + 1, j), cv::Size(i, j), cv::Size(width, height), m_aiEdgeCost, labels, source, cost_curr);
				}
				if (j != height - 1) {
					depth_estimation_by_graph_cut_sWeight_add_nodes(g, nodes, cv::Size(i, j + 1), cv::Size(i, j), cv::Size(width, height), m_aiEdgeCost, labels, source, cost_curr);
				}
			}
		}
		//printf("nodes and egde set \n");

		//resolve the maximum flow/minimum cut problem
		g.maxflow();

		//update the depth labels, nodes that are still connected to the source will receive a new depth label
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				const int pp = r * width + c;
				if (g.what_segment(nodes[pp]) != Graph::SOURCE)
					labels(r, c) = ushort(source);
			}
		}
		nodes.clear();
		
		/*
		cv::namedWindow("labels", cv::WINDOW_NORMAL);
		cv::imshow("labels", labels);
		cv::waitKey(0);
		*/

	}

	cv::Mat depth;
	labels.convertTo(depth, CV_8U, 1.0);

	return depth;
}
/*
*  function to check if 2 cost cubes have identical values with some tolerance
*/
inline bool costs_are_equals(const std::vector<cv::Mat>& cube1, const std::vector<cv::Mat>& cube2, float tolerance = 1e-6f) {
	if (cube1.size() != cube2.size()) {
		printf("Cubes do not have the same size\n");
		return false;
	} 

	for (int z = 0; z < cube1.size(); ++z)
	{
		const cv::Mat& A = cube1[z];
		const cv::Mat& B = cube2[z];

		if (A.size() != B.size() || A.type() != B.type()) {
			printf("Cubes do not have the same size or type\n");
			return false;
		}

		for (int y = 0; y < A.rows; ++y)
		{
			for (int x = 0; x < A.cols; ++x)
			{
				float a = A.at<float>(y, x);
				float b = B.at<float>(y, x);

				if (std::fabs(a - b) > tolerance) {
					printf("Z-> %d - Cube[Y-> %d][X-> %d] - Host: %f | Cuda: %f \n",z,y,x,a,b);
					return false;
				} 
			}
		}
	}
	return true;
}
template <class T>
inline bool copyDataForDevice(std::vector<cam> cam_vector, T* p_ref, T* p_cam, uint8_t* luma, unsigned ref_idx) {
	auto ref = cam_vector.at(ref_idx);
	int idx = 0;
	const int width = 1920;
	const int height = 1080;
	// prep data array for device
	memcpy(p_ref, ref.p.K_inv.data(), 9 * sizeof(T));
	memcpy(p_ref + 9, ref.p.R_inv.data(), 9 * sizeof(T));
	memcpy(p_ref + 18, ref.p.t_inv.data(), 3 * sizeof(T));
	for (auto& cam : cam_vector) {
		printf("Cam: %d\n", idx);
		int offset = height * width * idx;
		memcpy(luma + offset, cam.YUV[0].data, height * width * sizeof(uint8_t));
		printf("Memcpy of luma: %d\n", idx);
		if (!cam_vector.at(idx).YUV[0].isContinuous()) {
			printf("Cam %d is not continuous\n", idx);
			memcpy(luma + offset, cam.YUV[0].clone().data, height * width * sizeof(uint8_t));
		}
		if (cam.name == ref.name) {
			++idx;
			continue;
		}
		memcpy(p_cam + (21 * (idx - 1)), cam.p.K.data(), 9 * sizeof(T));
		memcpy(p_cam + 9 + (21 * (idx - 1)), cam.p.R.data(), 9 * sizeof(T));
		memcpy(p_cam + 18 + (21 * (idx - 1)), cam.p.t.data(), 3 * sizeof(T));
		idx++;
	}

	for (int cam_num = 0; cam_num < cam_vector.size(); cam_num++) {
		bool isRef = cam_num == ref_idx;
		double* c = isRef ? p_ref : p_cam + 21 * (cam_num - 1);
		auto K_v = isRef ? cam_vector.at(cam_num).p.K_inv : cam_vector.at(cam_num).p.K;
		auto R_v = isRef ? cam_vector.at(cam_num).p.R_inv : cam_vector.at(cam_num).p.R;
		auto t_v = isRef ? cam_vector.at(cam_num).p.t_inv : cam_vector.at(cam_num).p.t;
		for (int K = 0; K < 9; ++K) {
			if (K_v.at(K) != c[K]) {
				printf("Wrong K value at cam: %d on index: %d\n", cam_num, K);
				printf("---Expected: %f got %f\n", K_v.at(K), c[K]);
				return false;
			}
		}
		c += 9;
		for (int R = 0; R < 9; ++R) {
			if (R_v.at(R) != c[R]) {
				printf("Wrong K value at cam: %d on index: %d\n", cam_num, R);
				printf("---Expected: %f got %f\n", R_v.at(R), c[R]);
				return false;
			}
		}
		c += 9;
		for (int t = 0; t < 3; ++t) {
			if (t_v.at(t) != c[t]) {
				printf("Wrong t value at cam: %d on index: %d\n", cam_num, t);
				printf("---Expected: %f got %f\n", t_v.at(t), c[t]);
				return false;
			}
		}
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int offset = height * width * cam_num;
				idx = (i * width + j) + offset;
				if (cam_vector.at(cam_num).YUV[0].at<uint8_t>(i, j) != luma[idx]) {
					printf("Wrong value at Mat[%d][%d]: %d - Array[%d]: %d\n", i, j,
						cam_vector.at(cam_num).YUV[0].at<uint8_t>(i, j), idx, luma[idx]);
					return false;
				}
			}
		}
	}
	return true;
}

std::vector<cv::Mat> load_or_compute_cost_cube(const std::string& filename, cam const ref, std::vector<cam> const& cam_vector, int window) {
	std::ifstream inFile(filename, std::ios::binary);
	if (inFile) {
		printf("Cache found. Loading cost cube from %s\n", filename.c_str());
		size_t planes;
		int height, width;

		inFile.read(reinterpret_cast<char*>(&planes), sizeof(planes));
		inFile.read(reinterpret_cast<char*>(&height), sizeof(height));
		inFile.read(reinterpret_cast<char*>(&width), sizeof(width));

		std::vector<cv::Mat> cost_cube(planes);
		for (size_t i = 0; i < planes; ++i) {
			cost_cube[i] = cv::Mat(height, width, CV_32FC1);
			inFile.read(reinterpret_cast<char*>(cost_cube[i].data), cost_cube[i].total() * cost_cube[i].elemSize());
		}
		return cost_cube;
	}
	else {
		printf("No cache found. Computing cost cube with sweeping_plane...\n");
		std::vector<cv::Mat> cost_cube = sweeping_plane(ref, cam_vector, window);
		store_cost_cube(cost_cube, filename);
		return cost_cube;
	}
}
int main()
{
	// Read cams
	std::vector<cam> cam_vector = read_cams("data");
	std::vector<cv::Mat> cost_cube;
	cam ref = cam_vector.at(0);
	// prep data for device
	const int width = 1920;
	const int height = 1080;
	float cuda_ms_time_d = 0, cuda_ms_time_f = 0;
	float total_cuda_time_d = 0, total_cuda_time_f = 0;
	double p_ref[21];
	float p_ref_f[21];
	double p_cam[63];
	float p_cam_f[63];
	uint8_t* luma = new uint8_t[width*height*4];
	std::vector<cv::Mat> v_cost_cube_cuda_d(ZPlanes), v_cost_cube_cuda_f(ZPlanes);
	float* cost_cube_cuda_d, *cost_cube_cuda_f;
	cv::Mat depth;
	bool copySucess = copyDataForDevice(cam_vector,p_ref, p_cam, luma, 0);
	if (copySucess) {
		printf("All values are correctly copied!\n");
		for (int i = 0; i < 21; ++i) p_ref_f[i] = static_cast<float>(p_ref[i]);
		for (int i = 0; i < 63; ++i) p_cam_f[i] = static_cast<float>(p_cam[i]);
	}
	else {
		printf("Error in value copy\n");
	}
	// calling CUDAfunction sweeping plane double
	for (int i = 0; i < 5; ++i) {
		cost_cube_cuda_d = wrap_sweeping_plane_device(p_ref, p_cam, luma, width, height, ZPlanes, cam_vector.size(), 5, cuda_ms_time_d);
		total_cuda_time_d += cuda_ms_time_d;
	} 

	// calling CUDAfunction sweeping plane float
	for (int i = 0; i < 5; ++i) {
		cost_cube_cuda_f = wrap_sweeping_plane_device(p_ref_f, p_cam_f, luma, width, height, ZPlanes, cam_vector.size(), 5, cuda_ms_time_f);
		total_cuda_time_f += cuda_ms_time_f;
	}
	//cost_cube_cuda_d = wrap_sweeping_plane_device(p_ref, p_cam, luma, width, height, ZPlanes, cam_vector.size(), 5, total_cuda_time);
	//total_cuda_time *= 5;
	//cost_cube_cuda_d = wrap_sweeping_plane_device(p_ref, p_cam, luma, width, height , ZPlanes, cam_vector.size(), 5, cuda_ms_time_d);
	
	for (int zi = 0; zi < ZPlanes; zi++){
		v_cost_cube_cuda_d[zi] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
		v_cost_cube_cuda_f[zi] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				v_cost_cube_cuda_d[zi].at<float>(y, x) = cost_cube_cuda_d[y * width + x + (zi * width * height)];
				v_cost_cube_cuda_f[zi].at<float>(y, x) = cost_cube_cuda_f[y * width + x + (zi * width * height)];
			}
		}
	}
	//if (costs_are_equals(v_cost_cube_cuda_d, v_cost_cube_cuda_f)) printf("Values are similar for both kernels\n");
	//else printf("Wrong cuda values for kernels\n");
	printf("cost cuda copied\n");
	// //Sweeping algorithm for camera 0 on host
	//const long  avg_cpu = 270201; // computed over 5 time and kepts here because it is really long to compute
	auto start = std::chrono::high_resolution_clock::now();
	////for(int i  = 0; i<5; ++i) cost_cube = sweeping_plane(ref, cam_vector, 5); //used to do an avg of execution time
	//cost_cube = load_or_compute_cost_cube("cost_cube.cache", ref, cam_vector, 5);
	cost_cube = sweeping_plane(ref, cam_vector, 5);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	//double mag_order_d = static_cast<double>(duration.count()) / static_cast<double>(total_cuda_time);
	double speed_up_d =  static_cast<double>(duration.count()) / static_cast<double>(total_cuda_time_d/5);
	double speed_up_f = static_cast<double>(duration.count()) / static_cast<double>(total_cuda_time_f/5);
	double mag_order_d = log10(speed_up_d);
	double mag_order_f =  log10(speed_up_f);
	////double mag_order_f = static_cast<double>(duration.count()) / static_cast<double>(cuda_ms_time_f);
	//printf("Host function execution time: %lld ms\n", duration.count() / 5);
	printf("Host function execution time: %lld ms\n", duration.count());
	printf("Device double function execution time: %f ms\n", total_cuda_time_d/5);
	printf("Faster by %f order of magnitude\n",mag_order_d);
	printf("Host function execution time: %lld ms\n", duration.count());
	printf("Device float  function execution time: %f ms\n", total_cuda_time_f/5);
	printf("Faster by %f order of magnitude\n", mag_order_f);
	//printf("Faster by %f order of magnitude\n", log10(mag_order_f));

	if (costs_are_equals(cost_cube, v_cost_cube_cuda_d)) printf("Values are similar\n");
	else printf("Wrong cuda values\n");

	//
	// Use graph cut to generate depth map 
	// Cleaner results, long compute time
	//depth = depth_estimation_by_graph_cut_sWeight(cost_cube);
	//cv::imwrite("./depth_map_host.png", depth);
	//depth = depth_estimation_by_graph_cut_sWeight(v_cost_cube_cuda_d);
	//cv::imwrite("./depth_map_dev_d.png", depth);
	depth = depth_estimation_by_graph_cut_sWeight(v_cost_cube_cuda_f);
	cv::imwrite("./depth_map_dev_f.png", depth);
	// Find min cost and generate depth map
	// Faster result, low quality
	//cv::Mat depth = find_min(cost_cube);


	cv::namedWindow("Depth", cv::WINDOW_NORMAL);
	cv::imshow("Depth", depth);
	cv::waitKey(0);

	//cv::imwrite("./depth_map.png", depth);
	//printf("%f", depth.at<uchar>(0, 0));
	delete[] luma;
	return 0;
}