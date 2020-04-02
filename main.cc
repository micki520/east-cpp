/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.

#include <fstream>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "./text_detector/lanms.h"
#include "./text_detector/utils.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;

using namespace cv;
using namespace std;
using namespace tensorflow;
using namespace lanms;

#ifdef __LOG__
const char* logfile = "d:\\east_test.log";
FILE* logfd = NULL;
#endif

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = data.ToString();
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
                DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } else if (tensorflow::StringPiece(file_name).ends_with(".bmp")) {
    image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

std::unique_ptr<tensorflow::Session> session;

__declspec(dllexport) int __cdecl tensorStartup(std::string graph_path) {
  int argc = 1;
  char* argv[] = {"tensorinfer"};

  tensorflow::port::InitMain("tensorinfer", &argc, ((char***)(&argv)));

  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return (-1);
  }

  return (0);
}

__declspec(dllexport) int __cdecl tensorCleanup() {
  session->Close();
  return (0);
}



// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* indices, Tensor* scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const std::vector<Tensor>& outputs,
                      const string& labels_file_name) {
  std::vector<string> labels;
  size_t label_count;
  Status read_labels_status =
      ReadLabelsFile(labels_file_name, &labels, &label_count);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return read_labels_status;
  }
  const int how_many_labels = std::min(5, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);
    LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
  }
  return Status::OK();
}

// This is a testing function that returns whether the top label index is the
// one that's expected.
Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
                     bool* is_expected) {
  *is_expected = false;
  Tensor indices;
  Tensor scores;
  const int how_many_labels = 1;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  if (indices_flat(0) != expected) {
    LOG(ERROR) << "Expected label #" << expected << " but got #"
               << indices_flat(0);
    *is_expected = false;
  } else {
    *is_expected = true;
  }
  return Status::OK();
}

void visualize_det_output(std::vector<Tensor>& outputs) {
  std::cout << outputs[0].DebugString() << std::endl;

  int dim = outputs[0].dims();
  std::cout << "num dim: " << dim << " " << outputs[0].NumElements()
            << std::endl;

  auto out_score_mat = outputs[0].matrix<float>();
  std::cout << out_score_mat.NumDimensions << std::endl;

  const Eigen::Tensor<float, out_score_mat.NumDimensions>::Dimensions&
      out_score_dim = out_score_mat.dimensions();
  LOG(INFO) << "score dimensions: " << out_score_mat.NumDimensions << " "
            << out_score_dim[0];

  // tensor to cv mat
  // cv::Mat vis_score = tensor_to_cv_mat(outputs[0]);
  // std::cout<<vis_score.rows<<" "<<vis_score.cols<<std::endl;
  // double min_v, max_v;
  // cv::minMaxLoc(vis_score, &min_v, &max_v);
}

typedef struct detection {
	float xa;
	float ya;
	float xb;
	float yb;
	float xc;
	float yc;
	float xd;
	float yd;
	float score;	
} detection;


void decode_netout(vector<float> out_vector0, vector<float> out_vector1, vector<Mat>& boxes,
                   float score_thresh, int grid_h, int grid_w, 
                   int input_width, int input_height) {
	vector<vector<vector<float>>> net(grid_h, vector<vector<float>>(grid_w, vector<float>(6, 0)));

	for (int i = 0; i < grid_h; i++) {
		for (int j = 0; j < grid_w; j++) {            //combine score and poly
			net[i][j][0] = out_vector0[i * grid_w + j ];
			for (int k = 0; k < 5; k++) {
				net[i][j][k+1] = out_vector1[((i * grid_w + j) * 5 + k)];
			}
		}
	}

	float objectness;
	//float aa, bb, cc, dd, angle, scores;
	float scores;
	for (int i = 0; i < (grid_h * grid_w); i++) {
		float row = i / grid_w;
		float col = i % grid_w;

		scores = out_vector0[i];
		if(scores >= score_thresh){
			Mat box = Mat_<float>(8, 1);   
			box.at<float>(0) = out_vector0[i];
			box.at<float>(1) = col;
			box.at<float>(2) = row;
			box.at<float>(3) = net[row][col][1];
			box.at<float>(4) = net[row][col][2];
			box.at<float>(5) = net[row][col][3];			
			box.at<float>(6) = net[row][col][4];
			box.at<float>(7) = net[row][col][5];	
			boxes.push_back(box);
		}
	}
}

vector<detection> restore_rectangle_rbox(vector<Mat> boxes) {
	detection rbox;
	vector<detection> rboxes;
	float origin_x, origin_y;
	float angle_0;
	Mat p = Mat_<float>(5, 2);   
	Mat rotate_matrix_x = Mat_<float>(2, 1); 
	Mat rotate_matrix_y = Mat_<float>(2, 1);
	Mat p_rotate_x = Mat_<float>(5,1);
	Mat p_rotate_y = Mat_<float>(5,1);

	Mat p_rotate = Mat_<float>(5,2);

	Mat p3_in_origin = Mat_<float>(2,1);
	Mat new_p0 = Mat_<float>(2,1);
	Mat new_p1 = Mat_<float>(2,1);
	Mat new_p2 = Mat_<float>(2,1);
	Mat new_p3 = Mat_<float>(2,1);

	if (boxes.size() == 0) {
		rboxes.push_back(rbox);
		return rboxes;
	} 
	else {
		
		for (int i = 0; i < boxes.size(); i++) {
			origin_x = boxes[i].at<float>(1) * 4;
			origin_y = boxes[i].at<float>(2) * 4;

			float d0 = boxes[i].at<float>(3);
			float d1 = boxes[i].at<float>(4);
			float d2 = boxes[i].at<float>(5);
			float d3 = boxes[i].at<float>(6);		

			angle_0 =  boxes[i].at<float>(7);
			if(angle_0 >= 0) {
				p.at<float>(0,0) = 0;
				p.at<float>(0,1) = -d0 - d2;
				p.at<float>(1,0) =  d1 + d3;
				p.at<float>(1,1) = -d0 - d2;
				p.at<float>(2,0) =  d1 + d3;
				p.at<float>(2,1) = 0;
				p.at<float>(3,0) = 0;
				p.at<float>(3,1) = 0;

				p.at<float>(4,0) =  d3;
				p.at<float>(4,1) = -d2;		

				rotate_matrix_x.at<float>(0,0) = cos(angle_0);
				rotate_matrix_x.at<float>(1,0) = sin(angle_0);
				rotate_matrix_y.at<float>(0,0) = -sin(angle_0);
				rotate_matrix_y.at<float>(1,0) = cos(angle_0);	

				p_rotate_x = p * rotate_matrix_x;
				p_rotate_y = p * rotate_matrix_y;

				p3_in_origin.at<float>(0,0) =  origin_x - p_rotate_x.at<float>(4,0);
				p3_in_origin.at<float>(1,0) =  origin_y - p_rotate_y.at<float>(4,0);

			}
			else {     //angle_0 < 0
				p.at<float>(0,0) = 0;
				p.at<float>(0,1) = -d0 - d2;
				p.at<float>(1,0) =  d1 + d3;
				p.at<float>(1,1) = -d0 - d2;
				p.at<float>(2,0) =  d1 + d3;
				p.at<float>(2,1) = 0;
				p.at<float>(3,0) = 0;
				p.at<float>(3,1) = 0;

				p.at<float>(4,0) =  d3;
				p.at<float>(4,1) = -d2;		

				rotate_matrix_x.at<float>(0,0) = cos(angle_0);
				rotate_matrix_x.at<float>(1,0) = sin(angle_0);
				rotate_matrix_y.at<float>(0,0) = -sin(angle_0);
				rotate_matrix_y.at<float>(1,0) = cos(angle_0);	

				p_rotate_x = p * rotate_matrix_x;
				p_rotate_y = p * rotate_matrix_y;				

				p3_in_origin.at<float>(0,0) =  origin_x - p_rotate_x.at<float>(4,0);
				p3_in_origin.at<float>(1,0) =  origin_y - p_rotate_y.at<float>(4,0);
			}

			rbox.score = boxes[i].at<float>(0);     //get score
			rbox.xa = p_rotate_x.at<float>(0,0) + p3_in_origin.at<float>(0,0);
			rbox.ya = p_rotate_y.at<float>(0,0) + p3_in_origin.at<float>(1,0);
			rbox.xb = p_rotate_x.at<float>(1,0) + p3_in_origin.at<float>(0,0);
			rbox.yb = p_rotate_y.at<float>(1,0) + p3_in_origin.at<float>(1,0);
			rbox.xc = p_rotate_x.at<float>(2,0) + p3_in_origin.at<float>(0,0);
			rbox.yc = p_rotate_y.at<float>(2,0) + p3_in_origin.at<float>(1,0);
			rbox.xd = p_rotate_x.at<float>(3,0) + p3_in_origin.at<float>(0,0);
			rbox.yd = p_rotate_y.at<float>(3,0) + p3_in_origin.at<float>(1,0);

			rboxes.push_back(rbox);			
		}

		return rboxes;
	}

}

std::vector<std::vector<float>> polys2floats(std::vector<lanms::Polygon> &polys) {
  std::vector<std::vector<float>> ret;
  for (size_t i = 0; i < polys.size(); i ++) {
    auto &p = polys[i];
    auto &poly = p.poly;
    ret.emplace_back(std::vector<float>{
        float(static_cast<double>(poly[0].X) / 10000.0), float(static_cast<double>(poly[0].Y) / 10000.0),
        float(static_cast<double>(poly[1].X) / 10000.0), float(static_cast<double>(poly[1].Y) / 10000.0),
        float(static_cast<double>(poly[2].X) / 10000.0), float(static_cast<double>(poly[2].Y) / 10000.0),
        float(static_cast<double>(poly[3].X) / 10000.0), float(static_cast<double>(poly[3].Y) / 10000.0),
        float(p.score),
        });
   } 
  return ret;
}


//input_height, input_width必须是32的倍数（例如:320*320）
__declspec(dllexport) int __cdecl tensorRun(std::vector<cv::Mat>& imgArray, int input_height, int input_width,std::vector<double>& scores, string input_layer,
                                          string output_layer0, string output_layer1, float score_thresh, float nms_thresh) {

  int grid_h = input_height / 4;
  int grid_w = input_width / 4;
  //string output_layer0 = output_layer0 ;
  //string output_layer1 = output_layer1;

  int batchSize = imgArray.size();
  int input_depth = 3;

  /* image resizing */
  std::vector<cv::Mat> imgArrayB;
  std::vector<float> ratio_h;
  std::vector<float> ratio_w;

  // creating a Tensor for storing the data
  Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({batchSize, input_height, input_width, input_depth}));
  auto input_tensor_mapped = input_tensor.tensor<float, 4>();

  for (int bz = 0; bz < batchSize; ++bz) {

    int new_h = imgArray.at(bz).rows;
    int new_w = imgArray.at(bz).cols;

    float ratio_h_tmp = float(input_height)/new_h;
    float ratio_w_tmp = float(input_width )/new_w;

    ratio_h.push_back(ratio_h_tmp);
    ratio_w.push_back(ratio_w_tmp);

    new_h = int(input_height);
    new_w = int(input_width);

    Size dsize = cv::Size(new_w, new_h);
    Mat resized;
    cv::resize(imgArray.at(bz), resized, dsize, 0, 0, cv::INTER_LINEAR);  

    cv::Mat image2;
    resized.convertTo(image2, CV_32FC1);
    //we assume that the image is unsigned char dtype
    const float *source_data = (float*)(image2.data); 

    for (int y = 0; y < input_height; ++y) {
      const float* source_row = source_data + (y * input_width * input_depth);
      for (int x = 0; x < input_width; ++x) {
        const float* source_pixel = source_row + (x * input_depth);
        float b = *(source_pixel)    ;
        float g = *(source_pixel + 1);
        float r = *(source_pixel + 2);
        input_tensor_mapped(bz, y, x, 0) = r;   
        input_tensor_mapped(bz, y, x, 1) = g;
        input_tensor_mapped(bz, y, x, 2) = b;
      }
    }     
  }
  // Actually run the image through the model.
  std::vector<Tensor> outputs0, outputs1;
  Status run_status = session->Run({{input_layer, input_tensor}}, {output_layer0}, {}, &outputs0);
         run_status = session->Run({{input_layer, input_tensor}}, {output_layer1}, {}, &outputs1);

  auto scores_flat0 = outputs0[0].flat<float>();
  auto poly_flat0 = outputs1[0].flat<float>();

  LOG(INFO) <<"number of outputs0:"<<scores_flat0.size();
  LOG(INFO) <<"number of outputs1:"<<poly_flat0.size();

  for (int b = 0; b < batchSize; b++) {
    vector<float> out_vector0, out_vector1;
    float eachBatchSize = scores_flat0.size() / batchSize;    //=grid_h*grid_w
    for (int i = 0; i < eachBatchSize; i++) {
      out_vector0.push_back(scores_flat0(b * eachBatchSize + i));
    }
    for (int i = 0; i < eachBatchSize * 5; i++) {
      out_vector1.push_back(poly_flat0(b * eachBatchSize * 5 + i));
    }

    vector<Mat> boxes;
    decode_netout(out_vector0, out_vector1, boxes, score_thresh, grid_h, grid_w, input_width, input_height);
    vector<detection> crboxes = restore_rectangle_rbox(boxes);

    int box_size = boxes.size();
    int size = box_size * 9;
    float *bBox = new float[size];
    for (int i = 0; i < crboxes.size(); i++) {
      bBox[i * 9 + 0] = crboxes[i].xa * 10000.0;
      bBox[i * 9 + 1] = crboxes[i].ya * 10000.0;
      bBox[i * 9 + 2] = crboxes[i].xb * 10000.0;
      bBox[i * 9 + 3] = crboxes[i].yb * 10000.0;
      bBox[i * 9 + 4] = crboxes[i].xc * 10000.0;
      bBox[i * 9 + 5] = crboxes[i].yc * 10000.0;
      bBox[i * 9 + 6] = crboxes[i].xd * 10000.0;
      bBox[i * 9 + 7] = crboxes[i].yd * 10000.0;
      bBox[i * 9 + 8] = crboxes[i].score;
    }

    auto ptr = static_cast<float *>(bBox); 
    vector<lanms::Polygon> rboxes_out = lanms::merge_quadrangle_n9( ptr , box_size, nms_thresh);
    vector<vector<float>> rboxes_out_vec = polys2floats( rboxes_out );   
    //1.此处需要再加一个过滤score较低的boxes
    ofstream out("out_test.txt");  
    if (out.is_open()) {  
    out << "This is EAST detect output:\n\n" << b << endl;  
    for (int i = 0; i < rboxes_out_vec.size(); i++) {
      out << (rboxes_out_vec[i][0]) << ",";  
      out << (rboxes_out_vec[i][1]) << ","; 
      out << (rboxes_out_vec[i][2]) << ",";  
      out << (rboxes_out_vec[i][3]) << ",";  
      out << (rboxes_out_vec[i][4]) << ",";  
      out << (rboxes_out_vec[i][5]) << ",";  
      out << (rboxes_out_vec[i][6]) << ",";  
      out << (rboxes_out_vec[i][7]) << "," << endl;  
      out << (rboxes_out_vec[i][8]) << endl;  
    }
     out.close();  
    }

	#if 1
    Point points[7][4];
    points[0][0] = Point(rboxes_out_vec[0][0]/ratio_w.at(0), rboxes_out_vec[0][1]/ratio_h.at(0));
    points[0][1] = Point(rboxes_out_vec[0][2]/ratio_w.at(0), rboxes_out_vec[0][3]/ratio_h.at(0));
    points[0][2] = Point(rboxes_out_vec[0][4]/ratio_w.at(0), rboxes_out_vec[0][5]/ratio_h.at(0));
    points[0][3] = Point(rboxes_out_vec[0][6]/ratio_w.at(0), rboxes_out_vec[0][7]/ratio_h.at(0));
    points[1][0] = Point(rboxes_out_vec[1][0]/ratio_w.at(0), rboxes_out_vec[1][1]/ratio_h.at(0));
    points[1][1] = Point(rboxes_out_vec[1][2]/ratio_w.at(0), rboxes_out_vec[1][3]/ratio_h.at(0));
    points[1][2] = Point(rboxes_out_vec[1][4]/ratio_w.at(0), rboxes_out_vec[1][5]/ratio_h.at(0));
    points[1][3] = Point(rboxes_out_vec[1][6]/ratio_w.at(0), rboxes_out_vec[1][7]/ratio_h.at(0));
    points[2][0] = Point(rboxes_out_vec[2][0]/ratio_w.at(0), rboxes_out_vec[2][1]/ratio_h.at(0));
    points[2][1] = Point(rboxes_out_vec[2][2]/ratio_w.at(0), rboxes_out_vec[2][3]/ratio_h.at(0));
    points[2][2] = Point(rboxes_out_vec[2][4]/ratio_w.at(0), rboxes_out_vec[2][5]/ratio_h.at(0));
    points[2][3] = Point(rboxes_out_vec[2][6]/ratio_w.at(0), rboxes_out_vec[2][7]/ratio_h.at(0));
    points[3][0] = Point(rboxes_out_vec[3][0]/ratio_w.at(0), rboxes_out_vec[3][1]/ratio_h.at(0));
    points[3][1] = Point(rboxes_out_vec[3][2]/ratio_w.at(0), rboxes_out_vec[3][3]/ratio_h.at(0));
    points[3][2] = Point(rboxes_out_vec[3][4]/ratio_w.at(0), rboxes_out_vec[3][5]/ratio_h.at(0));
    points[3][3] = Point(rboxes_out_vec[3][6]/ratio_w.at(0), rboxes_out_vec[3][7]/ratio_h.at(0));  
    points[4][0] = Point(rboxes_out_vec[4][0]/ratio_w.at(0), rboxes_out_vec[4][1]/ratio_h.at(0));
    points[4][1] = Point(rboxes_out_vec[4][2]/ratio_w.at(0), rboxes_out_vec[4][3]/ratio_h.at(0));
    points[4][2] = Point(rboxes_out_vec[4][4]/ratio_w.at(0), rboxes_out_vec[4][5]/ratio_h.at(0));
    points[4][3] = Point(rboxes_out_vec[4][6]/ratio_w.at(0), rboxes_out_vec[4][7]/ratio_h.at(0));  
    points[5][0] = Point(rboxes_out_vec[5][0]/ratio_w.at(0), rboxes_out_vec[5][1]/ratio_h.at(0));
    points[5][1] = Point(rboxes_out_vec[5][2]/ratio_w.at(0), rboxes_out_vec[5][3]/ratio_h.at(0));
    points[5][2] = Point(rboxes_out_vec[5][4]/ratio_w.at(0), rboxes_out_vec[5][5]/ratio_h.at(0));
    points[5][3] = Point(rboxes_out_vec[5][6]/ratio_w.at(0), rboxes_out_vec[5][7]/ratio_h.at(0));  
    points[6][0] = Point(rboxes_out_vec[6][0]/ratio_w.at(0), rboxes_out_vec[6][1]/ratio_h.at(0));
    points[6][1] = Point(rboxes_out_vec[6][2]/ratio_w.at(0), rboxes_out_vec[6][3]/ratio_h.at(0));
    points[6][2] = Point(rboxes_out_vec[6][4]/ratio_w.at(0), rboxes_out_vec[6][5]/ratio_h.at(0));
    points[6][3] = Point(rboxes_out_vec[6][6]/ratio_w.at(0), rboxes_out_vec[6][7]/ratio_h.at(0));  

    const Point* pts[] = {points[0],points[1],points[2],points[3],points[4],points[5],points[6]};
    int npts[] = {4,4,4,4,4,4,4};
    polylines(imgArray.at(0),pts,npts,7,true,Scalar(255),5,8,0);
    namedWindow("Poly");
    imshow("Poly", imgArray.at(0));
    waitKey();
    #endif


    #if 0
    Point points[2][4];
    points[0][0] = Point(rboxes_out_vec[0][0] / ratio_w.at(0),rboxes_out_vec[0][1] / ratio_h.at(0));
    points[0][1] = Point(rboxes_out_vec[0][2] / ratio_w.at(0),rboxes_out_vec[0][3] / ratio_h.at(0));
    points[0][2] = Point(rboxes_out_vec[0][4] / ratio_w.at(0),rboxes_out_vec[0][5] / ratio_h.at(0));
    points[0][3] = Point(rboxes_out_vec[0][6] / ratio_w.at(0),rboxes_out_vec[0][7] / ratio_h.at(0));
    points[1][0] = Point(rboxes_out_vec[1][0] / ratio_w.at(0),rboxes_out_vec[1][1] / ratio_h.at(0));
    points[1][1] = Point(rboxes_out_vec[1][2] / ratio_w.at(0),rboxes_out_vec[1][3] / ratio_h.at(0));
    points[1][2] = Point(rboxes_out_vec[1][4] / ratio_w.at(0),rboxes_out_vec[1][5] / ratio_h.at(0));
    points[1][3] = Point(rboxes_out_vec[1][6] / ratio_w.at(0),rboxes_out_vec[1][7] / ratio_h.at(0));

    const Point* pts[] = {points[0], points[1]};
    int npts[] = {4, 4};
    polylines(imgArray.at(0), pts, npts, 2, true, Scalar(255), 5, 8, 0);
    namedWindow("Poly");
    imshow("Poly", imgArray.at(0));
    waitKey();
	#endif
  }

  return (0);
}


__declspec(dllexport) int __cdecl tensorRun(
  std::vector<std::string>& image_paths, int input_height, int input_width,
  std::vector<double>& scores, string input_layer, string output_layer0,
  string output_layer1, float score_thresh, float nms_thresh) {
  int batchSize = image_paths.size();
  int input_depth = 3;

  std::vector<cv::Mat> imgArray;

  for (int i = 0; i < batchSize; i++) {
    string image_path = image_paths[i];

    cv::Mat img = cv::imread(image_path);
    imgArray.push_back(img);
  }

  return tensorRun(imgArray, input_height, input_width, scores, input_layer, output_layer0, output_layer1, score_thresh, nms_thresh);
}

int main(int argc, char* argv[]) {
    // These are the command-line flags the program can understand.
    // They define where the graph and input data is located, and what kind of
    // input the model expects. If you train your own model, or use something
    // other than inception_v3, then you'll need to update these.
    std::string image = "D:/tools/tensorflow-1.7.1/tensorflow/examples/label_image/data/img_1.jpg";
    std::string image1 = "D:/tools/tensorflow-1.7.1/tensorflow/examples/label_image/data/img_3.jpg";
    std::string image2 = "D:/tools/tensorflow-1.7.1/tensorflow/examples/label_image/data/img_6.jpg";
    std::string graph = "D:/tools/tensorflow-1.7.1/tensorflow/examples/label_image/data/east_resnet_v1_50_0121.pb";

    int input_width = 320;
    int input_height = 320;

    float score_thresh = 0.80;
    float nms_thresh = 0.20;
    int grid_h = input_height / 4;
    int grid_w = input_width / 4;
    string input_layer = "input_images";
    string output_layer0 = "feature_fusion/Conv_7/Sigmoid";
    string output_layer1 = "feature_fusion/concat_3";

    if (tensorStartup(graph) < 0) {
      return (-1);
    }

    // Get the image from disk as a float array of numbers, resized and normalized
    // to the specifications the main graph expects.
    std::vector<string> image_paths;

    image_paths.push_back(image1);
    image_paths.push_back(image1);
    image_paths.push_back(image1);

    std::vector<double> scores;
    if (tensorRun(image_paths, input_height, input_width, scores, input_layer,
      output_layer0, output_layer1, score_thresh, nms_thresh) < 0) {
      return (-1);
    }
  

  return 0;
}


#if 0
int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
  string image =
      "D:/tools/tensorflow-1.7.1/tensorflow/examples/label_image/data/img_1.jpg";
  string graph =
      "D:/tools/tensorflow-1.7.1/tensorflow/examples/label_image/data/east_resnet_v1_50_0121.pb";
  string labels =
      "D:/tools/tensorflow-1.7.1/tensorflow/examples/label_image/data/imagenet_slim_labels.txt";

  int32 input_width = 512;
  int32 input_height = 512;
  float input_mean = 0;
  float input_std = 255;

  string input_layer = "input_images";

  string output_layer0 = "feature_fusion/Conv_7/Sigmoid";
  string output_layer1 = "feature_fusion/concat_3";

  bool self_test = false;
  string root_dir = "";
  std::vector<Flag> flag_list = {
      Flag("image", &image, "image to be processed"),
      Flag("graph", &graph, "graph to be executed"),
      Flag("labels", &labels, "name of file containing labels"),
      Flag("input_width", &input_width, "resize image to this width in pixels"),
      Flag("input_height", &input_height, "resize image to this height in pixels"),
      Flag("input_layer", &input_layer, "name of input layer"),
      Flag("output_layer0", &output_layer0, "name of output layer0"),
      Flag("output_layer1", &output_layer1, "name of output layer1"),
      Flag("self_test", &self_test, "run a self test"),
      Flag("root_dir", &root_dir, "interpret image and graph file names relative to this directory"),
  };
  
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  cv::Mat resized_image;
  std::vector<Tensor> resized_tensors;
  cv::Mat image_mat = cv::imread(image);
  float ratio_h=0, ratio_w=0;
  int resize_h=0,resize_w=0;
  resize_image_max_len(image_mat, resized_image, ratio_h, ratio_w, resize_h, resize_w, 768);

  int grid_h = resize_h / 4;
  int grid_w = resize_w / 4;

  auto resized_tensor = cv_mat_to_tensor(resized_image);

  // Actually run the image through the model.
  std::vector<Tensor> outputs0,outputs1;
  Status run_status = session->Run({{input_layer, resized_tensor}},{output_layer0}, {}, &outputs0);
  run_status = session->Run({{input_layer, resized_tensor}},{output_layer1}, {}, &outputs1);

  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  auto scores_flat0 = outputs0[0].flat<float>();
  auto poly_flat0   = outputs1[0].flat<float>();

  LOG(INFO) <<"number of outputs0:"<<scores_flat0.size();
  LOG(INFO) <<"number of outputs1:"<<poly_flat0.size();

  auto out_tex_boxes = outputs0[0].flat_outer_dims<float>();

  auto detection_boxes = outputs1[0].flat_outer_dims<float>();
  int num_box = detection_boxes.dimension(0);
  int dim = detection_boxes.dimension(1);
  int num_data = num_box * dim; 

  LOG(INFO) << "num_box: "<< num_box;  
  LOG(INFO) << "dim: "<< dim;
  LOG(INFO) << "num_data: "<< num_data;

  Mat east_rbox;
  vector<Mat> boxes;
  vector<detection> nboxes;
  int num_grid = grid_h * grid_w;

  vector<float> out_vector0, out_vector1;

  cout << resize_h << " ";
  cout << resize_w << " ";
  cout << num_grid << " ";

  for (int i = 0; i < num_grid; i++)       //valid pixels /16
  {
    out_vector0.push_back(scores_flat0(i));
    for(int j = 0; j < 5; j++)
    {
      out_vector1.push_back(poly_flat0(5 * i + j));
    }
  }

  int num_classes = out_vector0.size();
  LOG(INFO) <<"num_classes:"<<scores_flat0.size();

  decode_netout(out_vector0, out_vector1, boxes, 0.60, grid_h, grid_w, resize_w, resize_h);     //score_thres

  vector<detection> crboxes = restore_rectangle_rbox(boxes);

  int box_size = boxes.size();
  float nms_th1 = 0.2;
  int size;

  size = box_size * 9;
  float *bBox = new float[size];
  for (int i = 0; i < crboxes.size(); i++) {
   bBox[i * 9 + 0] = crboxes[i].xa * 10000.0;
   bBox[i * 9 + 1] = crboxes[i].ya * 10000.0;
   bBox[i * 9 + 2] = crboxes[i].xb * 10000.0;
   bBox[i * 9 + 3] = crboxes[i].yb * 10000.0;
   bBox[i * 9 + 4] = crboxes[i].xc * 10000.0;
   bBox[i * 9 + 5] = crboxes[i].yc * 10000.0;
   bBox[i * 9 + 6] = crboxes[i].xd * 10000.0;
   bBox[i * 9 + 7] = crboxes[i].yd * 10000.0;
   bBox[i * 9 + 8] = crboxes[i].score;
  }
   
  auto ptr = static_cast<float *>(bBox); 
  vector<lanms::Polygon> rboxes_out = lanms::merge_quadrangle_n9( ptr , box_size, nms_th1);
  vector<vector<float>> rboxes_out_vec = polys2floats_new( rboxes_out );   

  //****************************************
  //此处需要再加一个过滤score较低的boxes

  //****************************************

  Mat img = imread(image);  //获取原始图片信息
  Point points[5][4];
  points[0][0] = Point(rboxes_out_vec[0][0]/ratio_w, rboxes_out_vec[0][1]/ratio_h);
  points[0][1] = Point(rboxes_out_vec[0][2]/ratio_w, rboxes_out_vec[0][3]/ratio_h);
  points[0][2] = Point(rboxes_out_vec[0][4]/ratio_w, rboxes_out_vec[0][5]/ratio_h);
  points[0][3] = Point(rboxes_out_vec[0][6]/ratio_w, rboxes_out_vec[0][7]/ratio_h);

  points[1][0] = Point(rboxes_out_vec[1][0]/ratio_w, rboxes_out_vec[1][1]/ratio_h);
  points[1][1] = Point(rboxes_out_vec[1][2]/ratio_w, rboxes_out_vec[1][3]/ratio_h);
  points[1][2] = Point(rboxes_out_vec[1][4]/ratio_w, rboxes_out_vec[1][5]/ratio_h);
  points[1][3] = Point(rboxes_out_vec[1][6]/ratio_w, rboxes_out_vec[1][7]/ratio_h);

  points[2][0] = Point(rboxes_out_vec[2][0]/ratio_w, rboxes_out_vec[2][1]/ratio_h);
  points[2][1] = Point(rboxes_out_vec[2][2]/ratio_w, rboxes_out_vec[2][3]/ratio_h);
  points[2][2] = Point(rboxes_out_vec[2][4]/ratio_w, rboxes_out_vec[2][5]/ratio_h);
  points[2][3] = Point(rboxes_out_vec[2][6]/ratio_w, rboxes_out_vec[2][7]/ratio_h);

  points[3][0] = Point(rboxes_out_vec[3][0]/ratio_w, rboxes_out_vec[3][1]/ratio_h);
  points[3][1] = Point(rboxes_out_vec[3][2]/ratio_w, rboxes_out_vec[3][3]/ratio_h);
  points[3][2] = Point(rboxes_out_vec[3][4]/ratio_w, rboxes_out_vec[3][5]/ratio_h);
  points[3][3] = Point(rboxes_out_vec[3][6]/ratio_w, rboxes_out_vec[3][7]/ratio_h);  

  points[4][0] = Point(rboxes_out_vec[4][0]/ratio_w, rboxes_out_vec[4][1]/ratio_h);
  points[4][1] = Point(rboxes_out_vec[4][2]/ratio_w, rboxes_out_vec[4][3]/ratio_h);
  points[4][2] = Point(rboxes_out_vec[4][4]/ratio_w, rboxes_out_vec[4][5]/ratio_h);
  points[4][3] = Point(rboxes_out_vec[4][6]/ratio_w, rboxes_out_vec[4][7]/ratio_h);  

  const Point* pts[] = {points[0],points[1],points[2],points[3],points[4]};
  int npts[] = {4,4,4,4,4};
  polylines(img,pts,npts,5,true,Scalar(255),5,8,0);
  namedWindow("Poly");
  imshow("Poly", img);
  waitKey();
  //fillPoly(img,pts,npts,5,Scalar(255),8,0,Point());
  //imshow("Poly", img);
  waitKey();

  ofstream out("out_test.txt");  
  if (out.is_open()) {  
     out << "This is EAST detect output:\n\n";  
     for (int i = 0; i < rboxes_out_vec.size(); i++) {
      out << (rboxes_out_vec[i][0]) << ",";  
      out << (rboxes_out_vec[i][1]) << ","; 
      out << (rboxes_out_vec[i][2]) << ",";  
      out << (rboxes_out_vec[i][3]) << ",";  
      out << (rboxes_out_vec[i][4]) << ",";  
      out << (rboxes_out_vec[i][5]) << ",";  
      out << (rboxes_out_vec[i][6]) << ",";  
      out << (rboxes_out_vec[i][7]) << "," << endl;  
      out << (rboxes_out_vec[i][8]) << endl;  
   }
     out.close();  
  }  

  return 0;
}


#endif