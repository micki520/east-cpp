#include "utils.h" 

void resize_image_max_len(const cv::Mat& image,
                          cv::Mat& resized_image,
                          float& ratio_h,
                          float& ratio_w,
                          int& resize_h,
                          int& resize_w,
                          int max_side_len){
  int height = image.rows;
  int width = image.cols;
  float ratio = 1;
  if(std::max(height, width) > max_side_len)
    ratio = height > width ? float(max_side_len)/height: float(max_side_len)/width;
  resize_h = int(height * ratio);
  resize_w = int(width  * ratio);
  resize_h = resize_h%32 == 0? resize_h : (resize_h/32 - 1) * 32;
  resize_w = resize_w%32 == 0? resize_w : (resize_w/32 - 1) * 32;

  //resize_h = 320;
  //resize_w = 320;
  cv::resize(image, resized_image, cv::Size(resize_w, resize_h));
  
  ratio_h = float(resize_h)/height;
  ratio_w = float(resize_w)/width;
}


tensorflow::Tensor cv_mat_to_tensor(const cv::Mat& image){
  int height = image.rows;
  int width = image.cols;
  int depth = 3;
  tensorflow::Tensor res_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, height, width, 3}));

  cv::Mat image2;
  image.convertTo(image2, CV_32FC1);
  //we assume that the image is unsigned char dtype
  const float *source_data = (float*)(image2.data); 

  auto tensor_mapped = res_tensor.tensor<float, 4>();
  for (int y = 0; y < height; ++y) {
    const float* source_row = source_data + (y * width * depth);
    for (int x = 0; x < width; ++x) {
      const float* source_pixel = source_row + (x * depth);
      float b = *(source_pixel)    ;
      float g = *(source_pixel + 1);
      float r = *(source_pixel + 2);
      tensor_mapped(0, y, x, 0) = r;   //add @0127
      tensor_mapped(0, y, x, 1) = g;
      tensor_mapped(0, y, x, 2) = b;
    }
  }
  return res_tensor;
}


void batch_resize_image_max_len(const std::vector<cv::Mat>& imgArray,
                          std::vector<cv::Mat>& resized_image,
                          std::vector<float>& ratio_h,
                          std::vector<float>& ratio_w,
                          std::vector<int>& resize_h,
                          std::vector<int>& resize_w,
                          int max_side_len){
  int batchSize = imgArray.size();

  for (int bz = 0; bz < batchSize; bz++) {
    int height = imgArray.at(bz).rows;
    int width = imgArray.at(bz).cols;
    float ratio = 1;
    if(std::max(height, width) > max_side_len)
      ratio = height > width ? float(max_side_len)/height: float(max_side_len)/width;
    int tmp_resize_h = int(height * ratio);
    int tmp_resize_w = int(width  * ratio);
    //resize_h.at(bz) = tmp_resize_h%32 == 0? tmp_resize_h : (tmp_resize_h/32 - 1) * 32;
    //resize_w.at(bz) = tmp_resize_w%32 == 0? tmp_resize_w : (tmp_resize_w/32 - 1) * 32;
    resize_h.at(bz) = 512;
    resize_w.at(bz) = 512;
    cv::resize(imgArray.at(bz), resized_image.at(bz), cv::Size(resize_w.at(bz), resize_h.at(bz)));

    ratio_h.at(bz) = float(resize_h.at(bz))/height;
    ratio_w.at(bz) = float(resize_w.at(bz))/width;  
  }
}

tensorflow::Tensor batch_cv_mat_to_tensor(const std::vector<cv::Mat>& imgResizeArray){

    int batchSize = imgResizeArray.size();
    int height = imgResizeArray.at(0).rows;
    int width = imgResizeArray.at(0).cols;
    int depth = 3;
    tensorflow::Tensor res_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({batchSize, height, width, 3}));
    auto tensor_mapped = res_tensor.tensor<float, 4>();    

    for(int bz = 0; bz < batchSize; bz++) {
      cv::Mat image2;
      imgResizeArray.at(bz).convertTo(image2, CV_32FC1);
      //we assume that the image is unsigned char dtype
      const float *source_data = (float*)(image2.data); 

      for (int y = 0; y < height; ++y) {
        const float* source_row = source_data + (y * width * depth);
        for (int x = 0; x < width; ++x) {
          const float* source_pixel = source_row + (x * depth);
          float b = *(source_pixel)    ;
          float g = *(source_pixel + 1);
          float r = *(source_pixel + 2);
          tensor_mapped(bz, y, x, 0) = r;   //add @0127
          tensor_mapped(bz, y, x, 1) = g;
          tensor_mapped(bz, y, x, 2) = b;
        }
      }       
    }
  
  return res_tensor;
}


void resize_image_fix_height(const cv::Mat& image,
                             cv::Mat& resized_image,
                             float& ratio,
                             int fixed_height){
  int height = image.rows;
  int width = image.cols;
  ratio = float(fixed_height)/height;
  int resize_h = fixed_height;
  int resize_w = int(width * ratio);
  cv::resize(image, resized_image, cv::Size(resize_w, resize_h));  
}

void pad_image_width(const cv::Mat& image,
                     cv::Mat& padded_image,
                     int target_width){
  int height = image.rows;
  int width = image.cols;
  int borderType = cv::BORDER_CONSTANT;
  if(width > target_width)
    cv::resize(image, padded_image, cv::Size(target_width, height));
  else if(width < target_width){
    int pad_len = target_width - width;
    copyMakeBorder(image, padded_image, 0, 0, 0, pad_len, borderType, cv::Scalar(0,0,0));
  }else
    padded_image = image.clone();
}



cv::Mat tensor_to_cv_mat(const tensorflow::Tensor tensor){
  auto tensor_data = tensor.flat<float>();
  //assume it is a 4d tensor
  auto tensor_shape = tensor.shape();
  int height = tensor_shape.dim_size(1);
  int width = tensor_shape.dim_size(2);
  std::cout<<" height "<<height << " width "<< width<<std::endl;

  cv::Mat res_mat = cv::Mat(height, width, CV_32FC1, cv::Scalar(0));
  float *res_data = (float*)(res_mat.data); 
  float min_val=100000, max_val=0;
  //(TODO) is there any other ways to copy the data into tensor?
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      res_data[width*y+x] = float(tensor_data(y*width+x)) * 255;
      min_val = std::min(min_val, tensor_data(y*width+x));
      max_val = std::max(max_val, tensor_data(y*width+x));
    }
  }
  std::cout<<"min max tensor value: "<<min_val<<" "<<max_val<<std::endl;
  return res_mat;
}



