#include <vector>

#include "caffe/layers/euclidean_gaze_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanGazeLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void EuclideanGazeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  //Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  //Dtype loss = dot / bottom[0]->num() / Dtype(2);
  //top[0]->mutable_cpu_data()[0] = loss;
  
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  
  for (int i = 0; i < num; ++i) {
    // Accuracy
    float data_x = (-1)*cos(bottom_data[i * 2 + 0])*sin(bottom_data[i * 2 + 1]);
    float data_y = (-1)*sin(bottom_data[i * 2 + 0]);
    float data_z = (-1)*cos(bottom_data[i * 2 + 0])*cos(bottom_data[i * 2 + 1]);
    float norm_data = sqrt(data_x*data_x + data_y*data_y + data_z*data_z);
    
    float label_x = (-1)*cos(bottom_label[i * 2 + 0])*sin(bottom_label[i * 2 + 1]);
    float label_y = (-1)*sin(bottom_label[i * 2 + 0]);
    float label_z = (-1)*cos(bottom_label[i * 2 + 0])*cos(bottom_label[i * 2 + 1]);
    float norm_label = sqrt(label_x*label_x + label_y*label_y + label_z*label_z);

    float angle_value = (data_x*label_x+data_y*label_y+data_z*label_z) / (norm_data*norm_label);
    if (angle_value > 1.0)
      angle_value = 1.0;
    else if (angle_value < -1.0)
      angle_value = -1.0;
    accuracy += (acos(angle_value)*180)/3.1415926;
  }
  top[0]->mutable_cpu_data()[0] = accuracy/num;
}

template <typename Dtype>
void EuclideanGazeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanGazeLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanGazeLossLayer);
REGISTER_LAYER_CLASS(EuclideanGazeLoss);

}  // namespace caffe
