#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_gaze_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyGazeLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_gaze_param().top_k();
}

template <typename Dtype>
void AccuracyGazeLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
      << "top_k must be less than or equal to the number of classes.";
  //CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void AccuracyGazeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  //int dim = bottom[0]->count() / bottom[0]->num();
  //vector<Dtype> maxval(top_k_+1);
  //vector<int> max_id(top_k_+1);
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

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / num;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AccuracyGazeLayer);
REGISTER_LAYER_CLASS(AccuracyGaze);

}  // namespace caffe
