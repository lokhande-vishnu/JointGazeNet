#ifndef CAFFE_JMMD_LOSS_LAYER_HPP_
#define CAFFE_JMMD_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class JMMDLossLayer : public LossLayer<Dtype> {
 public:
  explicit JMMDLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "JMMDLoss"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 4; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  /*virtual Dtype get_label_kernel(MMDParameter_LabelKernelMode kernel_mode, int source_label, */
          //const Dtype* target_p, const int label_dim);
  //virtual Dtype get_label_kernel(MMDParameter_LabelKernelMode kernel_mode, const Dtype* target_p1, 
          //const Dtype* target_p2, const int label_dim);
  //virtual Dtype get_label_kernel(MMDParameter_LabelKernelMode kernel_mode, const int source_label1,
          /*const int source_label2);*/

  int source_num_;
  int target_num_;
  int total_num_;
  int dim_;
  int kernel_num_;
  Dtype kernel_mul_;
  Dtype gamma_;
  Dtype sigma_;
  Blob<Dtype> diff_;
  vector<Blob<Dtype>*> kernel_val_;
  Blob<Dtype> diff_multiplier_;
  Dtype loss_weight_;
  Blob<Dtype> delta_;
  int label_kernel_num_;
  Dtype label_kernel_mul_;
  int train_iter_num_;
};

}  // namespace caffe

#endif  // CAFFE_JMMD_LAYER_HPP_
