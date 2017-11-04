
#include <vector>

// #include "caffe/filler.hpp"
#include "caffe/layers/diff_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void DiffLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
						 const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  // diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),bottom[0]->height(), bottom[0]->width());
  }
  
  template <typename Dtype>
  void DiffLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
						     const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();
    const Dtype* bottom_common = bottom[0]->cpu_data();
    const Dtype* bottom_private = bottom[1]->cpu_data();
    int num_batch = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    Dtype loss = 0.0;
    
    for (int i = 0; i < num_batch; i++) {
      for (int j = 0; j < num_batch; j++) {
	loss += caffe_cpu_dot(dim, bottom_common[i*dim], bottom_private[j*dim]);
      }
    }
    top[0]->mutable_cpu_data()[0] = loss;
  }

  template <typename Dtype>
  void DiffLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
						      const vector<bool>& propagate_down,
						      const vector<Blob<Dtype>*>& bottom) {

    const Dtype l = top[0]->cpu_diff();
    const Dtype* HC_data = bottom[0]->cpu_diff();
    const Dtype* HP_data = bottom[1]->cpu_diff();
    int num_batch = bottom[0]->num();
    int num_chan = bottom[0]->channels;

    int M = bottom[0]->shape(3);
    int K = bottom[0]->shape(2);
    int N = bottom[0]->shape(3);
    Blob<Dtype> alpha;
    alpha.Reshape(bottom[0]->num(), bottom[0]->channels, bottom[0]->width, bottom[1]->width);
    Blob<Dtype> beta;
    beta.Reshape(bottom[0]->num(), bottom[0]->channels, bottom[0]->width, bottom[1]->width);
    for(int b = 0; b < num_batch; b++) {
      for(int c = 0; c < num_chan; c++) {
	if (propagate_down[0]) {
	  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
				M, N, K,
				(Dtype)2.0,
				HC_data[b][c],
				HP_data[b][c],
				(Dtype)0.0,
				alpha[b][c]);
	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
				M, K, N,
				(Dtype)1.0,
				alpha[b][c],
				HP_data[b][c],
				(Dtype)0.0,
				bottom[0]->mutable_cpu_diff()[b][c]);	  
	}

	if (propagate_down[1]) {
	  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
				M, N, K,
				(Dtype)2.0,
				HC_data[b][c],
				HP_data[b][c],
				(Dtype)0.0,
				beta[b][c]);
	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
				K, N, M,
				(Dtype)1.0,
				HC_data[b][c],
				beta[b][c],
				(Dtype)0.0,
				bottom[1]->mutable_cpu_diff()[b][c]);	  

	}
      }
    }
  }

  #ifdef CPU_ONLY
  STUB_GPU(DiffLossLayer);
  #endif

  INSTANTIATE_CLASS(DiffLossLayer);
  REGISTER_LAYER_CLASS(DiffLoss);

}  // namespace caffe
