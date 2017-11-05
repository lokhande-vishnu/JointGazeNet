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
  alpha_.Reshape(bottom[0]->num(), bottom[0]->channels(),bottom[0]->height(), bottom[0]->width());
  }
  
  template <typename Dtype>
  void DiffLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
						     const vector<Blob<Dtype>*>& top) {

    Forward_cpu(bottom, top);
    /*
    const Dtype* bottom_common = bottom[0]->gpu_data();
    const Dtype* bottom_private = bottom[1]->gpu_data();
    int batch_size = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    Dtype loss = 0.0;

    printf("bathc-dim ******* %d %d\n", batch_size, dim);
    printf("common******* %d %d %d %d\n", bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
    printf("private******* %d %d %d %d\n", bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
    printf("size ==  %lu %lu %lu\n",sizeof(bottom_common) ,sizeof(bottom_common[0]) , (sizeof(bottom_common)/sizeof(bottom_common[0])));
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
	for (int k = 0; k < batch_size; k++) {    
	  loss += 1; // bottom_common[k*dim + i] * bottom_private[k*dim + j] ;
	}
      }
    }
    top[0]->mutable_cpu_data()[0] = loss / batch_size;
    */
  }

  template <typename Dtype>
  void DiffLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
						      const vector<bool>& propagate_down,
						      const vector<Blob<Dtype>*>& bottom) {
    Backward_cpu(top, propagate_down, bottom);
/*
    const Dtype l = top[0]->cpu_diff()[0];
    const Dtype* bottom_common = bottom[0]->gpu_data();
    const Dtype* bottom_private = bottom[1]->gpu_data();
    Dtype* alpha_data = alpha_.mutable_gpu_data();
    
    int batch_size = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    
    for (int i = 0; i < batch_size * dim; i++) {
      alpha_data[i] = 0;
    }
    
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
	for (int k = 0; k < batch_size; k++) {    
	  alpha_data[i*dim + j] = bottom_common[k*dim + i] * bottom_private[k*dim + j] ;
	  //	  caffe_cpu_axpby(
	  //		  1,
	  //		  bottom_common[k*dim + i],
	  //		  bottom_private[k*dim + j],
	  //		  Dtype(0),
	  //		  alpha[i*dim + j]);
	}
      }
    }

    if (propagate_down[0]) {
      for (int i = 0; i < batch_size; i++) {
	for (int j = 0; j < batch_size; j++) {
	  for (int k = 0; k < dim; k++) {    
	    (bottom[0]->mutable_gpu_diff())[i*dim + j] = 2.0 * l * bottom_private[i*dim + k] * alpha_data[j*dim + k] / batch_size;
	  }
	}
      }
      
    }
    
    if (propagate_down[1]) {
      for (int i = 0; i < batch_size; i++) {
	for (int j = 0; j < dim; j++) {
	  for (int k = 0; k < dim; k++) {    
	    (bottom[0]->mutable_gpu_diff())[i*dim + j] = 2.0 * l * bottom_common[i*dim + k] * alpha_data[k*dim + j] / batch_size;
	  }
	}
      }
    }
*/
  }

INSTANTIATE_LAYER_GPU_FUNCS(DiffLossLayer);

}  // namespace caffe
