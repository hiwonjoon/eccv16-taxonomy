/*
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SuperCategoryFMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top[depth_-1]->mutable_gpu_data());
	for(int i = 0; i < depth_-1; ++i)
		caffe_gpu_set(top[i]->count(), (Dtype)0., top[i]->mutable_gpu_data());

	for(int m = 0; m < M_; ++m) {
		for( int i = depth_-2; i >= 0; --i ) {
			Blob<Dtype> * tops = top[i];
			Blob<Dtype> * bottoms = top[i+1];

			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j) {
				Tree * node = serialized_tree_[base_idx + j];
				const std::vector<shared_ptr<Tree> >* children = node->GetChildren();

				Dtype * top_data = &tops->mutable_gpu_data()[tops->offset(m,node->GetLabel())];

				for(std::vector<shared_ptr<Tree> >::const_iterator it = children->begin(); it != children->end(); ++it) {
					int offset = bottoms->offset(m,(*it)->GetLabel());
					const Dtype * bottom_data = &bottoms->gpu_data()[offset];
					caffe_gpu_axpy(H_*W_,(Dtype)(1.),bottom_data,top_data);
				}

				caffe_gpu_scal(H_*W_,(Dtype)(1./children->size()),top_data);
			}
		}
	}
}

template <typename Dtype>
void SuperCategoryFMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	if( propagate_down[0] == false )
		return;

	for(int m = 0; m < M_; ++m) {
		for( int i = 0; i < depth_-1; ++i ) {
			Blob<Dtype> * tops = top[i];
			Blob<Dtype> * bottoms = top[i+1];

			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j) {
				Tree * node = serialized_tree_[base_idx + j];
				const std::vector<shared_ptr<Tree> >* children = node->GetChildren();
				const Dtype * top_diff = &tops->gpu_diff()[tops->offset(m,node->GetLabel())];
				for(std::vector<shared_ptr<Tree> >::const_iterator it = children->begin(); it != children->end(); ++it) {
					int offset = bottoms->offset(m,(*it)->GetLabel());
					Dtype * bottom_diff = &bottoms->mutable_gpu_diff()[offset];

					caffe_gpu_axpy(H_*W_,(Dtype)(1./children->size()),top_diff,bottom_diff);	
				}

			}
		}
	}
	caffe_copy(bottom[0]->count(), top[depth_-1]->gpu_diff(), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(SuperCategoryFMLayer);

}  // namespace caffe
*/
