#include <vector>
#include <limits>
#include <queue>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {
//Layer Implementation
template <typename Dtype>
void SuperCategoryFMPostLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	op_ = this->layer_param_.eltwise_param().operation();

	SuperCategoryParameter * super_param = this->layer_param_.mutable_super_category_param();
	if( super_param->file_name().empty() == false ) {
		ReadProtoFromTextFileOrDie(super_param->file_name().c_str(), super_param->mutable_root());
	}

	Tree::MakeTree(&root_, &super_param->root());
	depth_ = root_.Depth() - 1;
	root_.MakeBalance(depth_);
	Tree::GiveIndex(&root_, serialized_tree_);
	Tree::GetNodeNumPerLevelAndGiveLabel(node_num_per_level_, base_index_per_level_, &this->root_,serialized_tree_,label_to_index_);

	CHECK_EQ(depth_,bottom.size());

	M_ = bottom[depth_-1]->shape(0);
	H_ = bottom[depth_-1]->shape(2);
	W_ = bottom[depth_-1]->shape(3);
	for(int i = 0; i < depth_; ++i) {
		CHECK_EQ(bottom[i]->shape(0),M_);
		CHECK_EQ(bottom[i]->shape(1),node_num_per_level_[i]);
		CHECK_EQ(bottom[i]->shape(2),H_);
		CHECK_EQ(bottom[i]->shape(3),W_);
	}
}

template <typename Dtype>
void SuperCategoryFMPostLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(depth_,top.size());
	for(int i = 0; i < depth_; ++i) {
		top[i]->ReshapeLike(*bottom[i]); // Top for output data
	}
}

template <typename Dtype>
void SuperCategoryFMPostLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	for(int i = 0; i < depth_; ++i)
		caffe_copy(top[i]->count(), bottom[i]->cpu_data(), top[i]->mutable_cpu_data());

	switch (op_) {
	case EltwiseParameter_EltwiseOp_SUM :
		for(int m = 0; m < M_; ++m) {
			for(int i = 0; i < depth_-1; ++i) {
				Blob<Dtype> * tops = top[i];
				Blob<Dtype> * bottoms = top[i+1];
					
				int base_idx = base_index_per_level_[i];
				for(int j = 0; j < node_num_per_level_[i]; ++j) {
					Tree * node = serialized_tree_[base_idx + j];
					const std::vector<shared_ptr<Tree> >* children = node->GetChildren();
					const Dtype * top_data = &tops->cpu_data()[tops->offset(m,node->GetLabel())];

					for(auto it = children->cbegin(); it != children->cend(); ++it) {
						int offset = bottoms->offset(m,(*it)->GetLabel());
						Dtype * bottom_data = &bottoms->mutable_cpu_data()[offset];
						caffe_axpy(H_*W_,(Dtype)1.,top_data,bottom_data);
					}
				}
			}
		}
		break;
	case EltwiseParameter_EltwiseOp_MINUS :
		for(int m = 0; m < M_; ++m) {
			for(int i = 0; i < depth_-1; ++i) {
				Blob<Dtype> * tops = bottom[i];
				Blob<Dtype> * bottoms = top[i+1];
					
				int base_idx = base_index_per_level_[i];
				for(int j = 0; j < node_num_per_level_[i]; ++j) {
					Tree * node = serialized_tree_[base_idx + j];
					const std::vector<shared_ptr<Tree> >* children = node->GetChildren();
					const Dtype * top_data = &tops->cpu_data()[tops->offset(m,node->GetLabel())];

					for(auto it = children->cbegin(); it != children->cend(); ++it) {
						int offset = bottoms->offset(m,(*it)->GetLabel());
						Dtype * bottom_data = &bottoms->mutable_cpu_data()[offset];
						caffe_axpy(H_*W_,(Dtype)-1.,top_data,bottom_data);
					}
				}
			}
		}
		break;
	case EltwiseParameter_EltwiseOp_MINUS_REVERSE :
		for(int m = 0; m < M_; ++m) {
			for(int i = 0; i < depth_-1; ++i) {
				Blob<Dtype> * tops = bottom[i];
				Blob<Dtype> * bottoms = top[i+1];
					
				int base_idx = base_index_per_level_[i];
				for(int j = 0; j < node_num_per_level_[i]; ++j) {
					Tree * node = serialized_tree_[base_idx + j];
					const std::vector<shared_ptr<Tree> >* children = node->GetChildren();
					const Dtype * top_data = &tops->cpu_data()[tops->offset(m,node->GetLabel())];

					for(auto it = children->cbegin(); it != children->cend(); ++it) {
						int offset = bottoms->offset(m,(*it)->GetLabel());
						Dtype * bottom_data = &bottoms->mutable_cpu_data()[offset];
						caffe_cpu_axpby(H_*W_,(Dtype)1.,top_data,(Dtype)-1.,bottom_data);
					}
				}
			}
		}
		break;
	default:
        LOG(FATAL) << "Unknown elementwise operation.";
		break;
	}
}

template <typename Dtype>
void SuperCategoryFMPostLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	//if( propagate_down[0] == false )
	//	return;

	for(int i = 0; i < depth_; ++i) {
		if( propagate_down[i] )
			caffe_copy(bottom[i]->count(), top[i]->cpu_diff(), bottom[i]->mutable_cpu_diff());
	}

	switch (op_) {
	case EltwiseParameter_EltwiseOp_SUM :
		for(int m = 0; m < M_; ++m) {
			for( int i = depth_-1; i > 0; --i ) {
				if( propagate_down[i] != true )
					continue;

				Blob<Dtype> * tops = bottom[i-1];
				Blob<Dtype> * bottoms = bottom[i];

				int base_idx = base_index_per_level_[i-1];
				for(int j = 0; j < node_num_per_level_[i-1]; ++j) {
					Tree * node = serialized_tree_[base_idx + j];
					const std::vector<shared_ptr<Tree> >* children = node->GetChildren();
					Dtype * top_diff = &tops->mutable_cpu_diff()[tops->offset(m,node->GetLabel())];
					for(auto it = children->cbegin(); it != children->cend(); ++it) {
						int offset = bottoms->offset(m,(*it)->GetLabel());
						const Dtype * bottom_diff = &bottoms->cpu_diff()[offset];

						caffe_axpy(H_*W_,(Dtype)(1.),bottom_diff,top_diff);
					}
				}
			}
		}
		break;
	case EltwiseParameter_EltwiseOp_MINUS :
		for(int m = 0; m < M_; ++m) {
			for( int i = depth_-1; i > 0; --i ) {
				if( propagate_down[i] != true )
					continue;

				Blob<Dtype> * tops = bottom[i-1];
				Blob<Dtype> * bottoms = top[i];

				int base_idx = base_index_per_level_[i-1];
				for(int j = 0; j < node_num_per_level_[i-1]; ++j) {
					Tree * node = serialized_tree_[base_idx + j];
					const std::vector<shared_ptr<Tree> >* children = node->GetChildren();
					Dtype * top_diff = &tops->mutable_cpu_diff()[tops->offset(m,node->GetLabel())];
					for(auto it = children->cbegin(); it != children->cend(); ++it) {
						int offset = bottoms->offset(m,(*it)->GetLabel());
						const Dtype * bottom_diff = &bottoms->cpu_diff()[offset];

						caffe_axpy(H_*W_,(Dtype)(-1.),bottom_diff,top_diff);
					}
				}
			}
		}
		break;
	case EltwiseParameter_EltwiseOp_MINUS_REVERSE :
    for(int i = 1; i < depth_; ++i ) {
			caffe_scal(bottom[i]->count(), (Dtype)(-1.), bottom[i]->mutable_cpu_diff());
    }
		for(int m = 0; m < M_; ++m) {
			for( int i = depth_-1; i > 0; --i ) {
				if( propagate_down[i] != true )
					continue;

				Blob<Dtype> * tops = bottom[i-1];
				Blob<Dtype> * bottoms = top[i];

				int base_idx = base_index_per_level_[i-1];
				for(int j = 0; j < node_num_per_level_[i-1]; ++j) {
					Tree * node = serialized_tree_[base_idx + j];
					const std::vector<shared_ptr<Tree> >* children = node->GetChildren();
					Dtype * top_diff = &tops->mutable_cpu_diff()[tops->offset(m,node->GetLabel())];
					for(auto it = children->cbegin(); it != children->cend(); ++it) {
						int offset = bottoms->offset(m,(*it)->GetLabel());
						const Dtype * bottom_diff = &bottoms->cpu_diff()[offset];

						caffe_axpy(H_*W_,(Dtype)(1.),bottom_diff,top_diff);
					}
				}
			}
		}
/*		for(int m = 0; m < M_; ++m) {
			for( int i = depth_-1; i > 0; --i ) {
				if( propagate_down[i] != true )
					continue;

				Blob<Dtype> * tops = top[i];
				Blob<Dtype> * bottoms = bottom[i-1];

				int base_idx = base_index_per_level_[i-1];
				for(int j = 0; j < node_num_per_level_[i-1]; ++j) {
					Tree * node = serialized_tree_[base_idx + j];
					const std::vector<shared_ptr<Tree> >* children = node->GetChildren();
					Dtype * bottom_diff = &bottoms->mutable_cpu_diff()[bottoms->offset(m,node->GetLabel())];

					for(auto it = children->cbegin(); it != children->cend(); ++it) {
						int offset = tops->offset(m,(*it)->GetLabel());
						const Dtype * top_diff = &tops->cpu_diff()[offset];

						caffe_axpy(H_*W_,(Dtype)(-1.),top_diff,bottom_diff);
					}
				}
			}
		}
    */
		break;
	default:
        LOG(FATAL) << "Unknown elementwise operation.";
		break;
	}
}

#ifdef CPU_ONLY
STUB_GPU(SuperCategoryFMPostLayer);
#endif

INSTANTIATE_CLASS(SuperCategoryFMPostLayer);

REGISTER_LAYER_CLASS(SuperCategoryFMPost);
}  // namespace caffe


