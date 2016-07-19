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
//Tree Implementation
int Tree::Depth() const {
	int max_depth = 0;
	for(int i = 0; i < this->children.size(); i++) {
	  int depth = this->children[i]->Depth();
	  if( max_depth < depth ) max_depth = depth;
	}
	return max_depth + 1;
}
void Tree::MakeBalance(int remain) {
	if( remain == 0 ) return;
	if( children.size() == 0 ) {
	  Tree * root = this;
	  int label = root->label;
	  for(int i = 0; i < remain; ++i ) {
		  root->InsertChild(shared_ptr<Tree>(new Tree()));
		  root->SetLabel(-1);
		  root = root->children[0].get();
	  }
	  root->SetLabel(label);
	}
	else {
	  for(int i = 0; i < children.size(); ++i)
		  children[i]->MakeBalance(remain-1);
	}
}
//Tree helper
void Tree::GiveIndex(Tree * root, std::vector<Tree *>& serialized_tree) {
	int cnt = 0;
	std::queue<Tree *> queue;
	queue.push(root);
	while( queue.size() != 0 ) {
	  Tree * node = queue.front();
	  node->index = cnt++;

	  serialized_tree.push_back(node);
	  for(int i = 0; i < node->children.size(); ++i)
		  queue.push(node->children[i].get());
	  queue.pop();
	}
}
void Tree::GetNodeNumPerLevelAndGiveLabel(std::vector<int>& node_num, std::vector<int>& base_index,Tree * root, std::vector<Tree *>& serialized_tree, std::vector<int>& label_to_index) { 
	Tree * right_root = root;
	int depth = root->Depth();
	node_num.resize(depth-1);
	base_index.resize(depth-1);
	for(int i = 0; i < depth-1; ++i)
	{
	  node_num[i] = right_root->children[right_root->children.size()-1]->GetIndex() - root->children[0]->GetIndex() + 1;
	  base_index[i] = root->children[0]->index;
	  root = root->children[0].get();
	  right_root = right_root->children[right_root->children.size()-1].get();

	  if( i < depth-2 ){ //label for last layer is already made
		  for(int j = base_index[i]; j < base_index[i]+node_num[i]; ++j)
			  serialized_tree[j]->label = j - base_index[i];
	  }
	  else {
		  label_to_index.resize(node_num[i]);
		  for(int index = 0; index < node_num[i]; ++index) {
			  int label = serialized_tree[index+base_index[i]]->GetLabel();
			  CHECK_LT(label,node_num[i]);
			  label_to_index[label] = index;
		  }
	  }

	}
}
void Tree::MakeTree(Tree * node, const SuperCategoryParameter::TreeScheme * node_param){
	if( node_param->children_size() == 0 ){
		CHECK_NE(node_param->label(),-1);
		node->SetLabel(node_param->label());
	}
	else {
		CHECK_EQ(node->label,-1);
		for(int i = 0; i < node_param->children_size(); ++i) {
			shared_ptr<Tree> child(new Tree());
			node->InsertChild(child);
			MakeTree(child.get(), &node_param->children(i));
		}
	}
}

//Layer Implementation
template <typename Dtype>
void SuperCategoryLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	SuperCategoryParameter * super_param = this->layer_param_.mutable_super_category_param();
	if( super_param->file_name().empty() == false ) {
		ReadProtoFromTextFileOrDie(super_param->file_name().c_str(), super_param->mutable_root());
	}

	Tree::MakeTree(&root_, &super_param->root());
	depth_ = root_.Depth();
	root_.MakeBalance(depth_-1);
	Tree::GiveIndex(&root_, serialized_tree_);
	Tree::GetNodeNumPerLevelAndGiveLabel(node_num_per_level_, base_index_per_level_, &this->root_,serialized_tree_,label_to_index_);

	N_ = bottom[0]->count(0,1);
	CHECK_EQ(*node_num_per_level_.rbegin(), bottom[0]->count(1));

	this->temp_.Reshape(N_,1,1,1);
}

template <typename Dtype>
void SuperCategoryLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	SuperCategoryParameter * super_param = this->layer_param_.mutable_super_category_param();
	if( super_param->file_name().empty() == false ) {
		ReadProtoFromTextFileOrDie(super_param->file_name().c_str(), super_param->mutable_root());
	}

	N_ = bottom[0]->count(0,1);

	Tree::MakeTree(&root_, &super_param->root());
	depth_ = root_.Depth();
	root_.MakeBalance(depth_-1);
	Tree::GiveIndex(&root_, serialized_tree_);
	Tree::GetNodeNumPerLevelAndGiveLabel(node_num_per_level_, base_index_per_level_, &this->root_,serialized_tree_,label_to_index_);
}

template <typename Dtype>
void SuperCategoryLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	CHECK_EQ(top.size(), depth_-1);

	mark_.resize(top.size());
	for( int i = 0; i < depth_-1; ++i) {
		std::vector<int> shape;
		shape.push_back(N_);
		shape.push_back(node_num_per_level_[i]);
		top[i]->Reshape(shape); // Top for output data
		mark_[i].reset(new Blob<int> (shape));// Marking for Maxpoolling backprop
	}
}

template <typename Dtype>
void SuperCategoryLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	CHECK_EQ(top.size(), depth_-1);

	int i = 0;
	for( i = 0; i < depth_-1; ++i) {
		std::vector<int> shape;
		shape.push_back(N_);
		top[i]->Reshape(shape); // Top for label
	}
	CHECK_EQ(bottom[0]->count(), N_);
}

template <typename Dtype>
void SuperCategoryLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	//For Data
	for(int n = 0; n < N_; ++n) {
		for(int i = depth_-2; i >= 0; --i)
		{
			int node_cnt;
			if( i == depth_-2)
				node_cnt = node_num_per_level_[i];
			else
				node_cnt = node_num_per_level_[i+1];

			Blob<Dtype> * bottoms;
			if( i == depth_-2 )
				bottoms = bottom[0];
			else
				bottoms  = top[i+1];

			Dtype * top_data = &top[i]->mutable_cpu_data()[node_num_per_level_[i]*n];
			int * mark_data = &mark_[i]->mutable_cpu_data()[node_num_per_level_[i]*n];
			const Dtype * bottom_data = &bottoms->cpu_data()[node_cnt*n]; //is equal.

			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j ) {
				Tree * node = serialized_tree_[base_idx + j];
				const std::vector<shared_ptr<Tree> > * children = node->GetChildren();
				if( children->size() == 0 )
				{
					CHECK_EQ(i, depth_-2);
					//caffe_mul<Dtype>(N_,&blob_data[N_*j], &bottom_data[N_*j], &top_data[N_*j]);
					top_data[j] = bottom_data[j];
				}
				else{
					int node_label = node->GetLabel();
					top_data[node_label] = -1 * std::numeric_limits<Dtype>::max();
					for(auto it = children->cbegin(); it != children->cend(); ++it) {
						int label = (*it)->GetLabel();
						//caffe_mul<Dtype>(N_,&blob_data[idx*N_],&bottom_data[idx*N_],temp_.mutable_cpu_data());
						//caffe_add<Dtype>(N_,temp_.cpu_data(),&top_data[j*N_],&top_data[j*N_]);
						if( top_data[node_label] < bottom_data[label] )
						{
							top_data[node_label] = bottom_data[label];
							mark_data[node_label] = label;
						}
					}
				}
			}
		}
	}
}

template <typename Dtype>
void SuperCategoryLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	//For Label
	for(int n = 0; n < N_; ++n) {
		int idx = label_to_index_[static_cast<int>(bottom[0]->cpu_data()[n])] + *(base_index_per_level_.rbegin());
		const Tree * node = serialized_tree_[idx];
		for(int i = depth_-2; i >= 0; --i) {
			top[i]->mutable_cpu_data()[n] = node->GetLabel();
			node = node->GetParent();
		}
		CHECK_EQ(top[depth_-2]->cpu_data()[n],bottom[0]->cpu_data()[n]);
	}
}

template <typename Dtype>
void SuperCategoryLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());

	if( propagate_down[0] ) {

		for(int n = 0; n < N_; ++n) {
			for(int i = 0; i < depth_-1; ++i) {

				int node_cnt;
				if( i == depth_-2)
					node_cnt = node_num_per_level_[i];
				else
					node_cnt = node_num_per_level_[i+1];

				const Dtype * top_diff = &top[i]->cpu_diff()[n*node_num_per_level_[i]];
				const int * mark_data = &mark_[i]->cpu_data()[n*node_num_per_level_[i]];
				Dtype * bottom_diff;
				if( i + 1 == depth_-1 ){
					bottom_diff = &bottom[0]->mutable_cpu_diff()[n*node_cnt];
				}
				else {
					bottom_diff = &top[i+1]->mutable_cpu_diff()[n*node_cnt];
				}

				int base_idx = base_index_per_level_[i];
				for(int j = 0; j < node_num_per_level_[i]; ++j) {
					Tree * node = serialized_tree_[base_idx + j];
					const std::vector<shared_ptr<Tree> > * children = node->GetChildren();
					if( children->size() == 0 ) { //this layer is connected with bottom layer
						//caffe_mul<Dtype>(N_,&top_diff[j*N_],&bottom_data[j*N_],&blob_diff[j*N_]);
						//caffe_mul<Dtype>(N_,&top_diff[j*N_],&blob_data[j*N_],&bottom_diff[j*N_]);
						CHECK_EQ(i, depth_-2);
						bottom_diff[j] = top_diff[j];
					}
					else {
						int node_label = node->GetLabel();
						int label = mark_data[node_label];
						bottom_diff[label] += top_diff[node_label];
					}
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(SuperCategoryLayer);
STUB_GPU(SuperCategoryLabelLayer);
#endif

INSTANTIATE_CLASS(SuperCategoryLayer);
REGISTER_LAYER_CLASS(SuperCategory);

INSTANTIATE_CLASS(SuperCategoryLabelLayer);
REGISTER_LAYER_CLASS(SuperCategoryLabel);
}  // namespace caffe
