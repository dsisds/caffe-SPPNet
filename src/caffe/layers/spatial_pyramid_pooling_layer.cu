// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void spp_Forward_region_core(int nthreads, Dtype* dst, const Dtype* src, const int num,
	const int channels, const int height_ori, const int width_ori, const int height, 
	const int width, const int h_off, const int w_off){
	CUDA_KERNEL_LOOP(index, nthreads){
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;
		int dstIdx = (n*channels + c) * height * width;
		dstIdx += h*width + w;
		int srcIdx = (n*channels + c) * height_ori * width_ori;
		srcIdx += (h+h_off) * width + w + w_off;
		*(dst + dstIdx) = *(src+srcIdx);
	}
}
template <typename Dtype>
void SpatialPyramidPoolingLayer<Dtype>::spp_Forward_region_gpu(
	const vector<Blob<Dtype>*>& bottom,
	vector<Blob<Dtype>*>* top, const vector<int>& bbox){
	assert(bottom.size()==0);
	vector<Blob<Dtype>*> bottom_t;
	int height = bbox[2]-bbox[0]+1;
	int width = bbox[3]-bbox[1]+1;
	int channels = bottom[0]->channels();
	int height_ori = bottom[0]->height();
	int width_ori = bottom[0]->width();
	int h_off = bbox[0];
	int w_off = bbox[1];

	assert(height <= bottom[0]->height());
	assert(width <= bottom[0]->width());
	Blob<Dtype>* bt = new Blob<Dtype>(bottom[0]->num(), bottom[0]->channels(), height, width);
	bottom_t.push_back(bt);
	Dtype* bt_data = bt->mutable_gpu_data();
	int count = bt->count();
	const Dtype *bottom_data = bottom[0]->gpu_data();
	spp_Forward_region_core<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, 
		bt_data, bottom_data, bottom[0]->num(), channels, height_ori, width_ori, 
		height, width, h_off, w_off);
	Forward_cpu(bottom_t, top);
	delete bt;
	bottom_t.clear();
}

INSTANTIATE_CLASS(SpatialPyramidPoolingLayer);


}  // namespace caffe
