#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
//#include <mat.h>
#include "caffe/nms.hpp"

using namespace caffe;
using std::vector;
using std::string;
using std::cout;

template <typename Dtype>
struct img_data{
	string name;
	int height;
	int width;
	int height_ori;
	int width_ori;
	vector<Dtype> data;
	int num;
	vector<vector<int> > bboxes;
};

template<typename Dtype>
int readWindowsFromFile(std::string filename, vector<struct img_data<Dtype> >& windows){
	std::ifstream fin(filename.c_str());
	std::cout << "Listfile " << filename << std::endl;
	while(!fin.eof()){
		string imgfile;
		int numWindows;
		fin >> imgfile >> numWindows;
	//	std::cout << "Imagefile:"<<imgfile<<" NumWindows:" << numWindows << std::endl;
		if (imgfile==""){
			return 0;
		}
		cv::Mat im=cv::imread(imgfile,CV_LOAD_IMAGE_COLOR);
		if(!im.data){
			std::cout << "Couldn't open or find file " << imgfile << std::endl;
			return 1;
		}
		//vector<vector<int> > win;
		//int label;
		//float score;
		struct img_data<Dtype> winImg;
		winImg.name = imgfile;
		winImg.num = numWindows;
		for(int i=0;i<numWindows;++i){
			vector<int> window(4,0);
			fin >> window[0] >> window[1] >> window[2] >> window[3];
			winImg.bboxes.push_back(window);
		}

		assert(numWindows == winImg.bboxes.size());
		windows.push_back(winImg);
	}
	return 0;
}

template <typename Dtype>
cv::Mat Blob2Mat(Blob<Dtype>& data_mean){
	const Dtype* mean = data_mean.cpu_data();

	const int height = data_mean.height();
	const int width = data_mean.width();
	const int channels = data_mean.channels();

	CHECK_EQ( channels, 3);
	cv::Mat M(height, width, CV_32FC3);
	for(int c=0; c< channels;++c){
		for(int h=0;h<height;++h){
			for(int w=0;w<width;++w){
				M.at<cv::Vec3f>(h,w)[c] = static_cast<float>(mean[(c*height+h)*width+w]);
			}
		}
	}
	//LOG(ERROR) << "ret[0][0]:" << M.at<cv::Vec3f>(0,0)[0] << " mean[0]:" << mean[0];

	return M;
}


template <typename Dtype>
void readImagesToCache(vector<struct img_data<Dtype> >& cache, const vector<struct img_data<Dtype> >& data, int crop_size, cv::Mat& data_mean, int cache_size, 
		int* Index, int *num, float scale, bool multiview){
	int maxIdx = (*Index+cache_size)<data.size()?(*Index+cache_size):data.size();
	*num = 0;
	//img_size.clear();
	assert(multiview == false);
	for(int i=*Index;i<maxIdx;++i){
		struct img_data<Dtype> cache_t;
		cache_t.name = data[i].name;
		cache_t.num = data[i].num;
		cache_t.bboxes = data[i].bboxes;
		cv::Mat img=cv::imread(data[i].name, CV_LOAD_IMAGE_COLOR);
		
		int height = img.rows;
		int width = img.cols;
		//data[i].height_ori = height;
		//data[i].width_ori = width;
		cache_t.height_ori = height;
		cache_t.width_ori = width;
		int target_h, target_w;
		if(height > width){
			target_w = crop_size;
			target_h = int(height*(float(target_w)/width));
		}else{
			target_h = crop_size;
			target_w = int(width*(float(target_h)/height));
		}
		cv::Size cv_target_size(target_w, target_h);
		cv::resize(img, img, cv_target_size, 0 , 0, cv::INTER_LINEAR);
		cv::Mat resized_mean;
		cv::resize(data_mean, resized_mean, cv_target_size, 0 , 0, cv::INTER_CUBIC);
		//const int mean_width = resized_mean.cols;
		//const int mean_height = resized_mean.rows;
		int channels=img.channels();
		if(!multiview){
			vector<Dtype>& data_t = cache_t.data;
			data_t.resize(channels*target_h*target_w, 0);
			//vector<Dtype> data_t(channels*target_h*target_w,0);
			for (int c = 0; c < channels; ++c) {
          			for (int h = 0; h < target_h; ++h) {
            				for (int w = 0; w < target_w; ++w) {
              					int top_index = (c * target_h + h) * target_w + w;
			  					Dtype pixel=static_cast<Dtype>(img.at<cv::Vec3b>(h,w)[c]);
              					/*data_t[top_index] = (pixel - 
									static_cast<Dtype>(resized_mean.at<cv::Vec3f>(h+h_off, w+w_off)[c])) * scale;*/
								data_t[top_index] = Dtype(pixel);
							}
					}
			}
			/*vector<int> img_size_t(2,0);
			img_size_t[0] = target_h;
			img_size_t[1] = target_w;
			img_size.push_back(img_size_t);*/
		//	data[i].height = target_h;
		//	data[i].width = target_w;
			cache_t.height = target_h;
			cache_t.width = target_w;
			cache.push_back(cache_t);
			//data.push_back(data_t);
		}
		else{
			/*int hc = (target_h - crop_size) / 2;
			int wc = (target_w - crop_size) / 2;
			//LOG(ERROR) << " targeth:" << target_h << " target_w:" << target_w << " mean_h:" << mean_height << " mean_w:" << mean_width;
			for(int m=0;m<9;m++){ 
				int im=m/3;
				int jm=m%3;
				int h_off = im*hc;
				int w_off = jm*wc;
				vector<Dtype> data_t(channels*crop_size*crop_size,0);
				// norm copy
				for(int c=0;c<channels;++c){
					for(int h=0;h<crop_size;++h){
						for(int w=0;w<crop_size;++w){
							int top_index = (c * crop_size + h) * crop_size + w;
			  				Dtype pixel=static_cast<Dtype>(img.at<cv::Vec3b>(h+h_off,w+w_off)[c]);
              				data_t[top_index] = (pixel - static_cast<Dtype>(resized_mean.at<cv::Vec3f>(h+h_off, w+w_off)[c])) * scale;
						}
					}
				}
				data.push_back(data_t);
				//Copy mirrored version
				vector<Dtype> data_tm(channels*crop_size*crop_size,0);
				for(int c=0;c<channels;++c){
					for(int h=0;h<crop_size;++h){
						for(int w=0;w<crop_size;++w){
							int top_index = (c * crop_size + h) * crop_size + (crop_size - 1 - w);
			  				Dtype pixel=static_cast<Dtype>(img.at<cv::Vec3b>(h+h_off,w+w_off)[c]);
              				data_tm[top_index] = (pixel - static_cast<Dtype>(resized_mean.at<cv::Vec3f>(h+h_off, w+w_off)[c])) * scale;
						}
					}
				}
				data.push_back(data_tm);
			}*/
			LOG(ERROR) << "multiview is not implemented.";
		}
		(*num)++;
	}
	assert((*Index + *num)==maxIdx);
	*Index = maxIdx;
}

template <typename Dtype>
int readImagesToBlob(Blob<Dtype>& data, vector<struct img_data<Dtype> >& cache, int batch_size, int batchIdx){
	Dtype* top_data = data.mutable_cpu_data();
	int startIdx = batch_size * batchIdx;
	int num = 0;
	for(int i=0;i<batch_size && (i+startIdx)<cache.size();++i){
		vector<Dtype>& cache_t = cache[i+startIdx].data;
		for(int j=0;j<cache_t.size();++j){
			top_data[i*cache_t.size()+j] = cache_t[j];
		}
		++num;
	}
	return num;
}

/*
template<typename Dtype>
int freeWin(vector<win_img*>& windows){
	for(int i=0;i<windows.size();++i){
		delete windows[i];
	}
	windows.clear();
}*/


string num2str(double i)
{
	std::stringstream ss;
	ss << i;
	return ss.str();
}

template <typename Dtype>
int detect(int argc, char** argv){
	namespace bf=boost::filesystem;
	if (argc < 9){
		LOG(ERROR)<< "Usage: "<<argv[0]<<" pretrained_net_param feature_extraction_proto_file extract_feature_blob_name spp_layer_name filelist meanfile savepath mode";
		return 1;
	}
	int mode=atoi(argv[8]);
	if (mode==0){
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	}else{
		LOG(ERROR)<< "Using GPU";
		uint device_id = 0;
		LOG(ERROR) << "Using Device_id=" << device_id;
		Caffe::SetDevice(device_id);
		Caffe::set_mode(Caffe::GPU);
	}
	Caffe::set_phase(Caffe::TEST);
	string extract_feature_blob_name=argv[3];
	string spp_layer_name = argv[4];
	string tst_filelist=argv[5];
	string mean_file = argv[6];
	string save_path = argv[7];

	shared_ptr<Net<Dtype> > feature_extraction_net(new Net<Dtype>(argv[2]));
	feature_extraction_net->CopyTrainedLayersFrom(argv[1]);
	shared_ptr<Blob<Dtype> > feature_blob=feature_extraction_net->blob_by_name(extract_feature_blob_name);
	int layerIdx = feature_extraction_net->layerIdx_by_name(spp_layer_name);
	if(layerIdx == -1){
		LOG(ERROR) << "Can't find layer:" << spp_layer_name;
	}else{
		LOG(ERROR) << "LayerIdx:" << layerIdx << " continue...";
	}
	shared_ptr<Layer<Dtype> > spp_layer = feature_extraction_net->layer_by_name(spp_layer_name);

	shared_ptr<Blob<Dtype> > data_blob = feature_extraction_net->blob_by_name("data");
	LOG(ERROR) << "batch size:" << data_blob->num();
	int batch_size = data_blob->num();
	int channels = data_blob->channels();
	int crop_size = data_blob->height();
	vector<struct img_data<Dtype> > windows;
	LOG(ERROR) << "load bboxes";
	if(!readWindowsFromFile<Dtype>(tst_filelist, windows)){
		std::cout<< "parse Data Done." << std::endl;
	}else{
		std::cout<<"parse Data failed."<<std::endl;
		return 1;
	}

	Blob<Dtype> data_mean;
	BlobProto blob_proto;
	std::cout << "reading data_mean from " << mean_file << std::endl;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
	data_mean.FromProto(blob_proto);
	cv::Mat mat_mean = Blob2Mat<Dtype>(data_mean);
	CHECK_EQ(data_mean.num(), 1);
	CHECK_EQ(data_mean.width(), data_mean.height());
	CHECK_EQ(data_mean.channels(), 3);
	std::cout << "prepare parameters" << std::endl;

	float scale = 1.0;	
	std::cout << "processing windows..." << std::endl;
	bf::path output_path(save_path);
	Blob<Dtype>* bottom = new Blob<Dtype>(batch_size, 3, crop_size, crop_size);
	vector<Blob<Dtype>*> bottomV;
	bottomV.push_back(bottom);
	
	int numCaches = ceil(float(windows.size()) / batch_size);
	Dtype* feature_blob_data;
	Dtype* im_blob_ori;
	int num=0;
	int startIdx = 0;
	std::ofstream fo(output_path.string().c_str());
	bool multivew = false;

	LOG(ERROR) << "cachesize:" << batch_size << " numCaches:" << numCaches;
	clock_t start_processing, end_processing;
	start_processing = clock();
	for(int cacheIdx = 0;cacheIdx < numCaches;cacheIdx++){
		LOG(ERROR) << "processing:" << cacheIdx << "/" << numCaches;
		vector<struct img_data<Dtype> > cache;
		//vector< vector<Dtype> > resultcache;
		//clock_t start_cache, end_cache;
		//start_cache = clock();

		readImagesToCache(cache, windows, crop_size, mat_mean, batch_size, &startIdx, &num, scale, multivew);
		//end_cache = clock();
		//LOG(ERROR) << "readImageToCache:" << (end_cache-start_cache) << "ms";
		//start_cache = clock();
		int nBatches = ceil(float(cache.size()) / batch_size);
		//LOG(ERROR) << "nBatches:"<< nBatches << " cache:" << cache.size();
		for(int batchIdx = 0;batchIdx < nBatches; batchIdx++){
			time_t start_epoch, end_epoch;
			start_epoch = time(NULL);
			LOG(ERROR) << "ResetLayer: height" <<cache[batchIdx].height << " width:" << cache[batchIdx].width ;
			bottom->Reshape(bottom->num(), bottom->channels(), cache[batchIdx].height, cache[batchIdx].width);
			feature_extraction_net->ResetLayers(layerIdx, cache[batchIdx].height, cache[batchIdx].width);
			int n=readImagesToBlob(*bottom, cache, batch_size, batchIdx);
			float loss = 0.0;
			LOG(ERROR) << "forward";
			const vector<Blob<Dtype>*>& result_t =  feature_extraction_net->Forward(bottomV, layerIdx-1, &loss);
			vector<vector<int> >& bboxes = cache[batchIdx].bboxes;
			for(int i=0;i< bboxes.size();++i){
				vector<int> bbox(4,0);
				int width_ori = cache[batchIdx].width_ori;
				int width_resized = cache[batchIdx].width;
				float ratio = float(width_resized) / float(width_ori);
				for(int k=0;k<4;++k){
					bbox[k] = int(ratio*bboxes[i][k]);
				}
				//vector<int> 
				feature_extraction_net->Forward_onelayer(layerIdx, bbox, &loss);
				const vector<Blob<Dtype>*>& result = feature_extraction_net->Forward_continue(layerIdx + 1, &loss);

				//SetTopAct<Dtype>(feature_blob);
				int height_t = feature_blob->height();
				int width_t = feature_blob->width();
				LOG(ERROR) << "feature_blob:" << height_t << " " << width_t;
			}
		}
	}
	return 0;
}

int main(int argc, char** argv){
	return detect<float>(argc, argv);
}
