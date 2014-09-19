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
#include <mat.h>
#include "caffe/nms.hpp"
#include <time.h>

using namespace caffe;
using std::vector;
using std::string;
using std::cout;

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
	LOG(ERROR) << "ret[0][0]:" << M.at<cv::Vec3f>(0,0)[0] << " mean[0]:" << mean[0];

	return M;
}

template <typename Dtype>
void readImagesToCache(vector<vector<Dtype> >& data, vector<string>& images, int crop_size, cv::Mat& data_mean, int cache_size, int* Index, int *num, float scale, bool multiview){
	//const Dtype* mean=data_mean.cpu_data();
	
	//const int mean_width = data_mean.cols;
	//const int mean_height = data_mean.rows;

	
	int maxIdx = (*Index+cache_size)<images.size()?(*Index+cache_size):images.size();
	*num = 0;
	for(int i=*Index;i<maxIdx;++i){
		cv::Mat img=cv::imread(images[i], CV_LOAD_IMAGE_COLOR);
		
		int height = img.rows;
		int width = img.cols;
		int target_h, target_w;
		if(height > width){
			target_w = 256;
			target_h = int(height*(float(target_w)/width));
		}else{
			target_h = 256;
			target_w = int(width*(float(target_h)/height));
		}
		cv::Size cv_target_size(target_w, target_h);
		cv::resize(img, img, cv_target_size, 0 , 0, cv::INTER_LINEAR);
		cv::Mat resized_mean;
		cv::resize(data_mean, resized_mean, cv_target_size, 0 , 0, cv::INTER_CUBIC);
		const int mean_width = resized_mean.cols;
		const int mean_height = resized_mean.rows;
		int channels=img.channels();
		if(!multiview){
			int h_off = (target_h - crop_size) / 2;
			int w_off = (target_w - crop_size) / 2;
			vector<Dtype> data_t(channels*crop_size*crop_size,0);
			for (int c = 0; c < channels; ++c) {
          			for (int h = 0; h < crop_size; ++h) {
            				for (int w = 0; w < crop_size; ++w) {
              					int top_index = (c * crop_size + h) * crop_size + w;
			  					Dtype pixel=static_cast<Dtype>(img.at<cv::Vec3b>(h+h_off,w+w_off)[c]);
              					data_t[top_index] = (pixel - static_cast<Dtype>(resized_mean.at<cv::Vec3f>(h+h_off, w+w_off)[c])) * scale;
							}
					}
			}
			data.push_back(data_t);
		}
		else{
			int hc = (target_h - crop_size) / 2;
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
			}
		}
		(*num)++;
	}
	assert((*Index + *num)==maxIdx);
	*Index = maxIdx;
}

template <typename Dtype>
int readImagesToBlob(Blob<Dtype>& data, vector< vector<Dtype> >& cache, int batch_size, int batchIdx){
	Dtype* top_data = data.mutable_cpu_data();
	int startIdx = batch_size * batchIdx;
	int num = 0;
	for(int i=0;i<batch_size && (i+startIdx)<cache.size();++i){
		vector<Dtype>& cache_t = cache[i+startIdx];
		for(int j=0;j<cache_t.size();++j){
			top_data[i*cache_t.size()+j] = cache_t[j];
		}
		++num;
	}
	return num;
}


string num2str(double i)
{
	std::stringstream ss;
	ss << i;
	return ss.str();
}

void loadSvmModel(string& model, vector<vector<double> >& W, vector<double>& Bias){
	//std::ifstream fin(model.c_str(), std::ios::binary);
	mxArray *pW = NULL;  
	mxArray *pB = NULL;
	MATFile *pmatFile = NULL;
	//int dim;
	//int N;
	pmatFile = matOpen(model.c_str(),"r");
	pW = matGetVariable(pmatFile, "W");
	pB = matGetVariable(pmatFile, "B");
	int dimW=mxGetM(pW);
	int numW=mxGetN(pW);
	
	int numB=mxGetN(pB);
	assert(numW==numB);
	//fin.read((char*)&dim, sizeof(int));
	//fin.read((char*)&N, sizeof(int));
	std::cout << "dim:"<<dimW << " N:" << numW << std::endl;
	/*for(int i=0;i<N;++i){
		double b;
		vector<double> wv;
		for(int j=0;j<dim;++j){
			double Wt;
			fin.read((char*)&Wt, sizeof(double));
			wv.push_back(Wt);
		}
		fin.read((char*)&b, sizeof(double));
		Bias.push_back(b);
		W.push_back(wv);
	}
	fin.close();*/
	double* dataW, *dataB;
	dataW = (double*) mxGetData(pW);
	dataB = (double*) mxGetData(pB);
	for(int i=0;i<numW;++i){
		vector<double> wv;
		for(int j=0;j<dimW;j++){
			wv.push_back(dataW[i*dimW+j]);
		}
		Bias.push_back(dataB[i]);
		W.push_back(wv);
	}
	//for(int i=0;i<dimW;++i){
	//	std::cout << W[0][i] << " ";
	//}
	//std::cout << std::endl;
	matClose(pmatFile);
	mxFree(dataW);
	mxFree(dataB);
}


int readFromFile(string infile, vector<string>& images){
	std::ifstream fin(infile.c_str());
	while(!fin.eof()){
		string ts;
		fin >> ts;
		if(ts == ""){
			return 0;
		}
		images.push_back(ts);
	}
	std::cout << "images:" << images.size() << std::endl;
	return 0;
}
		

template <typename Dtype>
int ex_feature(int argc, char** argv){
	namespace bf=boost::filesystem;
	if (argc < 7){
		LOG(ERROR)<< "Usage: "<<argv[0]<<" pretrained_net_param feature_extraction_proto_file extract_feature_blob_name filelist meanfile savefile";
		return 1;
	}
	//LOG(ERROR) << "Using CPU";
	//Caffe::set_mode(Caffe::CPU);
	 //using gpu
	LOG(ERROR)<< "Using GPU";
	uint device_id = 0;
	LOG(ERROR) << "Using Device_id=" << device_id;
	Caffe::SetDevice(device_id);
	Caffe::set_mode(Caffe::GPU);
	
	Caffe::set_phase(Caffe::TEST);
	string extract_feature_blob_name=argv[3];
	//string svm_model = argv[3];
	string tst_filelist=argv[4];
	string mean_file = argv[5];
	string save_path = argv[6];
	LOG(ERROR) << "load cnn model";
	shared_ptr<Net<Dtype> > feature_extraction_net(new Net<Dtype>(argv[2]));
	feature_extraction_net->CopyTrainedLayersFrom(argv[1]);
	shared_ptr<Blob<Dtype> > feature_blob=feature_extraction_net->blob_by_name(extract_feature_blob_name);

	shared_ptr<Blob<Dtype> > data_blob = feature_extraction_net->blob_by_name("data");
	LOG(ERROR) << "batch size:" << data_blob->num();
	int batch_size = data_blob->num();
	int channels = data_blob->channels();
	int height = data_blob->height();
	int width = data_blob->width();
	CHECK_EQ(height, width);
	int crop_size = height;

	vector<string> images;
	if(!readFromFile(tst_filelist, images)){
		std::cout<< "parse Data Done." << std::endl;
	}else{
		std::cout<<"parse Data failed."<<std::endl;
		return 1;
	}
	Blob<Dtype> data_mean;
	//std::string mean_file = argv[5];
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
	bf::path output_path(save_path);
	Blob<Dtype>* bottom = new Blob<Dtype>(batch_size, 3, crop_size, crop_size);
	vector<Blob<Dtype>*> bottomV;
	bottomV.push_back(bottom);
	int numCaches = ceil(float(images.size()) / batch_size);
	Dtype* feature_blob_data;
	int num=0;
	int startIdx = 0;
	bf::path ftrfile = output_path;
	//ftrfile.replace_extension(".ftr");
	std::ofstream fo(ftrfile.string().c_str());
	bool multivew = true;
	LOG(ERROR) << "cachesize:" << batch_size << " numCaches:" << numCaches;
	clock_t start_processing, end_processing;
	start_processing = clock();
	for(int cacheIdx = 0;cacheIdx < numCaches;cacheIdx++){
		LOG(ERROR) << "processing:" << cacheIdx << "/" << numCaches;
		vector< vector<Dtype> > cache;
		vector< vector<Dtype> > resultcache;
		clock_t start_cache, end_cache;
		start_cache = clock();
		readImagesToCache(cache, images, crop_size, mat_mean, batch_size, &startIdx, &num, scale, multivew);
		end_cache = clock();
		LOG(ERROR) << "readImageToCache:" << (end_cache-start_cache) << "ms";
		start_cache = clock();
		int nBatches = ceil(float(cache.size()) / batch_size);
		//LOG(ERROR) << "nBatches:"<< nBatches << " cache:" << cache.size();
		for(int batchIdx = 0;batchIdx < nBatches; batchIdx++){
			time_t start_epoch, end_epoch;
			start_epoch = time(NULL);
			int n=readImagesToBlob(*bottom, cache, batch_size, batchIdx);
			float loss = 0.0;
			const vector<Blob<Dtype>*>& result =  feature_extraction_net->Forward(bottomV, &loss);
			int feature_num = feature_blob->num();
			int feature_dim = feature_blob->count() / feature_num;
			int start_idx=batch_size*batchIdx;
			for(int s=0;s<n;++s){
				feature_blob_data = feature_blob->mutable_cpu_data()+feature_blob->offset(s);
				vector<Dtype> result_t;
				for(int d=0;d<feature_dim;++d){
					result_t.push_back(feature_blob_data[d]);
				}
				resultcache.push_back(result_t);
			}
			end_epoch = time(NULL);
			LOG(ERROR) << "forward batch(" << batch_size << "):" << difftime(end_epoch,start_epoch) << "s";
			//LOG(ERROR) << "BatchIdx:" << batchIdx << " n:" << n << " resultcache:" << resultcache.size();
		}
		end_cache = clock();
		LOG(ERROR) << "forward cache:" << end_cache-start_cache << "ms";

		int imgIdx = startIdx - num;
		for(int s=0;s<num;++s){
			if(multivew){
				fo << images[imgIdx+s] << " " << 9*2 << " " << resultcache[0].size() << std::endl;
				for(int m=0;m<9*2;++m){
					vector<Dtype>& ftr=resultcache[s*9*2+m];
					for(int d=0;d<ftr.size()-1;++d){
						fo << ftr[d] << " ";
					}
					fo << ftr[ftr.size()-1] << std::endl;
				}
			}else{
				fo << images[imgIdx+s] << " " << 1 << " " << resultcache[0].size() << std::endl;
				vector<Dtype>& ftr=resultcache[s];
				for(int d=0;d<ftr.size()-1;++d){
						fo << ftr[d] << " ";
				}
				fo << ftr[ftr.size()-1] << std::endl;
			}
		}
	}
	fo.close();
	end_processing = clock();
	LOG(ERROR) << "total time:" << float(end_processing-start_processing)/CLOCKS_PER_SEC << "s";
	return 0;
}


int main(int argc, char** argv){
	return ex_feature<float>(argc, argv);
}
