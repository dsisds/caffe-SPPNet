#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <opencv2/opencv.hpp>

#include "caffe/nms.hpp"

using std::vector;
using std::pair;

namespace caffe {

int nms(vector<box>& boxes, vector<box>& pick, float overlap){
	if (boxes.size()<1){
		return 0;
	}
	sort(boxes.begin(), boxes.end());

	while(boxes.size()>0){
		int last = boxes.size()-1;
		pick.push_back(boxes[last]);
		boxes.pop_back();
		box& Bbox=pick[pick.size()-1];

		vector<box>::iterator iter=boxes.begin();
		while(iter != boxes.end()){
			float op=compute_overlap(Bbox, *iter);
			if(op>overlap){
				iter = boxes.erase(iter);
			}else{
				iter++;
			}
		}
	}
	return 0;
}

float compute_overlap(box& b1, box& b2){
	vector<int> win1=b1.getWindow();
	vector<int> win2=b2.getWindow();
	int xx1=win1[0]>win2[0]?win1[0]:win2[0];
	int yy1=win1[1]>win2[1]?win1[1]:win2[1];
	int xx2=win1[2]<win2[2]?win1[2]:win2[2];
	int yy2=win1[3]<win2[3]?win1[3]:win2[3];
	
	int w=(xx2-xx1+1)>0?(xx2-xx1+1):0;
	int h=(yy2-yy1+1)>0?(yy2-yy1+1):0;

	int inter=w*h;
	int area1 = (win1[2]-win1[0]+1)*(win1[3]-win1[1]+1);
	int area2 = (win2[2]-win2[0]+1)*(win2[3]-win2[1]+1);
	float o = float(inter)/float(area1+area2-inter);

	return o;
}

int nms(vector<vector<int> >& boxes, vector<double>& scores, vector<pair<vector<int>, double> >& pick, float overlap){
	vector<box> boxes_t;
	for(int i=0;i<boxes.size();++i){
		double score = scores[i];
		box box_t(score, boxes[i], 0);
		boxes_t.push_back(box_t);
	}
	vector<box> pick_t;
	int status = nms(boxes_t, pick_t, overlap);
	for(int i=0;i<pick_t.size();++i){
		vector<int>& win = pick_t[i]._window;
		double score=pick_t[i]._score;
		pick.push_back(make_pair(win, score));
	}
	for (int i=0;i<boxes_t.size();++i)
		delete &boxes_t[i];
	boxes_t.clear();
	return 0;
}
}