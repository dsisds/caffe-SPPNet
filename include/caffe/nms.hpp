#ifndef NMS_H_
#define NMS_H_
#include <string>
#include <vector>

namespace caffe{
class box{
public:
	float _score;
	std::vector<int> _window;
	int _label;
public:
	box(float score, std::vector<int> window, int label){
		_score = score;
		for(int i=0;i<4;i++)
			_window.push_back(window[i]);
		_label=label;
	}
	box();
	float getScore(){ return _score;}
	std::vector<int> getWindow(){ return _window;}
	void operator =(box a){
		_score = a.getScore();
		std::vector<int> win = a.getWindow();
		_window.clear();
		for (int i=0;i<4;i++)
			_window.push_back(win[i]);
		_label=a._label;
	}
	void setScore(float score){
		_score = score;
	}
	void setWindow(std::vector<int> window){
		_window.clear();
		for(int i=0;i<4;++i)
			_window.push_back(window[i]);
	}
	bool operator< (const box& a)const{
		return this->_score < a._score;
	}
};

int nms(std::vector<box>& boxes, std::vector<box>& pick, float overlap);
float compute_overlap(box& b1, box& b2);
int nms(std::vector<std::vector<int> >& boxes, std::vector<double>& scores, std::vector<std::pair<std::vector<int>, double> >& pick, float overlap);
}
#endif
