#ifndef __THREEPERCEPTRON_H_INCLUDED__
#define __THREEPERCEPTRON_H_INCLUDED__

#include "csvFile.h"

#include <random>

class ThreePerceptron : public csvFile
{
private:
	// 素子と重みの行列計算において, Eigenデフォルトの列ベクトルのままだと, 『素子 * 重み([2. 1] * [2, 2])』の形で計算できないため, 行ベクトルを用いる.
	// (転置を用いて実装もできるが, コードの可読性が保たれない)
	Eigen::RowVectorXd in;
	Eigen::RowVectorXd hid;
	Eigen::RowVectorXd out;

	Eigen::MatrixXd w_in_hid;
	Eigen::MatrixXd w_hid_out;

	// Eigenはpush_backが出来ない弱みがあるため, もしpush_backの必要性が出たなら, std::vectorによって入力した後に, Eigenに渡すようにする.
	Eigen::RowVectorXd theta_hid;
	Eigen::RowVectorXd theta_out;

	double alpha;

	double errorTotal_out;

	double error_threshold;

	double p;

	Eigen::RowVectorXd sigmoid(Eigen::RowVectorXd vec);
	void forwardPropagation(int i);
	void backPropagation(int i);

	double calc_error(int i);

	void calc_p_errorTotal();

public:
	std::vector<Eigen::MatrixXd> w_in_hid_log;
	std::vector<Eigen::MatrixXd> w_hid_out_log;

	std::vector<Eigen::RowVectorXd> theta_hid_log;
	std::vector<Eigen::RowVectorXd> theta_out_log;

	std::vector<double> p_log;
	std::vector<double> errorTotal_out_log;

	ThreePerceptron();
	~ThreePerceptron();

	void learn();
};

#endif
