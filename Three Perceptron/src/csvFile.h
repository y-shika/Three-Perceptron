#ifndef __CSVFILE_H_INCLUDED__
#define __CSVFILE_H_INCLUDED__

#include <Eigen/Eigen>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define INNUM 2
#define HIDNUM 2
#define OUTNUM 1

class csvFile
{
	// 継承クラスでprivate変数使いたいなら, private -> protectedにする.
private:
	std::string inputFileName;
	std::string outputFileName1;
	std::string outputFileName2;
	std::string outputFileName3;
	void split(std::string& line);

public:
	// RowVectorと同じことだが, std::vectorに入れ子にした場合の初期化(大きさの指定)の方法が不明だったため, マクロを用いてヘッダで指定した.
	std::vector<Eigen::Matrix<double, 1, INNUM>> Teach_in;
	std::vector<Eigen::Matrix<double, 1, OUTNUM>> Teach_out;

	csvFile();
	//~csvFile();

	void read();
	void write(std::vector<Eigen::MatrixXd> w_in_hid_log,
		std::vector<Eigen::MatrixXd> w_hid_out_log,
		std::vector<Eigen::RowVectorXd> theta_hid_log,
		std::vector<Eigen::RowVectorXd> theta_out_log,
		std::vector<double> p_log,
		std::vector<double> errorTotal_out_log);
};

#endif