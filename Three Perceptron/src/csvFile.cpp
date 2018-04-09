#include "csvFile.h"

csvFile::csvFile() {
	// 相対パスで指定している. "data"フォルダにcsvファイルを置いておく.
	inputFileName = "data\\exor_data.csv";
	outputFileName1 = "data\\w_DataLog.csv";
	outputFileName2 = "data\\theta_DataLog.csv";
	outputFileName3 = "data\\p_error_DataLog.csv";
}

void csvFile::read() {
	std::ifstream inputFile(inputFileName);
	if (inputFile.fail()) {
		std::cout << "ファイルを開けません.\n";

		// キー入力待ち
		getchar();

		exit(0);
	}
	else std::cout << "ファイルを開きました.\n";

	std::string line;

	getline(inputFile, line);

	// 入力を一行ごとに分割
	while (getline(inputFile, line)) {
		split(line);
	}
	inputFile.close();
}

// 行中での分割
// ?? 教師信号.csvから入力されるデータから, 桁落ちしてしまうがそれはいいのか?
// TODO : データの桁落ちを回避する.
void csvFile::split(std::string& line) {
	std::istringstream lineStream(line);

	int inoutCount = 0;
	Eigen::Matrix<double, 1, INNUM> inData;
	Eigen::Matrix<double, 1, OUTNUM> outData;
	std::string field;
	while (getline(lineStream, field, ',')) {
		switch (inoutCount)
		{
		case 0:
			inoutCount++;
			break;

		case 1:
			inData[0] = std::stod(field);
			inoutCount++;
			break;

		case 2:
			inData[1] = std::stod(field);
			Teach_in.push_back(inData);
			inoutCount++;
			break;

		case 3:
			outData[0] = std::stod(field);
			Teach_out.push_back(outData);
			inoutCount++;
			break;

		default:
			break;
		}
	}
}

void csvFile::write(std::vector<Eigen::MatrixXd> w_in_hid_log,
	std::vector<Eigen::MatrixXd> w_hid_out_log,
	std::vector<Eigen::RowVectorXd> theta_hid_log,
	std::vector<Eigen::RowVectorXd> theta_out_log,
	std::vector<double> p_log,
	std::vector<double> errorTotal_out_log) {

	std::ofstream outputFile1(outputFileName1);
	std::ofstream outputFile2(outputFileName2);
	std::ofstream outputFile3(outputFileName3);

	outputFile1 << "w_in_hid_log[" << INNUM << "][" << HIDNUM << "], w_hid_out_log[" << HIDNUM << "][" << OUTNUM << "]" << std::endl;
	for (int i = 0; i < w_in_hid_log.size(); i++)
		outputFile1 << w_in_hid_log[i](0, 0) << ", " << w_in_hid_log[i](0, 1) << ", " << w_in_hid_log[i](1, 0) << ", " << w_in_hid_log[i](1, 1) << ", " << w_hid_out_log[i](0, 0) << w_hid_out_log[i](1, 0) << std::endl;

	outputFile2 << "theta_hid_log[" << HIDNUM << "], theta_out_log[" << OUTNUM << "]" << std::endl;
	for (int i = 0; i < theta_hid_log.size(); i++)
		outputFile2 << theta_hid_log[i][0] << ", " << theta_hid_log[i][1] << ", " << theta_out_log[i][0] << std::endl;

	outputFile3 << "p_log, errorTotal_out_log" << std::endl;
	for (int i = 0; i < p_log.size(); i++)
		outputFile3 << p_log[i] << ", " << errorTotal_out_log[i] << std::endl;

	outputFile1.close();
	outputFile2.close();
	outputFile3.close();
}