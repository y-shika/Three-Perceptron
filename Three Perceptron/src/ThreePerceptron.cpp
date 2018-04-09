#include "ThreePerceptron.h"

ThreePerceptron::ThreePerceptron() {
	in.resize(INNUM);
	hid.resize(HIDNUM);
	out.resize(OUTNUM);

	w_in_hid.resize(INNUM, HIDNUM);
	w_hid_out.resize(HIDNUM, OUTNUM);

	theta_hid.resize(HIDNUM);
	theta_out.resize(OUTNUM);

	delta_hid.resize(HIDNUM);
	delta_out.resize(OUTNUM);

	// 乱数生成器 [-1.0, 1.0)
	std::random_device rnd;
	std::mt19937_64 mt(rnd());
	std::uniform_real_distribution<> rand100(-1.0, 1.0);

	for (int i_in = 0; i_in < INNUM; i_in++)
		for (int i_hid = 0; i_hid < HIDNUM; i_hid++)
			w_in_hid(i_in, i_hid) = rand100(mt);

	for (int i_hid = 0; i_hid < HIDNUM; i_hid++)
		for (int i_out = 0; i_out < OUTNUM; i_out++)
			w_hid_out(i_hid, i_out) = rand100(mt);

	for (int i_hid = 0; i_hid < HIDNUM; i_hid++) theta_hid[i_hid] = rand100(mt);

	for (int i_out = 0; i_out < OUTNUM; i_out++) theta_out[i_out] = rand100(mt);

	alpha = 0.05;

	errorTotal_out = 0;

	error_threshold = 0.015;

	p = 0;

	read();
}

ThreePerceptron::~ThreePerceptron() {
	write(w_in_hid_log, w_hid_out_log, theta_hid_log, theta_out_log, p_log, errorTotal_out_log);
}

// データの数だけ学習を回してから, 正答率を計算し, 学習終了か否かを判定するアルゴリズム
void ThreePerceptron::learn() {
	bool flagLearned = false;
	double error_out = 0;
	int learnCount = 0;

	// 初期値で成功する場合を考慮.
	calc_p_errorTotal();
	std::cout << learnCount << ", " << p << ", " << errorTotal_out << std::endl;
	if (p >= 98.5 && errorTotal_out < 4.5) flagLearned = true;

	while (!flagLearned) {
		for (int i = 0; i < Teach_in.size(); i++) {
			forwardPropagation(i);
			error_out = calc_error(i);

			// バックプロパゲーションによる学習
			if (error_out >= error_threshold) {
				while (error_out >= error_threshold) {
					backPropagation(i);
					forwardPropagation(i);
					error_out = calc_error(i);
				}
				learnCount++;
			}
		}

		calc_p_errorTotal();
		std::cout << learnCount << ", " << p << ", " << errorTotal_out << std::endl;

		// ORにして条件を緩めている.
		if (p >= 98.5 || errorTotal_out < 4.5) flagLearned = true;
	}
}

//　学習をするたびに正答率を計算し, 学習終了か否かを判定するアルゴリズム
//void ThreePerceptron::learn() {
//	bool flagLearned = false;
//	double error_out = 0;
//	int learnCount = 0;
//
//	// 初期値で成功する場合を考慮.
//	calc_p_errorTotal();
//	std::cout << learnCount << ", " << p << ", " << errorTotal_out << std::endl;
//	if (p >= 98.5 && errorTotal_out < 4.5) flagLearned = true;
//
//	while (!flagLearned) {
//		for (int i = 0; i < Teach_in.size(); i++) {
//			forwardPropagation(i);
//			error_out = calc_error(i);
//			if (error_out < error_threshold) continue;
//
//			// バックプロパゲーションによる学習
//			while (error_out >= error_threshold) {
//				backPropagation(i);
//				forwardPropagation(i);
//				error_out = calc_error(i);
//			}
//			learnCount++;
//
//			calc_p_errorTotal();
//			std::cout << learnCount << ", " << p << ", " << errorTotal_out << std::endl;
//
//			// ORにして条件を緩めている.
//			if (p >= 98.5 || errorTotal_out < 4.5) {
//				flagLearned = true;
//				break;
//			}
//		}
//	}
//}

// forwardpropagationと同じタイミングで呼ばなければ, outの値が教師データと一致しない.
double ThreePerceptron::calc_error(int i) {
	double error_out = 0;

	error_out = (out - Teach_out[i]).cwiseAbs2().sum();
	return error_out;
}

void ThreePerceptron::calc_p_errorTotal() {
	int correctCount = 0;
	double error_out = 0;
	errorTotal_out = 0;

	for (int i = 0; i < Teach_in.size(); i++) {
		forwardPropagation(i);
		error_out = calc_error(i);
		errorTotal_out = errorTotal_out + error_out;

		if (error_out < error_threshold) correctCount++;
	}

	p = correctCount * 100 / Teach_out.size() / OUTNUM;

	p_log.push_back(p);
	errorTotal_out_log.push_back(errorTotal_out);
}

Eigen::RowVectorXd ThreePerceptron::sigmoid(Eigen::RowVectorXd vec) {
	Eigen::RowVectorXd _vec;

	_vec = 1.0 / (1.0 + exp(-vec.array()));
	return _vec;
}

void ThreePerceptron::forwardPropagation(int i) {
	in = Teach_in[i];

	hid = sigmoid(in * w_in_hid - theta_hid);
	out = sigmoid(hid * w_hid_out - theta_out);
}

void ThreePerceptron::backPropagation(int i) {
	Eigen::Matrix<double, INNUM, HIDNUM> delta_w_in_hid;
	Eigen::Matrix<double, HIDNUM, OUTNUM> delta_w_hid_out;

	Eigen::Matrix<double, 1, HIDNUM> delta_theta_hid;
	Eigen::Matrix<double, 1, OUTNUM> delta_theta_out;

	delta_out = (out - Teach_out[i]).array() * (1 - out.array()).array() * out.array();
	delta_hid = (w_hid_out * delta_out.transpose()).transpose().array() * (1 - hid.array()).array() * hid.array();

	delta_w_in_hid = alpha * (delta_hid.transpose() * in).transpose().array();
	delta_w_hid_out = alpha * (delta_out.transpose() * hid).transpose().array();

	delta_theta_hid = -alpha * delta_hid.array();
	delta_theta_out = -alpha * delta_out.array();

	w_in_hid = w_in_hid - delta_w_in_hid;
	w_hid_out = w_hid_out - delta_w_hid_out;

	theta_hid = theta_hid - delta_theta_hid;
	theta_out = theta_out - delta_theta_out;

	w_in_hid_log.push_back(w_in_hid);
	w_hid_out_log.push_back(w_hid_out);
	theta_hid_log.push_back(theta_hid);
	theta_out_log.push_back(theta_out);
}