#include <iostream>
#include <dense>
#include <Windows.h>
#include <vector>
#include <random>
#include <time.h>

namespace GNN
{
    float sigmoid(float num);

    float d_sigmoid(float num);


    struct Solution
    {
        Eigen::Matrix<float, 5, 32> w1;
        Eigen::Matrix<float, 32, 64> w2;
        Eigen::Matrix<float, 64, 32> w3;
        Eigen::Matrix<float, 32, 16> w4;
        Eigen::Matrix<float, 16, 3> w5;

        float fitness = 0;
    };

    Eigen::Matrix<float, 1, 3> feed(Solution sol, Eigen::MatrixXf input);

    float evalFitness(Eigen::VectorXf pred, Eigen::VectorXf y);

    Eigen::MatrixXf crossover(Eigen::MatrixXf par1, Eigen::MatrixXf par2);

    int roulette(std::vector<Solution> solutions);
}