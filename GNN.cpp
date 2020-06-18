#include "GNN.h"


float GNN::sigmoid(float num)
{
    return 1 / (1 + exp(-num));
    //return max(0, num);
}
float GNN::d_sigmoid(float num)
{
    return sigmoid(num) * (1 - sigmoid(num));
    /*
    if (num < 0)
    {
        return 0;
    }
    else
    {
        return 1;
    }
    */
}


auto sig = std::ptr_fun(GNN::sigmoid);
auto d_sig = std::ptr_fun(GNN::d_sigmoid);


Eigen::Matrix<float, 1, 3> GNN::feed(Solution sol, Eigen::MatrixXf input)
{
    //matrix multiplication
    Eigen::MatrixXf z1 = (input * sol.w1);
    Eigen::MatrixXf a1 = z1.unaryExpr(sig);

    Eigen::MatrixXf z2 = (a1 * sol.w2);
    Eigen::MatrixXf a2 = z2.unaryExpr(sig);

    Eigen::MatrixXf z3 = (a2 * sol.w3);
    Eigen::MatrixXf a3 = z3.unaryExpr(sig);

    Eigen::MatrixXf z4 = (a3 * sol.w4);
    Eigen::MatrixXf a4 = z4.unaryExpr(sig);

    Eigen::MatrixXf z5 = (a4 * sol.w5);
    Eigen::Matrix<float, 1, 3> a5 = z5.unaryExpr(sig);

    return a5;
}

float GNN::evalFitness(Eigen::VectorXf pred, Eigen::VectorXf y)
{
    pred = pred.cwiseAbs();
    y = y.cwiseAbs();
    Eigen::VectorXf errors = pred - y;
    errors = errors.cwiseAbs();
    return errors.sum() / y.size();
}

Eigen::MatrixXf GNN::crossover(Eigen::MatrixXf par1, Eigen::MatrixXf par2)
{
    Eigen::MatrixXf child(par1.rows(), par1.cols());
    std::uniform_int_distribution<> ranPoint(1, par1.size() - 1);
    std::random_device device;
    std::mt19937 seedval(device());

    int midPoint = ranPoint(seedval);

    for (int i = 0; i < midPoint; i++)
    {
        child(i) = par1(i);
    }

    for (int i = midPoint; i < par1.size(); i++)
    {
        child(i) = par2(i);
    }

    return child;
}

int GNN::roulette(std::vector<Solution> solutions)
{
    //summing all elements
    int sum = 0;
    for (int i = 0; i < solutions.size(); i++)
    {
        //inversing numbers, the smaller they are the larger they become
        sum += solutions[i].fitness;
    }

    //getting random number
    std::uniform_int_distribution<> ranParent(0, sum);
    std::random_device device;
    std::mt19937 seedval(device());

    sum = 0;
    int ranNum = ranParent(seedval);

    for (int i = 0; i < solutions.size(); i++)
    {
        sum += solutions[i].fitness;

        if (sum >= ranNum)
        {
            return i;
        }
    }

    return -1;
}
