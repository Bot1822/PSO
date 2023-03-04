#include"pso.h"
#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>

using namespace std;
using namespace cv;

double fitnessfunction(Particle& p){
    double fitness;
    double t = p.x[0];

    fitness = 1-p.x[0]*p.x[0];
    return fitness;
}

int main() {
    PsoAlgorithm pso(1, 4, fitnessfunction, 0.99999999999);
    double max[1], min[1];
    max[0] = 1;
    min[0] = -1;

    pso.setSearchScope(min, max);
    pso.initial();

    for (size_t i = 0; i < 50; ++i) pso.search_once();
    cout << pso.result_position[0] << endl;

    pso.run(1000);

    cout << pso.result_position[0] << endl;

    return 0;
}

