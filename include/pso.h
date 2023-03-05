#pragma once

#ifndef _PSO_H
#define _PSO_H


struct Particle
{
    double *x;	//粒子所在位置，即为问题候选解
    double *v;	//粒子运动速度
    double fitness;		//粒子适应度
    double *pbest; //粒子所经历过的最优位置

};

struct ParticleSwarm
{
    Particle *particles; //一群粒子构成的集合
    double *gbest; //当前粒子群进化中的全局最优解
    double gfitness; //当前粒子群进化中的全局最优解适应度
};

class PsoAlgorithm
{
public:
    int time_to_end; //允许几次循环无更好结果
    double result_threshold;
    int dimension; //粒子维度
    int particle_number; //粒子数量
    double w; //惯性参数
    double cp; //个体参数
    double cg; //全局参数
    double wall; //碰撞反弹系数
    double maxspeedratio;
    double *positionMin; //粒子位置的最小界
    double *positionMax; //粒子位置的最大界
    ParticleSwarm particle_swarm;
    double (*fitnessFunctionPtr)(Particle&); //粒子适应度函数
    double result_fitness;
    double* result_position; //粒子所在位置，即为问题候选解

    PsoAlgorithm();
    PsoAlgorithm(int _dimension, int _particlenumber, double (*fitnessFunctionPtr)(Particle&), 
                 double _result_threshold = 0.8, double _w = 0.9, double _cp = 1.6, double _cg = 2, double _wall = 0.8, int _time_to_end = 5);
    PsoAlgorithm(int _dimension, int _particlenumber,
                 double _result_threshold = 0.8, double _w = 0.9, double _cp = 1.6, double _cg = 2, double _wall = 0.8, int _time_to_end = 5);
    ~PsoAlgorithm();

    void printParticle(Particle* particle);
    virtual double fitnessFunction(Particle& particle);
    void setSearchScope(double *_positionMin, double *_positionMax, double maxspeedratio = 0.15); //设置寻找范围
    void initial(); //粒子初始化
    void update(); // 更新x,v
    void refreshFitness(); // 更新fitness,gbest,pbest
    void search_once(); // 优化一轮
    void run(int round); //
};

#endif