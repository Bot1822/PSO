#pragma once

#ifndef _PSO_H
#define _PSO_H

#include <vector>
#include <cmath>

// 返回0-1的随机数
inline double rand0_1() {return ((1.0*rand())/RAND_MAX);}

// 粒子描述符，包含粒子的位置、速度、适应度、最优位置
typedef struct Particle Particle;
struct Particle
{
    double *x;	//粒子所在位置，即为问题候选解
    double *v;	//粒子运动速度
    double pbest_fitness;		//粒子适应度
    double *pbest; //粒子所经历过的最优位置
};

// 定义粒子空间
typedef struct ParticleSpace ParticleSpace;
struct ParticleSpace
{
    int dimension; //粒子维度
    std::vector<double> lower_bound; //粒子的最小界
    std::vector<double> upper_bound; //粒子的最大界

    ParticleSpace &operator=(const ParticleSpace &particle_space) {
        this->dimension = particle_space.dimension;
        this->lower_bound = particle_space.lower_bound;
        this->upper_bound = particle_space.upper_bound;
        return *this;
    };
};

// 粒子群描述符，包含粒子群中粒子的个数、粒子维度、粒子群中粒子的集合、粒子群中粒子的全局最优解、粒子群中粒子的全局最优解适应度
class ParticleSwarm
{
 public:
    ParticleSpace* particle_space_; //粒子群移动空间
    std::vector<Particle> particles; //粒子群中粒子的集合
    double *gbest_position; //当前粒子群进化中的全局最优解
    double gbest_fitness; //当前粒子群进化中的全局最优解适应度

    // 在给定空间中随机初始化粒子群
    void initParticleSwarm();
    // 设置粒子空间
    void setParticleSpace(const ParticleSpace &particle_space);
    // 获取粒子位置均值与标准差
    void getParticleMeanAndStd(std::vector<double> &mean, std::vector<double> &stddeviation);

    ParticleSwarm(int particle_number);
    ParticleSwarm(int particle_number, const ParticleSpace &particle_space);
    ~ParticleSwarm();
};

// 粒子群位置和速度更新模式
class MoveMode
{
public:
    // 惯性参数
    double inertia_weight_ = 0.9;
    // 个体参数
    double personal_weight_ = 1.6;
    // 群体参数
    double social_weight = 2;
    // 最大速度系数
    double max_velocity_ = 0.3;
    // 碰撞反弹衰减系数
    double rebound_decay_ = 0.8;
    // 更新粒子群的位置和速度
    void updateParticleSwarm(ParticleSwarm &particle_swarm);

    MoveMode();
    MoveMode(double inertia_weight, double personal_weight, double global_weight, double max_velocity, double rebound_decay);
    ~MoveMode();
};
// 一个用于计算适应度的基类
class BaseProblem
{
public:
    // 计算适应度值的函数
    virtual double calculateFitness(Particle &particle) = 0;
    // 更新粒子群的局部最优适应度和全局最优适应度
    void updateFitness(ParticleSwarm &particle_swarm);
    // 构造函数
    BaseProblem(){};
    // 析构函数
    ~BaseProblem(){};
};

// 基类优化器
class BaseOptimizer
{
 public:
    BaseProblem* base_problem;
    MoveMode* move_mode;

    // 构造函数
    BaseOptimizer();
    BaseOptimizer(BaseProblem* base_problem, MoveMode* move_mode);
    // 析构函数
    ~BaseOptimizer();
    // 优化函数
    void step(ParticleSwarm &particle_swarm);
};
// 一个用于测试的粒子群算法类
class BasePSO
{
public:
    ParticleSwarm* particle_swarm_; //粒子群
    BaseOptimizer* base_optimizer_; //优化器

    // 冷启动（给粒子初速度，初始化全局最优适应与局部最优）
    void initPSO();
    // 优化一轮
    void step();
    // 优化多轮
    void optimize(int times);
    // 优化直到满足条件
    bool optimizeUntil(double threshold, int times);
    // 获取最优解
    double* getResult();
    // 获取最优解的适应度
    double getResultFitness();
    // 设置粒子群
    void setParticleSwarm(ParticleSwarm* particle_swarm);
    // 设置优化器
    void setOptimizer(BaseOptimizer* base_optimizer);
    // 构造函数
    BasePSO();
    BasePSO(ParticleSwarm* particle_swarm, BaseOptimizer* base_optimizer);
    // 析构函数
    ~BasePSO();
};

#endif