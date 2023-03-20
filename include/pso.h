#pragma once

#ifndef _PSO_H
#define _PSO_H

#include <iostream>
#include <assert.h>
#include <vector>
#include <cfloat>
#include <cmath>

// 返回0-1的随机数
double rand0_1();

// 粒子描述符，包含粒子的位置、速度、适应度、最优位置
struct Particle
{
    double *x;	//粒子所在位置，即为问题候选解
    double *v;	//粒子运动速度
    double pbest_fitness;		//粒子适应度
    double *pbest; //粒子所经历过的最优位置
};

// 定义粒子空间
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
    ParticleSpace particle_space_; //粒子群移动空间
    std::vector<Particle> particles; //粒子群中粒子的集合
    double *gbest_position; //当前粒子群进化中的全局最优解
    double gbest_fitness; //当前粒子群进化中的全局最优解适应度

    // 在给定空间中随机初始化粒子群
    void initParticleSwarm(){
        for (Particle &particle : particles) {
            particle.x = new double[particle_space_.dimension];
            particle.v = new double[particle_space_.dimension];
            particle.pbest = new double[particle_space_.dimension];
            for (int j = 0; j < particle_space_.dimension; j++) {
                particle.x[j] = particle.pbest[j] = 
                    rand0_1() * (particle_space_.upper_bound[j] - particle_space_.lower_bound[j]) + particle_space_.lower_bound[j];
                particle.v[j] = 0;
            }
        }
    };

    // 设置粒子空间
    void setParticleSpace(const ParticleSpace &particle_space) {
        this->particle_space_ = particle_space;
        initParticleSwarm();
    };

    // 获取粒子位置均值与标准差
    void getParticleMeanAndStd(std::vector<double> &mean, std::vector<double> &stddeviation) {
        mean.resize(particle_space_.dimension, 0);
        stddeviation.resize(particle_space_.dimension, 0);
        for (Particle &particle : particles) {
            for (int i = 0; i < particle_space_.dimension; i++) {
                mean[i] += particle.x[i];
            }
        }
        for (int i = 0; i < particle_space_.dimension; i++) {
            mean[i] /= particles.size();
        }
        for (Particle &particle : particles) {
            for (int i = 0; i < particle_space_.dimension; i++) {
                stddeviation[i] += (particle.x[i] - mean[i]) * (particle.x[i] - mean[i]);
            }
        }
        for (int i = 0; i < particle_space_.dimension; i++) {
            stddeviation[i] = sqrt(stddeviation[i] / particles.size());
        }
    };

    // 构造函数
    ParticleSwarm(int particle_number) {
        particles.resize(particle_number);
        gbest_position = new double[particle_space_.dimension];
    };
    ParticleSwarm(int particle_number, const ParticleSpace &particle_space) 
    : particle_space_(particle_space){
        particles.resize(particle_number);
        gbest_position = new double[particle_space_.dimension];
        initParticleSwarm();
    };
    // 析构函数
    ~ParticleSwarm() {
        std::cout << "In ParticleSwarm destructor!\nDeleting particles..." << std::endl;
        for (Particle &particle : particles) {
            delete[] particle.x;
            delete[] particle.v;
        }
        std::cout << "Deleting gbest_position..." << std::endl;
        delete[] gbest_position;
    };
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
    void updateParticleSwarm(ParticleSwarm &particle_swarm) {
        // 各维度最大速度
        double max_v[particle_swarm.particle_space_.dimension];
        for (int i = 0; i < particle_swarm.particle_space_.dimension; i++) {
            max_v[i] = max_velocity_ * (particle_swarm.particle_space_.upper_bound[i] - particle_swarm.particle_space_.lower_bound[i]);
        }
        for (Particle &particle : particle_swarm.particles) {
            // 更新速度，限制最大速度
            double max_v_ratio = 1;
            for (int i = 0; i < particle_swarm.particle_space_.dimension; i++) {
                particle.v[i] = inertia_weight_ * particle.v[i] 
                              + personal_weight_ * rand0_1() * (particle.pbest[i] - particle.x[i]) 
                              + social_weight * rand0_1() * (particle_swarm.gbest_position[i] - particle.x[i]);
                double temp_ratio = fabs(particle.v[i]) / max_v[i];
                max_v_ratio = max_v_ratio > temp_ratio ? max_v_ratio : temp_ratio;
            }
            if (max_v_ratio > 1) {
                for (int i = 0; i < particle_swarm.particle_space_.dimension; i++) {
                    particle.v[i] /= max_v_ratio;
                }
            }
            // 更新位置
            for (int i = 0; i < particle_swarm.particle_space_.dimension; i++) {
                particle.x[i] += particle.v[i];
            }
            // 碰撞反弹，如果粒子超出了粒子空间，则将速度反向并衰减
            for (int i = 0; i < particle_swarm.particle_space_.dimension; i++) {
                if (particle.x[i] >= particle_swarm.particle_space_.upper_bound[i]) {
                    particle.v[i] = -particle.v[i] * rebound_decay_;
                    particle.x[i] = particle_swarm.particle_space_.upper_bound[i] 
                                  - rebound_decay_ * (particle.x[i] - particle_swarm.particle_space_.upper_bound[i]);
                }
                else if (particle.x[i] <= particle_swarm.particle_space_.lower_bound[i]) {
                    particle.v[i] = -particle.v[i] * rebound_decay_;
                    particle.x[i] = particle_swarm.particle_space_.lower_bound[i] 
                                  + rebound_decay_ * (particle_swarm.particle_space_.lower_bound[i] - particle.x[i]);
                }
            }
        }
    };
    // 构造函数
    MoveMode(){};
    MoveMode(double inertia_weight, double personal_weight, double global_weight, double max_velocity, double rebound_decay) 
        : inertia_weight_(inertia_weight), personal_weight_(personal_weight), social_weight(global_weight), 
          max_velocity_(max_velocity), rebound_decay_(rebound_decay) {};
    // 析构函数
    ~MoveMode(){
        std::cout << "In MoveMode destructor!" << std::endl;
    }
};
// 一个用于计算适应度的基类
class BaseProblem
{
public:
    // 一些计算适应度值所需要的参数

    // 计算适应度值的函数
    virtual double CalculateFitness(Particle &particle) {
        std::cout << "BaseProblem::CalculateFitness" << std::endl;
        std::cerr << "Have not implemented BaseProblem::CalculateFitness" << std::endl;
        exit(1);
    };
    // 更新粒子群的局部最优适应度和全局最优适应度
    void updateFitness(ParticleSwarm &particle_swarm){
        for (Particle &particle : particle_swarm.particles) {
            // 更新粒子群的局部最优适应度
            double fitness = CalculateFitness(particle);
            if (fitness > particle.pbest_fitness) {
                particle.pbest_fitness = fitness;
                for (int i = 0; i < particle_swarm.particle_space_.dimension; i++) {
                    particle.pbest[i] = particle.x[i];
                }
            }
            // 更新粒子群的全局最优适应度
            if (fitness > particle_swarm.gbest_fitness) {
                particle_swarm.gbest_fitness = fitness;
                for (int j = 0; j < particle_swarm.particle_space_.dimension; j++) {
                    particle_swarm.gbest_position[j] = particle.x[j];
                }
            }
        }
    };
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
    BaseOptimizer(){};
    BaseOptimizer(BaseProblem* base_problem, MoveMode* move_mode) : base_problem(base_problem), move_mode(move_mode) {};
    // 析构函数
    ~BaseOptimizer(){};
    // 优化函数
    void step(ParticleSwarm &particle_swarm) {
        move_mode->updateParticleSwarm(particle_swarm);
        base_problem->updateFitness(particle_swarm);
    };
};
// 一个用于测试的粒子群算法类
class BasePSO
{
public:
    ParticleSwarm* particle_swarm_; //粒子群
    BaseOptimizer* base_optimizer_; //优化器

    // 冷启动（给粒子初速度，初始化全局最优适应与局部最优）
    void initPSO(){
        // 设置初始速度范围
        double init_speed_range_ = base_optimizer_->move_mode->max_velocity_;
        for (Particle &particle : particle_swarm_->particles) {
            // 初始化粒子速度
            for (int i = 0; i < particle_swarm_->particle_space_.dimension; i++) {
                particle.v[i] = rand0_1() * init_speed_range_ 
                              * (particle_swarm_->particle_space_.upper_bound[i] 
                              -  particle_swarm_->particle_space_.lower_bound[i]);
            }
            // 初始化粒子适应度和个体最优解
            particle.pbest_fitness = base_optimizer_->base_problem->CalculateFitness(particle);
            // 初始化粒子群的全局最优解
            if (particle.pbest_fitness > particle_swarm_->gbest_fitness) {
                particle_swarm_->gbest_fitness = particle.pbest_fitness;
                for (int i = 0; i < particle_swarm_->particle_space_.dimension; i++) {
                    particle_swarm_->gbest_position[i] = particle.x[i];
                }
            }
        }
    };
    // 优化一轮
    void step(){
        base_optimizer_->step(*particle_swarm_);
    };
    // 优化多轮
    void optimize(int times){
        for (int i = 0; i < times; i++) {
            step();
        }
    };
    // 优化直到满足条件
    bool optimizeUntil(double threshold, int times){
        int count = 0;
        while (count < times) {
            step();
            if (particle_swarm_->gbest_fitness > threshold) {
                return true;
            }
            count++;
        }
        return false;
    };
    // 获取最优解
    double* getResult(){
        return particle_swarm_->gbest_position;
    };
    // 获取最优解的适应度
    double getResultFitness(){
        return particle_swarm_->gbest_fitness;
    };
    // 设置粒子群
    void setParticleSwarm(ParticleSwarm* particle_swarm){
        particle_swarm_ = particle_swarm;
    };
    // 设置优化器
    void setOptimizer(BaseOptimizer* base_optimizer){
        base_optimizer_ = base_optimizer;
    };
    // 构造函数
    BasePSO(){};
    BasePSO(ParticleSwarm* particle_swarm, BaseOptimizer* base_optimizer) : 
        particle_swarm_(particle_swarm), base_optimizer_(base_optimizer){};
    // 析构函数
    ~BasePSO(){
        delete particle_swarm_;
        delete base_optimizer_;
    };
};

double rand0_1();

#endif