#include "pso.h"
#include <iostream>
#include <cstdlib>
#include <cassert>

void ParticleSwarm::initParticleSwarm() {
    for (Particle &particle : particles) {
        particle.x = new double[particle_space_->dimension];
        particle.v = new double[particle_space_->dimension];
        particle.pbest = new double[particle_space_->dimension];
        for (int j = 0; j < particle_space_->dimension; j++) {
            particle.x[j] = particle.pbest[j] = 
                rand0_1() * (particle_space_->upper_bound[j] - particle_space_->lower_bound[j]) + particle_space_->lower_bound[j];
            particle.v[j] = 0;
        }
    }
}

void ParticleSwarm::setParticleSpace(const ParticleSpace &particle_space) {
    this->particle_space_ = new ParticleSpace(particle_space);
    initParticleSwarm();
}

void ParticleSwarm::getParticleMeanAndStd(std::vector<double> &mean, std::vector<double> &stddeviation) {
    mean.resize(particle_space_->dimension, 0);
    stddeviation.resize(particle_space_->dimension, 0);
    for (Particle &particle : particles) {
        for (int i = 0; i < particle_space_->dimension; i++) {
            mean[i] += particle.x[i];
        }
    }
    for (int i = 0; i < particle_space_->dimension; i++) {
        mean[i] /= particles.size();
    }
    for (Particle &particle : particles) {
        for (int i = 0; i < particle_space_->dimension; i++) {
            stddeviation[i] += (particle.x[i] - mean[i]) * (particle.x[i] - mean[i]);
        }
    }
    for (int i = 0; i < particle_space_->dimension; i++) {
        stddeviation[i] = sqrt(stddeviation[i] / particles.size());
    }
}

ParticleSwarm::ParticleSwarm(int particle_number) {
    particles.resize(particle_number);
    gbest_position = new double[particle_space_->dimension];
}
ParticleSwarm::ParticleSwarm(int particle_number, const ParticleSpace &particle_space) 
: particle_space_(new ParticleSpace(particle_space)){
    particles.resize(particle_number);
    gbest_position = new double[particle_space_->dimension];
    initParticleSwarm();
}
ParticleSwarm::~ParticleSwarm() {
    std::cout << "In ParticleSwarm destructor!\nDeleting particles..." << std::endl;
    for (Particle &particle : particles) {
        delete[] particle.x;
        delete[] particle.v;
    }
    std::cout << "Deleting gbest_position..." << std::endl;
    delete[] gbest_position;
    std::cout << "Deleting particle_space_..." << std::endl;
    delete particle_space_;
}


void MoveMode::updateParticleSwarm(ParticleSwarm &particle_swarm) {
    // 各维度最大速度
    double max_v[particle_swarm.particle_space_->dimension];
    for (int i = 0; i < particle_swarm.particle_space_->dimension; i++) {
        max_v[i] = max_velocity_ * (particle_swarm.particle_space_->upper_bound[i] - particle_swarm.particle_space_->lower_bound[i]);
    }
    for (Particle &particle : particle_swarm.particles) {
        // 更新速度，限制最大速度
        double max_v_ratio = 1;
        for (int i = 0; i < particle_swarm.particle_space_->dimension; i++) {
            particle.v[i] = inertia_weight_ * particle.v[i] 
                            + personal_weight_ * rand0_1() * (particle.pbest[i] - particle.x[i]) 
                            + social_weight * rand0_1() * (particle_swarm.gbest_position[i] - particle.x[i]);
            double temp_ratio = fabs(particle.v[i]) / max_v[i];
            max_v_ratio = max_v_ratio > temp_ratio ? max_v_ratio : temp_ratio;
        }
        if (max_v_ratio > 1) {
            for (int i = 0; i < particle_swarm.particle_space_->dimension; i++) {
                particle.v[i] /= max_v_ratio;
            }
        }
        // 更新位置
        for (int i = 0; i < particle_swarm.particle_space_->dimension; i++) {
            particle.x[i] += particle.v[i];
        }
        // 碰撞反弹，如果粒子超出了粒子空间，则将速度反向并衰减
        for (int i = 0; i < particle_swarm.particle_space_->dimension; i++) {
            if (particle.x[i] >= particle_swarm.particle_space_->upper_bound[i]) {
                particle.v[i] = -particle.v[i] * rebound_decay_;
                particle.x[i] = particle_swarm.particle_space_->upper_bound[i] 
                                - rebound_decay_ * (particle.x[i] - particle_swarm.particle_space_->upper_bound[i]);
            }
            else if (particle.x[i] <= particle_swarm.particle_space_->lower_bound[i]) {
                particle.v[i] = -particle.v[i] * rebound_decay_;
                particle.x[i] = particle_swarm.particle_space_->lower_bound[i] 
                                + rebound_decay_ * (particle_swarm.particle_space_->lower_bound[i] - particle.x[i]);
            }
        }
    }
}

MoveMode::MoveMode(){}
MoveMode::MoveMode(double inertia_weight, double personal_weight, double global_weight, double max_velocity, double rebound_decay) 
    : inertia_weight_(inertia_weight), personal_weight_(personal_weight), social_weight(global_weight), 
        max_velocity_(max_velocity), rebound_decay_(rebound_decay) {}
MoveMode::~MoveMode(){
    std::cout << "In MoveMode destructor!" << std::endl;
}



void BaseProblem::updateFitness(ParticleSwarm &particle_swarm){
    for (Particle &particle : particle_swarm.particles) {
        // 更新粒子群的局部最优适应度
        double fitness = calculateFitness(particle);
        if (fitness > particle.pbest_fitness) {
            particle.pbest_fitness = fitness;
            for (int i = 0; i < particle_swarm.particle_space_->dimension; i++) {
                particle.pbest[i] = particle.x[i];
            }
        }
        // 更新粒子群的全局最优适应度
        if (fitness > particle_swarm.gbest_fitness) {
            particle_swarm.gbest_fitness = fitness;
            for (int j = 0; j < particle_swarm.particle_space_->dimension; j++) {
                particle_swarm.gbest_position[j] = particle.x[j];
            }
        }
    }
}

BaseOptimizer::BaseOptimizer(){}
BaseOptimizer::BaseOptimizer(BaseProblem *base_problem, MoveMode *move_mode) : base_problem(base_problem), move_mode(move_mode){};
BaseOptimizer::~BaseOptimizer(){
    std::cout << "In BaseOptimizer destructor!" << std::endl;
    delete base_problem;
    delete move_mode;
}

void BaseOptimizer::step(ParticleSwarm &particle_swarm) {
    assert(base_problem != nullptr && move_mode != nullptr);
    move_mode->updateParticleSwarm(particle_swarm);
    base_problem->updateFitness(particle_swarm);
}


void BasePSO::initPSO(){
    // 设置初始速度范围
    double init_speed_range_ = base_optimizer_->move_mode->max_velocity_;
    for (Particle &particle : particle_swarm_->particles) {
        // 初始化粒子速度
        for (int i = 0; i < particle_swarm_->particle_space_->dimension; i++) {
            particle.v[i] = rand0_1() * init_speed_range_ 
                            * (particle_swarm_->particle_space_->upper_bound[i] 
                            -  particle_swarm_->particle_space_->lower_bound[i]);
        }
        // 初始化粒子适应度和个体最优解
        particle.pbest_fitness = base_optimizer_->base_problem->calculateFitness(particle);
        // 初始化粒子群的全局最优解
        if (particle.pbest_fitness > particle_swarm_->gbest_fitness) {
            particle_swarm_->gbest_fitness = particle.pbest_fitness;
            for (int i = 0; i < particle_swarm_->particle_space_->dimension; i++) {
                particle_swarm_->gbest_position[i] = particle.x[i];
            }
        }
    }
}
void BasePSO::step(){
    base_optimizer_->step(*particle_swarm_);
}
void BasePSO::optimize(int times){
    for (int i = 0; i < times; i++) {
        step();
    }
}
bool BasePSO::optimizeUntil(double threshold, int times){
    int count = 0;
    while (count < times) {
        step();
        if (particle_swarm_->gbest_fitness > threshold) {
            return true;
        }
        count++;
    }
    return false;
}
std::vector<double> BasePSO::getResult(){
    std::vector<double> result;
    for (int i = 0; i < particle_swarm_->particle_space_->dimension; i++) {
        result.push_back(particle_swarm_->gbest_position[i]);
    }
    return result;
}
double BasePSO::getResultFitness(){
    return particle_swarm_->gbest_fitness;
}
void BasePSO::setParticleSwarm(ParticleSwarm* particle_swarm){
    particle_swarm_ = particle_swarm;
}
void BasePSO::setOptimizer(BaseOptimizer* base_optimizer){
    base_optimizer_ = base_optimizer;
}

BasePSO::BasePSO(){}
BasePSO::BasePSO(ParticleSwarm* particle_swarm, BaseOptimizer* base_optimizer) : 
    particle_swarm_(particle_swarm), base_optimizer_(base_optimizer){}
BasePSO::~BasePSO(){
    std::cout << "In BasePSO destructor!" << std::endl;
    delete particle_swarm_;
    delete base_optimizer_;
}