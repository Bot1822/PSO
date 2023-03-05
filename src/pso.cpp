#include <pso.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

double rand0_1() {return ((1.0*rand())/RAND_MAX);}

PsoAlgorithm::PsoAlgorithm() {

    }

PsoAlgorithm::PsoAlgorithm(int _dimension, int _particlenumber, double (*_fitnessFunction)(Particle&), 
                           double _result_threshold, double _w, double _cp, double _cg, double _wall, int _time_to_end)
    : dimension(_dimension), particle_number(_particlenumber), fitnessFunctionPtr(_fitnessFunction), 
      result_threshold(_result_threshold), w(_w), cp(_cp), cg(_cg), wall(_wall), time_to_end(_time_to_end) {}
      
PsoAlgorithm::PsoAlgorithm(int _dimension, int _particlenumber, double _result_threshold, double _w, double _cp, double _cg, double _wall, int _time_to_end)
    : dimension(_dimension), particle_number(_particlenumber),
      result_threshold(_result_threshold), w(_w), cp(_cp), cg(_cg), wall(_wall), time_to_end(_time_to_end) {}

PsoAlgorithm::~PsoAlgorithm() {
    std::cout << "PsoAlgorithm析构开始";
    delete positionMax;
    delete positionMin;
    delete result_position;
    delete particle_swarm.gbest;

    for (int i = 0; i < particle_number; ++i) {
        delete particle_swarm.particles[i].x;
        delete particle_swarm.particles[i].v;
        delete particle_swarm.particles[i].pbest;
    }
    delete particle_swarm.particles;
    std::cout << "PsoAlgorithm析构结束";
}

void PsoAlgorithm::setSearchScope(double *_positionMin, double *_positionMax, double _maxspeedratio) {
    positionMin = new double[dimension];
    positionMax = new double[dimension];
    maxspeedratio = _maxspeedratio;
    for(int i = 0; i < dimension; ++i) {
        positionMax[i] = _positionMax[i];
        positionMin[i] = _positionMin[i];
    }
}

void PsoAlgorithm::initial() {
    
    srand(time(NULL));
    result_position = new double[dimension];

    // 开辟并初始化一块粒子群空间
    particle_swarm.particles = new Particle[particle_number];
    particle_swarm.gbest = new double[dimension];
    particle_swarm.gfitness = 0;
    for(int i = 0; i < particle_number; ++i) {
        Particle &particle = particle_swarm.particles[i];

        particle.x = new double[dimension];
        particle.v = new double[dimension];
        particle.pbest = new double[dimension];
        // 位置与速度赋初值
        for(int j = 0; j < dimension; ++j) {
            particle.x[j] = particle.pbest[j] = 
                rand0_1() * (positionMax[j] - positionMin[j]) + positionMin[j];
            particle.v[j] = 
                rand0_1() * maxspeedratio * (positionMax[j] - positionMin[j]);
        }
        particle.fitness = fitnessFunction(particle);
        if(particle.fitness > particle_swarm.gfitness) {
            particle_swarm.gfitness = particle.fitness;
            for(int j = 0; j < dimension; ++j) 
                particle_swarm.gbest[j] = particle.pbest[j];
        }
    }
    
    result_fitness = particle_swarm.gfitness;
}

void PsoAlgorithm::update() {
    for(int i = 0; i < particle_number; ++i) {
        Particle &particle = particle_swarm.particles[i];
        for(int j = 0; j < dimension; ++j) {
            double temp_v, temp_x;
            // 速度更新
            temp_v = w * particle.v[j]
                          + cp * rand0_1() * (particle.pbest[j] - particle.x[j])
                          + cg * rand0_1() * (particle_swarm.gbest[j] - particle.x[j]);
            // 位置更新
            temp_x = particle.x[j] + particle.v[j];
            // 设置撞墙反弹
            // while(particle.x[j] >= positionMax[j] || particle.x[j] <= positionMin[j]) {
            //     if(particle.x[j] >= positionMax[j]) {
            //         particle.x[j] = (2 * positionMax[j] - particle.x[j]) * wall;
                    
            //     }
            //     else {
            //         particle.x[j] = (2 * positionMin[j] - particle.x[j]) * wall;
                    
            //     }
            //     particle.v[j] = -particle.v[j] * wall;
            // }
            while (temp_x >= positionMax[j] || temp_x <= positionMin[j])
            {
                if (temp_x >= positionMax[j])
                {
                    temp_x = positionMax[j] - (temp_x - positionMax[j]) * wall;
                }
                else
                {
                    temp_x = positionMin[j] + (positionMin[j] - temp_x) * wall;
                }
                temp_v = -temp_v * wall;
            }
            
            particle.v[j] = temp_v;
            particle.x[j] = temp_x;
        }
    }
    
}

void PsoAlgorithm::refreshFitness() {
    for(int i = 0; i < particle_number; ++i) {
        Particle &particle = particle_swarm.particles[i];
        double _fitness = fitnessFunction(particle);
        if(_fitness > particle.fitness) {
            particle.fitness = _fitness;
            for(int j = 0; j < dimension; ++j) particle.pbest[j] = particle.x[j];
        }
        if(_fitness > particle_swarm.gfitness) {
            particle_swarm.gfitness = particle.fitness;
            for(int j = 0; j < dimension; ++j) 
                particle_swarm.gbest[j] = particle.pbest[j];
        }
    }
}

void PsoAlgorithm::search_once()
{
    update();
    refreshFitness();
    result_fitness = particle_swarm.gfitness;
    for(int j = 0; j < dimension; ++j) result_position[j] = particle_swarm.gbest[j];
}

void PsoAlgorithm::run(int round) {
    int times = time_to_end;
    for(int i = 0; i < round; ++i) {
        times = time_to_end;
        update();
        // 适应值状态更新
        refreshFitness();
        if(result_fitness > result_threshold && particle_swarm.gfitness == result_fitness) {
            if(times) --times;
            else break;
        }
        result_fitness = particle_swarm.gfitness;
    }

    for(int j = 0; j < dimension; ++j) result_position[j] = particle_swarm.gbest[j];
}

void PsoAlgorithm::printParticle(Particle *particle)
{
    std::cout << "particle:" << std::endl;
    std::cout << "x: ";
    for(int i = 0; i < dimension; ++i) std::cout << particle->x[i] << " ";
    std::cout << std::endl;
    std::cout << "v: ";
    for(int i = 0; i < dimension; ++i) std::cout << particle->v[i] << " ";
    std::cout << std::endl;
    std::cout << "pfitness: " << particle->fitness << std::endl;
}

double PsoAlgorithm::fitnessFunction(Particle &particle)
{
    return fitnessFunctionPtr(particle);
}