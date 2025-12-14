#pragma once
#include <vector>
#include <glm/glm.hpp>

struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    float life;
    float size;

    Particle() : position(0.0f), velocity(0.0f), life(1.0f), size(0.5f) {}
};

class ParticleSystem {
public:
    ParticleSystem(int count);

    void update(float deltaTime);
    const std::vector<Particle>& getParticles() const { return particles; }

private:
    void respawnParticle(Particle& particle);

    std::vector<Particle> particles;
};