#include "ParticleSystem.h"
#include <cstdlib>   // rand()
#include <cmath>
static float randFloat(float a, float b) {
    return a + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (b - a)));
}
ParticleSystem::ParticleSystem(int count) {
    particles.resize(count);
    for (auto& p : particles) {
        respawnParticle(p);
    }
}
void ParticleSystem::update(float deltaTime) {
    for (auto& p : particles) {
        p.life -= deltaTime;
        if (p.life > 0.0f) {
            // Simple Euler integration
            p.position += p.velocity * deltaTime;
            // Slight upward drift over time
            p.velocity.y += 0.2f * deltaTime;
            // Optional: shrink as they die
            p.size = glm::max(0.0f, p.size - deltaTime * 0.3f);
        }
        else {
            respawnParticle(p);
        }
    }
}
void ParticleSystem::respawnParticle(Particle& particle) {
    // Start near origin with slight random offset
    particle.position = glm::vec3(
        randFloat(-0.2f, 0.2f),
        randFloat(-0.1f, 0.1f),
        randFloat(-0.2f, 0.2f)
    );
    // Randomized initial velocity
    particle.velocity = glm::vec3(
        randFloat(-0.3f, 0.3f),
        randFloat(0.4f, 1.2f),
        randFloat(-0.3f, 0.3f)
    );
    // Reset life and size
    particle.life = randFloat(0.7f, 1.4f);
    particle.size = randFloat(0.3f, 0.6f);
}