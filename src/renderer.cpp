#include "Renderer.h"
#include <iostream>
#include <random>
#include <cfloat>
#include <glad/glad.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <map>

Renderer::Renderer()
    : pointVAO(0), pointVBO(0),
    backgroundVAO(0), backgroundVBO(0),
    particleVAO(0), particleVBO(0),
    cloudVAO(0), cloudVBO(0),
    selectedPointIndex(-1),
    ambientRotationEnabled(false),
    showDebugInfo(false),
    ambientRotationAngle(0.0f)
{
}

Renderer::~Renderer() {
    cleanup();
}

bool Renderer::initialize(const std::vector<SoundPoint>& points,
    const std::vector<ClusterColor>& colors)
{
    soundPoints = points;

    // Load shaders
    pointShader = std::make_unique<Shader>("shaders/point-v.glsl", "shaders/point-f.glsl");
    backgroundShader = std::make_unique<Shader>("shaders/bg-v.glsl", "shaders/bg-f.glsl");
    particleShader = std::make_unique<Shader>("shaders/particle-v.glsl", "shaders/particle-f.glsl");

    setupBuffers();
    setupCloudParticles(); 

    std::cout << "Renderer initialized with " << soundPoints.size() << " sound points\n";
    return true;
}

void Renderer::setupCloudParticles() {
    // Group points by cluster
    std::map<int, std::vector<const SoundPoint*>> clusters;
    for (const auto& p : soundPoints) {
        clusters[p.clusterID].push_back(&p);
    }

    cloudParticles.clear();

    // Create cloud particles for each cluster
    std::mt19937 rng(42);
    for (const auto& cluster : clusters) {
        if (cluster.second.empty()) continue;

        // Calculate cluster center
        glm::vec3 center(0.0f);
        for (const auto* p : cluster.second) {
            center += p->position;
        }
        center /= cluster.second.size();

        // Get cluster color
        glm::vec3 color = cluster.second[0]->color;

        // Create 150-300 particles per cluster
        int particleCount = 150 + (rng() % 150);
        for (int i = 0; i < particleCount; i++) {
            CloudParticle cp;

            // Random offset from center (spherical cloud)
            float theta = (rng() % 360) * 3.14159f / 180.0f;
            float phi = (rng() % 180) * 3.14159f / 180.0f;
            float radius = 8.0f + (rng() % 1000) / 100.0f; // 8-18 units

            cp.position = center + glm::vec3(
                radius * sin(phi) * cos(theta),
                radius * sin(phi) * sin(theta),
                radius * cos(phi)
            );

            cp.color = color;
            cp.size = 0.3f + (rng() % 100) / 200.0f; // 0.3-0.8
            cp.alpha = 0.1f + (rng() % 30) / 200.0f; // 0.1-0.25

            cloudParticles.push_back(cp);
        }
    }

    // Setup cloud VAO/VBO
    std::vector<float> cloudData;
    for (const auto& cp : cloudParticles) {
        cloudData.push_back(cp.position.x);
        cloudData.push_back(cp.position.y);
        cloudData.push_back(cp.position.z);
        cloudData.push_back(cp.color.r);
        cloudData.push_back(cp.color.g);
        cloudData.push_back(cp.color.b);
        cloudData.push_back(cp.size);
        cloudData.push_back(cp.alpha);
    }

    glGenVertexArrays(1, &cloudVAO);
    glGenBuffers(1, &cloudVBO);

    glBindVertexArray(cloudVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cloudVBO);
    glBufferData(GL_ARRAY_BUFFER, cloudData.size() * sizeof(float),
        cloudData.data(), GL_STATIC_DRAW);

    // pos
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);

    // color
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
        (void*)(3 * sizeof(float)));

    // size
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
        (void*)(6 * sizeof(float)));

    // alpha
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
        (void*)(7 * sizeof(float)));

    glBindVertexArray(0);

    std::cout << "Created " << cloudParticles.size() << " cloud particles\n";
}

void Renderer::setupBuffers() {
    // POINTS
    std::vector<float> pointData;
    for (const auto& p : soundPoints) {
        pointData.push_back(p.position.x);
        pointData.push_back(p.position.y);
        pointData.push_back(p.position.z);
        pointData.push_back(p.color.r);
        pointData.push_back(p.color.g);
        pointData.push_back(p.color.b);
        pointData.push_back(15.0f); // Larger for visibility
    }

    glGenVertexArrays(1, &pointVAO);
    glGenBuffers(1, &pointVBO);
    glBindVertexArray(pointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
    glBufferData(GL_ARRAY_BUFFER, pointData.size() * sizeof(float),
        pointData.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float),
        (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float),
        (void*)(6 * sizeof(float)));
    glBindVertexArray(0);

    // BACKGROUND QUAD
    float quad[] = {
        -1.f,  1.f, 0.f,
        -1.f, -1.f, 0.f,
         1.f, -1.f, 0.f,
        -1.f,  1.f, 0.f,
         1.f, -1.f, 0.f,
         1.f,  1.f, 0.f
    };

    glGenVertexArrays(1, &backgroundVAO);
    glGenBuffers(1, &backgroundVBO);
    glBindVertexArray(backgroundVAO);
    glBindBuffer(GL_ARRAY_BUFFER, backgroundVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);
}

void Renderer::render(const glm::mat4& view, const glm::mat4& projection, float time) {
    renderBackground(view, projection);
    renderCloudParticles(view, projection, time); 
    renderPoints(view, projection, time);
}

void Renderer::renderCloudParticles(const glm::mat4& view,
    const glm::mat4& projection,
    float time) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // Additive for glow
    glDepthMask(GL_FALSE); // Don't write to depth buffer

    particleShader->use();
    particleShader->setMat4("view", view);
    particleShader->setMat4("projection", projection);
    particleShader->setFloat("time", time);

    glBindVertexArray(cloudVAO);
    glDrawArrays(GL_POINTS, 0, cloudParticles.size());
    glBindVertexArray(0);

    glDepthMask(GL_TRUE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Renderer::renderBackground(const glm::mat4&, const glm::mat4&) {
    glDisable(GL_DEPTH_TEST);
    backgroundShader->use();
    backgroundShader->setVec3("colorTop", glm::vec3(0.05f, 0.02f, 0.1f)); // Darker
    backgroundShader->setVec3("colorBottom", glm::vec3(0.02f, 0.05f, 0.15f));
    glBindVertexArray(backgroundVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glEnable(GL_DEPTH_TEST);
}

void Renderer::renderPoints(const glm::mat4& view,
    const glm::mat4& projection,
    float time) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // Additive for intense glow

    pointShader->use();
    pointShader->setMat4("view", view);
    pointShader->setMat4("projection", projection);
    pointShader->setFloat("time", time);

    glBindVertexArray(pointVAO);

    for (size_t i = 0; i < soundPoints.size(); i++) {
        bool isSelected = (selectedPointIndex == (int)i);
        pointShader->setBool("isSelected", isSelected);
        pointShader->setFloat("glowIntensity", isSelected ? 5.0f : 1.5f); // Much stronger
        glDrawArrays(GL_POINTS, i, 1);
    }

    glBindVertexArray(0);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

int Renderer::selectPoint(float ndcX, float ndcY,
    const glm::mat4& view,
    const glm::mat4& projection,
    const glm::vec3& cameraPos) {
    glm::mat4 invProj = glm::inverse(projection);
    glm::mat4 invView = glm::inverse(view);

    glm::vec4 rayClip(ndcX, ndcY, -1.f, 1.f);
    glm::vec4 rayEye = invProj * rayClip;
    rayEye = glm::vec4(rayEye.x, rayEye.y, -1.f, 0.f);
    glm::vec3 rayWorld = glm::normalize(glm::vec3(invView * rayEye));

    float minDist = FLT_MAX;
    int closest = -1;
    float radius = 8.0f; // Larger selection radius

    for (size_t i = 0; i < soundPoints.size(); i++) {
        glm::vec3 diff = soundPoints[i].position - cameraPos;
        float projLen = glm::dot(diff, rayWorld);

        if (projLen > 0) {
            glm::vec3 closestPoint = cameraPos + rayWorld * projLen;
            float d = glm::distance(closestPoint, soundPoints[i].position);

            if (d < radius && projLen < minDist) {
                minDist = projLen;
                closest = (int)i;
            }
        }
    }

    selectedPointIndex = closest;
    if (closest >= 0) {
        std::cout << "Selected point " << closest << "\n";
    }

    return closest;
}

void Renderer::cleanup() {
    if (pointVAO) glDeleteVertexArrays(1, &pointVAO);
    if (pointVBO) glDeleteBuffers(1, &pointVBO);
    if (backgroundVAO) glDeleteVertexArrays(1, &backgroundVAO);
    if (backgroundVBO) glDeleteBuffers(1, &backgroundVBO);
    if (cloudVAO) glDeleteVertexArrays(1, &cloudVAO);
    if (cloudVBO) glDeleteBuffers(1, &cloudVBO);
}