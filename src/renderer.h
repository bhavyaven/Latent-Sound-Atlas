#ifndef RENDERER_H
#define RENDERER_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include "Shader.h"
#include "SoundPoint.h"

struct CloudParticle {
    glm::vec3 position;
    glm::vec3 color;
    float size;
    float alpha;
};

class Renderer {
public:
    Renderer();
    ~Renderer();

    bool initialize(const std::vector<SoundPoint>& points,
        const std::vector<ClusterColor>& colors);

    void render(const glm::mat4& view, const glm::mat4& projection, float time);

    int selectPoint(float ndcX, float ndcY,
        const glm::mat4& view,
        const glm::mat4& projection,
        const glm::vec3& cameraPos);

    void toggleAmbientRotation() { ambientRotationEnabled = !ambientRotationEnabled; }
    void toggleDebugInfo() { showDebugInfo = !showDebugInfo; }

    const SoundPoint& getPoint(int index) const { return soundPoints[index]; }
    const std::vector<SoundPoint>& getSoundPoints() const { return soundPoints; }

    void cleanup();

private:
    // OpenGL objects
    unsigned int pointVAO, pointVBO;
    unsigned int backgroundVAO, backgroundVBO;
    unsigned int particleVAO, particleVBO;
    unsigned int cloudVAO, cloudVBO;

    // Shaders
    std::unique_ptr<Shader> pointShader;
    std::unique_ptr<Shader> backgroundShader;
    std::unique_ptr<Shader> particleShader;

    // Data
    std::vector<SoundPoint> soundPoints;
    std::vector<CloudParticle> cloudParticles;
    int selectedPointIndex;

    // State
    bool ambientRotationEnabled;
    bool showDebugInfo;
    float ambientRotationAngle;

    // Setup functions
    void setupBuffers();
    void setupCloudParticles();

    // Render functions
    void renderBackground(const glm::mat4& view, const glm::mat4& projection);
    void renderPoints(const glm::mat4& view, const glm::mat4& projection, float time);
    void renderCloudParticles(const glm::mat4& view, const glm::mat4& projection, float time);
};

#endif // RENDERER_H