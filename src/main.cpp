/*
 * Latent Sound Atlas
 *
 * Ethereal Fantasy 3D Visualization with OpenGL
 */

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <vector>
#include <memory>

#include "Config.h"
#include "DataLoader.h"
#include "Renderer.h"
#include "Camera.h"
#include "OSCManager.h"

 // Global state
std::unique_ptr<Camera> camera;
std::unique_ptr<Renderer> renderer;
std::unique_ptr<OSCManager> oscManager;

bool firstMouse = true;
float lastX = WINDOW_WIDTH / 2.0f;
float lastY = WINDOW_HEIGHT / 2.0f;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Callback functions
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void processInput(GLFWwindow* window);

int main() {
    std::cout << "=============================================================\n";
    std::cout << "  LATENT SOUND ATLAS - Ethereal 3D Visualization\n";
    std::cout << "=============================================================\n\n";

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4); // MSAA

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create window
    GLFWwindow* window = glfwCreateWindow(
        WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, nullptr, nullptr
    );

    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);

    // Capture mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // Load OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    }

    // OpenGL configuration
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE); // Enable MSAA
    glEnable(GL_PROGRAM_POINT_SIZE);

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << "\n";
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << "\n\n";

    // Initialize systems
    std::cout << "Initializing systems...\n";

    // Load sound point data
    DataLoader dataLoader;
    if (!dataLoader.loadJSON("opengl_data/sound_map_data.json")) {
        std::cerr << "Failed to load sound map data\n";
        return -1;
    }
    std::cout << "✓ Loaded " << dataLoader.getSoundPoints().size() << " sound points\n";

    // Initialize camera
    camera = std::make_unique<Camera>(
        glm::vec3(0.0f, 0.0f, 150.0f),  // Start far back
        glm::vec3(0.0f, 1.0f, 0.0f),    // Up vector
        -90.0f,                          // Look forward
        0.0f                             // Level pitch
    );

    // Initialize renderer
    renderer = std::make_unique<Renderer>();
    if (!renderer->initialize(dataLoader.getSoundPoints(), dataLoader.getClusterColors())) {
        std::cerr << "Failed to initialize renderer\n";
        return -1;
    }
    std::cout << "✓ Renderer initialized\n";

    // Initialize OSC manager
    oscManager = std::make_unique<OSCManager>("127.0.0.1", 57120);
    if (oscManager->isConnected()) {
        std::cout << "✓ OSC socket created successfully\n";
        std::cout << "  Sending to 127.0.0.1:57120\n";
        std::cout << "  ** SuperCollider must be running! **\n";

    }
    else {
        std::cout << "⚠ OSC connection failed (SuperCollider not running?)\n";
    }

    std::cout << "\n=============================================================\n";
    std::cout << "  Entering main loop...\n";
    std::cout << "=============================================================\n\n";
	std::cout << "May need to navigate camera to see points initially.\n\n";
    std::cout << "Controls:\n";
    std::cout << "  W/A/S/D - Move camera\n";
    std::cout << "  Q/E - Move up/down\n";
    std::cout << "  Right Click + Drag - Look around\n";
    std::cout << "  Left Click - Select and play sound\n";
    std::cout << "  ESC - Exit\n\n";

    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Input
        processInput(window);

        // Update camera
        camera->updateMatrices(WINDOW_WIDTH, WINDOW_HEIGHT);

        // Render
        glClearColor(
            BACKGROUND_COLOR_BOTTOM[0],
            BACKGROUND_COLOR_BOTTOM[1],
            BACKGROUND_COLOR_BOTTOM[2],
            1.0f
        );
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        renderer->render(camera->getViewMatrix(), camera->getProjectionMatrix(), currentFrame);

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    renderer->cleanup();
    glfwTerminate();

    std::cout << "\n=============================================================\n";
    std::cout << "  Application closed gracefully\n";
    std::cout << "=============================================================\n";

    return 0;
}

// Callback implementations
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos;

        lastX = xpos;
        lastY = ypos;

        camera->processMouseMovement(xoffset, yoffset);
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        // Get mouse position in NDC
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        // Convert to NDC
        float x = (2.0f * xpos) / WINDOW_WIDTH - 1.0f;
        float y = 1.0f - (2.0f * ypos) / WINDOW_HEIGHT;

        // Perform ray casting to find nearest point
        int selectedIndex = renderer->selectPoint(
            x, y,
            camera->getViewMatrix(),
            camera->getProjectionMatrix(),
            camera->getPosition()
        );

        if (selectedIndex >= 0) {
            const auto& point = renderer->getPoint(selectedIndex);
            std::cout << "Selected: " << point.filename
                << " (" << point.category << "/" << point.synth << ")\n";

            // Send OSC message to SuperCollider
            oscManager->playSoundByIndex(selectedIndex, renderer->getSoundPoints());
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    camera->processMouseScroll(static_cast<float>(yoffset));
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_SPACE) {
            renderer->toggleAmbientRotation();
        }
        else if (key == GLFW_KEY_F1) {
            renderer->toggleDebugInfo();
        }
    }
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    // Camera movement
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera->processKeyboard(CameraMovement::FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera->processKeyboard(CameraMovement::BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera->processKeyboard(CameraMovement::LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera->processKeyboard(CameraMovement::RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        camera->processKeyboard(CameraMovement::UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        camera->processKeyboard(CameraMovement::DOWN, deltaTime);
}