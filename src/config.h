#pragma once

// Window settings
constexpr int WINDOW_WIDTH = 1920;
constexpr int WINDOW_HEIGHT = 1080;
constexpr const char* WINDOW_TITLE = "Latent Sound Atlas";

// Rendering settings
constexpr float POINT_BASE_SIZE = 0.8f;
constexpr float POINT_SELECTED_SIZE = 1.5f;
constexpr float GLOW_INTENSITY = 0.7f;

// Camera settings
constexpr float CAMERA_SPEED = 20.0f;
constexpr float CAMERA_SENSITIVITY = 0.1f;
constexpr float CAMERA_FOV = 45.0f;
constexpr float CAMERA_NEAR = 0.1f;
constexpr float CAMERA_FAR = 500.0f;

// OSC settings
constexpr const char* OSC_HOST = "127.0.0.1";
constexpr int OSC_PORT = 57120; // SuperCollider default

// Animation settings
constexpr float AMBIENT_ROTATION_SPEED = 0.00f;
constexpr float PULSE_SPEED = 2.0f;
constexpr int PARTICLE_COUNT = 1000;

// Colors (ethereal palette)
constexpr float BACKGROUND_COLOR_TOP[3] = { 0.05f, 0.02f, 0.15f };    // Deep purple
constexpr float BACKGROUND_COLOR_BOTTOM[3] = { 0.02f, 0.05f, 0.12f }; // Deep blue
constexpr float AMBIENT_LIGHT[3] = { 0.3f, 0.3f, 0.4f };