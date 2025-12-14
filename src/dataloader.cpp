#include "dataloader.h"
#include "soundpoint.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <map>

// Helper functions
static std::string extractSynthFromFilename(const std::string& filename) {
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower.find("harp") != std::string::npos) return "harp";
    if (lower.find("flute") != std::string::npos) return "flute";
    if (lower.find("ocarina") != std::string::npos) return "ocarina";
    if (lower.find("pad") != std::string::npos) return "pad";
    if (lower.find("sparkle") != std::string::npos) return "sparkle";
    if (lower.find("chime") != std::string::npos) return "chime";
    if (lower.find("brook") != std::string::npos) return "brook";
    if (lower.find("rain") != std::string::npos) return "rain";
    if (lower.find("fire") != std::string::npos) return "fire";
    if (lower.find("leaves") != std::string::npos) return "leaves";
    if (lower.find("whoosh") != std::string::npos) return "whoosh";
    if (lower.find("bird") != std::string::npos) return "bird";
    if (lower.find("owl") != std::string::npos) return "owl";

    return "harp";
}

static std::string extractMood(const std::string& filename) {
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower.find("cozy") != std::string::npos) return "cozy";
    if (lower.find("mystic") != std::string::npos) return "mystic";
    if (lower.find("tense") != std::string::npos) return "tense";

    return "cozy";
}

static float generateFrequency(const std::string& synth, const std::string& mood, int seed) {
    std::mt19937 rng(seed);

    float cozy_notes[] = { 261.63f, 293.66f, 329.63f, 392.00f, 440.00f, 523.25f };
    float mystic_notes[] = { 293.66f, 329.63f, 349.23f, 392.00f, 440.00f, 493.88f };
    float tense_notes[] = { 220.00f, 246.94f, 261.63f, 293.66f, 329.63f, 349.23f };

    float* scale;
    int scale_size = 6;

    if (mood == "cozy") scale = cozy_notes;
    else if (mood == "mystic") scale = mystic_notes;
    else scale = tense_notes;

    int note_index = rng() % scale_size;
    float base_freq = scale[note_index];

    if (synth == "pad" || synth == "owl") base_freq *= 0.5f;
    else if (synth == "sparkle" || synth == "chime" || synth == "bird") base_freq *= 2.0f;
    else if (synth == "harp" || synth == "flute" || synth == "ocarina") {
        if (rng() % 3 == 0) base_freq *= 2.0f;
    }

    return base_freq;
}

static float generateAmplitude(const std::string& synth, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.3f, 0.7f);

    float base_amp = dist(rng);

    if (synth == "brook" || synth == "rain" || synth == "leaves" || synth == "bird") {
        base_amp *= 0.7f;
    }
    else if (synth == "sparkle" || synth == "chime") {
        base_amp *= 0.8f;
    }

    return base_amp;
}

bool DataLoader::loadJSON(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filepath << std::endl;
        return false;
    }

    nlohmann::json j;
    file >> j;

    soundPoints.clear();

    // Group points by cluster
    std::map<int, std::vector<SoundPoint>> clusterPoints;

    // First pass: load and group by cluster
    for (const auto& point : j["points"]) {
        SoundPoint sp;
        sp.id = point["id"];
        sp.filename = point["filename"];
        sp.clusterID = point["cluster_id"];
        sp.color = glm::vec3(
            point["color"]["rgb"][0],
            point["color"]["rgb"][1],
            point["color"]["rgb"][2]
        );

        // Original position (for grouping)
        sp.position = glm::vec3(
            point["coordinates"]["x"],
            point["coordinates"]["y"],
            point["coordinates"]["z"]
        );

        sp.synth = extractSynthFromFilename(sp.filename);
        sp.category = getCategoryFromSynth(sp.synth);
        std::string mood = extractMood(sp.filename);

        sp.freq = generateFrequency(sp.synth, mood, sp.id);
        sp.amp = generateAmplitude(sp.synth, sp.id + 1000);

        if (sp.synth == "harp" || sp.synth == "flute" || sp.synth == "ocarina") {
            sp.duration = 2.0f + (sp.id % 3) * 0.5f;
        }
        else if (sp.synth == "sparkle" || sp.synth == "chime") {
            sp.duration = 4.0f + (sp.id % 4);
        }
        else if (sp.synth == "brook" || sp.synth == "rain" || sp.synth == "fire" || sp.synth == "leaves") {
            sp.duration = 8.0f + (sp.id % 3) * 2.0f;
        }
        else {
            sp.duration = 2.0f;
        }

        sp.pan = -0.5f + (sp.id % 10) * 0.1f;

        clusterPoints[sp.clusterID].push_back(sp);
    }

    // Arrange clusters in a circular layout around origin
    int numClusters = clusterPoints.size();
    float clusterRadius = 40.0f; // Distance from center
    float angleStep = (2.0f * 3.14159f) / numClusters;

    int clusterIndex = 0;
    for (auto& clusterPair : clusterPoints) {
        float angle = clusterIndex * angleStep;
        glm::vec3 clusterCenter(
            cos(angle) * clusterRadius,
            0.0f,
            sin(angle) * clusterRadius
        );

        // Position points within cluster (tight grouping)
        std::mt19937 rng(clusterPair.first);
        std::uniform_real_distribution<float> offsetDist(-5.0f, 5.0f);

        for (auto& sp : clusterPair.second) {
            sp.position = clusterCenter + glm::vec3(
                offsetDist(rng),
                offsetDist(rng),
                offsetDist(rng)
            );
            soundPoints.push_back(sp);
        }

        clusterIndex++;
    }

    std::map<std::string, int> synthCounts;
    for (const auto& sp : soundPoints) {
        synthCounts[sp.synth]++;
    }

    std::cout << "\n=== Loaded " << soundPoints.size() << " sound points ===" << std::endl;
    std::cout << "Arranged in " << numClusters << " clusters\n";
    std::cout << "Synth distribution:" << std::endl;
    for (const auto& pair : synthCounts) {
        std::cout << "  " << pair.first << ": " << pair.second << std::endl;
    }

    std::cout << "\nSample audio parameters:" << std::endl;
    for (size_t i = 0; i < 5 && i < soundPoints.size(); i++) {
        std::cout << "  " << i << ". " << soundPoints[i].synth
            << " → freq=" << soundPoints[i].freq << " Hz"
            << ", amp=" << soundPoints[i].amp
            << ", dur=" << soundPoints[i].duration << "s" << std::endl;
    }
    std::cout << std::endl;

    return true;
}