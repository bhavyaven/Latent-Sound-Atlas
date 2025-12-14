#ifndef SOUNDPOINT_H
#define SOUNDPOINT_H

#include <string>
#include <glm/glm.hpp>

// Sound point structure for 3D visualization
struct SoundPoint {
    // Identification
    int id;
    std::string filename;

    // 3D position
    glm::vec3 position;

    // Visual properties
    int clusterID;
    glm::vec3 color;

    // Audio categorization
    std::string category;  // "melodic", "magic", "nature", "sfx"
    std::string synth;     // "harp", "flute", "sparkle", etc.

    // Audio parameters
    float freq;            // Frequency in Hz
    float amp;             // Amplitude (0.0 - 1.0)
    float duration;        // Duration in seconds
    float pan;             // Pan (-1.0 to 1.0)

    // Optional audio features (for future use)
    float spectralCentroid;
    float rmsEnergy;
    float zeroCrossingRate;

    // Constructor with defaults
    SoundPoint()
        : id(0)
        , filename("")
        , position(0.0f)
        , clusterID(0)
        , color(1.0f)
        , category("melodic")
        , synth("harp")
        , freq(440.0f)
        , amp(0.5f)
        , duration(2.0f)
        , pan(0.0f)
        , spectralCentroid(0.0f)
        , rmsEnergy(0.0f)
        , zeroCrossingRate(0.0f)
    {
    }
};

 // Cluster color structure
struct ClusterColor {
    int clusterID;
    glm::vec3 color;
    std::string name;
    std::string category;
};

// Helper function to get category from synth type
inline std::string getCategoryFromSynth(const std::string& synth) {
    if (synth == "harp" || synth == "flute" || synth == "ocarina" || synth == "pad") {
        return "melodic";
    }
    else if (synth == "sparkle" || synth == "chime") {
        return "magic";
    }
    else if (synth == "brook" || synth == "rain" || synth == "fire" || synth == "leaves") {
        return "nature";
    }
    else if (synth == "whoosh" || synth == "bird" || synth == "owl") {
        return "sfx";
    }
    return "unknown";
}

#endif // SOUNDPOINT_H