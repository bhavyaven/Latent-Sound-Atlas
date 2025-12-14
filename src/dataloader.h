#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "SoundPoint.h"

class DataLoader {
public:
    DataLoader() = default;

    bool loadJSON(const std::string& filepath);

	const std::vector<SoundPoint>& getSoundPoints() const { return soundPoints; }
	const std::vector<ClusterColor>& getClusterColors() const { return clusterColors; }

private:
    std::vector<SoundPoint> soundPoints;
    std::vector<ClusterColor> clusterColors;
};
