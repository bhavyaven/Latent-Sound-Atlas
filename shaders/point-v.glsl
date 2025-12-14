#version 450 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aSize;

out vec3 Color;
out float PointSize;

uniform mat4 view;
uniform mat4 projection;
uniform float time;

void main()
{
    // REMOVED: Ambient rotation - points stay in fixed positions
    // Use the original position directly
    vec3 worldPos = aPos;
    
    gl_Position = projection * view * vec4(worldPos, 1.0);
    
    // Size attenuation based on distance
    // Prevent points from becoming too small or disappearing
    float dist = length(gl_Position.xyz);
    gl_PointSize = aSize * max(50.0, 200.0 / max(1.0, dist));
    
    Color = aColor;
    PointSize = aSize;
}