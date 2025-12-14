#version 450 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aSize;
layout (location = 3) in float aAlpha;

out vec3 Color;
out float PointSize;
out float Alpha;

uniform mat4 view;
uniform mat4 projection;
uniform float time;

void main()
{
    // STATIC positions - no movement
    vec3 worldPos = aPos;
    
    gl_Position = projection * view * vec4(worldPos, 1.0);
    
    // Size based on distance
    float dist = length(gl_Position.xyz);
    gl_PointSize = aSize * max(80.0, 300.0 / max(1.0, dist));
    
    Color = aColor;
    PointSize = aSize;
    Alpha = aAlpha;
}