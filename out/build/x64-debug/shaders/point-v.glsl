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
    vec3 worldPos = aPos;
    gl_Position = projection * view * vec4(worldPos, 1.0);
    
    float dist = length((view * vec4(worldPos, 1.0)).xyz);
    gl_PointSize = aSize * (10.0 / (1.0 + dist * 0.02)); // Changed from 80.0
    
    Color = aColor;
    PointSize = aSize;
}