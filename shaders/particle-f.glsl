#version 450 core
out vec4 FragColor;

in vec3 Color;
in float PointSize;
in float Alpha;

uniform float time;

void main()
{
    // Calculate distance from center
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    // Soft cloud-like falloff
    if (dist > 0.5)
        discard;
    
    // Very soft gradient from center
    float cloudDensity = 1.0 - smoothstep(0.0, 0.5, dist);
    cloudDensity = pow(cloudDensity, 2.5); // Softer falloff
    
    // Gentle pulsing
    float pulse = 1.0 + 0.05 * sin(time * 0.5 + PointSize * 10.0);
    
    // Final color with cluster color
    vec3 finalColor = Color * (0.8 + cloudDensity * 0.4) * pulse;
    
    // Very subtle alpha
    float finalAlpha = Alpha * cloudDensity * 0.5;
    
    FragColor = vec4(finalColor, finalAlpha);
}