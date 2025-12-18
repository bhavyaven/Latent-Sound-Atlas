#version 450 core
out vec4 FragColor;

in vec3 Color;
in float PointSize;

uniform float time;
uniform bool isSelected;
uniform float glowIntensity;

void main()
{
    // Distance from center
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float dist = length(coord);
    
    // Discard outside circle
    if (dist > 1.0)
        discard;
    
    float core = 1.0 - smoothstep(0.0, 0.6, dist);
    core = pow(core, 3.0); // Sharp falloff
    
    // Minimal glow - only at edges
    float edgeGlow = 1.0 - smoothstep(0.8, 1.0, dist);
    
    // Pulse for selected
    float pulse = 1.0;
    if (isSelected) {
        pulse = 1.0 + 0.5 * sin(time * 4.0);
    }
    
    // Brightness focused in center
    float brightness = core * glowIntensity * pulse;
    
    // Color with much less intensity to prevent overlap bloom
    vec3 finalColor = Color * (0.8 + brightness * 0.7);
    
    // Sharp alpha falloff - transparent edges
    float alpha = core * 0.9 + edgeGlow * 0.3;
    
    // Ensure selected points are brighter
    if (isSelected) {
        finalColor *= 1.3;
        alpha = min(alpha * 1.2, 1.0);
    }
    
    FragColor = vec4(finalColor, alpha);
}