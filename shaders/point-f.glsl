#version 450 core
out vec4 FragColor;

in vec3 Color;
in float PointSize;

uniform float time;
uniform bool isSelected;
uniform float glowIntensity;

void main()
{
    // Calculate distance from center of point
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    // Larger visible radius
    if (dist > 0.5)
        discard;
    
    // Core sphere - solid center
    float core = 1.0 - smoothstep(0.0, 0.2, dist);
    
    // Inner glow ring
    float innerGlow = 1.0 - smoothstep(0.2, 0.35, dist);
    innerGlow = pow(innerGlow, 1.5);
    
    // Outer glow halo
    float outerGlow = 1.0 - smoothstep(0.35, 0.5, dist);
    outerGlow = pow(outerGlow, 2.5);
    
    // Dramatic pulse for selected points
    float pulse = 1.0;
    if (isSelected) {
        pulse = 1.5 + 0.8 * sin(time * 8.0);
        // Add rapid shimmer
        pulse += 0.3 * sin(time * 20.0);
    } else {
        // Subtle breathing effect for non-selected
        pulse = 1.0 + 0.1 * sin(time * 2.0);
    }
    
    // Combine glow layers
    float totalGlow = core + (innerGlow * 0.8) + (outerGlow * 0.5);
    
    // Color intensity based on glow
    vec3 finalColor = Color * (1.2 + totalGlow * glowIntensity * 2.0 * pulse);
    
    // Add white hot center for selected points
    if (isSelected && dist < 0.15) {
        finalColor = mix(finalColor, vec3(1.5), core * pulse);
    }
    
    // Alpha based on distance with strong falloff
    float alpha = mix(1.0, 0.3, smoothstep(0.0, 0.5, dist));
    
    // Boost alpha for selected
    if (isSelected) {
        alpha = mix(1.0, 0.6, smoothstep(0.0, 0.5, dist));
    }
    
    FragColor = vec4(finalColor, alpha);
}