#version 450 core
out vec4 FragColor;

in vec2 TexCoord;

uniform vec3 colorTop;
uniform vec3 colorBottom;
uniform float time;

void main() {
    // Animated gradient
    float t = TexCoord.y;
    t += sin(time * 0.2 + TexCoord.x * 3.14) * 0.05;
    
    vec3 color = mix(colorBottom, colorTop, t);
    
    // Add subtle stars/sparkles
    float star = fract(sin(dot(TexCoord * 100.0, vec2(12.9898, 78.233))) * 43758.5453);
    if (star > 0.998) {
        color += vec3(0.3) * (sin(time + star * 100.0) * 0.5 + 0.5);
    }
    
    FragColor = vec4(color, 1.0);
}