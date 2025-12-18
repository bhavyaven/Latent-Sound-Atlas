#version 330 core
out vec4 FragColor;
in float Alpha;
void main()
{
    // Soft circular particle
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;

    float softness = 1.0 - (dist * 2.0);
    vec3 color = vec3(0.8, 0.9, 1.0); // Soft blue-white

    FragColor = vec4(color, Alpha * softness * 0.3);
}