#version 450
#define PI 3.14159265359

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 colour;
layout(location = 2) in vec2 tex_coord;

layout(location = 0) out vec4 fragColour;
layout(location = 1) out vec2 fragTexCoords;

layout(push_constant) uniform TimeData {
    uint stepCount;
} time;

void main() {
    float periodFactor = 0.01;

    gl_Position = vec4(position, 1.0);
    fragColour = vec4(
        sin(time.stepCount * periodFactor / (2 * PI)) / 2 + 0.5,
        sin(time.stepCount * periodFactor / (2 * PI) + 2 * PI / 3) / 2 + 0.5,
        sin(time.stepCount * periodFactor / (2 * PI) + 4 * PI / 3) / 2 + 0.5,
        1.0
    );
    fragTexCoords = tex_coord;
}