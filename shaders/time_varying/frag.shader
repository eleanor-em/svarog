#version 450

layout(location = 0) in vec4 fragColour;
layout(location = 1) in vec2 fragTexCoords;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    f_color = fragColour * texture(tex, fragTexCoords);
}