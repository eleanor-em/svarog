#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 colour;
layout(location = 2) in vec2 tex_coord;

layout(location = 0) out vec4 fragColour;
layout(location = 1) out vec2 fragTexCoords;

void main() {
    gl_Position = vec4(position, 1.0);
    fragColour = colour;
    fragTexCoords = tex_coord;
}