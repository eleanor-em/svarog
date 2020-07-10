use vulkano::device::Device;
use std::sync::Arc;
use stable_eyre::Report;
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use crate::gfx::core::VulkanInterface;

pub struct VctRenderInstance {
    pub position: VkVertex, // in GPU space
    pub colour: VkColour,
    pub tex_coord: VkTexCoord,
}

pub fn into_vert_cols_tex<V: VulkanInterface>(vk: &V, instances: Vec<VctRenderInstance>)
                                              -> Result<Vec<Arc<(dyn vulkano::buffer::BufferAccess + std::marker::Send + std::marker::Sync + 'static)>>, Report> {
    let mut positions = Vec::with_capacity(instances.len());
    let mut colours = Vec::with_capacity(instances.len());
    let mut tex_coords = Vec::with_capacity(instances.len());

    for instance in instances {
        positions.push(instance.position);
        colours.push(instance.colour);
        tex_coords.push(instance.tex_coord);
    }

    Ok(vec![
        vk.iter_buffer(positions.into_iter())?,
        vk.iter_buffer(colours.into_iter())?,
        vk.iter_buffer(tex_coords.into_iter())?,
        ])
}

pub struct VcRenderInstance {
    pub position: VkVertex, // in GPU space
    pub colour: VkColour,
}

pub fn into_vert_cols<V: VulkanInterface>(vk: &V, instances: Vec<VcRenderInstance>)
                                          -> Result<Vec<Arc<(dyn vulkano::buffer::BufferAccess + std::marker::Send + std::marker::Sync + 'static)>>, Report> {
    let mut positions = Vec::with_capacity(instances.len());
    let mut colours = Vec::with_capacity(instances.len());

    for instance in instances {
        positions.push(instance.position);
        colours.push(instance.colour);
    }

    Ok(vec![
        vk.iter_buffer(positions.into_iter())?,
        vk.iter_buffer(colours.into_iter())?,
    ])
}

#[derive(Default, Debug, Copy, Clone)]
pub struct VkVertex {
    pub position: [f32; 3],
}

impl From<[f32; 3]> for VkVertex {
    fn from(position: [f32; 3]) -> Self {
        Self { position }
    }
}

vulkano::impl_vertex!(VkVertex, position);

#[derive(Default, Debug, Copy, Clone)]
pub struct VkColour {
    pub colour: [f32; 4]
}

impl From<[f32; 4]> for VkColour {
    fn from(colour: [f32; 4]) -> Self {
        Self { colour }
    }
}

vulkano::impl_vertex!(VkColour, colour);

#[derive(Default, Debug, Copy, Clone)]
pub struct VkTexCoord {
    pub tex_coord: [f32; 2],
}

impl From<[f32; 2]> for VkTexCoord {
    fn from(tex_coord: [f32; 2]) -> Self {
        Self { tex_coord }
    }
}

vulkano::impl_vertex!(VkTexCoord, tex_coord);

#[inline(always)]
pub fn default_sampler(device: Arc<Device>) -> Result<Arc<Sampler>, Report> {
    Ok(Sampler::new(
        device,
        Filter::Linear,
        Filter::Linear,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0,
        1.0,
        0.0,
        0.0,
    )?)
}