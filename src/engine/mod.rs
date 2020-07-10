// use stable_eyre::Report;
// use vulkano::command_buffer::AutoCommandBufferBuilder;
// use crate::gfx::core::{VulkanInitialiser, VulkanBackend};
//
// pub trait GameObject {
//     fn init(&mut self, vk: &mut VulkanBackend) -> Result<(), Report>;
//
//     fn render(&mut self, vk: &mut VulkanBackend, command_buffer: &mut AutoCommandBufferBuilder) -> Result<(), Report>;
// }
//
// pub fn init<T: GameObject>(mut root: T) -> Result<(), Report> {
//     let vk = VulkanInitialiser::new()?;
//     let mut starter = vk.finalise(1)?;
//     let vk = &mut starter.vk;
//
//     root.init(vk);
//
//     starter.begin(move |vk| {
//         let framebuffer = vk.next_framebuffer(vk.default_pass())?;
//         let mut commands = vk.new_command_buffer()?;
//         commands.begin_render_pass(framebuffer, false, vec![[0.0, 0.0, 0.0, 1.0].into()])?
//
//         root.render(vk, &mut commands)?;
//         Ok(commands)
//     });
//
//     Ok(())
// }