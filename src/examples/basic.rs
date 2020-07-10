use crate::gfx::util;
use stable_eyre::Report;
use rand::Rng;
use std::cell::RefCell;
use time::{Instant, Duration};
use crate::gfx::core::{VulkanInitialiser, VulkanInterface};
use std::sync::Arc;

pub fn main() -> Result<(), Report> {
    let vk = VulkanInitialiser::new()?;
    let mut starter = vk.finalise(1)?;
    let vk = &mut starter.vk;

    let image = vk.load_texture(include_bytes!("../../res/image.png"))?;
    let sampler = util::default_sampler(vk.device())?;
    let set = Arc::new(vk.vert_col_tex().descriptor_set(0)?
        .add_sampled_image(image.clone(), sampler.clone())?
        .build()?
    );

    let tri_buffers = RefCell::new(util::into_vert_cols(&vk, generate_tris())?);
    let quad_buffers = RefCell::new(util::into_vert_cols_tex(&vk, generate_quads())?);

    let now = RefCell::new(Instant::now());

    starter.begin(move |vk| {
        if now.borrow().elapsed() > Duration::milliseconds(300) {
            tri_buffers.replace(util::into_vert_cols(&vk, generate_tris())?);
            quad_buffers.replace(util::into_vert_cols_tex(&vk, generate_quads())?);

            now.replace(Instant::now());
        }

        let fb = vk.next_framebuffer(vk.default_pass())?;

        let mut commands = vk.new_command_buffer()?;
        commands.begin_render_pass(fb, false, vec![[1.0, 0.0, 1.0, 1.0].into()])?
            .draw(vk.vert_col().inner(), vk.dynamic(), tri_buffers.borrow().clone(), (), ())?
            .draw(vk.vert_col_tex().inner(), vk.dynamic(), quad_buffers.borrow().clone(), set.clone(), ())?
            .end_render_pass()?;

        Ok(commands)
    });

    Ok(())
}

fn generate_tris() -> Vec<util::VcRenderInstance> {
    let mut instances = Vec::new();

    let mut rng = rand::thread_rng();
    for _ in 0..(3 * 400) {
        let x: f32 = rng.gen();
        let y: f32 = rng.gen();

        let position = util::VkVertex {
            position: [2.0 * (x - 0.5), 2.0 * (y - 0.5), rng.gen()]
        };
        let colour = util::VkColour {
            colour: [rng.gen(), rng.gen(), rng.gen(), rng.gen()]
        };

        instances.push(util::VcRenderInstance {
            position,
            colour,
        });
    }

    instances
}

fn generate_quads() -> Vec<util::VctRenderInstance> {
    let mut instances = Vec::new();

    let mut rng = rand::thread_rng();
    for _ in 0..20 {
        let w = 0.1;
        let h = 0.1;

        let x = 2.0 * (rng.gen::<f32>() - 0.5);
        let y = 2.0 * (rng.gen::<f32>() - 0.5);

        // top-left
        instances.push(util::VctRenderInstance {
            position: [x, y, rng.gen()].into(),
            colour: [rng.gen(), rng.gen(), rng.gen(), rng.gen()].into(),
            tex_coord: [0.0, 0.0].into(),
        });
        // top-right
        instances.push(util::VctRenderInstance {
            position: [x + w, y, rng.gen()].into(),
            colour: [rng.gen(), rng.gen(), rng.gen(), rng.gen()].into(),
            tex_coord: [1.0, 0.0].into(),
        });
        // bottom-left
        instances.push(util::VctRenderInstance {
            position: [x, y + h, rng.gen()].into(),
            colour: [rng.gen(), rng.gen(), rng.gen(), rng.gen()].into(),
            tex_coord: [0.0, 1.0].into(),
        });
        // bottom-left
        instances.push(util::VctRenderInstance {
            position: [x, y + h, rng.gen()].into(),
            colour: [rng.gen(), rng.gen(), rng.gen(), rng.gen()].into(),
            tex_coord: [0.0, 1.0].into(),
        });
        // top-right
        instances.push(util::VctRenderInstance {
            position: [x + w, y, rng.gen()].into(),
            colour: [rng.gen(), rng.gen(), rng.gen(), rng.gen()].into(),
            tex_coord: [1.0, 0.0].into(),
        });
        // bottom-right
        instances.push(util::VctRenderInstance {
            position: [x + w, y + h, rng.gen()].into(),
            colour: [rng.gen(), rng.gen(), rng.gen(), rng.gen()].into(),
            tex_coord: [1.0, 1.0].into(),
        });
    }

    instances
}