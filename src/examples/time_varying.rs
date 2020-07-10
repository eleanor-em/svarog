use crate::gfx::util;
use stable_eyre::Report;
use rand::Rng;
use crate::gfx::core::{VulkanInitialiser, VulkanInterface};
use std::sync::Arc;
use vulkano::descriptor::DescriptorSet;

struct TimeVarying {
    x: f32,
    y: f32,
    v_x: f32,
    v_y: f32,
    set: Arc<dyn DescriptorSet>,
}

pub fn main() -> Result<(), Report> {
    let vk = VulkanInitialiser::new()?;
    let mut starter = vk.finalise(1)?;
    let vk = &mut starter.vk;

    let image = vk.load_texture(include_bytes!("../../res/dvd.png"))?;
    let sampler = util::default_sampler(vk.device())?;
    let set = Arc::new(vk.vert_col_tex().descriptor_set(0)?
        .add_sampled_image(image.clone(), sampler.clone())?
        .build()?
    );

    let vs = vs::Shader::load(vk.device())?;
    let fs = fs::Shader::load(vk.device())?;

    let pipeline = Arc::new(vk.new_vert_col_tex()?
        .vertex_shader(vs.main_entry_point(), ())
        .fragment_shader(fs.main_entry_point(), ())
        .build(vk.device())?);

    let mut x = vk.window_dims()[0] / 2.0;
    let mut y = vk.window_dims()[1] / 2.0;

    let mut v_x = 0.12;
    let mut v_y = v_x;

    let mut t = 0;

    starter.begin(move |vk| {
        let (sx, sy) = vk.window_to_vk(x, y);
        let (sw, sh) = vk.size_to_vk(image.dimensions().width() as f32 / 4., image.dimensions().height() as f32 / 4.);
        let quad = gen_quad(sx, sy, sw, sh);

        let framebuffer = vk.next_framebuffer(vk.default_pass())?;
        let mut commands = vk.new_command_buffer()?;
        commands.begin_render_pass(framebuffer, false, vec![[0.0, 0.0, 0.0, 1.0].into()])?
            .draw(pipeline.clone(),
                  vk.dynamic(),
                  util::into_vert_cols_tex(&vk, quad)?,
                  set.clone(),
                  [t])?
            .end_render_pass()?;

        t += 1;

        x += v_x;
        y += v_y;

        if x < 0. || x > vk.window_dims()[0] {
            v_x = -v_x;
        }
        if y < 0. || y > vk.window_dims()[1] {
            v_y = -v_y;
        }

        Ok(commands)
    });

    Ok(())
}

fn gen_quad(x: f32, y: f32, w: f32, h: f32) -> Vec<util::VctRenderInstance> {
    let mut instances = Vec::new();
    let mut rng = rand::thread_rng();
    let x = x - w / 2.0;
    let y = y - h / 2.;

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

    instances
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/time_varying/vert.shader",
    }
}


mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/time_varying/frag.shader",
    }
}