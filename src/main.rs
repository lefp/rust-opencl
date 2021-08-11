use ocl::{
    Platform, Device, DeviceType, Context, Queue, Program, Kernel, Buffer,
};
use std::f32::consts::PI as PI32;

fn main() {
    let plat = any_platform_with_substr("nvidia")
        .expect("failed to find specified platform");
    #[cfg(debug_assertions)]
    println!(
        "using platform '{}'", plat.name().expect("failed to get platform name")
    );

    let dev = any_gpu_device(&plat)
        .expect("failed to find any GPU device in this platform");
    #[cfg(debug_assertions)]
    println!(
        "using device '{}'", dev.name().expect("failed to get device name")
    );

    let context = Context::builder().platform(plat).devices(&dev)
        .build().expect("failed to create context");
    let queue = Queue::new(&context, dev, None)
        .expect("failed to create command queue");
    let prog = Program::builder().devices(&dev).src_file("src/test_program.cl")
        .build(&context).expect("failed to build program");

    let mut data = {
        const SIZE: u32 = 10_000;
        (0..SIZE).map(|x| x as f32 * PI32 / SIZE as f32).collect::<Vec<f32>>()
    };
    let buff = Buffer::<f32>::builder().queue(queue.clone()).len(data.len())
        .build().expect("failed to build buffer");
    buff.write(&data).enq().expect("failed to write source data to buffer");

    let kernel = Kernel::builder()
        .program(&prog)
        .name("calc_sin")
        .queue(queue.clone())
        .arg(&buff)
        .build().expect("failed to build kernel");
        // TODO? .global_work_size([]);
    unsafe {
        kernel.cmd().global_work_size(buff.len())
            .enq().expect("failed to enqueue kernel");
    }

    buff.read(&mut data).enq().expect("failed to write output to host");
    println!("{:?}", data);
}

fn any_platform_with_substr(substr: &str) -> Option<Platform> {
    let substr = &substr.to_ascii_lowercase();
    Platform::list().into_iter()
        .find(|plat| {
            plat.name().unwrap_or_else(|_| {
                    println!("WARN: ignoring nameless platform");
                    "".to_string()
                })
                .to_ascii_lowercase()
                .contains(substr)
        })
}
fn any_device_with_substr(substr: &str, platform: &Platform) -> Option<Device> {
    let substr = &substr.to_ascii_lowercase();
    Device::list_all(platform).expect("failed to list devices").into_iter()
        .find(|dev| {
            dev.name().unwrap_or_else(|_| {
                println!("WARN: ignoring nameless device");
                "".to_string()
            })
            .to_ascii_lowercase()
            .contains(substr)
        })
}
fn any_gpu_device(platform: &Platform) -> Option<Device> {
    Device::list(platform, Some(DeviceType::new().gpu()))
        .expect("failed to list devices")
        .into_iter().next()
}