mod cl_util;

use ocl::{
    Platform, Device, Context, Queue, Program, Kernel,
    enums::{ImageChannelOrder, ImageChannelDataType, MemObjectType},
    flags::MemFlags,
};
use image::{
    RgbaImage,
    io::Reader as ImageReader,
};
use cl_util::*;

fn main() {
    let (_plat, _dev, _ctx, queue, prog) = setup_env();
    let im =
        ImageReader::open("gecko.jpg").expect("failed to open image")
            .decode().expect("failed to decode image")
            .into_rgba8();
    let im_dims = im.dimensions();
    let im_cl = ocl::Image::<u8>::builder()
        // .context(&ctx)
        .queue(queue.clone())
        .flags(
            MemFlags::READ_ONLY |
            MemFlags::HOST_NO_ACCESS |
            MemFlags::COPY_HOST_PTR
        )
        .copy_host_slice(&im)
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(im_dims)
        .build().expect("failed to build input cl image");
    
    let mut out = RgbaImage::new(im_dims.0, im_dims.1);
    let out_cl = ocl::Image::<u8>::builder()
        // .context(&ctx)
        .queue(queue.clone())
        .flags(
            MemFlags::WRITE_ONLY |
            MemFlags::HOST_READ_ONLY
        )
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(im_dims)
        .build().expect("faield to build output cl image");
    
    let kernel = Kernel::builder()
        .program(&prog)
        .name("invert")
        .queue(queue.clone())
        .arg(&im_cl)
        .arg(&out_cl)
        .build().expect("failed to build kernel");
    // no need to manually copy to `im_cl`: it has `COPY_HOST_PTR` set
    unsafe {
        kernel.cmd().global_work_size(&im_dims)
            .enq().expect("failed to enqueue kernel");
    }

    // `read` is blocking by default
    out_cl.read(&mut out).enq().expect("failed to read output cl image");
    out.save("out.png").expect("failed to write output image to disk");
}

fn setup_env() -> (Platform, Device, Context, Queue, Program) {
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

    let ctx = Context::builder().platform(plat).devices(&dev)
        .build().expect("failed to create context");
    let queue = Queue::new(&ctx, dev, None)
        .expect("failed to create command queue");
    let prog = Program::builder().devices(&dev).src_file("src/program.cl")
        .build(&ctx).expect("failed to build program");
    
    (plat, dev, ctx, queue, prog)
}