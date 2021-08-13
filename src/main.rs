mod cl_util;

use ocl::{
    Platform, Device, Context, Queue, Program, Kernel, OclPrm,
    enums::{ImageChannelOrder, ImageChannelDataType, MemObjectType},
    flags::MemFlags,
};
use image::io::Reader as ImageReader;
use cl_util::*;

struct CorrelationMatrix {
    dims: [u32; 2],
    vals: Vec<f32>,
}
impl CorrelationMatrix {
    fn new(dims: [u32; 2], vals: Vec<f32>) -> Self {
        assert_eq!(dims[0] % 2, 1, "matrix dimensions must be odd");
        assert_eq!(dims[1] % 2, 1, "matrix dimensions must be odd");
        assert_eq!(
            vals.len() as u32, dims[0] * dims[1],
            "number of matrix elements ({}) does not match dimensions ({}x{})",
            vals.len(), dims[0], dims[1],
        );
        Self { vals, dims }
    }
}

fn main() {
    let (_plat, _dev, _ctx, queue, prog) = setup_env();
    let mut im =
        ImageReader::open("gecko.jpg").expect("failed to open image")
            .decode().expect("failed to decode image")
            .into_rgba8();
    let im_dims = im.dimensions();
    let im_cl = ocl::Image::<u8>::builder()
        .queue(queue.clone())
        .flags(
            MemFlags::READ_WRITE |
            MemFlags::COPY_HOST_PTR
        )
        .copy_host_slice(&im)
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(im_dims)
        .build().expect("failed to build input cl image");
    
    let intermediate_cl = ocl::Image::<u8>::builder()
        .queue(queue.clone())
        .flags(
            MemFlags::READ_WRITE |
            MemFlags::HOST_NO_ACCESS
        )
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(im_dims)
        .build().expect("failed to build output cl image");
    
    // sigma = 1
    let gaus_matrix = CorrelationMatrix::new([7, 7], vec![
        0.0000, 0.0002, 0.0011, 0.0018, 0.0011, 0.0002, 0.0000,
        0.0002, 0.0029, 0.0131, 0.0216, 0.0131, 0.0029, 0.0002,
        0.0011, 0.0131, 0.0586, 0.0966, 0.0586, 0.0131, 0.0011,
        0.0018, 0.0216, 0.0966, 0.1592, 0.0966, 0.0216, 0.0018,
        0.0011, 0.0131, 0.0586, 0.0966, 0.0586, 0.0131, 0.0011,
        0.0002, 0.0029, 0.0131, 0.0216, 0.0131, 0.0029, 0.0002,
        0.0000, 0.0002, 0.0011, 0.0018, 0.0011, 0.0002, 0.0000,
    ]);
    let gaus_kernel = correlation_kernel(
        &im_cl, &intermediate_cl, &gaus_matrix, &prog, queue.clone()
    );
    
    let lap_matrix = CorrelationMatrix::new([3, 3], vec![
        0f32,  1., 0.,
        1.,   -4., 1.,
        0.,    1., 0.,
    ]);
    let lap_kernel = correlation_kernel(
        &intermediate_cl, &im_cl, &lap_matrix, &prog, queue.clone()
    );

    unsafe {
        gaus_kernel.cmd()
            .enq().expect("failed to enqueue kernel");
        lap_kernel.cmd()
            .enq().expect("failed to enqueue kernel");
    }

    // `read` is blocking by default
    im_cl.read(&mut im).enq().expect("failed to read output cl image");
    im.save("out.png").expect("failed to write output image to disk");
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

fn correlation_kernel<T: OclPrm> (
    in_cl_image: &ocl::Image<T>, out_cl_image: &ocl::Image<T>,
    matrix: &CorrelationMatrix,
    prog: &Program, queue: Queue,
) -> Kernel {
    assert_eq!(
        in_cl_image.dims(), out_cl_image.dims(),
        "input and output images have different dimensions"
    );

    // integer division is intentional
    let half_size_x = (matrix.dims[0] / 2) as i32; // kernel takes a signed int
    let half_size_y = (matrix.dims[1] / 2) as i32;

    let matrix_cl = ocl::Image::<f32>::builder()
        .queue(queue.clone())
        .flags(
            MemFlags::READ_ONLY | 
            MemFlags::HOST_NO_ACCESS | 
            MemFlags::COPY_HOST_PTR
        )
        .copy_host_slice(&matrix.vals)
        .channel_order(ImageChannelOrder::Luminance)
        .channel_data_type(ImageChannelDataType::Float)
        .image_type(MemObjectType::Image2d)
        .dims(matrix.dims)
        .build().expect("failed to create correlation matrix on device");

    Kernel::builder()
        .program(&prog)
        .name("correlate2d")
        .queue(queue.clone())
        .arg(in_cl_image).arg(out_cl_image)
        // TODO we have a reference to a local variable... will `matrix_cl`
        // survive on the gpu after this function returns?
        .arg(&matrix_cl).arg(half_size_x).arg(half_size_y)
        .global_work_size(in_cl_image.dims())
        .build().expect("failed to build kernel")
}