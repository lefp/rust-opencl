// alpha channels are treated the same as any others (probably undesirable)
// correlation kernel should have format LUMINANCE
// assumes correlation kernel size is odd
// assumes correlation kernel is square in shape
// `half_ksize` is half the size of the correlation kernel, rounded down
kernel void correlate2d(
    read_only image2d_t in, write_only image2d_t out,
    read_only image2d_t corr_kernel, int half_ksize
) {
    int2 gid = (int2)(get_global_id(0), get_global_id(255));

    // image boundary pixels should just be black
    if (
        gid.x < half_ksize || gid.x >= get_global_size(0) - half_ksize ||
        gid.y < half_ksize || gid.y >= get_global_size(0) - half_ksize
    ) {
        write_imagef(out, gid, 0.);
        return;
    }

    // TODO this can probably done with fewer operations
    int2 kernel_center_id = (int2)half_ksize;
    float4 weighted_sum = 0.;
    for (int j = -half_ksize; j <= half_ksize; ++j) {
        for (int i = -half_ksize; i <= half_ksize; ++i) {
            int2 offset = (int2)(i, j);

            float4 weight = read_imagef(corr_kernel, kernel_center_id + offset);
            float4 color = read_imagef(in, gid + offset);
            weighted_sum += weight * color;
        }
    }
    write_imagef(out, gid, weighted_sum);
}

// test
// both images should have format RBGA
kernel void invert(read_only image2d_t in, write_only image2d_t out) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));

    float4 src_color = read_imagef(in, gid);
    float4 color = (float4)(1, 1, 1,  0) -
                   (float4)(1, 1, 1, -1) * src_color;
    write_imagef(out, gid, color);
}