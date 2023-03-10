// correlation matrix should have format LUMINANCE
// assumes correlation matrix size is odd
// assumes correlation matrix is square in shape
// `half_msize` is half the size of the correlation matrix, rounded down
kernel void correlate2d(
    read_only image2d_t in, write_only image2d_t out,
    read_only image2d_t corr_matrix, int half_msize_x, int half_msize_y
) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));

    // image boundary pixels should just be 0
    if (
        gid.x < half_msize_x || gid.x >= get_global_size(0) - half_msize_x ||
        gid.y < half_msize_y || gid.y >= get_global_size(1) - half_msize_y
    ) {
        write_imagef(out, gid, 0);
        return;
    }

    // TODO this can probably done with fewer operations
    int2 mat_center_id = (int2)(half_msize_x, half_msize_y);
    float4 weighted_sum = 0;
    for (int j = -half_msize_y; j <= half_msize_y; ++j) {
        for (int i = -half_msize_x; i <= half_msize_x; ++i) {
            int2 offset = (int2)(i, j);

            float4 weight = read_imagef(corr_matrix, mat_center_id + offset);
            float4 color = read_imagef(in, gid + offset);
            weighted_sum += weight * color;
        }
    }
    write_imagef(out, gid, weighted_sum);
}

// both images should have format RBGA
kernel void invert(read_only image2d_t in, write_only image2d_t out) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));

    float4 src_color = read_imagef(in, gid);
    float4 color = (float4)(1, 1, 1,  0) -
                   (float4)(1, 1, 1, -1) * src_color;
    write_imagef(out, gid, color);
}