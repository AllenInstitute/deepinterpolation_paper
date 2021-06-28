import h5py

path_in = r"Z:\603576132\movie_2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5"
path_out = r"Z:\603576132\movie_lowerdim_2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5"

with h5py.File(path_in, "r") as file_handle_in:

    input_shape = file_handle_in['data_proc'].shape
    with h5py.File(path_out, "r") as file_handle_out:

        dset_out = file_handle_out.create_dataset(
            "data",
            shape= input_shape[0:3],
            chunks=(1, input_shape[1], input_shape[2]),
            dtype="float16",
        )

        for index_frame in range(input_shape[0]):
            dset_out[index_frame, :, :] = file_handle_in["data"][index_frame, :, :, 0]
            print(index_frame)