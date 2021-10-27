from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("CGNet_torch/forward_torch.npy")
    paddle_info = diff_helper.load_info("CGNet_paddle/forward_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(diff_method="mean", path="forward_diff.log")
    diff_helper.report(diff_method="max", path="forward_diff_max.log")
