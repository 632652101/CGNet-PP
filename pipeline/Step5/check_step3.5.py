from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("CGNet-PP/pipeline/Step3.5/CGNet_torch/loss_torch.npy")
    paddle_info = diff_helper.load_info("CGNet-PP/pipeline/Step3.5/CGNet_paddle/loss_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(path="CGNet-PP/pipeline/Step3/loss_diff.log")
