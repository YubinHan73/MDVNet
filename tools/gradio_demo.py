import gradio as gr
from mmseg.apis import multi_gpu_test, single_gpu_test, inference_segmentor, init_segmentor

def process_images(model, dataset, input_image):
    if model == 'LSKANet' and dataset == 'Endovis2018':
        config_file = ('/data2/hyb/SegNetwork_other/SegNeXt-main/'
                       'tools/work_dirs/0_LSKANet/edv2018/LSKANet_hr18_LSKA_MAFF_BGH.py')
        check_points = ('/data2/hyb/SegNetwork_other/SegNeXt-main/'
                        'tools/work_dirs/0_LSKANet/LSKANet_hr48_cadist3_new/best_mIoU_epoch_106.pth')
        model = init_segmentor(config_file, check_points, device='cuda:0')
        result = inference_segmentor(model, input_image)
        model.show_result(input_image, result, show=False,
                          out_file='/data2/hyb/DataSurgery/test.png')
        output_image = '/data2/hyb/DataSurgery/test.png'

    return input_image,  output_image


iface = gr.Interface(
    fn=process_images,  # 指定处理函数
    inputs=[
        gr.Dropdown(choices=["UNet", "DeepLabv3+", "UPerNet", "HRNetV2", "ResNet",
                             "MobileNetV2", "OCRNet", "PSPNet", "TDNet", "TMAet",
                             "SegFormer", "SegNeXt", "STswinCL", "MFFNet", "LSKANet"],
                    label="网络模型"),
        gr.Radio(choices=["Endovis2018", "CaDIS", "MILS"], label="分割数据集"),
        gr.Image(label="手术图像"),
    ],
    outputs=[
            gr.Image(label="手术图像"),
            gr.Image(label="分割结果")
    ]
)

# 启动界面
iface.launch(share=True)
