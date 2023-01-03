'''
    Camera流的model主体是5个部分：
    img_backbone, img_neck,
    pts_bbox_head,
    train_cfg, test_cfg
'''

# _base_没有使用
_base_ = [
    '../../_base_/datasets/nusc_cam_tf.py',

]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
optimizer = dict(type='AdamW', lr=0.00001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

voxel_size = [0.075, 0.075, 0.2]
# 使用了voxel_size：模型的bbox，train_cfg，test_cfg
# 这是camera流的代码，为何会使用voxel？
out_size_factor = 8
final_dim=(900, 1600) # HxW
downsample=8
dataset_type = 'NuScenesDataset'  # 数据集 NuScenes
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,  # 似乎可以扩展到radar
    use_map=False,
    use_external=False)
num_views = 6
imc = 256
model = dict(
    type='BEVF_TransFusion',
    camera_stream=True, 
    grid=0.6, 
    num_views=6,
    final_dim=final_dim,
    downsample=downsample, 
    imc=imc, 
    lic=256 * 2,
    pc_range = point_cloud_range,
    img_backbone=dict(
        type='CBSwinTransformer',
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                ape=False,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                use_checkpoint=False),
    img_neck=dict(
        type='FPNC',
        final_dim=final_dim,
        downsample=downsample, 
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        outC=imc,
        use_adp=True,
        num_outs=5),
    pts_bbox_head=dict(
        type='TransFusionHead',
        fuse_img=False,
        num_views=num_views,
        in_channels_img=256,
        out_size_factor_img=4,
        num_proposals=200,
        auxiliary=True,
        in_channels=imc,
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=1,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        )))


"""Ir_config 学习率配置
policy政策：cyclic，采用循环调整学习率的策略
target_ratio：在每次循环中学习率从10降低到0.0001
cyclic_times：循环次数，循环1次
step_ratio_up：学习率步长比例，具体尚不明
"""
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)

# momentum_config 动量配置，作用尚不明，参数含义类比Ir_config
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 20

"""log_config 日志配置
interval：每50步记录一次
hooks：定义了两种日志记录器，分别记录文本日历和可视化图表日志
"""
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])



# 以下大多是分布式训练配置
"""dist_params 参数
backend：后端，使用nccl，一个用于分布式训练/多GPU训练的库
"""
dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = None
resume_from = None
workflow = [('train', 1)]  # chatgpt：表示要执行训练操作，并将其参数设置为1
gpu_ids = range(0, 8)  # gpu的id
data = dict(
    samples_per_gpu=4,  # 每个gpu的样本数，用于调整batch大小和训练时间
    workers_per_gpu=6,)  # 每个gpu的工作线程数，勇于挑战训练时间



""".pth载入
看文件名，此处载入的应当是README.md中第一个训练得到的权重参数
"""
load_img_from = 'models/mask_rcnn_dbswin-t_fpn_3x_nuim_cocopre.pth'
# fp16 = dict(loss_scale=32.0)

# 输出训练参数的路径
checkpoint_config = dict(interval=1, out_dir='/model')
work_dir = '/model'

"""custom_hooks 定制的hook
# 在dev_aug分支中，没有此行代码
MindFreeHook：触发型hook，当特定事件发生时执行hook
"""
custom_hooks = [
    dict(type='MindFreeHook')
]
