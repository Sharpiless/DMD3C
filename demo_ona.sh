python demo_ona.py gpus=[0] name=BP_KITTI ++chpt=dmd3c_distillation_depth_anything_v2.pth \
    net=PMP num_workers=1 \
    data=KITTI data.testset.mode=test data.testset.height=352 \
    test_batch_size=1 metric=RMSE ++save=true
