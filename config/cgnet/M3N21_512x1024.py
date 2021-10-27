model = dict(
    type='CGNet',
    backbone=dict(
        m=3,
        n=21,
        classes=19,
        dropOutFlag=False,
        pretrained="weights/M3N21_512x1024_top1.pdparams"
    )
)

data = dict(
    set=dict(
        train=dict(
            root="data/Cityscapes",
            list_path="data/list/Cityscapes/cityscapes_train_list.txt",
            crop_size=(512, 1024),
            scale=True,
            mirror=True
        ),
        val=dict(
            root="data/Cityscapes",
            list_path="data/list/Cityscapes/cityscapes_val_list.txt",
        ),
        trainval=dict(
            root="data/Cityscapes",
            list_path="data/list/Cityscapes/cityscapes_trainval_list.txt",
            crop_size=(512, 1024),
            scale=True,
            mirror=True
        )
    ),
    loader=dict(
        train=dict(
            batch_size=8,
            shuffle=True,
            num_workers=1,
            use_shared_memory=True,
            drop_last=True
        ),
        val=dict(
            batch_size=1,
            shuffle=False,
            num_workers=0,
            use_shared_memory=True,
            drop_last=False
        )
    )
)


train = dict(
    max_epochs=360,
    opt=dict(
        learning_rate=0.001,
        max_epoch=360,
        last_epoch=-1,
        verbose=False
    ),
    resume=dict(
        last_epoch=-1,
        lr=0.001
    )
)
