from .nucleus_kaggle_stage1 import nucleus_stage1
import dataset.extend_transforms as extend_transforms
import dataset.joint_transforms as joint_transforms
import torchvision.transforms as standard_transforms

def factory(dataset_name):
    if dataset_name == 'nucleus_stage1':
        mean_std = mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_joint_transform = joint_transforms.Compose([
            joint_transforms.Scale_((256, 256)),
            joint_transforms.RandomHorizontallyFlip()
        ])
        input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            #standard_transforms.Normalize(*mean_std)
        ])
        target_transform = extend_transforms.MaskToTensor()
        dataset = nucleus_stage1(joint_transform = train_joint_transform,
            transform = input_transform, target_transform = target_transform)
    return dataset
