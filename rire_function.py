import numpy as np
import torch
import cv2

n_di = [1, 2]
sub_patch_dim_i = [(256, 128), (32, 64)]
max_rows_global = []
for i in range(len(n_di) - 1):
    if i == len(n_di) - 2:
        max_rows_global.append(n_di[i + 1])
    else:
        max_rows_global.append(np.prod(n_di[i + 1:]))


def rire_structure(images, labels, sub_patch_dim_i, n_di):
    all_patches = {}
    all_patches[0] = images
    mean = (0.4914, 0.4822, 0.4465)

    for i in range(len(n_di)):
        if i == 0:
            continue
        else:
            im_height, im_width = sub_patch_dim_i[i - 1]
        patch_height, patch_width = sub_patch_dim_i[i]

        patches = []

        for j in range(n_di[i]):
            while True:
                images_cp = images.copy()
                xp = np.random.randint(0, im_width)
                yp = np.random.randint(0, im_height)
                if xp + patch_width < im_width and yp + patch_height < im_height:
                    images_cp[:, yp: yp + patch_height, xp: xp + patch_width, 0] = mean[0]
                    images_cp[:, yp: yp + patch_height, xp: xp + patch_width, 1] = mean[1]
                    images_cp[:, yp: yp + patch_height, xp: xp + patch_width, 2] = mean[2]
                    patch = images_cp
                    patches.append(patch)
                    break

        patches = np.transpose(patches, [1, 0, 2, 3, 4])
        images = np.reshape(patches, [-1, 256, 128, 3])

        all_patches[i] = images

    new_labels = {}
    for i in range(len(max_rows_global)):
        lbls = []
        for k in range(len(all_patches[0])):
            lbl = np.tile([labels[k]], [max_rows_global[i]])
            lbls.extend(lbl)

        new_labels[len(n_di) - 1 - i] = lbls
    new_labels[0] = labels

    return all_patches, new_labels


if __name__ == "__main__":
    images = []
    targets = []
    img = cv2.imread('20004_finishline.jpg')
    img = cv2.resize(img, (128, 256))
    images.append(img)
    targets.append(20004)

    img = cv2.imread('20009_finishline.jpg')
    img = cv2.resize(img, (128, 256))
    images.append(img)
    images = np.array(images)
    targets.append(20009)

    images = np.array(images)
    patches, labels = rire_structure(images, targets, sub_patch_dim_i, n_di)

    sub_images = {}
    for i in range(len(n_di)):
        imgappend = []

        for j in range(len(patches[i])):
            img = np.expand_dims(patches[i][j], axis=0)
            img = torch.from_numpy(np.transpose(img, [0, 3, 1, 2]))

            if i == len(n_di) - 1:
                # if i != 0:
                imgappend.append(np.squeeze(img.data.cpu().numpy()))
            else:
                imgtile = np.tile(img.data.cpu().numpy(), [max_rows_global[i], 1, 1, 1])
                imgappend.extend(imgtile)

        sub_images[i] = np.array(imgappend)

    img0 = torch.from_numpy(images.transpose(0, 3, 1, 2)).cuda()
    img1 = torch.from_numpy(sub_images[1]).cuda()

    targets0 = torch.from_numpy(np.array(targets)).cuda()

    targets1 = labels[1]
    targets1 = torch.from_numpy(np.array(targets1)).cuda()

    print(img0.shape, img1.shape)
    print(targets0.shape, targets1.shape)

