#by hanlestudy@163.com
import numpy as np
import cv2

imgs_test = np.load('pred_img.npy') #读入.npy文件
# gtruth_masks = np.load('gt.npy')
# origin_imgs = np.load('ori.npy')
print(imgs_test.shape)
for i in range(imgs_test.shape[0]):
    # cv2.imshow('pred', np.squeeze(imgs_test[i]))
    # cv2.waitKey(0)
    cv2.imwrite("result/unet_detail/pred_{}.png".format(i), np.squeeze(imgs_test[i])*255)

    # cv2.imshow('gtruth_mask', np.squeeze(gtruth_masks[0]))
    # # cv2.waitKey(0)
    # cv2.imwrite("gt/gtruth_{}.png".format(i), np.squeeze(gtruth_masks[i]) * 255)
    # cv2.imshow('orith_image',origin_imgs[0])
    # cv2.waitKey(0)
    # cv2.imwrite("ori/origin_{}.png".format(i), origin_imgs[i])
# import cv2
# cv2.imshow('pred', np.squeeze(imgs_test[0]))
# cv2.waitKey(0)
# fig,ax = plt.subplots(10,1,figsize=[15,15])
# for idx in range(imgs_test.shape[0]):
#     # ax[idx, 0].imshow(np.uint8(np.squeeze((orig_imgs[idx]))))
#     # ax[idx, 1].imshow(np.squeeze(gtruth_masks[idx]), cmap='gray')
#     ax[idx, 0].imshow(np.squeeze(imgs_test[idx]), cmap='gray')
#
# plt.savefig('sample_results.png')
