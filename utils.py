import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage




class ShowResult:
    def mask_preprocessing(self, mask):
        mask = mask.squeeze().cpu().detach().numpy()
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))
        mask_WT = np.rot90(montage(mask[0]))
        mask_TC = np.rot90(montage(mask[1]))
        mask_ET = np.rot90(montage(mask[2]))
        return mask_WT, mask_TC, mask_ET
    def image_preprocessing(self, image):
        image = image.squeeze().cpu().detach().numpy()
        image = np.moveaxis(image, (0, 1, 2, 3), (0, 3, 2, 1))
        flair_img = np.rot90(montage(image[0]))
        return flair_img
    def plot(self, image, prediction,_id):
        image = self.image_preprocessing(image)
        pr_mask_WT, pr_mask_TC, pr_mask_ET = self.mask_preprocessing(prediction)
        _, axes = plt.subplots(figsize = (10, 10))
        axes.axis("off")
        axes.imshow(image, cmap ='bone')
        axes.imshow(np.ma.masked_where(pr_mask_WT == False, pr_mask_WT),
                  cmap='cool_r', alpha=0.6)
        axes.imshow(np.ma.masked_where(pr_mask_TC == False, pr_mask_TC),
                  cmap='YlGnBu', alpha=0.6)
        axes.imshow(np.ma.masked_where(pr_mask_ET == False, pr_mask_ET),
                  cmap='cool', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'./static/data/{_id}_out.png')