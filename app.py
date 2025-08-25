from my_model import ResAttUnetDync
from utils import ShowResult
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib
import torch
from PIL import Image
from flask import Flask
from flask import request
from flask import render_template
warnings.simplefilter("ignore")
plt.style.use("ggplot")
matplotlib.use('agg')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


weight_path='weight/weight.pth'

MODEL = ResAttUnetDync(in_channels=4, n_classes=3,
                       n_channels=16).to(device=device)
MODEL.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
show_result = ShowResult()
app = Flask(__name__)
upload_folder = "./static/data/"
location='./result/'
def get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth):
            x1 = (height - crop_height) // 2
            x2 = x1 + crop_height
            y1 = (width - crop_width) // 2
            y2 = y1 + crop_width
            z1 = (depth - crop_depth) // 2
            z2 = z1 + crop_depth
            return x1, y1, z1, x2, y2, z2



def center_crop(data:np.ndarray, crop_height, crop_width, crop_depth):
    height, width, depth = data.shape[:3]
    if height < crop_height or width < crop_width or depth < crop_depth:
        raise ValueError
    x1, y1, z1, x2, y2, z2 = get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth)
    data = data[x1:x2, y1:y2, z1:z2]
    return data

def get_survival_time(age,wt,tc,et):
    return "Survival Time: Low"


def process(images_name,_id,age):
    result=dict()
    images=[]
    for file_path in images_name:
        print(file_path)
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        data=center_crop(data,128,128,128)
        data_min = np.min(data)
        img=(data - data_min) / (np.max(data) - data_min)

        images.append(img)
    img = np.stack(images)
    img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
    result['Id']=_id
    result['image']=img
    
    imgs=result['image']
    imgs=torch.from_numpy(imgs).unsqueeze(0)
    print(imgs.shape)
    logits = MODEL(imgs.float().to(device))
    probs = torch.sigmoid(logits)
    
    predictions = (probs >= 0.33).float()
    predictions =  predictions.cpu()
    
    wt=predictions[0][0].detach().numpy()
    tc=predictions[0][1].detach().numpy()
    et=predictions[0][2].detach().numpy()
    
    
    survival = get_survival_time(age,wt,tc,et)
    
    wt = nib.Nifti1Image(wt, affine=np.eye(4))
    tc = nib.Nifti1Image(tc, affine=np.eye(4))
    et = nib.Nifti1Image(et, affine=np.eye(4))
    
    
    
    nib.save(wt, f'{location+_id}_wt.nii')
    nib.save(tc, f'{location+_id}_tc.nii')
    nib.save(et, f'{location+_id}_et.nii')
    

    show_result.plot(torch.from_numpy(result['image']), predictions, _id)
    
    return survival

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_t1 = request.files["t1"]
        image_t2w = request.files["t2w"]
        image_t1ce = request.files["t1ce"]
        image_flair = request.files["flair"]
        in_age=request.form['age']
        print(in_age,type(in_age))

        images=[image_flair,image_t1,image_t1ce,image_t2w]
        
        imgs_loc=[]
        
        for image_file in images :
            image_location = os.path.join(
                upload_folder,
                image_file.filename
            )

            image_file.save(image_location)

            image_name = os.path.basename(image_location)
            image_name = image_name.split('.')[0]

            imgs_loc.append(image_location)

        survival= process(imgs_loc, image_name[:12],int(in_age))
        
        print(f"{image_name[:12]}_out.png")
        
        im=Image.open(f'static/data/{image_name[:12]}_out.png')
        width, height = im.size
        left = 25
        top = 5
        right = width-175
        bottom = height
        im1 = im.crop((left, top, right, bottom))
        im1.save(f'static/data/{image_name[:12]}_out.png')
        
        return render_template("index.html", image_loc=(f"{image_name[:12]}_out.png"), prediction=image_name[:12], survival=survival)

    return render_template("index.html", prediction=0, image_loc=None)



if __name__ == "__main__":

    app.run(debug=True,host=5000)
