from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request, Form
import utils
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


# making app object 
app = FastAPI()

app.mount("/static", StaticFiles(directory= "static"), name="static")   #
template = Jinja2Templates(directory= "templates")    #

@app.get("/")
def home(request:Request):
    return template.TemplateResponse("index.html", {"request":request})

@app.post("/")
async def home_prediction(request:Request, file:UploadFile= File(...)):
    # get features from image 
    result = None
    error = None
    try:
        result= utils.get_result(img_file=file)
    except Exception as e:
        error = e
   
    return template.TemplateResponse("index.html", {"request":request, "result":result, "error":error})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return utils.get_result(image_file=file, is_api=True)


# test
# demo_img_path = "./static/images/necrosis_1.jpg"
# demo_image = utils.get_features(demo_img_path)
# print(demo_image.shape)

# cls = utils.prediction(demo_image)
# print(cls)
