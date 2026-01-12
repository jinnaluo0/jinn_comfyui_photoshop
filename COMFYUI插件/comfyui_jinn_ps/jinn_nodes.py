import logging
import torch
import numpy as np
import os,time,json
import folder_paths
from PIL import Image



#CUSTOM_NODES_DIR = folder_paths.folder_names_and_paths["custom_nodes"][0][0] 
#MODELS_DIR =  folder_paths.models_dir
#OUTPUT_PREVIEW="output_preview"
#NODE_FILE = os.path.abspath(__file__)

#python，如何去除一张图片四周的白色空白区域,输入的图片是torch.Tensor类型的数据，变量名称为image，查看image.shape内容为torch.Size([1, 768, 768, 3])
class JinnAlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False
any_type = JinnAlwaysEqualProxy("*")
class JinnCropImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.9, "min": 0, "max": 1, "step": 0.01},
                ),
                
            }
        }
    CATEGORY = "jinn"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_white_margins"

    def crop(self, image):

        logging.info(f"{type(image)},{image.shape}")
        print(type(image))
        print(image.shape)
        out = image
        return (image,)


    def remove_white_margins(self,image,threshold):
        # 将 image 转换为 NumPy 数组，移除 batch 维度
        logging.info(f"{type(image)},{image.shape}")

        image_np = image.squeeze(0).numpy()  # 形状为 (H, W, C)

        # 转换为灰度图像，取 RGB 的平均值
        gray_image = np.mean(image_np, axis=-1)
        gray_image[0:5,:]=1;gray_image[-1:-5,:]=1;gray_image[:,0:5]=1;gray_image[:,-1:-5]=1
        # 创建一个阈值掩码，标记白色区域（值接近 255）
        mask = gray_image < threshold

        # 找到非白色区域的边界
        print(image_np[50,50],image[0,50,50])
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        # 获取非白色区域的边界
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]

        #logging.info(f"{min_row},{max_row},{min_col},{max_col}")
        # 使用边界裁剪图像
        cropped_image = image[:, min_row:max_row + 1, min_col:max_col + 1, :]
        return (cropped_image,)

    def remove_white_margins3(self,image):
        # 如果图像是torch.Tensor类型，我们先将其转换成numpy数组
        image_np = image.permute(0, 2, 3, 1).cpu().numpy()[0]  # [768, 768, 3]

        # 定义白色阈值（RGB值都为255）
        white_threshold = 230

        # 查找非白色像素的边界
        mask = np.any(image_np < white_threshold, axis=-1)
        coords = np.argwhere(mask)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=1)
        # 裁剪图像
        cropped_image_np = image_np[x_min:x_max+1, y_min:y_max+1, :]

        # 转换回torch.Tensor
        cropped_image_tensor = torch.from_numpy(cropped_image_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, new_height, new_width]

        return (cropped_image_tensor,)

    

class JinnLoadFileFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": '', "multiline": False}),
                "file_tyle": ("STRING", {"default": '.txt', "multiline": False}),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }
    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("txt", "file_name")
    FUNCTION = "load_file"
    CATEGORY = "jinn"

    def load_file(self, folder_path='',file_tyle='.txt',index=0):
        txt_files = [
            file
            for file in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, file)) and file.endswith(file_tyle)
        ]
        txt_files.sort(key=lambda x:x[1])
        index=index%len(txt_files)
        file_name=txt_files[index]
        file_path=os.path.join(folder_path, file_name)
        filename_without_ext = os.path.splitext(file_name)[0]
        with open(file_path,encoding='utf-8') as f:
            txt=f.read()[:2000]
        return (txt,filename_without_ext)
        


class JinnJsonToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "JSON": (any_type, {"default": '', "multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "func"
    CATEGORY = "jinn"
    OUTPUT_TOOLTIPS = ("测试.", )
    DESCRIPTION = ""
    def func(self, JSON):
        text= json.dumps(JSON, indent=2)
        return (text,)

    
class JinnTextNoCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "func"
    CATEGORY = "jinn"
    OUTPUT_TOOLTIPS = ("测试.", )
    DESCRIPTION = "禁止缓存"
    

    def func(self, text=''):
        return (text,)
    @classmethod
    def IS_CHANGED(cls, text):
        return time.time()#这个函数可以让节点每次都执行，不会被缓存

    @classmethod
    def VALIDATE_INPUTS(cls, text):
        return True
class JinnFloatBatchIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            "float_batch": ("FLOAT",  {"forceInput": True}),
            "start": ("INT", {"forceInput": True} ),
            }
        }
    RETURN_TYPES = ("FLOAT","STRING")
    RETURN_NAMES = ("FLOAT","STRING")
    FUNCTION = "func"
    CATEGORY = "jinn"

    def func(self, float_batch,start):
        s='\n'.join([str(round(v,2)) for v in float_batch])
        return (float_batch[start],s)
        


class JinnSaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "output_path": ("STRING", {"default": '', "multiline": False}),
                "filename": ("STRING", {"default": "ComfyUI", "tooltip": ""})
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "jinn"
    DESCRIPTION = "绝对路径覆盖保存"


    def save_images(self, images,output_path, filename):
        #if output_path in [None, '', "none", "."]:output_path = self.output_dir
        #if not os.path.isabs(output_path):output_path = os.path.join(self.output_dir, output_path)
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(os.path.join(output_path, filename), compress_level=self.compress_level)
            
            results.append({
                "filename": filename,
                "subfolder": self.output_dir,
                "type": self.type
            })

        return { "ui": { "images": results } }


class JinnLoadOpenposeJSONNode:#将文本显示在界面上，挺麻烦
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_str": ("STRING", {"multiline": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("POSE_KEYPOINT",)
    FUNCTION = "load_json"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "jinn"

    def load_json(self, json_str, unique_id=None, extra_pnginfo=None):
        text=json.loads(json_str)
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next((x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),None,)
                if node:node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}



import json

class JinnKontextPreset:
    data = {
    "prefix": "You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.",
    "presets": [
        {
        "name": "随机传送",
        "brief": "Teleport the subject to a random location, scenario and/or style. Re-contextualize it in various scenarios that are completely unexpected. Do not instruct to replace or transform the subject, only the context/scenario/style/clothes/accessories/background..etc."
        },
        {
        "name": "移动镜头",
        "brief": "Move the camera to reveal new aspects of the scene. Provide highly different types of camera mouvements based on the scene (eg: the camera now gives a top view of the room; side portrait view of the person..etc )."
        },
        {
        "name": "重新打光",
        "brief": "Suggest new lighting settings for the image. Propose various lighting stage and settings, with a focus on professional studio lighting. Some suggestions should contain dramatic color changes, alternate time of the day, remove or include some new natural lights...etc"
        },
        {
        "name": "产品精修",
        "brief": "Turn this image into the style of a professional product photo. Describe a variety of scenes (simple packshot or the item being used), so that it could show different aspects of the item in a highly professional catalog. Suggest a variety of scenes, light settings and camera angles/framings, zoom levels, etc. Suggest at least 1 scenario of how the item is used."
        },
        {
        "name": "缩放",
        "brief": "Zoom {{SUBJECT}} of the image. If a subject is provided, zoom on it. Otherwise, zoom on the main subject of the image. Provide different level of zooms."
        },
        {
        "name": "上色",
        "brief": "Colorize the image. Provide different color styles / restoration guidance."
        },
        {
        "name": "海报风格",
        "brief": "Create a movie poster with the subjects of this image as the main characters. Take a random genre (action, comedy, horror, etc) and make it look like a movie poster. Sometimes, the user would provide a title for the movie (not always). In this case the user provided: . Otherwise, you can make up a title based on the image. If a title is provided, try to fit the scene to the title, otherwise get inspired by elements of the image to make up a movie. Make sure the title is stylized and add some taglines too. Add lots of text like quotes and other text we typically see in movie posters."
        },
        {
        "name": "卡通风格",
        "brief": "Turn this image into the style of a cartoon or manga or drawing. Include a reference of style, culture or time (eg: mangas from the 90s, thick lined, 3D pixar, etc)"
        },
        {
        "name": "移除文字",
        "brief": "Remove all text from the image."
        },
        {
        "name": "修改发型",
        "brief": "Change the haircut of the subject. Suggest a variety of haircuts, styles, colors, etc. Adapt the haircut to the subject's characteristics so that it looks natural. Describe how to visually edit the hair of the subject so that it has this new haircut."
        },
        {
        "name": "生成健美身型",
        "brief": "Ask to largely increase the muscles of the subjects while keeping the same pose and context. Describe visually how to edit the subjects so that they turn into bodybuilders and have these exagerated large muscles: biceps, abdominals, triceps, etc. You may change the clothes to make sure they reveal the overmuscled, exagerated body."
        },
        {
        "name": "移除家俱",
        "brief": "Remove all furniture and all appliances from the image. Explicitely mention to remove lights, carpets, curtains, etc if present."
        },
        {
        "name": "室内设计",
        "brief": "You are an interior designer. Redo the interior design of this image. Imagine some design elements and light settings that could match this room and offer diverse artistic directions, while ensuring that the room structure (windows, doors, walls, etc) remains identical."
        }
    ],
    "suffix": "Your response must consist of concise instruction ready for the image editing AI. Do not add any conversational text,Do not include tags like [img-0], explanations, or deviations; only the instructions."
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": ([preset["name"] for preset in cls.data.get("presets", [])],),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Prompt",)
    FUNCTION = "get_preset"
    CATEGORY = "jinn"
    
    @classmethod
    def get_brief_by_name(cls, name):
        for preset in cls.data.get("presets", []):
            if preset["name"] == name:
                return preset["brief"]
        return None

    def get_preset(cls, preset):
        brief = "The Brief:"+cls.get_brief_by_name(preset)
        fullString = cls.data.get("prefix")+'\n'+brief+'\n'+cls.data.get("suffix")
        return (fullString,)

from .jinn_utils  import QWenImageCaptioner
class JinnQWenImageCaptioner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api": ("STRING",{"multiline": False, "default": ""}),
                "model_name": (list(QWenImageCaptioner.load_models().keys()),),
                "image_url": ("STRING",{"multiline": False, "default": ""}),
                "user_prompt": ("STRING", {"multiline": True, "default": QWenImageCaptioner.DEFAULT_PROMPT, })
            }
        }
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("提示词",'价格(分)',)
    FUNCTION = "generate_image_captions"
    #OUTPUT_NODE = True
    CATEGORY = "jinn"
    def generate_image_captions(self, api,model_name,image_url, user_prompt):
        try:
            return QWenImageCaptioner.generate_image_captions(api,model_name,image_url, user_prompt)
        except Exception as e:
            logging.error(f"访问通义千问时发生错误", exc_info=True)
            return (f'访问通义千问时发生错误,{e}',-1)   
    






NODE_CLASS_MAPPINGS = {
    #"JinnCropImage": JinnCropImage,
    #"JinnLoadFileFromFolder": JinnLoadFileFromFolder,
    #"JinnTextNoCache": JinnTextNoCache,
    #"JinnFloatBatchIterator": JinnFloatBatchIterator,
    #"JinnSaveImage": JinnSaveImage,
    "JinnJsonToText": JinnJsonToText,
    #"JinnKontextPreset": JinnKontextPreset,
    "JinnQWenImageCaptioner": JinnQWenImageCaptioner,

    
}
NODE_DISPLAY_NAME_MAPPINGS = {
    #"JinnCropImage": "JinnCropImage",
    #"JinnLoadFileFromFolder": "JinnLoadFileFromFolder",
    #"JinnTextNoCache": "JinnTextNoCache",
    #"JinnFloatBatchIterator": "JinnFloatBatchIterator",
    #"JinnSaveImage": "JinnSaveImage",
    "JinnJsonToText": "JinnJsonToText",
    #"JinnKontextPreset": "JinnKontextPreset",
    "JinnQWenImageCaptioner": "JinnQWenImageCaptioner",
    
}


