import sys,os,json,random,time,base64,subprocess
from datetime import datetime
from PIL import Image
import folder_paths
from collections import OrderedDict
import logging
from aiohttp import web


DIR_UPLOAD="psupload"
PATH_UPLOAD = os.path.join(folder_paths.get_input_directory(),DIR_UPLOAD)
PATH_THIS = os.path.dirname(os.path.abspath(__file__))
PATH_CONFIG= os.path.join(PATH_THIS,f'config')
PATH_TMP= os.path.join(PATH_THIS,f'tmp')


def errorResp(msg,e=None):
    if e:
        logging.info(f"{e}", exc_info=True)
        #logging.info(traceback.format_exc())
    return web.json_response({"comfy_error": (f'{msg} {str(e)}' if e else msg)})


class LRUCacheDict:
    def __init__(self, max_size=200):
        self.cache = OrderedDict()
        self.max_size = max_size
    def get(self, key):
        if key in self.cache:
            # 将访问的键移动到末尾，表示最近使用
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    def put(self, key, value):
        if key in self.cache:
            # 如果键已存在，更新值并移到末尾
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            # 超出容量，删除最近最少使用的项（最前面的）
            self.cache.popitem(last=False)
        # 插入新项或更新旧值
        self.cache[key] = value
    def remove(self, key):
        """主动从缓存中删除指定的键"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self):
        """清空整个缓存"""
        self.cache.clear()
    def __contains__(self, key):
        return key in self.cache
    
APICACHE = LRUCacheDict(max_size=50) #API缓存

class JinnComfyWorkflowWrapper(dict):
    def __init__(self, dic: dict):
        super().__init__(dic)
    def list_nodes(self) :
        return [node["_meta"]["title"] for node in super().values()]
    def set_node_param(self, title: str, param: str, value):
        smth_changed = False
        for node in super().values():
            if node["_meta"]["title"] == title:
                node["inputs"][param] = value
                smth_changed = True
        if not smth_changed:
            raise ValueError(f"Node '{title}' not found.")
    def set_node_param_byid(self, nodeid: str, field: str, value):
        smth_changed = False
        node=self[nodeid]
        node["inputs"][field] = value
        smth_changed = True
    def get_node_param(self, title: str, param: str):
        for node in super().values():
            if node["_meta"]["title"] == title:
                return node["inputs"][param]
        raise ValueError(f"Node '{title}' not found.")

    def get_node_id(self, title: str) -> str:
            for id, node in super().items():
                if node["_meta"]["title"] == title:
                    return id
            raise ValueError(f"Node '{title}' not found.")
    
    def get_node_ids(self, title: str) -> str:#qjs
            res=[]
            for id, node in super().items():
                if node["_meta"]["title"] == title:
                    res.append(id)
            return res
    
ENCRYPT_KEY = 0xB3  # 加密密钥（1字节范围：0x00 ~ 0xFF）
def encrypt_save__(path,jsondata):
    text=json.dumps(jsondata, ensure_ascii=False)
    data = text.encode('utf-8')
    data= bytes(b ^ ENCRYPT_KEY for b in data)
    with open(path, 'wb') as f:f.write(data)

def encrypt_read__(path):
    with open(path, 'rb') as f:data = f.read()
    data = bytes(b ^ ENCRYPT_KEY for b in data)
    text = data.decode('utf-8')
    jsondata=json.loads(text)
    return jsondata

def save__(path,jsondata):
    text=json.dumps(jsondata, ensure_ascii=False)
    with open(path, 'w') as f:f.write(text)

def read__(path):
    with open(path, 'r') as f:jsondata=json.load(f)
    return jsondata

#创建ID
def create_id():
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        random_suffix = random.randint(1000, 9999)
        filename = f"{timestamp}_{random_suffix}"
        return filename

class WfGroupUtil:
    def __init__(self,db_path):
        self._cached_groups = None  # 缓存 workflows 到内存中
        self.db_path=db_path
        self.init_group()
    def get_db_path(self):
        return self.db_path
    
    def save_groups__(self, groups):
        save__(self.get_db_path(), groups)
        self._cached_groups = groups  # 同步更新缓存
    def init_group(self):
        if not os.path.isfile(self.get_db_path()):
            self.save_groups__({"items": {}})
    def _load_cached_groups(self):
        """如果缓存为空，则从磁盘加载"""
        if self._cached_groups is None:
            self._cached_groups = read__(self.get_db_path())

    def get_groups(self):
        self._load_cached_groups()
        return self._cached_groups
    
    def add_group(self,wfgroupname):
        self._load_cached_groups()
        groups = self.get_groups()
        wfgroupid=create_id()
        groups['items'][wfgroupid]={'name':wfgroupname}
        self.save_groups__(groups)
        return wfgroupid
    
    def edit_group(self, wfgroupid, wfgroupname):
        groups = self.get_groups()
        if wfgroupid not in groups['items']:raise Exception(f'工作流分组"{wfgroupid}"不存在')
        groups['items'][wfgroupid]['name']=wfgroupname
        self.save_groups__(groups)
    def del_group(self, wfgroupid):
        groups = self.get_groups()
        if wfgroupid not in groups['items']:raise Exception(f'工作流分组"{wfgroupid}"不存在')
        workflows = workflowsUtil.get_workflows()
        for item in workflows["items"]:
            if wfgroupid == item.get('wfgroupid',None):
                raise Exception(f'该工作流分组不为空,如:{item["wfname"]}')
        del groups['items'][wfgroupid]
        self.save_groups__(groups)
    def sort(self,wfgroupid1,wfgroupid2):
        groups = self.get_groups()
        items = list(groups['items'].items())
        key_order = [k for k, v in items]
        if wfgroupid1 not in key_order or wfgroupid2 not in key_order:
            raise Exception(f'修改分组顺序时出错')

        index1 = key_order.index(wfgroupid1)
        index2 = key_order.index(wfgroupid2)
        items.insert(index2, items.pop(index1))
        groups['items']=dict(items)


        
    
wfGroupUtil=WfGroupUtil(os.path.join(PATH_THIS,f'jinn_wf_group.dat'))

class WorkflowsUtil:
    def __init__(self,api_dir,db_path,db_path_del):
        self._cached_workflows = None  # 缓存 workflows 到内存中
        self.api_dir=api_dir
        self.db_path=db_path
        self.db_path_del=db_path_del
        self.api_path=os.path.join(PATH_THIS, api_dir)
    def get_api_dir(self):
        return self.api_dir
    def get_db_path_del(self):
        return self.db_path_del
    def get_db_path(self):
        return self.db_path
    
    

    def _load_cached_workflows(self):
        """如果缓存为空，则从磁盘加载"""
        if self._cached_workflows is None:
            self._cached_workflows = encrypt_read__(self.get_db_path())


    def save_workflows__(self, workflows):
        encrypt_save__(self.get_db_path(), workflows)
        self._cached_workflows = workflows  # 同步更新缓存

    def init_workflows(self):
        if not os.path.exists(self.api_path): os.makedirs(self.api_path)
        if not os.path.isfile(self.get_db_path()):
            self.save_workflows__({"items": []})
        if not os.path.isfile(self.get_db_path_del()):
            encrypt_save__(self.get_db_path_del(), {"items": []})

    def add_workflows(self, wfname,wfcode,apidata,params,wfidrel,wfgroupid):
        kind=params["output"]["kind"]
        self._load_cached_workflows()
        workflows = self._cached_workflows
        wfid=create_id()
        workflows['items'].append({
            'wfname': wfname,
            'wfcode':wfcode,
            "output_kind": kind,
            "wfid": wfid,
            "wfidrel": wfidrel,
            "wfgroupid": wfgroupid,
        })
        self.save_workflows__(workflows)

        self.save_api(apidata,wfname)
        self.save_params(wfname,params)
        return wfid

    def get_workflows(self):
        self._load_cached_workflows()
        return self._cached_workflows
    def get_workflows_t2i(self):
        workflows=self.get_workflows()
        return [item for i, item in enumerate(workflows["items"]) if item["output_kind"] =='OUTPUTTEXT']
    def get_workflows_recover(self):
        workflows_del = encrypt_read__(self.get_db_path_del())
        return workflows_del["items"]

    def workflows_recover(self, wfname, wfid):
        if self.workflow_exists(wfname):
            raise Exception(f'"{wfname}"已存在，请修改名称后再进行恢复操作')

        workflows_del = encrypt_read__(self.get_db_path_del())
        for i, item in enumerate(workflows_del["items"]):
            if wfid == item['wfid']:
                break
        else:
            raise Exception('没有找到被删除的工作流')

        item_del = workflows_del["items"].pop(i)
        del item_del["deltime"]

        self._load_cached_workflows()
        workflows = self._cached_workflows
        workflows['items'].append(item_del)
        self.save_workflows__(workflows)
        encrypt_save__(self.get_db_path_del(), workflows_del)

    def workflows_visible(self, wfid, visible):
        workflows = self.get_workflows()
        item = next((item for i, item in enumerate(workflows["items"]) if item["wfid"] == wfid), None)
        if item is not None:
            item["visible"] = visible
        self.save_workflows__(workflows)

    def edit_workflows_group(self,wfid,wfgroupid):
        workflows = self.get_workflows()
        for index,item in enumerate(workflows["items"]):
            if wfid == item['wfid']:
                item['wfgroupid']=wfgroupid
                workflows["items"].append(workflows["items"].pop(index))#换到新的分组，同时放到最后面
                break
        else:
            raise Exception('没有找到名称')
        self.save_workflows__(workflows)

    def edit_workflows_group_sort(self,wfid1,wfid2,wfgroupid):
        workflows = self.get_workflows()
        index1 = next((i for i, item in enumerate(workflows["items"]) if item["wfid"] == wfid1), None)
        index2 = next((i for i, item in enumerate(workflows["items"]) if item["wfid"] == wfid2), None)
        if index1 is not None and index2 is not None :
            workflows["items"][index1]['wfgroupid']=wfgroupid
            workflows["items"].insert(index2, workflows["items"].pop(index1))
        self.save_workflows__(workflows)

    def edit_workflows(self, wfnameold, wfnamenew,wfcode, kind,wfidrel,wfgroupid):
        workflows = self.get_workflows()
        changed = False
        for item in workflows["items"]:
            if wfnameold == item['wfname']:
                if wfnameold != wfnamenew:
                    item['wfname'] = wfnamenew;changed = True
                if kind != item.get('output_kind', None):
                    item['output_kind'] = kind;changed = True
                if wfcode != item.get('wfcode', None):
                    item['wfcode'] = wfcode;changed = True
                if wfidrel != item.get('wfidrel', None):
                    item['wfidrel'] = wfidrel;changed = True
                if wfgroupid != item.get('wfgroupid', None) and wfcode:#只有内置工作流在修改时，才能修改分组
                    item['wfgroupid'] = wfgroupid;changed = True
                if 'group' in item:
                    del item['group'];changed = True
                if 'wfgoupid' in item:
                    del item['wfgoupid'];changed = True
                break
        else:
            raise Exception('没有找到名称')

        if changed:
            self.save_workflows__(workflows)

    def del_workflows__(self, wfname, index):
        workflows = self.get_workflows()
        item = workflows["items"].pop(index)
        self.save_workflows__(workflows)
        APICACHE.remove(wfname)
        workflows_del = encrypt_read__(self.get_db_path_del())
        item["deltime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        workflows_del["items"].append(item)
        if len(workflows_del["items"]) > 50:
            del workflows_del["items"][0]
        encrypt_save__(self.get_db_path_del(), workflows_del)

    def del_workflows(self, wfname):
        workflows = self.get_workflows()
        for i, item in enumerate(workflows["items"]):
            if wfname == item['wfname']:
                self.del_workflows__(wfname, i)
                break
        else:
            raise Exception('没有找到名称')

    def workflow_exists(self, wfname):
        workflows = self.get_workflows()
        for item in workflows["items"]:
            if wfname == item['wfname']:
                return True
        return False

    def get_wfid(self, wfname):
        workflows = self.get_workflows()
        for item in workflows["items"]:
            if wfname == item['wfname']:
                return item.get('wfid', item['wfname'])
    
    def save_params(self, wfname,params):
        wfid=self.get_wfid(wfname)
        path= os.path.join(PATH_THIS,self.get_api_dir(),f'{wfid}_params.dat')
        encrypt_save__(path,params)

    def get_params(self, wfname):
        wfid=self.get_wfid(wfname)
        path= os.path.join(PATH_THIS,self.get_api_dir(),f'{wfid}_params.dat')
        params= encrypt_read__(path)
        return params
    
    def save_api(self,apidata,wfname):
        wfid=self.get_wfid(wfname)
        path= os.path.join(PATH_THIS,self.get_api_dir(),f'{wfid}_api.dat')
        encrypt_save__(path,apidata)
        APICACHE.remove(wfname)
    def get_api(self,wfname):
        cached_result = APICACHE.get(wfname)
        if cached_result is not None: return cached_result
        wfid=self.get_wfid(wfname)
        path= os.path.join(PATH_THIS,self.get_api_dir(),f'{wfid}_api.dat')
        api= encrypt_read__(path)
        if os.name != 'nt':repairWfAPIForPath(api) #ubuntu系统下，模型的路径是/,例如 flux/a.safetensors
        APICACHE.put(wfname, api)
        return api
    
    def get_removed(self):
        workflows = self.get_workflows()
        exclude_list =[item["wfid"] for item in workflows["items"]]
        path= os.path.join(PATH_THIS,self.get_api_dir())
        json_files = [f for f in os.listdir(path) if f.endswith('_api.dat')]
        matched_prefixes = []
        for filename in json_files:
            base_name = filename[:-8] 
            if base_name in exclude_list:continue
            #wfname=[item["wfname"] for item in workflows["items"] if item["wfid"]==base_name]
            matched_prefixes.append(base_name)
        return matched_prefixes
    def copy_workflows(self,wfname,wfcode):

        i = 1
        while True:
            newwfname=f'{i}_{wfname}'
            if not self.workflow_exists(newwfname):break
            i += 1

        if wfcode:
            apidata=workflowsUtilBuildin.get_api(wfname)
            params=workflowsUtilBuildin.get_params(wfname)
        else:
            apidata=self.get_api(wfname)
            params=self.get_params(wfname)
        wfid=self.add_workflows(newwfname,"", apidata,params,"","")
        getWfUtil(wfcode).save_api(apidata,wfname)
        getWfUtil(wfcode).save_params(wfname,params)
        return wfid


workflowsUtil=WorkflowsUtil('workflows',os.path.join(PATH_THIS,f'jinn_db.dat'),os.path.join(PATH_THIS,f'jinn_db_deleted.dat'))
workflowsUtilBuildin=WorkflowsUtil('workflows_buildin',os.path.join(PATH_THIS,f'jinn_db_buildin.dat'),os.path.join(PATH_THIS,f'jinn_db_buildin_deleted.dat'))
def getWfUtil(buildin):
    return workflowsUtilBuildin if buildin and buildin!="undefined" else workflowsUtil


#操作config文件
def get_config():
    try:
        with open(PATH_CONFIG,encoding='utf-8') as f:  config=json.load(f)
        return config
    except:
        return {}
def set_config(key,value):
    config=get_config()
    config[key]=value
    with open(PATH_CONFIG, 'w', encoding='utf-8') as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

def save_config(config):
    with open(PATH_CONFIG, 'w', encoding='utf-8') as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


# 缩放图片
async def resize_image(image, min_size,max_size ):
    width, height = image.size
    shortest_side = min(width, height)

    if max_size > 0 and shortest_side > max_size:
        if width <= height:
            new_width = max_size
            new_height = round(height * max_size / width)
        else:
            new_height = max_size
            new_width = round(width * max_size / height)
    elif min_size > 0 and shortest_side < min_size:
        if width <= height:
            new_width = min_size
            new_height = round(height * min_size / width)
        else:
            new_height = min_size
            new_width = round(width * min_size / height)
    else:
        return image


    # 缩放图片，使用 LANCZOS 插值
    resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img

#创建图像文件名
def recreateFileName(folder,filename):
    name, ext = os.path.splitext(filename)
    i = 1
    filepath = os.path.join(folder, filename)
    filepath = os.path.abspath(filepath)
    if not os.path.exists(filepath):return filename,filepath
    while os.path.exists(filepath):
        filename = f"{name}({i}){ext}"
        filepath = os.path.join(folder, filename)
        i += 1
    filepath = os.path.abspath(filepath)
    return filename,filepath

import math

# 给定的字典对象
def repairWfAPI(dic):
    for k, v in dic.items():
        if '_meta' in v and 'is_changed' in v['_meta']:
            value=v['_meta']['is_changed']
            if math.isnan(value):
                v['_meta']['is_changed'] = False
    
def repairWfAPIForPath(apidata):
    for nodeid, node in apidata.items():
        if "inputs" not in node:continue
        for field,value in node["inputs"].items():
            if isinstance(value,str) and '\\' in value:
                node["inputs"][field]=value.replace('\\','/')
        


#工作流检查时报错，分析报错原因
def check_validate_prompt_result(dic):
    for nodeid,v in dic[3].items():
        errors=v["errors"]
        class_name=v["class_type"]
        return f"{class_name}({nodeid}):{str(errors)[:200]}"
    return ""
def get_node_title(node):
    return node["_meta"]["title"] if "_meta" in node else node["class_type"]


#自动修改工作流参数，例如SEED
def change_params(apidata,wfcode,is50GPU):
    try:
        for nodeid,node in apidata.items():
            nodeclass=node["class_type"]
            if (nodeclass=='KSampler' or nodeclass=='KSamplerAdvanced') and "seed" in node["inputs"]:
                #种子的数值太小，会有问题的 种子位数和模型初始噪声的映射方式不同，导致风格差异 random.randint(10**14, 10**15 - 1)
                node["inputs"]["seed"]=random.randint(10**13, 10**15 - 1)#random.getrandbits(48)#random.randint(10**12, 10**15)
            #对于内置工作流+50系统显卡+nunchaku工作流
            if (wfcode and is50GPU and nodeclass=='NunchakuFluxDiTLoader') and "model_path" in node["inputs"]:
                model_path=node["inputs"]["model_path"]
                if 'svdq-int4' in model_path:
                    model_path=model_path.replace('svdq-int4','svdq-fp4')
                    node["inputs"]["model_path"]=model_path
                
    except:pass 


#批量生图时，会上传各种格式的文件，比如'.avif','.jfif'，而这些，comfyui不支持 ，需要进行转换AVIF 转 PNG 的函数
iio_module=None
def convertSpecialImage(filepath,filename):
    if filepath.lower().endswith(".avif"):
        try:
            global iio_module
            if not iio_module:
                import imageio.v3 as iio
                iio_module=iio
        except Exception as e:
            logging.error(f"加载 imageio.v3 失败: {e}", exc_info=True)
            return
        
        try:
            png_path = filepath[:-5] + ".png"
            image = iio_module.imread(filepath)
            iio_module.imwrite(png_path, image)
            filename=filename[:-5] + ".png"
            return filename
        except Exception as e:
            logging.error(f"AVIF 转换失败", exc_info=True)


#伪造Request
class FakeRequest:
    def __init__(self, data: dict):
        self._json_data = data

    async def json(self):
        return self._json_data


#语音输入
whisper_module = None
sounddevice_module=None
write_func=None
samplerate = 16000 
path_audio=os.path.join(PATH_UPLOAD,"audio.wav")
path_model=os.path.join(folder_paths.models_dir,"whisper")
def soundToText(duration,modelName):
    global whisper_module,sounddevice_module,write_func
    if whisper_module is None:
        from scipy.io.wavfile import write
        import sounddevice
        import whisper
        whisper_module=whisper
        write_func=write
        sounddevice_module=sounddevice
    
    recording = sounddevice_module.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sounddevice_module.wait()
    write_func(path_audio, samplerate, recording)
    model = whisper_module.load_model(modelName, download_root=path_model)  # 可改为 base,tiny、small、medium、large
    result = model.transcribe(path_audio, language="Chinese")
    return result["text"]

#获取上传图片的完整路径
def getFullImageFilePath(image_url):
    inputdir=folder_paths.get_input_directory()
    filepath = os.path.join(inputdir, os.path.normpath(image_url))
    return filepath

#通义千问
from http import HTTPStatus
class QWenImageCaptioner:
    DEFAULT_PROMPT='请详细描述这张图片，包括内容、布局、风格、灯光、细节等。您的回复将作为人工智能生成图像的提示。，您的回复应仅包括AI提示词，不包括图像的分析过程,请用中文回答'
    dashscope_module=None
    DEFAULT_MODELS={
        "qwen-vl-max":{"input_price":0.003,"output_price":0.009},
        "qwen-vl-plus":{"input_price":0.0015,"output_price":0.0045},
    }
    MODELS=None
    @classmethod
    def load_models(cls):
        if not QWenImageCaptioner.MODELS:
            config=get_config()
            if "qwen_vl_models" not in config:
                config["qwen_vl_models" ]=QWenImageCaptioner.DEFAULT_MODELS
                save_config(config)
            QWenImageCaptioner.MODELS=config["qwen_vl_models"]
        return QWenImageCaptioner.MODELS

    @classmethod
    def generate_image_captions(cls,api,model_name,image_url, user_prompt):
        if QWenImageCaptioner.dashscope_module is None:
            try:
                import dashscope
                QWenImageCaptioner.dashscope_module=dashscope
            except Exception as e:
                logging.error(f"未安装python模块dashscope", exc_info=True)
                return {"text":f'f未安装python模块dashscope,{e}',"price":-1,"success":False}
        QWenImageCaptioner.load_models()
        

        filepath =getFullImageFilePath(image_url)
        if not os.path.isfile(filepath): return {"text":f'图片不存在{filepath}',"price":-1,"success":False}

        QWenImageCaptioner.dashscope_module.api_key = api
        messages = [
            {
                "role": "system",
                "content": [{"text":QWenImageCaptioner.DEFAULT_PROMPT}]
            },
            {
                "role": "user",
                "content": [{"image": filepath},{"text": user_prompt}]
            }
        ]

        try:
            response =  QWenImageCaptioner.dashscope_module.MultiModalConversation.call(
                model=model_name,#qwen-vl-plus
                messages=messages
            )
        except Exception as e:
            s=f"访问通问千问时发生错误: {e}"
            if 'InvalidApiKey' in s:
                s=f"访问通问千问时发生错误,API-Key不正确，请检查。\n正确的API-Key例如：sk-3gaa2bcbabba428337fefccdb5449aff\n {e}"
            logging.error(s, exc_info=True)
            return {"text":s,"price":-1,"success":False}
      
        if response.status_code == HTTPStatus.OK:
            raw_prompt = response.output.choices[0].message.content[0]["text"]
            if isinstance(raw_prompt, list):
                raw_prompt = ', '.join(str(item) for item in raw_prompt)
            try:
                price=(response.usage.input_tokens*QWenImageCaptioner.MODELS[model_name]["input_price"]+response.usage.output_tokens*QWenImageCaptioner.MODELS[model_name]["output_price"])/10
            except:
                price=-1
            price=f'{price:.1f}'
            return {"text":raw_prompt,"price":price,"success":True}
        else:
            s=f"访问通问千问时发生未知错误: {response.code} - {response.message}"
            logging.error(s, exc_info=True)
            return {"text":s,"price":-1,"success":False}
        
#智谱  https://www.bigmodel.cn/pricing
class ZhipuAIUtil:
    zhipuai_module=None
    PROMPT_CHAT ='''
你来充当一位有艺术气息的FLUX prompt 助理。
## 任务
我用自然语言告诉你要生成的prompt的主题，你的任务是根据这个主题想象一幅完整的画面，然后生成详细的prompt，包含具体的描述、场景、情感和风格等元素，让FLUX可以生成高质量的图像。
## 背景介绍
FLUX是一款利用深度学习的文生图模型，支持通过使用 自然语言 prompt 来产生新的图像，描述要包含或省略的元素。
## Prompt 格式要求
下面我将说明 prompt 的生成步骤，这里的 prompt 可用于描述人物、风景、物体或抽象数字艺术图画。你可以根据需要添加合理的、但不少于5处的画面细节。
**示例：**
- **输入主题**：一条龙在山脉上空翱翔。
  **生成提示词**：一条威严的、长着翡翠鳞片的龙，眼睛闪烁着琥珀色的光芒，双翼展开，在白雪皑皑的山脉间翱翔。龙那强大的身躯主宰着整个场景，在巍峨的山峰上投下长长的影子。下方，一条瀑布飞流直下，落入深谷，水花在阳光的照耀下呈现出令人眼花缭乱的色彩。龙的鳞片闪烁着彩虹般的色泽，映照着周围的自然美景。天空是充满活力的蓝色，点缀着蓬松的白云，营造出一种敬畏和惊奇的感觉。这幅充满活力和视觉震撼力的画面，捕捉到了龙和山景的雄伟壮丽。
- **输入主题**：解释泡一杯茶的过程。
  **生成提示词**：一张详细的信息图，描绘泡一杯茶的步骤流程。信息图应具有视觉吸引力，插图清晰，文字简洁。它应从装满水的壶开始，到一杯热气腾腾的茶结束，突出显示加热水、选择茶叶、泡茶和享用最终产品等步骤。信息图的设计应既富有信息量又引人入胜，配色方案应与茶的主题相辅相成。文字应清晰易懂，信息丰富，清晰简洁地解释过程中的每个步骤。
**指导**：
1. **描述细节**：尽量提供具体的细节，如颜色、形状、位置等。
2. **情感和氛围**：描述场景的情感和氛围，如温暖、神秘、宁静等。
3. **风格和背景**：说明场景的风格和背景，如卡通风格、未来主义、复古等。
**限制**：
- 我给你的主题可能是用中文描述，你给出的prompt只用中文。
- 不要解释你的prompt，直接输出prompt。
- 不要输出其他任何非prompt字符，只输出prompt，也不要包含 **生成提示词**： 等类似的字符。
'''
    PROMPT_VISION='''
你是一个专业的图像描述专家，能够将图片内容转化为高质量的英文提示词，用于文本到图像的生成模型。
请仔细观察提供的图片，并生成一段详细、具体、富有创造性的英文短语，描述图片中的主体对象、场景、动作、光线、材质、色彩、构图和艺术风格。
要求：
语言：使用中文。
细节：尽可能多地描绘图片细节，包括但不限于物体、人物、背景、前景、纹理、表情、动作、服装、道具等。
角度：尽可能从多个角度丰富描述，例如特写、广角、俯视、仰视等，但不要直接写“角度”。连接：使用逗号（,）连接不同的短语，形成一个连贯的提示词。
人物：描绘人物时，使用第三人称（如 '一个女人', '一个男人'）。
质量词：在生成的提示词末尾，务必添加以下质量增强词：', 最佳品质，高分辨率，4K，高品质，杰作，逼真
'''
    PROMPT_TRANSLATE='''
    你是一个专业的翻译助手。请将用户提供的文本从{}翻译成{}。只输出翻译结果，不要包含任何解释性文字。
    '''

    PROMPT_KONTEXT='''
你来充当一位有艺术气息且擅长命令式指令的 FLUX prompt 助理。
## 任务
你的任务是根据给定的命令式指令生成符合要求的AI图片生成提示词。
## 背景介绍
FLUX 是一款利用深度学习的文生图模型，支持通过自然语言的命令式指令修改图像。
## 生成的 prompt 必须具备：
*   **清晰的动作指令：** （如更换、添加、融合、删除）
*   **场景细节：** （如材质、光影、比例、位置、色彩、姿态）
*   **情绪氛围：** （如自然、浪漫、活力、平静、清新）
*   **风格设定：** （如摄影风、插画风、电影感、产品摄影）
*   **质量词：** 在生成的提示词末尾，添加以下质量增强词：', 最佳品质，高分辨率，4K，高品质，杰作，逼真
## Prompt 示例
### 示例：
**输入内容：** 让图中女人的裙子变成红色。
**生成 prompt：** 将图像中女性裙子的颜色改为鲜红色。确保红色看起来自然，且具有逼真的面料质感、柔和的褶皱和适当的阴影。裙子应与女性的姿势和周围环境无缝融合。画面清晰，质量上乘
**输入内容：** 把图中的草地换成沙漠。
**生成 prompt：** 将图像中的草地替换为一片宽阔、阳光普照的沙漠景观。确保沙子的纹理、颜色和光照都符合现实。环境应显得干燥而广阔，原本在草地上的所有物体现在应正确地立于沙漠表面。画面清晰，质量上乘
## 限制：
*   你给出的 prompt 只用中文。
*   不要解释你的 prompt，直接输出 prompt。
*   不要输出任何非 prompt 字符，不要输出 "生成提示词","请根据以下主题生成命令式中文提示词："等类似内容。
以下为用于修改图片的命令式指令：
'''
    @classmethod
    def load_module(cls):       
        if cls.zhipuai_module is None:
            try:
                from zhipuai import ZhipuAI
                cls.zhipuai_module=ZhipuAI
                return None
            except ImportError:
                logging.warning("未安装 zhipuai 模块，尝试自动安装...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "zhipuai",
                        "-i", "http://mirrors.aliyun.com/pypi/simple/",
                        "--trusted-host", "mirrors.aliyun.com"
                    ])
                    from zhipuai import ZhipuAI
                    cls.zhipuai_module = ZhipuAI
                    return None
                except Exception as e:
                    logging.error("安装 zhipuai 模块失败", exc_info=True)
                    return {"text": "安装 zhipuai 模块失败", "price": -1, "success": False}
            except Exception as e:
                logging.error(f"未安装python的zhipuai模块", exc_info=True)
                return {"text":"未安装python的zhipuai模块","price":-1,"success":False}

    @classmethod
    def chat(cls,api_key, model_name='glm-4-flash-250414',text_input="请扩写关于一只小狗在草地上玩耍的图像生成提示词。",  temperature=0.9, top_p=0.7, max_tokens=1024, seed=0):
        try:
            res=cls.load_module()
            if res:return res
            client = cls.zhipuai_module(api_key=api_key)
        except Exception as e:
            logging.info(f"{e}", exc_info=True)
            return {"text":'智谱AI初始化失败',"price":-1,"success":False}
        try:
            messages = [{"role": "system", "content": ZhipuAIUtil.PROMPT_CHAT},{"role": "user", "content": text_input}]
            response = client.chat.completions.create(model=model_name,messages=messages,temperature=temperature,top_p=top_p,max_tokens=max_tokens,)
            response_text = response.choices[0].message.content
            return {"text":response_text,"price":-1,"success":True}
        except Exception as e:
            logging.info(f"{e}", exc_info=True)
            return {"text":f"调用智谱AI失败: {e}","price":-1,"success":False}
    @classmethod  
    def translate(cls,api_key, model_name='glm-4-flash-250414',text_input="",from_lang="汉语",to_lang="英语",  temperature=0.1, top_p=0.7, max_tokens=1024, seed=0):
        try:
            res=cls.load_module()
            if res:return res
            client = cls.zhipuai_module(api_key=api_key)
        except Exception as e:
            logging.info(f"{e}", exc_info=True)
            return {"text":'智谱AI初始化失败',"price":-1,"success":False}
        try:
            messages = [{"role": "system", "content": ZhipuAIUtil.PROMPT_TRANSLATE.format(from_lang,to_lang)},{"role": "user", "content": text_input}]
            response = client.chat.completions.create(model=model_name,messages=messages,temperature=temperature,top_p=top_p,max_tokens=max_tokens,)
            response_text = response.choices[0].message.content
            return {"text":response_text,"price":-1,"success":True}
        except Exception as e:
            logging.info(f"{e}", exc_info=True)
            return {"text":f"调用智谱AI失败: {e}","price":-1,"success":False}
        
    @classmethod
    def vision(cls,api_key, model_name='glm-4v-flash', image_url=""):
        try:
            res=cls.load_module()
            if res:return res
            client = cls.zhipuai_module(api_key=api_key)
        except Exception as e:
            logging.info(f"{e}", exc_info=True)
            return {"text":'智谱AI初始化失败',"price":-1,"success":False}
        
        filepath =getFullImageFilePath(image_url)
        if not os.path.isfile(filepath):return (f'图片不存在{filepath}',-1)
        with open(filepath, 'rb') as image_file:
            encoded_bytes = base64.b64encode(image_file.read())
            encoded_str = encoded_bytes.decode('utf-8')
        content_parts = [{"type": "text", "text": ZhipuAIUtil.PROMPT_VISION},{"type": "image_url", "image_url": {"url": encoded_str}}]
        try:
            response = client.chat.completions.create(model=model_name,messages=[{"role": "user", "content": content_parts}])
            response_content = str(response.choices[0].message.content)
            return {"text":response_content,"price":-1,"success":True}
        except Exception as e:
            logging.info(f"{e}", exc_info=True)
            return {"text":f"调用智谱AI失败: {e}","price":-1,"success":False}
        
    @classmethod
    def kontext(cls,api_key, model_name='glm-4v-flash', image_url="",user_prompt=""):
        try:
            res=cls.load_module()
            if res:return res
            client = cls.zhipuai_module(api_key=api_key)
        except Exception as e:
            logging.info(f"{e}", exc_info=True)
            return {"text":'智谱AI初始化失败',"price":-1,"success":False}
        
        filepath =getFullImageFilePath(image_url)
        if not os.path.isfile(filepath):return (f'图片不存在{filepath}',-1)
        with open(filepath, 'rb') as image_file:
            encoded_bytes = base64.b64encode(image_file.read())
            encoded_str = encoded_bytes.decode('utf-8')
        content_parts = [{"type": "text", "text": ZhipuAIUtil.PROMPT_KONTEXT+user_prompt},{"type": "image_url", "image_url": {"url": encoded_str}}]
        try:
            response = client.chat.completions.create(model=model_name,messages=[{"role": "user", "content": content_parts}])
            response_content = str(response.choices[0].message.content)
            return {"text":response_content,"price":-1,"success":True}
        except Exception as e:
            logging.info(f"{e}", exc_info=True)
            return {"text":f"调用智谱AI失败: {e}","price":-1,"success":False}
        