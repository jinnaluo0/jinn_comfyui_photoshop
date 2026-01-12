import os,json,uuid,traceback,logging,re,time,io,shutil,random,inspect
from datetime import datetime
import requests
from hashlib import md5,sha256,sha1

from PIL import Image,ImageChops
from aiohttp import web


import execution,folder_paths
import comfy.model_management
from server import PromptServer,create_cors_middleware
from nodes import NODE_CLASS_MAPPINGS

from .jinn_utils  import *

#nest_asyncio.apply()
#asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#EXTENSION_WEB_DIRS  WEB_DIRECTORY

#对提示词增加处理器,保存最后一次执行的的工作流API
LAST_RUN_API={"apidata":None,"lasttime":None}
def jinn_prompt_handler(json_data):
    try:
        prompt = json_data["prompt"]
        LAST_RUN_API["apidata"]=prompt
        LAST_RUN_API["lasttime"]=time.time()
    except Exception as e:
        logging.info(f"获取API时出错")
        logging.info(traceback.format_exc())
    return json_data
PromptServer.instance.add_on_prompt_handler(jinn_prompt_handler)

#判断 execution.validate_prompt的两个不同版本：async def validate_prompt(prompt_id, prompt):   async def validate_prompt(prompt):
VALIDATE_PROMPT_PARAM_NUM=1
try:
    sig = inspect.signature(execution.validate_prompt)
    param_count = len(sig.parameters)
    if param_count == 2:VALIDATE_PROMPT_PARAM_NUM=2
except Exception as e:
    logging.info(f"获取execution.validate_prompt参数数量失败")





#获取系统路由
#args.enable_cors_header='http://127.0.0.1:5500'
create_cors_middleware("*")#有用户反馈上传图片失败psareamask.png,403,Forbidden
routes = PromptServer.instance.routes


#获取原服务器响应方法
def find_handler(func,method):
    for route in routes:
        if route.path == func and route.method == method: return route.handler
    return None
view_handler = find_handler("/view","GET")
interrupt_handler = find_handler("/interrupt","POST")
prompt_handler = find_handler("/prompt","POST")

#通用HTTP错误响应
def errorResp(msg,e=None):
    if e:
        logging.info(f"请求过程中出现错误: {e}", exc_info=True)
        #logging.info(traceback.format_exc())
    return web.json_response({"comfy_error": (f'{msg} {str(e)}' if e else msg)})


#设置系统采样器预览方式
'''
def __jinn_set_output_preview(preview):
    if preview:
        args.preview_method = latent_preview.LatentPreviewMethod.Auto
    else:
        args.preview_method = latent_preview.LatentPreviewMethod.NoPreviews
try:
    config=get_config()
    __jinn_set_output_preview(config[OUTPUT_PREVIEW])
except Exception as e :
    logging.info(str(e))
'''   

#清理上传的文件夹
try:
    if os.path.exists(PATH_UPLOAD):shutil.rmtree(PATH_UPLOAD, ignore_errors=True)
    if not os.path.exists(PATH_UPLOAD):os.makedirs(PATH_UPLOAD)
except Exception as e :
    pass
getWfUtil(True).init_workflows()#创建工作流列表，参数目录等
getWfUtil(False).init_workflows()#创建工作流列表，参数目录等

#获取显卡类型
try:
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    is50GPU="GeForce RTX 50" in device_name
except Exception as e :
    is50GPU=False
    pass


def get_hardware_id():
    try:
        #mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(5, -1, -1)])
        b = uuid.getnode().to_bytes(6, byteorder='big')
        s = sha1(b).hexdigest()
        return s
    except Exception as e:
        raise Exception(f"硬件错误")

def generate_signature(user_id, hardware_id, secret_key):
    data_string = f"{user_id}{hardware_id}{secret_key}"
    return sha256(data_string.encode()).hexdigest()


USER_STATE= {
    "last_login": None,  # 或者比如：1728645600.0
    "failed_count":0, #请求服务器失败次数
    "role":""
}

        
    
    






BAIDU_URL = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
BAIDU_HEADERS = {'Content-Type': 'application/x-www-form-urlencoded'}



translation_cache = LRUCacheDict(max_size=50) #翻译缓存
@routes.post("/jinn_translate")
async def jinn_translate(request): 
    try:
        req =  await request.json()
        text,appid,appkey,from_lang,to_lang= req['text'],req['appid'], req['appkey'],req['from_lang'], req['to_lang']
        cache_key = (text, to_lang)
        cached_result = translation_cache.get(cache_key)
        if cached_result is not None:
            return web.json_response({"result":cached_result,"trans":True})
        if to_lang=="en" and not  bool(re.search(r'[\u4e00-\u9fff]', text)):return web.json_response({"result":text,"trans":False})
        if to_lang=="zh" and not  bool(re.search(r'[a-zA-Z]', text)):return web.json_response({"result":text,"trans":False})
        salt = random.randint(32768, 65536)
        sign=appid + text + str(salt) + appkey
        sign = md5(sign.encode('utf-8')).hexdigest()
        params = {'appid': appid, 'q': text, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
        r = requests.post(BAIDU_URL, params=params, headers=BAIDU_HEADERS)
        result = r.json()
        if 'error_code' in result: 
            error_code=result['error_code']
            error_msg=result.get('error_msg',"")
            return errorResp(f"翻译出错 {error_code} {error_msg}")
        lst=[]
        for dic in result['trans_result']:
            lst.append(dic['dst'])
        transText='\n'.join(lst)
        translation_cache.put(cache_key, transText)
        return web.json_response({"result":transText,"trans":True})
    except Exception as e:
        return errorResp("翻译出错",e)
    

#语音
@routes.post("/jinn_sound_to_text")
async def jinn_sound_to_text(request): 
    try:
        req =  await request.json()
        duration,modelName = int(req['duration']),req['model_name']
        text=soundToText(duration,modelName)
        return web.json_response({"result":text})
    except Exception as e:
        return errorResp("获取语音时出错",e)
    

#千问
@routes.post("/jinn_cloud_qwen_vl")
async def jinn_cloud_qwen_vl(request):
    try:
        req =  await request.json()
        api,model_name,image_url,user_prompt= req['api'],req['model_name'],req['image_url'],req['user_prompt']
        data= QWenImageCaptioner.generate_image_captions(api,model_name,image_url, user_prompt)
        return web.json_response(data)
    except Exception as e :
        return errorResp(f'Error{e}',e)
    
#智谱
@routes.post("/jinn_cloud_zhipu_chat")
async def jinn_cloud_zhipu_chat(request):
    try:
        req =  await request.json()
        api,model_name,user_prompt= req['api'],req['model_name'],req['user_prompt']
        data= ZhipuAIUtil.chat(api,model_name, user_prompt)
        return web.json_response(data)
    except Exception as e :
        return errorResp(f'Error{e}',e)
@routes.post("/jinn_cloud_zhipu_translate")
async def jinn_cloud_zhipu_translate(request):
    try:
        req =  await request.json()
        api,model_name,user_prompt,from_lang,to_lang= req['api'],req['model_name'],req['user_prompt'],req['from_lang'],req['to_lang']
        data= ZhipuAIUtil.translate(api,model_name, user_prompt,from_lang,to_lang)
        return web.json_response(data)
    except Exception as e :
        return errorResp(f'Error{e}',e)

@routes.post("/jinn_cloud_zhipu_vision")
async def jinn_cloud_zhipu_vision(request):
    try:
        req =  await request.json()
        api,model_name,image_url= req['api'],req['model_name'],req['image_url']
        data= ZhipuAIUtil.vision(api,model_name, image_url)
        return web.json_response(data)
    except Exception as e :
        return errorResp(f'Error{e}',e)
@routes.post("/jinn_cloud_zhipu_kontext")
async def jinn_cloud_zhipu_kontext(request):
    try:
        req =  await request.json()
        api,model_name,image_url,user_prompt= req['api'],req['model_name'],req['image_url'],req['user_prompt']
        data= ZhipuAIUtil.kontext(api,model_name, image_url,user_prompt)
        return web.json_response(data)
    except Exception as e :
        return errorResp(f'Error{e}',e)

#工作流分组
@routes.get("/jinn_wf_group_list")
async def jinn_wf_group_list(request): 
    try:
        #req =  await request.json()
        items=wfGroupUtil.get_groups()['items']
        return web.json_response(items)
    except Exception as e:
        return errorResp("获取工作流分组时出错",e)
    
@routes.post("/jinn_wf_group_add")
async def jinn_wf_group_add(request): 
    try:
        req =  await request.json()
        wfgroupname= req['wfgroupname']
        wfGroupUtil.add_group(wfgroupname)
        return web.json_response({"result":True})
    except Exception as e:
        return errorResp("添加工作流分组时出错",e)
@routes.post("/jinn_wf_group_edit")
async def jinn_wf_group_edit(request): 
    try:
        req =  await request.json()
        wfgroupid, wfgroupname= req['wfgroupid'], req['wfgroupname']
        wfGroupUtil.edit_group(wfgroupid, wfgroupname)
        return web.json_response({"result":True})
    except Exception as e:
        return errorResp("修改工作流分组时出错",e)
@routes.post("/jinn_wf_group_del")
async def jinn_wf_group_del(request): 
    try:
        req =  await request.json()
        wfgroupid, wfgroupname= req['wfgroupid'], req['wfgroupname']
        wfGroupUtil.del_group(wfgroupid)
        return web.json_response({"result":True})
    except Exception as e:
        return errorResp("删除工作流分组时出错",e)
    
@routes.post("/jinn_wf_group_sort")#排序工作流
async def jinn_wf_group_sort(request): 
    try:
        req =  await request.json()
        wfid1,wfid2,wfgroupid1,wfgroupid2,wfcode= req['wfid1'],req['wfid2'],req['wfgroupid1'],req['wfgroupid2'],req['wfcode']
        if wfid1=="-1" and wfid2=="-1":#分组到分组
            wfGroupUtil.sort(wfgroupid1,wfgroupid2)
        elif wfid2=="-1":#工作流到分组
            getWfUtil(wfcode).edit_workflows_group(wfid1,wfgroupid2)
        else:#工作流到工作流
            getWfUtil(wfcode).edit_workflows_group_sort(wfid1,wfid2,wfgroupid2)
        return web.json_response({"success":True})
    except Exception as e:
        return errorResp("排序工作流时出错",e)
    
#节点参数
@routes.get("/jinn_get_wf_nodefield_info/{class_type}/{field}")
async def jinn_get_wf_nodefield_info(request): 
    try:
        #http://127.0.0.1:8188/jinn_get_wf_nodefield_info/KSampler/sampler_name
        #http://127.0.0.1:8188/jinn_get_wf_nodefield_info/LoraLoader/lora_name
        class_type = request.match_info.get("class_type", None)
        field = request.match_info.get("field", None)
        if not class_type or not field:return errorResp("获取节点字段信息时出错")
        obj_class = NODE_CLASS_MAPPINGS[class_type]
        info=obj_class.INPUT_TYPES()
        for k,v in info.items():
            if field in v:fieldinfo=v[field];break
        else: return web.json_response({})
        if type(fieldinfo)==tuple and fieldinfo[0]=='COMBO' and type(fieldinfo[1]['options'])==list:
            fieldinfo=(fieldinfo[1]['options'],'COMBO')
        return web.json_response(fieldinfo)
    except Exception as e:
        return errorResp("获取节点字段信息时出错",e)
    
@routes.get("/jinn_get_all_wf_nodefield_info") #没有被调用
async def jinn_get_wf_nodefield_info(request): 
    try:
        #http://127.0.0.1:8188/jinn_get_all_wf_nodefield_info
        result=set()
        for obj_class in NODE_CLASS_MAPPINGS.values():
            try:
                info=obj_class.INPUT_TYPES()
                for k,v in info.items():
                    for field,fieldinfo in v.items():
                        fieldinfocls=type(fieldinfo)
                        if fieldinfocls==str:pass
                        field0cls=type(fieldinfo[0])
                        if field0cls==str: #字段类型
                            result.add(fieldinfo[0])
                        elif field0cls==list or field0cls==tuple :#字段枚举值
                            pass#result.add(str(field0cls))
                        else:#字段类型
                            logging.info(f'{str(field0cls)} {str(obj_class)} {str(field)}')
                        
            except : pass
        
        return web.json_response(list(result))
    except Exception as e:
        return errorResp("获取节点字段信息时出错",e)

@routes.get("/jinn_test")
async def jinn_test(request): 
    try:
        buildin = os.environ.get('JINN_BUILDIN')
        return web.json_response({"result":True,"role":"ADMIN","buildin":buildin=="1"})
    except Exception as e:
        return errorResp("出错",e)


@routes.post("/jinn_interrupt")
async def jinn_interrupt(request): 
    try:
        await interrupt_handler(request)
        return web.json_response({"result":True})
    except Exception as e:
        return errorResp("出错",e)
    
@routes.post("/jinn_prompt_result_text_byid")
async def jinn_prompt_result_text_byid(request): 
    try:
        req =  await request.json()
        prompt_id,nodeid,field = req['prompt_id'],req['nodeid'],req['field']


        history=PromptServer.instance.prompt_queue.get_history(prompt_id=prompt_id)
        #logging.info(str(history[prompt_id]["outputs"]))
        res=[]
        text = history[prompt_id]["outputs"][nodeid][field]
        if isinstance(text,list):text=text[0]
        elif isinstance(text,str):pass
        else:text=str(text)
        
        return web.json_response({"result":text})
    except Exception as e:
        return errorResp("获取结果图片时出错",e)
    
@routes.post("/jinn_prompt_result_byid")
async def jinn_prompt_result_byid(request): 
    try:
        req =  await request.json()
        prompt_id,nodeid,field = req['prompt_id'],req['nodeid'],req['field']
        history=PromptServer.instance.prompt_queue.get_history(prompt_id=prompt_id)
        #logging.info(str(history[prompt_id]["outputs"]))

        '''
        for i in range(5):
            try:
                images = history[prompt_id]["outputs"][nodeid]["images"]
            except:
                time.sleep(1)
        '''
        res=[]
        images = history[prompt_id]["outputs"][nodeid][field]
        for image in images:
            res.append({"filename": image["filename"], "subfolder": image["subfolder"], "type": image["type"]})
        return web.json_response(res)
    except Exception as e:
        return errorResp("获取结果图片时出错",e)
@routes.get("/jinn_get_wf_params")
async def jinn_get_wf_params(request): 
    try:
        wfname = request.query.get("wfname", None) 
        if not wfname:return errorResp("获取工作流条件时出错")
        wfcode = request.query.get("wfcode", "") 
        params=getWfUtil(wfcode).get_params(wfname)
        return web.json_response(params)
    except Exception as e:
        return errorResp("获取工作流条件时出错",e)
    
@routes.get("/jinn_get_wf_api_run")
async def jinn_get_wf_api_run(request): #获取最近一次执行的工作流API
    try:
        apidata=LAST_RUN_API["apidata"]
        if not apidata:return web.json_response({"result":False})
        mtime=LAST_RUN_API["lasttime"]
        lasttime = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        if time.time()-mtime>600:lasttime=lasttime+"(旧)"
        #如果apidata中包含NaN时，会报错
        #"20": {"inputs": {"image": "U\u76d8-\u84dd\u8272.jpg"}, "class_type": "LoadImage", "_meta": {"title": "\u52a0\u8f7d\u56fe\u50cf"}, "is_changed": NaN
        repairWfAPI(apidata)
        return web.json_response({"result":True,"apidata":apidata,"lasttime":lasttime})
    except Exception as e:
        return errorResp("获取工作流API时出错",e)
    

@routes.get("/jinn_get_wf_api")
async def jinn_get_wf_api(request): 
    try:
        wfname = request.query.get("wfname", None) 
        if not wfname:return errorResp("获取工作流条件时出错")
        wfcode = request.query.get("wfcode", "") 
        api=getWfUtil(wfcode).get_api(wfname)
        return web.json_response(api)
    except Exception as e:
        return errorResp("获取工作流API时出错",e)

@routes.get("/jinn_get_workflows_custom")
async def jinn_get_workflows_custom(request): 
    try:
        workflows=getWfUtil(False).get_workflows()
        return web.json_response(workflows)
    except Exception as e:
        return errorResp("获取工作流列表时出错",e)
    
@routes.get("/jinn_get_workflows_buildin")
async def jinn_get_workflows_buildin(request): 
    try:
        workflows=getWfUtil(True).get_workflows()
        return web.json_response(workflows)
    except Exception as e:
        return errorResp("获取工作流列表时出错",e)
    


@routes.get("/jinn_get_workflows_i2t")
async def jinn_get_workflows_i2t(request): 
    try:
        workflows=getWfUtil(False).get_workflows_t2i()
        workflows_buildin=getWfUtil(True).get_workflows_t2i()
        return web.json_response(workflows+workflows_buildin)
    except Exception as e:
        return errorResp("获取工作流列表时出错",e)
    
@routes.get("/jinn_get_workflows_recover_custom")
async def jinn_get_workflows_recover_custom(request): #获取待恢复的工作流
    try:
        workflows=getWfUtil(False).get_workflows_recover()
        return web.json_response(workflows)
    except Exception as e:
        return errorResp("",e)
@routes.get("/jinn_get_workflows_recover_buildin")
async def jinn_get_workflows_recover_buildin(request): #获取待恢复的工作流
    try:
        workflows=getWfUtil(True).get_workflows_recover()
        return web.json_response(workflows)
    except Exception as e:
        return errorResp("",e)
@routes.get("/jinn_get_workflows_removed_buildin")
async def jinn_get_workflows_removed_buildin(request): #
    try:
        workflows=getWfUtil(True).get_removed()
        return web.json_response(workflows)
    except Exception as e:
        return errorResp("",e)
    
@routes.post("/jinn_workflows_recover")#恢复工作流
async def jinn_workflows_recover(request): 
    try:
        data =  await request.json()
        wfname, wfid, wfcode = data['wfname'], data['wfid'], data['wfcode']
        getWfUtil(wfcode).workflows_recover(wfname,wfid)
        return web.json_response({"success":True})
    except Exception as e:
        return errorResp("",e)





@routes.post("/jinn_workflows_visible")#显示/隐藏工作流
async def jinn_workflows_visible(request): 
    try:
        req =  await request.json()
        wfid,wfcode,visible= req['wfid'],req['wfcode'],req['visible']
        getWfUtil(wfcode).workflows_visible(wfid,visible)
        return web.json_response({"success":True})
    except Exception as e:
        return errorResp("显示/隐藏工作流时出错",e)




@routes.post("/jinn_save_wf_new_run")
async def jinn_save_wf_new_run(request): 
    try:
        req =  await request.json()
        apidata,wfname,wfcode,params,wfidrel,wfgroupid = req['apidata'],req['wfname'],req['wfcode'],req['params'],req.get("wfidrel",""),req.get("wfgroupid","")
        if getWfUtil(wfcode).workflow_exists(wfname):return errorResp("工作流名称已存在")
        wfid=getWfUtil(wfcode).add_workflows(wfname,wfcode,apidata,params,wfidrel,wfgroupid)#先添加，先生成ID,冗余保存output_kind到db文件中，方便过滤出反推工作流
        
        LAST_RUN_API["apidata"]=None
        LAST_RUN_API["lasttime"]=None

        return web.json_response({"success":True,"wfid":wfid})
    except Exception as e:
        return errorResp("添加工作流时出错",e)
    
@routes.post("/jinn_save_wf_edit")
async def jinn_save_wf_edit(request): 
    try:
        req =  await request.json()
        wfnamenew,wfnameold,wfcode,params,apidata,wfidrel,wfgroupid = req['wfnamenew'],req['wfname'],req['wfcode'],req['params'],req['apidata'],req.get("wfidrel",""),req.get("wfgroupid","")
        if not getWfUtil(wfcode).workflow_exists(wfnameold):return errorResp("工作流不存在")
        if apidata:
            getWfUtil(wfcode).save_api(apidata,wfnameold)#修改时时替换了api文件，会传入apidata
            LAST_RUN_API["apidata"]=None
            LAST_RUN_API["lasttime"]=None
        if wfnameold!=wfnamenew and getWfUtil(wfcode).workflow_exists(wfnamenew):return errorResp(f"名称'{wfnamenew}'已存在")
        getWfUtil(wfcode).edit_workflows(wfnameold,wfnamenew,wfcode,params["output"]["kind"],wfidrel,wfgroupid)
        getWfUtil(wfcode).save_params(wfnamenew,params)#修改条件
        return web.json_response({"success":True})
    except Exception as e:
        return errorResp("修改工作流时出错",e)
    
@routes.post("/jinn_set_output_preview")
async def jinn_set_output_preview(request): 
    try:
        req =  await request.json()
        preview = req['preview']
        #__jinn_set_output_preview(preview)
        #set_config(OUTPUT_PREVIEW,preview)
        return web.json_response({"success":True})
    except Exception as e:
        return errorResp("修改预览方式时出错",e)
    



@routes.post("/jinn_save_wf_copy")
async def jinn_save_wf_copy(request): 
    try:
        req =  await request.json()
        wfname =req['wfname'];wfcode =req['wfcode']
        wfid=getWfUtil(False).copy_workflows(wfname,wfcode)
        return web.json_response({"success":True,"wfid":wfid})
    except Exception as e:
        return errorResp("复制工作流时出错",e)
    
@routes.post("/jinn_save_wf_del")
async def jinn_save_wf_del(request): 
    try:
        req =  await request.json()
        wfname =req['wfname'];wfcode =req['wfcode']
        if not getWfUtil(wfcode).workflow_exists(wfname):return errorResp("工作流不存在")
        getWfUtil(wfcode).del_workflows(wfname)
        return web.json_response({"success":True})
    except Exception as e:
        return errorResp("删除工作流时出错",e)

@routes.get("/jinn_get_queue/{prompt_id}")
async def jinn_get_queue(request):
    prompt_id = request.match_info.get("prompt_id", None)
    if not prompt_id:return errorResp("no prompt_id")
    #self.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))
    currqueue,queue = PromptServer.instance.prompt_queue.get_current_queue()
    for item in currqueue: 
        if item[1]==prompt_id:
            return web.json_response({"queue_index":0})
    for i in range(len(queue)): 
        if queue[i][1]==prompt_id:
            return web.json_response({"queue_index":len(currqueue)+i})
    else:
        return web.json_response({"queue_index":-1})


@routes.post("/jinn_upload_mask")
async def jinn_upload_mask(request):
    post = await request.post()
    image = post.get("image")
    maskfilename = post.get("maskfilename");maskfolder = post.get("maskfolder")
    imagefilename = post.get("imagefilename");imagefolder = post.get("imagefolder")
    minsize=int(post.get("minsize", "-1"));maxsize=int(post.get("maxsize", "-1"))
    bounds=post.get("bounds",None)
    if not image:return errorResp(f'没有上传蒙板数据')
    if not maskfilename or maskfilename=='blob':return errorResp(f'没有提供蒙板文件名')
    if not imagefilename or imagefilename=='blob':return errorResp(f'没有提供图片文件名')

    inputdir=folder_paths.get_input_directory()
    maskfiledir = os.path.join(inputdir, os.path.normpath(maskfolder))
    if not os.path.exists(maskfiledir):os.makedirs(maskfiledir)
    maskfilename,maskfilepath = recreateFileName(maskfiledir,maskfilename)
    imagefilepath = os.path.join(inputdir, imagefolder, imagefilename)
    if not os.path.isfile(imagefilepath):return web.Response(status=403)
    # 初始尝试调整图片大小，如果不需要调整（或者调整失败），则使用原始图片
    image = Image.open(io.BytesIO(image.file.read()))
    if bounds:
        left, top, right, bottom = map(int, bounds.split(','))
        image = image.crop((left, top, right, bottom))

    if maxsize > 0 or minsize > 0:
        image = await resize_image(image, minsize, maxsize)
    else:
        image = image.convert('RGBA')

    image = ImageChops.invert(image.convert('L'))#灰度转换取反
    with Image.open(imagefilepath) as original_pil:
        #metadata = PngInfo()
        if image.size != original_pil.size:image = image.resize(original_pil.size, Image.LANCZOS)
        original_pil = original_pil.convert('RGBA')
        original_pil.putalpha(image)
        original_pil.save(maskfilepath, compress_level=4)

    return web.json_response({"name" : maskfilename, "subfolder": maskfolder, "type": 'input'})
@routes.post("/jinn_upload_image")
async def jinn_upload_image(request):
    try:
        post = await request.post()
        image = post.get("image");filename = post.get("filename");subfolder = post.get("subfolder", "")
        bounds=post.get("bounds",None)#上传局部图像时，会传入bounds
        minsize=int(post.get("minsize", "-1"));maxsize=int(post.get("maxsize", "-1"))
        if not image:return errorResp(f'没有上传图片数据')
        if not filename or filename=='blob':return errorResp(f'没有提供图片文件名')
        inputdir=folder_paths.get_input_directory()
        filedir = os.path.join(inputdir, os.path.normpath(subfolder))
        filepath = os.path.abspath(os.path.join(filedir, filename))
        if os.path.commonpath((inputdir, filepath)) != inputdir:return errorResp(f'路径不正确')
        if not os.path.exists(filedir):os.makedirs(filedir)
        filename,filepath= recreateFileName(filedir,filename)
        if bounds:
            left, top, right, bottom = map(int, bounds.split(','))
            image = Image.open(io.BytesIO(image.file.read()))
            image = image.crop((left, top, right, bottom))
            if maxsize>0 or minsize>0:
                resized_img=await resize_image(image,minsize,maxsize)
                resized_img.save(filepath, format="PNG")
            else:
                image.save(filepath, format="PNG")
        else:
            if maxsize>0 or minsize>0:
                image = Image.open(io.BytesIO(image.file.read()))
                resized_img=await resize_image(image,minsize,maxsize)
                resized_img.save(filepath, format="PNG")
            else:
                with open(filepath, "wb") as f:f.write(image.file.read())
                newfilename=convertSpecialImage(filepath,filename)
                if newfilename:filename=newfilename
    

        return web.json_response({"name" : filename, "subfolder": subfolder, "type": 'input'})
    except Exception as e:
        return errorResp("上传图片时出错",e)
    
@routes.post("/jinn_upload_image_live")
async def jinn_upload_image_live(request):
    try:
        post = await request.post()
        image = post.get("image");filename = post.get("filename");subfolder = post.get("subfolder", "")

        inputdir=folder_paths.get_input_directory()
        filedir = os.path.join(inputdir, os.path.normpath(subfolder))
        filepath = os.path.abspath(os.path.join(filedir, filename))
        if not os.path.exists(filedir):os.makedirs(filedir)
        img = Image.open(io.BytesIO(image.file.read()))
        width, height = img.size
        left_half_img = img.crop((0, 0, width // 2, height))
        #检查新图像与原有图像的差异
        left_half_img.save(filepath, format="JPEG", quality=75)#quality的有效范围是1到95，75为默认值
        return web.json_response({"name" : filename, "subfolder": subfolder, "type": 'input'})
    except Exception as e:
        return errorResp("上传图片时出错",e)
    


@routes.post("/jinn_prompt_byid")
async def jinn_prompt_byid(request):
    try:
        req =  await request.json()
        client_id,wfname,wfcode,params,debug= req['client_id'],req['wfname'],req['wfcode'],req['params'],req['debug']
        if debug:
            logging.info(wfname)
            logging.info(params)
        prompt=getWfUtil(wfcode).get_api(wfname)
        prompt=JinnComfyWorkflowWrapper(prompt)
        for param in params['params']:
            try:
                value=param['valueen'] if param['kind']=="PROMPTTEXT" else param['value']
                prompt.set_node_param_byid(param['nodeid'],param['field'],value)
            except Exception as e :
                return errorResp(f"节点或字段不存在，请检查工作流条件 nodeid={param['nodeid']} field={param['field']}")
        change_params(prompt,wfcode,is50GPU)#50系显卡 nunchaku的模型要用fp4的
        resp=await prompt_handler(FakeRequest({"prompt":prompt,"client_id":client_id}))
        getWfUtil(wfcode).save_params(wfname,params)
        return resp
    except Exception as e :
        return errorResp(f'Error{e}',e)




@routes.post("/jinn_prompt_preview")
async def jinn_prompt_preview(request):
    try:
        req =  await request.json()
        images= req['images']
        inputdir=folder_paths.get_input_directory()
        loaded_images = []
        for filePath in images:
            imagefilepath = os.path.join(inputdir, filePath)
            try:
                image = Image.open(imagefilepath).convert('RGBA')
                loaded_images.append(image)
            except Exception as e:
                return  errorResp(f'加载图片失败{imagefilepath}')

        # Calculate dimensions
        gap = 10  # pixels between images
        max_height = max(img.size[1] for img in loaded_images)
        total_width = sum(img.size[0] for img in loaded_images) + gap * (len(loaded_images) - 1)
        
        # Create new image
        result = Image.new('RGBA', (total_width, max_height), (0, 0, 0, 0))
        
        # Paste images with gap
        current_x = 0
        for img in loaded_images:
            result.paste(img, (current_x, 0))
            current_x += img.size[0] + gap
        
        # Resize to 500px height maintaining aspect ratio
        aspect_ratio = result.size[0] / result.size[1]
        new_width = int(500 * aspect_ratio)
        result = result.resize((new_width, 500), Image.Resampling.LANCZOS)
        
        # Save to bytes buffer
        output_buffer = io.BytesIO()
        result.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        return web.Response(body=output_buffer.read(), content_type='image/png')# headers={"Content-Disposition": f"filename=\"{filename}\""}
    except Exception as e :
        return errorResp(f'Error{e}',e)


async def responseImage(request,width,height):
    response= await view_handler(request)
    response.headers['X-Image-Width'] = str(width)
    response.headers['X-Image-Height'] = str(height)
    return response
@routes.get("/jinn_view")
async def jinn_view(request):
    #
    try:
        needSize = request.rel_url.query["zoom"]
        if not needSize:return await responseImage(request,100,200) #不需要缩放
        docwidth = int(request.rel_url.query["docwidth"])
        docheight = int(request.rel_url.query["docheight"])
        

        if "filename" not in request.rel_url.query:return errorResp(f'没有提供filename')
        filename = request.rel_url.query["filename"]
        filename,output_dir = folder_paths.annotated_filepath(filename)
        if not filename:return errorResp(f'filename不正确{filename}')
        if filename[0] == '/' or '..' in filename:return errorResp(f'filename不正确{filename}')

        if output_dir is None:
            type = request.rel_url.query.get("type", "output")
            output_dir = folder_paths.get_directory_by_type(type)

        if output_dir is None: return errorResp(f'output_dir不正确{output_dir}')

        if "subfolder" in request.rel_url.query:
            full_output_dir = os.path.join(output_dir, request.rel_url.query["subfolder"])
            if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:return errorResp(f'output_dir不合法{output_dir}')
            output_dir = full_output_dir

        filename = os.path.basename(filename)
        file = os.path.join(output_dir, filename)
        if not os.path.isfile(file):return errorResp(f'没有找到图片{file}')
        
        img = Image.open(file)
        img_width, img_height = img.size
        return await responseImage(request,img_width,img_height)  #图片尺寸合格，不需要缩放

        
    except Exception as e :
        return errorResp(f'放大结果图片失败,{e}',e)
