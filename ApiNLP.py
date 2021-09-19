from fastapi import FastAPI
import uvicorn, json, os
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mqtt import FastMQTT, MQTTConfig
from Classification import TextClassificationPredict
path_cfg = os.getcwd()
config_file = "{}/config.json".format(path_cfg, )
config = json.loads(open(config_file, 'r').read())

############################################################
#Variables Decleare
############################################################
filename="ApiNLP"
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

############################################################
#MQTT-FastApi Config
############################################################
mqtt_config = MQTTConfig(host      = config['mqtt']['host'],
                         port      = config['mqtt']['port'],
                         keepalive = config['mqtt']['keepalive'],
                         username  = config['mqtt']['username'],
                         password  = config['mqtt']['password'])

mqtt = FastMQTT(
    config=mqtt_config
)

mqtt.init_app(app)

############################################################
#API xử lý model theo phương thức POST
############################################################
@app.get("/modelNLP/{idnode}/{text}")
async def funcModel(idnode: str, text: str): # khai báo dưới dạng parameter
    response = dict()
    try:
        tcp = TextClassificationPredict() # khởi tạo object
        data = tcp.classification(text)
        response.update({"idnode": str(idnode), "result": str(data[0])})
    except:
        response.update({"idnode": str(idnode), "result": "error"})
    finally:
        mqtt.publish(idnode, response['result']) #publishing mqtt topic
    return response

############################################################
#                                       MAIN FUNCTIONS
############################################################ 
if __name__ == "__main__":
    try:
        port = config['port']
        uvicorn.run(app='ApiNLP:app', host="0.0.0.0", port=port, reload=False, debug=True)  
    except:
        uvicorn.run(app='ApiNLP:app', host="0.0.0.0", port=5000, reload=False, debug=True)