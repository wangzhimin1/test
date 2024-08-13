#!/usr/bin/env python
# encoding: utf-8
import base64
import re
import os
import random
import time
import datetime
# import redis
import math
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import struct
from scipy.fftpack import fft, fftshift, fftfreq, ifft, hilbert, ifftshift
import numpy as np
import json
import logging
# import pandas as pd
import warnings
import threading
from scipy.signal import get_window, butter, filtfilt, lfilter, sosfilt
from KafkaProducer import KafkaProducer
import feature_calculate as fc
from scipy import fftpack
from math import ceil, floor
from Logger import logger
warnings.filterwarnings("ignore")
import config
from concurrent.futures import ThreadPoolExecutor
# from get_feature_api import get_signal_feature, get_vibration_signal_feature
# from kafka_sender import kafka_send



# redisClient = redis.StrictRedis(host="127.0.0.1", port=6379, db=10,
                                # decode_responses=True, charset='UTF-8',
                                # password="")

client = mqtt.Client("Client_" + str(round(random.uniform(0, 100), 2) * 100))
timewave_list = []
spectrum_list = []
orbit_list = []
orbit_dict = {}

had_send_wave_dict = dict()  # key=channelId; value=接收时间戳

def fft_filter(signal, lpf1, lpf2, fs):
    """
    频域滤波
    signal: np.ndarray 原始振动信号
    fs: float 采样频率
    lpf1: float 滤波频率-低频
    lpf2: float 滤波频率-高频
    :return
    y: np.ndarray 滤波信号
    """
    yy = fftpack.fft(signal)
    m = len(yy)
    k = m / fs
    for i in range(0, floor(k * lpf1)):
        yy[i] = 0
    for i in range(ceil(k * lpf2 - 1), m):
        yy[i] = 0
    y = 2 * np.real(fftpack.ifft(yy))
    return y

##速度有效值（10-1000Hz）
def vel_pass_rms(wave, fs):
    """""
    wave: 原始加速度信号
    fs: 加速度采样频率
    Return
    v_rms: 速度有效值
    """""
    wave = wave - np.mean(wave)
    data = wave[0 * fs:1 * fs]
    x1 = fft_filter(data, 10, 1000, fs)
    x2, x3 = acc2dis(x1, fs)
    x4 = fft_filter(x2, 10, 1000, fs)
    v_rms = np.sqrt(np.mean(x4 ** 2))
    return v_rms

## 高频加速度有效值（3-10KHz）
def acc_high_rms(wave, fs):
    """""
    wave: 原始加速度信号
    fs: 加速度采样频率
    Return
    a_high_rms: 高频加速度有效值
    """""
    wave = wave - np.mean(wave)
    data = wave[0 * fs:1 * fs]
    x1 = fft_filter(data, 3, 10000, fs)
    a_high_rms = np.sqrt(np.mean(x1 ** 2))
    return a_high_rms

## 低频加速度有效值（3-2KHz）
def acc_low_rms(wave, fs):
    """""
    wave: 原始加速度信号
    fs: 加速度采样频率
    Return
    a_low_rms: 低频加速度有效值
    """""
    wave = wave - np.mean(wave)
    data = wave[0 * fs:1 * fs]
    x1 = fft_filter(data, 3, 2000, fs)
    a_low_rms = np.sqrt(np.mean(x1 ** 2))
    return a_low_rms

# 加速度峰值（3-10KHz）
def calc_acc_p(signal, fs):
    """""
    signal: 振动加速度信号
    fs: 采样频率
    Return
    a_p: 加速度有效值
    """""
    x1 = fft_filter(signal, 3, 10000, fs)
    x2 = sorted(abs(x1), reverse=True)
    x3 = x2[0:100]
    acc_p = np.mean(x3)
    return acc_p

# 速度峰值(3-1KHz)
def calc_vel_p(signal, fs):
    """""
    signal: 振动加速度信号
    fs: 采样频率
    Return
    vel_p: 速度峰值
    """""
    x1 = fft_filter(signal, 10, 1000, fs)
    x2, x3 = acc2dis(x1, fs)
    x4 = sorted(abs(x2), reverse=True)
    x5 = x4[0:100]
    vel_p = np.mean(x5)
    return vel_p

# 振动冲击值（5K~10KHz）
def calc_vibration_impulse(signal, fs):
    """""
    signal: 振动加速度信号
    fs: 采样频率
    Return
    impulse: 振动冲击值
    """""
    x1 = fft_filter(signal, 3000, 5000, fs)
    x2 = hilbert_envelop(x1)
    x3 = fft_filter(x2, 3, 1000, fs)
    # impulse = np.sqrt(np.mean(x3 ** 2))
    return x3
	
def acc2dis(data: np.ndarray, fs: float):
    """
    采用时域积分的方式，将振动加速度信号转化为速度信号和位移信号
    Parameters
    ----------
    data: np.ndarray, 振动加速度信号
    fs: float, 采样频率

    Return
    ------
    s_ifft: np.ndarray, 积分速度信号
    d_ifft：np.ndarray, 积分位移信号
    """
    n = len(data)
    a_mul_dt = data / fs

    s = []
    total = a_mul_dt[0]
    s.append(total)
    for i in range(n - 1):
        total = total + a_mul_dt[i + 1]
        s.append(total)
    s_fft = np.fft.fft(s)
    s_fft[:2] = 0  # 去除直流分量
    s_fft[-3:] = 0  # 去除直流分量
    s_ifft = np.real(np.fft.ifft(s_fft))

    s_mut_dt = s_ifft / fs
    d = []
    total = s_mut_dt[0]
    d.append(total)
    for i in range(n - 2):
        total = total + s_mut_dt[i + 1]
        d.append(total)
    d_fft = np.fft.fft(d)
    d_fft[:2] = 0
    d_fft[-3:] = 0
    d_ifft = np.real(np.fft.ifft(d_fft))
    return s_ifft * 1000, d_ifft * 1000000  # 单位转换

def zero_pad(signal_array):
    """
    将原始信号转换成1024整数倍
    @param signal_array: 原始信号
    @return:
    """
    fftshift_signal = fft(np.array(signal_array))

    N = len(fftshift_signal)
    total_N = math.ceil(math.log2(N))
    num = pow(2, total_N) - N
    # print(num)
    y_tmp_1 = np.lib.pad(fftshift_signal[:int(N / 2)], (0, int(num / 2)), 'constant', constant_values=0)
    y_tmp_2 = np.lib.pad(fftshift_signal[int(N / 2):], (int(num / 2), 0), 'constant', constant_values=0)
    y = np.append(y_tmp_1, y_tmp_2)

    result = ifft(y)
    return list(result.real)


def hilbert_envelop(signal_series):
    """
    输入原始信号，输出包络时域信号
    """
    signal_array = get_array(signal_series)
    hx = hilbert(signal_array)  # 对信号进行希尔伯特变换。
    analytic_signal = signal_array - hx * 1j  # 进行hilbert变换后的解析信号
    hy = np.abs(analytic_signal)  # 计算信号的包络,也就是调制信号的模
    return hy


def on_subscribe(client, userdata, mid, granted_qos):
    """回调函数：当代理响应订阅请求时被调用"""
    print("订阅响应回调函数：", mid)


def on_publish(client, userdata, mid):
    """回调函数：当消息发送出去后"""
    logger.info("消息已经发送：" + str(mid))


def publish(topic, payload, qos=0):
    """
    发送函数
    :param topic: 主题
    :param payload: 发送内容
    :param qos: 消息质量，默认0
    :return:
    """
    client.publish(topic, payload, qos)


def active_trigger(data_watch_no):
    """
    主动触发命令
    :param data_watch_no: 终端号
    :return:
    """
    mes_dict = {}
    mes_dict['cmd'] = '08'
    mes_str = json.dumps(mes_dict, ensure_ascii=False)
    mes_base64 = str(base64.b64encode(mes_str.encode('utf-8')), "utf-8")
    publish(topic='cmd/' + data_watch_no, payload=mes_base64)


def start(host="127.0.0.1", port=1883, username="admin", password="password"):
    while True:
        try:
            client.on_connect = on_connect
            client.on_message = on_message
            client.on_publish = on_publish
            client.on_subscribe = on_subscribe
            client.username_pw_set(username, password)
            logger.info('host=' + host + ';port=' + str(port) + ';username=' + username + ';password=' + password)
            client.connect(host, port, 60)
            client.loop_forever()
        except Exception as e:
            print("Exception:", e)
            logger.error(e)
            # 等待一段时间后尝试重新连接
            time.sleep(5)  # 等待5秒后重新连接


def on_connect(client, userdata, flags, rc):
    """回调函数：连接服务器后触发，rc的值决定了连接成功或者不成功：
           0-成功，1-协议版本错误，2-无效的客户端标识，3-服务器无法使用，4-错误的用户名或密码，5-未经授权"""
    logger.info("Connected with result code :" + str(rc))
    topic_list = [("equipment/realTimeData/+", 0), ("equipment/secondTimeData/+", 0), ('wireless/realTimeData/+', 0),("publish_data", 1)]
    client.subscribe(topic_list)




def cal_rms_peak(signal):
    """
    :param signal:  listhilbert_envelop
            一秒钟信号的list ，长度为采样频率
    :return: tuple
        返回一个元组类型：(rms, peak)， 元组的第一个元素为信号的有效值
        第二个元素为信号的峰值
    """
    signal_array = np.asarray(signal).ravel()
    rms = np.sqrt(np.mean(np.power(signal_array, 2)))
    peak = (max(signal_array) - min(signal_array)) / 2
    return rms, peak


def get_array(array):
    """transform data to numpy.array
    """
    if isinstance(array, np.ndarray):
        if len(array.shape) == 1:
            return array
        elif len(array.shape) == 2 and (array.shape[0] == 1 or array.shape[1] == 1):
            return array.reshape(-1)
        else:
            raise TypeError("The dimension of the numpy.array must be 1 or 2")
    elif isinstance(array, (list, pd.Series)):
        array = np.array(array)
        return get_array(array)
    else:
        raise TypeError("Input must be a numpy.array, list or pandas.Series")


thread_pool = ThreadPoolExecutor(max_workers=50)

def on_message(client, userdata, msg):
    """回调函数：订阅的主题有消息时触发"""
    topic = str(msg.topic).split("/")
    print(msg.topic)
    
    if topic[0] == 'wireless':
        handle_wireless_message(msg)
    elif topic[0] == 'equipment' and topic[1] == 'realTimeData':
        handle_equipment_message(msg)
    elif topic[0] == 'publish_data':
        handle_publish_data_message(msg)

def handle_wireless_message(msg):
    try:
        message_dict = json.loads(msg.payload.decode())
        channelId = message_dict['channelId']
        source_single = list(eval(message_dict['waveData']))[:12800]
        
        kpi_name = "12.8K速度波形(2-5000)"
        kpi_name1 = "12.8K冲击波形(2-5000)"
        logger.info(f"无线波形长度：{len(source_single)};ChannelId={channelId} Kpi_name={kpi_name}; Kpi_name1={kpi_name1}")
        
        if len(source_single) >= 12800:
            thread_pool.submit(formate_and_publish, source_single, channelId, kpi_name)
            thread_pool.submit(evnformate_and_publish, source_single, channelId, kpi_name1)
    except Exception as e:
        logger.error(f"无线异常报错: {e}")

def handle_equipment_message(msg):
    try:
        message = msg.payload
        msg_tup = struct.unpack('<5i', message[:20])
        number = msg_tup[3]
        channelId = struct.unpack('16s', message[20:36])[0].decode('utf-8')
        channelId = re.sub(u'\u0000', "", channelId)
        source_single = list(struct.unpack('<{}f'.format((len(message) - 68) // 4), message[68:]))[0:number]
        
        kpi_name = "32K速度波形(2-12800)"
        kpi_name1 = "32K加速度波形(2-12800)"
        logger.info(f"长度：{len(source_single)}; Kpi_name={kpi_name};")
        
        thread_pool.submit(formate_and_publish, source_single, channelId, kpi_name)
        thread_pool.submit(formatefeatures_and_publish, source_single, channelId, kpi_name1)
    except Exception as e:
        logger.error(f"有线异常报错: {e}")

def handle_publish_data_message(msg):
    try:
        print("123123")
        message_dict = json.loads(msg.payload.decode())
        channelId = message_dict['channelId']
        source_single = message_dict['raw_data']
        
        # 读取 Excel 文件
        excel_data = pd.read_excel('channel_ids.xlsx')
        
        if channelId in excel_data['channelId'].values:
            source_single = (np.array(source_single) - np.mean(source_single)).tolist()
            future = thread_pool.submit(formatefeatures_and_publish, source_single, channelId)
            logger.info(f"处理 publish_data 主题: ChannelId={channelId}, Future={future}")
        else:
            logger.info(f"跳过处理 publish_data 主题: ChannelId={channelId} 未匹配")
    except Exception as e:
        logger.error(f"处理 publish_data 主题时发生异常: {e}")

def formate_and_publish(source_single, channelId, kpi_name):
    try:
        logger.info("formate and ready to publish...")
        source_single_size = len(source_single)
        return_dict = {
            'channelId': channelId,
            'engineering_unit': "mm/s",
            'data_length': source_single_size,
            'raw_data': fc.acc2dis(np.array(fc.fft_filter(source_single, 3, 1000, float(len(source_single)))), float(len(source_single)))[0].tolist()
        }
        host = config.mqtt_publish_host
        port = config.mqtt_publish_port
        topic = 'test1'
        client.connect(host, port, 60)
        payload = json.dumps(return_dict, ensure_ascii=False)
        client.publish(config.mqtt_publish_topic, payload, qos=0)
        logger.info(f"已经发送到速度 MQTT; ChannelId={channelId};")
    except Exception as e:
        logger.error(f"发送 MQTT 消息时出现异常: {e}")

def evnformate_and_publish(source_single, channelId, kpi_name):
    try:
        logger.info("formate and ready to publish...")
        source_single_size = len(source_single)
        return_dict = {
            'channelId': channelId,
            'engineering_unit': "mm/s",
            'data_length': source_single_size,
            'raw_data': calc_vibration_impulse(source_single, 12800).tolist()
        }
        # topic = config.mqtt_publish_topic
        host = config.mqtt_publish_host
        port = config.mqtt_publish_port
        topic = 'test2'
        payload = json.dumps(return_dict, ensure_ascii=False)
        client.connect(host, port, 60)
        client.publish(topic, payload, qos=0)
        logger.info(f"已经发送到 冲击MQTT; ChannelId={channelId};")
    except Exception as e:
        logger.error(f"发送 MQTT 消息时出现异常: {e}")
def test_read_1txt():
    """测试读取 1.txt 文件内容并解析"""
    try:
        with open("1.txt", "r") as f:
            raw_data = json.loads(f.read().replace("data:", ""))
        source_single = raw_data["raw_data"]
        channelId = raw_data["channelId"]
        # print("1.txt 中的 source_single:", source_single)
        print("ChannelId:", channelId)
        formatefeatures_and_publish(source_single, channelId)
    except FileNotFoundError as e:
        logger.error("未找到 1.txt 文件: {}".format(e))
    except json.JSONDecodeError as e:
        logger.error("1.txt 的 JSON 解码错误: {}".format(e))
    except Exception as e:
        logger.error("1.txt 处理异常: {}".format(e))


def formatefeatures_and_publish(source_single, channelId):
    v_rms = vel_pass_rms(source_single, len(source_single))
    a_high_rms = acc_high_rms(source_single, len(source_single))
    a_calc_acc_p = calc_acc_p(source_single, len(source_single))
    v_calc_vel_p = calc_vel_p(source_single, len(source_single))
    logger.info("formate and ready to publish features...")
    
    return_dict = {
        'channelId': channelId,
        'acc_value': a_high_rms,
        'rms_value': v_rms,
        'a_calc_acc_p': a_calc_acc_p,
        'v_calc_vel_p': v_calc_vel_p,
        'DTime' : datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[0:-3]
    }

    topic = config.mqtt_publish_topicfeatures
    host = config.mqtt_publish_hostfeatures
    port = config.mqtt_publish_portfeatures

    payload = json.dumps(return_dict, ensure_ascii=False)

    logger.info(f"准备发送到 MQTT {host}:{port}, 主题: {topic}, payload: {payload}")
    
    try:
        qos = 1
        client.connect(host, port, 60)
        result = client.publish(topic, payload, qos)
        
        # 检查 publish 调用的结果
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info(f"已经发送到 MQTT {host}; ChannelId={channelId}")
        else:
            logger.error(f"发送到 MQTT 失败，返回码: {result.rc}")
            
    except Exception as e:
        logger.error(f"发送 MQTT 消息时出现异常: {e}")

    client.disconnect()

        
def formate_and_publish1(source_single, channelId, kpi_name):
    print("formate and ready to publish1...")
    #print(source_single)
    source_single_size = len(source_single)
    return_dict = dict()
    return_dict['SN'] = ''
    return_dict['DataType'] = "S"
    return_dict['ImplType'] = "N"
    return_dict['Specialty'] = "Z"
    # return_dict['KpiId'] = str(source_single_size)+"速度波形"
    return_dict['KpiId'] = kpi_name
    return_dict['DTime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[0:-3]
    return_dict['SNInfo'] = {
        "HZ": source_single_size,
        "CL": source_single_size
    }


    device_info_path = "./device-info.xlsx"
    if os.path.exists(device_info_path):
            device_df = pd.read_excel(device_info_path, dtype=str)
            if len(device_df[device_df['channelId'] == channelId]):
                return_dict['DevId'] = str(device_df[device_df['channelId'] == channelId]['devId'].to_list()[0])
                print(return_dict['DevId'])
                return_dict['PointId'] = str(device_df[device_df['channelId'] == channelId]['pointId'].to_list()[0])
                length = float(len(source_single))
                # 只返回source_single数据
                return_dict['Value'] = source_single
                # return return_dict  # 返回整个字典
                
                # 发送至 MQTT
                topic = 'test1'
                print(topic)
                payload = json.dumps(return_dict, ensure_ascii=False)
                client.publish(topic, payload, qos)
                
                logger.info("已经发送到 MQTT; ChannelId={};".format(channelId))

            else:
                logger.error("channel: %s 不存在"% channelId)
    else:
        logger.error("设备信息表不存在(device-info)")



def pub_time_waveform(hostname, port, topic):
    payload = {}
    payload['test'] = '测试'
    mes = json.dumps(payload)
    while True:
        try:
            publish.single(topic, mes, qos=0, hostname=hostname, port=port,
                           client_id="test_waveform", auth={'username': "admin", 'password': "password"})
            time.sleep(1)
        except Exception as e:
            print(e)


def get_array(array):
    """transform data to numpy.array
    """
    if isinstance(array, np.ndarray):
        if len(array.shape) == 1:
            return array
        elif len(array.shape) == 2 and (array.shape[0] == 1 or array.shape[1] == 1):
            return array.reshape(-1)
        else:
            raise TypeError("The dimension of the numpy.array must be 1 or 2")
    elif isinstance(array, (list, pd.Series)):
        array = np.array(array)
        return get_array(array)
    else:
        raise TypeError("Input must be a numpy.array, list or pandas.Series")


def read_json_file(path):
    with open(path, 'r') as cs:
        data = json.load(cs)
    return data


if __name__ == '__main__':
   # test_read_1txt()   
   #start(host='47.100.165.139', port=51613, username="chaosadmin", password="k97E@U4DZPLP7rA*")
    # start(host=config.mqtt_subscribe_host, port=config.mqtt_subscribe_port,username = config.mqtt_publish_username,password = config.mqtt_publish_password)
    # start(host='49.4.31.101', port=18833, username="chaos", password="HSD272*#xkd")
    # device_info_path = "./device-info.xlsxx"
    # device_df = pd.read_excel(device_info_path, dtype=str)
    # print(device_df)
   mqtt_thread = threading.Thread(target=start, args=(config.mqtt_subscribe_host, config.mqtt_subscribe_port, config.mqtt_publish_username, config.mqtt_publish_password))
   mqtt_thread.start()


