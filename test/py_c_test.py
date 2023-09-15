#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   py_c_test.py
@Time    :   2023/09/15 03:00:13
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   py 调用测试
"""

# here put the import lib

import requests

url = "http://127.0.0.1:30025/uploads"

payload={}
files=[
  ('image',('vlcsnap-2023-08-10-02h42m20s009.png',open('C:\\Users\\liruilong\\Pictures\\vlcsnap-2023-08-10-02h42m20s009.png','rb'),'image/png'))
]
headers = {
  'Authorization': 'token'
}


response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)